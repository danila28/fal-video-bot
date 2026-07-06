"""Generation flow: idea → enhanced prompt → image → video prompt → video.

Pipeline router lives in `bot.handlers.common.generate_video_for_model`.
This module is concerned with the Telegram UI state machine only.
"""

import asyncio
import html
import logging
import os
import time

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, FSInputFile, InputMediaPhoto, Message
from bot.guard import IsAllowed
from bot.keyboards import (
    get_image_keyboard,
    get_persistent_keyboard,
    get_prompt_keyboard,
    get_publish_keyboard,
    get_video_keyboard,
    get_video_prompt_keyboard,
    GENERATE_BUTTON_TEXT,
)
from bot.states import GenerationState
from services.db import DBService
from services.gemini import GeminiService
from services.imagegen import ImageGenService
from services.kling import KlingService
from utils import container
from utils.consts import allowed_users, generate_hashtags_prompt
from utils.presets import DEFAULT_PLOT_PROMPT
from utils.tg import send_long_message
from bot.handlers.common import (
    DEFAULT_TARGET_DURATION,
    _T2V_VIDEO_MODELS,
    _build_image_prompt,
    _build_video_prompt,
    _format_video_prompt_message,
    _is_generating,
    _post_process_video,
    _show_video_prompt,
    _split_video_prompt,
    _substitute_prompt_vars,
    generate_video_for_model,
)

router = Router()
logger = logging.getLogger(__name__)


async def _send_image_preview(message, image_paths: list[str]) -> None:
    """Send generated images as a media group (album) or single photo."""
    valid = [p for p in image_paths if os.path.exists(p)]
    if not valid:
        return
    if len(valid) == 1:
        await message.answer_photo(FSInputFile(valid[0]), caption="image")
        return
    media = [
        InputMediaPhoto(
            media=FSInputFile(p),
            caption=f"image {i + 1}/{len(valid)}" if i == 0 else None,
        )
        for i, p in enumerate(valid)
    ]
    await message.answer_media_group(media=media)


# ─────────────────────────────────────────────
# ENTRY POINTS
# ─────────────────────────────────────────────

async def _begin_generation(message: Message, state: FSMContext):
    """Validate settings and start the generation flow."""
    if await _is_generating(state):
        await message.answer("⚠️ Generation in progress — please wait until it finishes.")
        return

    db = container.inject(DBService)

    # Models and style prompts all have working built-in defaults — the bot
    # generates out of the box with zero setup. Only publish accounts are required.
    chat_accounts = await db.get_chat_accounts(message.chat.id)
    if not chat_accounts:
        await message.answer("No accounts configured for this chat\nTap ⚙️ Settings → 📤 Accounts")
        return

    await state.clear()
    await message.answer("Send your idea 💡")
    await state.set_state(GenerationState.RAW_PROMPT)


@router.message(Command("start"), IsAllowed(allowed_users))
async def handle_start(message: Message, state: FSMContext):
    await message.answer(
        "Welcome! Tap 🎬 Generate video to start or ⚙️ Settings to configure the bot.",
        reply_markup=get_persistent_keyboard(),
    )
    await _begin_generation(message, state)


@router.message(F.text == GENERATE_BUTTON_TEXT, IsAllowed(allowed_users))
async def handle_generate_button(message: Message, state: FSMContext):
    await _begin_generation(message, state)


# ─────────────────────────────────────────────
# PROMPT FLOW
# ─────────────────────────────────────────────

@router.message(GenerationState.RAW_PROMPT, IsAllowed(allowed_users))
async def handle_raw_prompt(message: Message, state: FSMContext):
    try:
        gemini = container.inject(GeminiService)
        db = container.inject(DBService)
        settings = await db.get_settings(message.from_user.id, message.chat.id)
        system_prompt = _substitute_prompt_vars(
            settings.get("system_plot_prompt") or DEFAULT_PLOT_PROMPT,
            settings
        )
        enhance_prompt = await gemini.generate_text(
            message.text,
            system_prompt,
            settings.get("text_model") or "",
        )
        await state.update_data(raw_prompt=message.text, enhance_prompt=enhance_prompt)
        await send_long_message(
            message,
            f"Your prompt: {enhance_prompt}\nIf you want to improve it, send a corrected version or click Continue",
            keyboard=get_prompt_keyboard(),
        )
        await state.set_state(GenerationState.ENHANCE_PROMPT)
    except Exception as e:
        logger.error(f"Error enhancing prompt: {e}")
        await message.answer(f"❌ Error enhancing prompt: {e}\nTry again or send a different idea.")
        await state.set_state(GenerationState.RAW_PROMPT)


@router.message(GenerationState.ENHANCE_PROMPT, IsAllowed(allowed_users))
async def handle_edit_prompt(message: Message, state: FSMContext):
    await state.update_data(raw_prompt=message.text, enhance_prompt=message.text)
    await send_long_message(
        message,
        f"Your prompt: {message.text}\nIf you want to improve it, send a corrected version or click Continue",
        keyboard=get_prompt_keyboard(),
    )
    await state.set_state(GenerationState.ENHANCE_PROMPT)


@router.callback_query(GenerationState.ENHANCE_PROMPT, lambda c: c.data == "prompt_ok", IsAllowed(allowed_users))
async def handle_prompt_ok(callback: CallbackQuery, state: FSMContext):
    """Prompt OK → for T2V models skip image and go straight to video prompt;
    for I2V models generate image and show for review."""
    await callback.answer()
    await callback.message.edit_reply_markup(reply_markup=None)

    db = container.inject(DBService)
    settings = await db.get_settings(callback.from_user.id, callback.message.chat.id)
    video_model = settings.get("video_model") or ""

    if video_model in _T2V_VIDEO_MODELS:
        # ── T2V: no image needed — jump straight to video prompt ──────────────
        await state.update_data(image_paths=[], image_path=None, image_prompt="")
        await state.update_data(generation_in_progress=True, generation_started_at=time.time())
        await callback.message.answer("⏳ Generating video prompt…")
        try:
            gemini = container.inject(GeminiService)
            data = await state.get_data()
            enhance_prompt = data.get("enhance_prompt", "")
            text_model = settings.get("text_model") or ""
            video_prompt, hashtags = await asyncio.gather(
                _build_video_prompt(enhance_prompt, settings, gemini),
                gemini.generate_text(enhance_prompt, generate_hashtags_prompt, text_model),
            )
            scene, voiceover = _split_video_prompt(video_prompt, gemini)
            parsed_vp = GeminiService.parse_script_json(video_prompt)
            if parsed_vp and parsed_vp.get("caption"):
                hashtags = parsed_vp["caption"]
            await state.update_data(
                video_scene=scene,
                video_voiceover=voiceover,
                video_script_json=video_prompt,
                cached_hashtags=hashtags,
            )
            text = _format_video_prompt_message(scene, voiceover)
            await send_long_message(
                callback.message, text, keyboard=get_video_prompt_keyboard(), parse_mode="HTML"
            )
            await state.set_state(GenerationState.CONFIRM_VIDEO_PROMPT)
        except Exception as e:
            logger.error(f"Video prompt generation failed (T2V): {e}")
            await callback.message.answer(
                f"❌ Error generating prompt: {e}\nTry again.",
                reply_markup=get_prompt_keyboard(),
            )
            await state.set_state(GenerationState.ENHANCE_PROMPT)
        finally:
            await state.update_data(generation_in_progress=False)
        return

    # ── I2V: generate reference image(s) and show for review ────────────────
    image_count = int(settings.get("image_count", 1))
    await callback.message.answer(
        f"⏳ Generating {image_count} image{'s' if image_count > 1 else ''}…"
    )
    try:
        gemini = container.inject(GeminiService)
        imagegen = container.inject(ImageGenService)
        data = await state.get_data()
        enhance_prompt = data.get("enhance_prompt", "")

        image_prompt = await _build_image_prompt(enhance_prompt, settings, gemini)

        image_paths = await imagegen.generate_many(
            prompt=image_prompt,
            model=settings.get("image_model") or "",  # empty → ImageGenService default
            video_model=video_model or "seedance",
            count=image_count,
            notify=callback.message.answer,
        )

        await state.update_data(image_paths=image_paths, image_prompt=image_prompt)
        await _send_image_preview(callback.message, image_paths)
        await callback.message.answer("Image(s) created.\nContinue?", reply_markup=get_image_keyboard())
        await state.set_state(GenerationState.CONFIRM_IMAGE)
    except Exception as e:
        logger.error(f"Image generation failed: {e}")
        await callback.message.answer(
            f"❌ Image generation failed: {e}\nTry again or change the image model.",
            reply_markup=get_prompt_keyboard(),
        )
        await state.set_state(GenerationState.ENHANCE_PROMPT)


@router.callback_query(GenerationState.ENHANCE_PROMPT, lambda c: c.data == "prompt_regenerate", IsAllowed(allowed_users))
async def handle_prompt_regenerate(callback: CallbackQuery, state: FSMContext):
    await callback.answer()
    await callback.message.edit_reply_markup(reply_markup=None)
    await callback.message.answer("Creating upgraded prompt...")
    try:
        gemini = container.inject(GeminiService)
        db = container.inject(DBService)
        settings = await db.get_settings(callback.from_user.id, callback.message.chat.id)
        data = await state.get_data()
        raw_prompt = data.get("raw_prompt", "")
        system_prompt = _substitute_prompt_vars(
            settings.get("system_plot_prompt") or DEFAULT_PLOT_PROMPT,
            settings
        )
        enhance_prompt = await gemini.generate_text(
            raw_prompt,
            system_prompt,
            settings.get("text_model") or "",
        )
        await state.update_data(enhance_prompt=enhance_prompt)
        await send_long_message(
            callback.message,
            f"Your prompt: {enhance_prompt}\nIf you want to improve it, send a corrected version or click Continue",
            keyboard=get_prompt_keyboard(),
        )
    except Exception as e:
        await callback.message.answer(f"Error: {str(e)}\nTry again", reply_markup=get_prompt_keyboard())
    await state.set_state(GenerationState.ENHANCE_PROMPT)


# ─────────────────────────────────────────────
# IMAGE STAGE
# ─────────────────────────────────────────────

@router.callback_query(GenerationState.CONFIRM_IMAGE, lambda c: c.data == "image_prompt_change", IsAllowed(allowed_users))
async def handle_image_prompt_change(callback: CallbackQuery, state: FSMContext):
    await callback.answer()
    await callback.message.edit_reply_markup(reply_markup=None)
    data = await state.get_data()
    enhance_prompt = data.get("enhance_prompt", "")
    await send_long_message(
        callback.message,
        f"Your prompt: {enhance_prompt}\nSend a corrected version or click Continue",
        keyboard=get_prompt_keyboard(),
    )
    await state.set_state(GenerationState.ENHANCE_PROMPT)


@router.callback_query(GenerationState.CONFIRM_IMAGE, lambda c: c.data == "image_regenerate", IsAllowed(allowed_users))
async def handle_image_regenerate(callback: CallbackQuery, state: FSMContext):
    await callback.answer()
    await callback.message.edit_reply_markup(reply_markup=None)
    try:
        gemini = container.inject(GeminiService)
        imagegen = container.inject(ImageGenService)
        db = container.inject(DBService)
        settings = await db.get_settings(callback.from_user.id, callback.message.chat.id)
        image_count = int(settings.get("image_count", 1))
        await callback.message.answer(
            f"Creating {image_count} image{'s' if image_count > 1 else ''}…"
        )
        data = await state.get_data()
        enhance_prompt = data.get("enhance_prompt", "")
        image_prompt = await _build_image_prompt(enhance_prompt, settings, gemini)
        image_paths = await imagegen.generate_many(
            prompt=image_prompt,
            model=settings.get("image_model") or "",  # empty → ImageGenService default
            video_model=settings.get("video_model") or "seedance",
            count=image_count,
            notify=callback.message.answer,
        )
        await state.update_data(image_paths=image_paths, image_prompt=image_prompt)
        await _send_image_preview(callback.message, image_paths)
        await callback.message.answer("Image(s) created.\nContinue?", reply_markup=get_image_keyboard())
    except Exception as e:
        await callback.message.answer(f"Error: {str(e)}\nTry again", reply_markup=get_image_keyboard())
    await state.set_state(GenerationState.CONFIRM_IMAGE)


@router.callback_query(GenerationState.CONFIRM_IMAGE, lambda c: c.data == "image_ok", IsAllowed(allowed_users))
async def handle_image_ok(callback: CallbackQuery, state: FSMContext):
    """User approved the image → generate video prompt and show for review."""
    if await _is_generating(state):
        await callback.answer("Already generating — please wait.", show_alert=False)
        return
    await callback.answer()
    await callback.message.edit_reply_markup(reply_markup=None)
    await state.update_data(generation_in_progress=True, generation_started_at=time.time())
    try:
        gemini = container.inject(GeminiService)
        db = container.inject(DBService)
        settings = await db.get_settings(callback.from_user.id, callback.message.chat.id)
        data = await state.get_data()
        enhance_prompt = data.get("enhance_prompt", "")

        await callback.message.answer("⏳ Generating video prompt…")
        text_model = settings.get("text_model") or ""
        video_prompt, hashtags = await asyncio.gather(
            _build_video_prompt(enhance_prompt, settings, gemini),
            gemini.generate_text(enhance_prompt, generate_hashtags_prompt, text_model),
        )

        scene, voiceover = _split_video_prompt(video_prompt, gemini)
        parsed_vp = GeminiService.parse_script_json(video_prompt)
        if parsed_vp and parsed_vp.get("caption"):
            hashtags = parsed_vp["caption"]
        await state.update_data(
            video_scene=scene,
            video_voiceover=voiceover,
            video_script_json=video_prompt,
            cached_hashtags=hashtags,
        )

        text = _format_video_prompt_message(scene, voiceover)
        await send_long_message(
            callback.message, text, keyboard=get_video_prompt_keyboard(), parse_mode="HTML"
        )
        await state.set_state(GenerationState.CONFIRM_VIDEO_PROMPT)
    except Exception as e:
        await callback.message.answer(
            f"❌ Error generating prompt: {e}\nTry again?",
            reply_markup=get_image_keyboard(),
        )
        await state.set_state(GenerationState.CONFIRM_IMAGE)
    finally:
        await state.update_data(generation_in_progress=False)


# ─────────────────────────────────────────────
# VIDEO PROMPT REVIEW
# ─────────────────────────────────────────────

@router.callback_query(
    GenerationState.CONFIRM_VIDEO_PROMPT,
    lambda c: c.data == "vp_ok",
    IsAllowed(allowed_users),
)
async def handle_vp_ok(callback: CallbackQuery, state: FSMContext):
    if await _is_generating(state):
        await callback.answer("Already generating — please wait.", show_alert=False)
        return
    await callback.answer()
    await callback.message.edit_reply_markup(reply_markup=None)
    await state.update_data(generation_in_progress=True, generation_started_at=time.time())
    try:
        await _run_video_gen(callback, state, gen_type="initial")
    finally:
        await state.update_data(generation_in_progress=False)


@router.callback_query(
    GenerationState.CONFIRM_VIDEO_PROMPT,
    lambda c: c.data == "vp_scene_regen",
    IsAllowed(allowed_users),
)
async def handle_vp_scene_regen(callback: CallbackQuery, state: FSMContext):
    await callback.answer()
    await callback.message.edit_reply_markup(reply_markup=None)
    try:
        gemini = container.inject(GeminiService)
        db = container.inject(DBService)
        settings = await db.get_settings(callback.from_user.id, callback.message.chat.id)
        data = await state.get_data()
        await callback.message.answer("⏳ Regenerating scene…")
        video_prompt = await _build_video_prompt(data.get("enhance_prompt", ""), settings, gemini)
        new_scene, _ = _split_video_prompt(video_prompt, gemini)
        voiceover = data.get("video_voiceover", "")
        await state.update_data(video_scene=new_scene, video_script_json=video_prompt)
        await _show_video_prompt(callback.message, new_scene, voiceover)
    except Exception as e:
        await callback.message.answer(f"❌ Error: {e}")
    await state.set_state(GenerationState.CONFIRM_VIDEO_PROMPT)


@router.callback_query(
    GenerationState.CONFIRM_VIDEO_PROMPT,
    lambda c: c.data == "vp_vo_regen",
    IsAllowed(allowed_users),
)
async def handle_vp_vo_regen(callback: CallbackQuery, state: FSMContext):
    await callback.answer()
    await callback.message.edit_reply_markup(reply_markup=None)
    try:
        gemini = container.inject(GeminiService)
        db = container.inject(DBService)
        settings = await db.get_settings(callback.from_user.id, callback.message.chat.id)
        data = await state.get_data()
        await callback.message.answer("⏳ Regenerating voiceover…")
        video_prompt = await _build_video_prompt(data.get("enhance_prompt", ""), settings, gemini)
        _, new_voiceover = _split_video_prompt(video_prompt, gemini)
        scene = data.get("video_scene", "")
        await state.update_data(video_voiceover=new_voiceover, video_script_json=video_prompt)
        await _show_video_prompt(callback.message, scene, new_voiceover)
    except Exception as e:
        await callback.message.answer(f"❌ Error: {e}")
    await state.set_state(GenerationState.CONFIRM_VIDEO_PROMPT)


@router.callback_query(
    GenerationState.CONFIRM_VIDEO_PROMPT,
    lambda c: c.data == "vp_scene_edit",
    IsAllowed(allowed_users),
)
async def handle_vp_scene_edit(callback: CallbackQuery, state: FSMContext):
    await callback.answer()
    await callback.message.edit_reply_markup(reply_markup=None)
    data = await state.get_data()
    current = html.escape(data.get("video_scene", ""))
    await callback.message.answer(
        f"✏️ Send the new <b>scene description</b> (current shown below):\n\n<code>{current}</code>",
        parse_mode="HTML",
    )
    await state.set_state(GenerationState.EDIT_SCENE)


@router.message(GenerationState.EDIT_SCENE, IsAllowed(allowed_users))
async def handle_edit_scene_input(message: Message, state: FSMContext):
    data = await state.get_data()
    await state.update_data(video_scene=message.text, video_script_json=None)
    await _show_video_prompt(message, message.text, data.get("video_voiceover", ""))
    await state.set_state(GenerationState.CONFIRM_VIDEO_PROMPT)


@router.callback_query(
    GenerationState.CONFIRM_VIDEO_PROMPT,
    lambda c: c.data == "vp_vo_edit",
    IsAllowed(allowed_users),
)
async def handle_vp_vo_edit(callback: CallbackQuery, state: FSMContext):
    await callback.answer()
    await callback.message.edit_reply_markup(reply_markup=None)
    data = await state.get_data()
    current = html.escape(data.get("video_voiceover", ""))
    await callback.message.answer(
        f"✏️ Send the new <b>voiceover text</b> (current shown below):\n\n<code>{current}</code>",
        parse_mode="HTML",
    )
    await state.set_state(GenerationState.EDIT_VOICEOVER)


@router.message(GenerationState.EDIT_VOICEOVER, IsAllowed(allowed_users))
async def handle_edit_voiceover_input(message: Message, state: FSMContext):
    data = await state.get_data()
    await state.update_data(video_voiceover=message.text)
    await _show_video_prompt(message, data.get("video_scene", ""), message.text)
    await state.set_state(GenerationState.CONFIRM_VIDEO_PROMPT)


# ─────────────────────────────────────────────
# VIDEO GENERATION
# ─────────────────────────────────────────────

async def _run_video_gen(callback: CallbackQuery, state: FSMContext, gen_type: str):
    """Shared generation pipeline used by 'initial' and 'regen' runs."""
    user_id = callback.from_user.id
    chat_id = callback.message.chat.id
    try:
        gemini = container.inject(GeminiService)
        db = container.inject(DBService)
        settings = await db.get_settings(user_id, chat_id)
        data = await state.get_data()
        # Support both new image_paths (list) and legacy image_path (single string)
        image_paths = data.get("image_paths") or []
        if not image_paths:
            single = data.get("image_path")
            image_paths = [single] if single else []
        if gen_type == "initial":
            video_scene = data.get("video_scene", "")
        else:
            video_scene = data.get("video_scene") or data.get("video_prompt") or data.get("enhance_prompt", "")
        video_voiceover = data.get("video_voiceover", "")

        gen_result = await generate_video_for_model(
            gemini=gemini,
            settings=settings,
            video_prompt=video_scene,
            voiceover_text=video_voiceover,
            image_paths=image_paths,
            notify=callback.message.answer,
            script_json=data.get("video_script_json") or "",
        )

        subs_default = settings.get("subtitles_enabled", True)
        grade_enabled = settings.get("colour_grade_enabled", False)
        grade_params = GeminiService.parse_grade_params(settings.get("colour_grade_params"))

        result = await _post_process_video(
            gemini,
            gen_result["raw_video_path"],
            video_scene or gen_result.get("voiceover_text", ""),
            subs_default,
            video_model=settings.get("video_model") or "seedance",
            grade_enabled=grade_enabled,
            grade_params=grade_params,
            voice_id=settings.get("voice_id") or "",
            notify=callback.message.answer,
            voiceover_text=gen_result.get("voiceover_text", ""),
            word_timings=gen_result.get("word_timings") or [],
            tts_audio_path=gen_result.get("tts_audio_path", ""),
            sfx_enabled=settings.get("sfx_enabled", False),
            video_speed=float(settings.get("video_speed", 1.0)),
            music_path=settings.get("background_music_path") or "",
            music_volume=float(settings.get("background_music_volume", 0.18)),
        )

        video_file = result["video_path"]
        if not os.path.exists(video_file):
            raise FileNotFoundError(f"Generated video file not found: {video_file}")
        sent_msg = await callback.message.answer_video(FSInputFile(video_file), caption="video")
        await state.update_data(video_prompt=video_scene, **result)
        await callback.message.answer(
            "Video created!\nContinue?",
            reply_markup=get_video_keyboard(result["subtitles_on"]),
        )
        await db.log_generation(
            user_id=user_id, chat_id=chat_id, gen_type=gen_type,
            video_model=settings.get("video_model") or "",
            target_duration=settings.get("target_duration", DEFAULT_TARGET_DURATION),
        )
        if sent_msg.video:
            history_title = (video_scene or video_voiceover or data.get("enhance_prompt") or "")[:120]
            await db.add_video_history(user_id, chat_id, sent_msg.video.file_id, history_title, gen_type)
    except Exception as e:
        logger.error(f"Generation error: {e}", exc_info=True)
        db = container.inject(DBService)
        await db.log_generation(
            user_id=user_id, chat_id=chat_id, gen_type=gen_type,
            video_model="", success=False, error_text=str(e)[:500],
        )
        await callback.message.answer(f"Error: {str(e)}\nTry again", reply_markup=get_video_keyboard())
    await state.set_state(GenerationState.CONFIRM_VIDEO)


@router.callback_query(GenerationState.CONFIRM_VIDEO, lambda c: c.data == "video_ok", IsAllowed(allowed_users))
async def handle_video_ok(callback: CallbackQuery, state: FSMContext):
    if await _is_generating(state):
        await callback.answer("⚠️ Generation in progress — please wait.", show_alert=False)
        return
    await callback.answer()
    await callback.message.edit_reply_markup(reply_markup=None)
    await callback.message.answer("Do you want to publish?", reply_markup=get_publish_keyboard())
    await state.set_state(GenerationState.CONFIRM_PUBLISH)


@router.callback_query(GenerationState.CONFIRM_VIDEO, lambda c: c.data == "video_prompt_change", IsAllowed(allowed_users))
async def handle_video_prompt_change(callback: CallbackQuery, state: FSMContext):
    await callback.answer()
    await callback.message.edit_reply_markup(reply_markup=None)
    data = await state.get_data()
    enhance_prompt = data.get("enhance_prompt", "")
    await send_long_message(
        callback.message,
        f"Your prompt: {enhance_prompt}\nSend a corrected version or click Continue",
        keyboard=get_prompt_keyboard(),
    )
    await state.set_state(GenerationState.ENHANCE_PROMPT)


@router.callback_query(GenerationState.CONFIRM_VIDEO, lambda c: c.data == "video_regenerate", IsAllowed(allowed_users))
async def handle_video_regenerate(callback: CallbackQuery, state: FSMContext):
    if await _is_generating(state):
        await callback.answer("Already generating — please wait.", show_alert=False)
        return
    await callback.answer()
    await callback.message.edit_reply_markup(reply_markup=None)
    await state.update_data(generation_in_progress=True, generation_started_at=time.time())
    try:
        await _run_video_gen(callback, state, gen_type="regen")
    finally:
        await state.update_data(generation_in_progress=False)


@router.callback_query(GenerationState.CONFIRM_VIDEO, lambda c: c.data == "video_subtitles_toggle", IsAllowed(allowed_users))
async def handle_video_subtitles_toggle(callback: CallbackQuery, state: FSMContext):
    """Per-video subtitle toggle. Lazy-burns karaoke version on first turn-on."""
    await callback.answer()
    data = await state.get_data()
    raw = data.get("video_path_raw")
    base = data.get("video_path_base") or raw
    subbed = data.get("video_path_subbed")
    word_timings = data.get("word_timings") or []
    currently_on = data.get("subtitles_on", False)

    if not raw:
        await callback.message.answer("No video in current session.")
        return

    try:
        if currently_on:
            new_path = base
            new_state = False
        else:
            if not word_timings:
                await callback.answer(
                    "No word-level timings (TTS likely failed) — cannot render karaoke.",
                    show_alert=True,
                )
                return
            if subbed is None:
                await callback.message.answer("Burning karaoke subtitles…")
                gemini = container.inject(GeminiService)
                subbed = await gemini.burn_karaoke_subtitles(base, word_timings)
                if subbed == base:
                    await callback.message.answer("Failed to burn subtitles, see logs.")
                    return
            new_path = subbed
            new_state = True

        try:
            await callback.message.edit_reply_markup(reply_markup=None)
        except Exception:
            pass

        if not os.path.exists(new_path):
            await callback.message.answer(
                "❌ Video file was deleted from disk (cleanup may have removed it). "
                "Please regenerate the video."
            )
            return
        await callback.message.answer_video(FSInputFile(new_path), caption="video")
        await state.update_data(
            video_path=new_path,
            video_path_subbed=subbed,
            subtitles_on=new_state,
        )
        await callback.message.answer(
            f"Subtitles {'ON' if new_state else 'OFF'}.\nContinue?",
            reply_markup=get_video_keyboard(new_state),
        )
    except Exception as e:
        await callback.message.answer(f"Error: {e}", reply_markup=get_video_keyboard(currently_on))


@router.callback_query(GenerationState.CONFIRM_VIDEO, lambda c: c.data == "video_edit", IsAllowed(allowed_users))
async def handle_video_edit_click(callback: CallbackQuery, state: FSMContext):
    """Ask user for edit prompt, then switch to VIDEO_EDIT_PROMPT state."""
    if await _is_generating(state):
        await callback.answer("Already generating — please wait.", show_alert=False)
        return
    await callback.answer()
    data = await state.get_data()
    video_path = data.get("video_path_base") or data.get("video_path_raw")
    if not video_path or not os.path.exists(video_path):
        await callback.message.answer("❌ Video file not found — please regenerate first.")
        return
    await callback.message.answer(
        "✏️ <b>Describe what to change in the video</b>\n"
        "Examples: <i>change background to sunset beach</i>, "
        "<i>add snow falling</i>, <i>make it look like anime</i>",
        parse_mode="HTML",
    )
    await state.set_state(GenerationState.VIDEO_EDIT_PROMPT)


@router.message(GenerationState.VIDEO_EDIT_PROMPT, IsAllowed(allowed_users))
async def handle_video_edit_prompt(message: Message, state: FSMContext):
    """Upload current video to Atlas, run Kling Video-Edit, show result."""
    if await _is_generating(state):
        await message.answer("Already generating — please wait.")
        return
    await state.update_data(generation_in_progress=True, generation_started_at=time.time())
    data = await state.get_data()
    subtitles_on = data.get("subtitles_on", False)
    try:
        video_path = data.get("video_path_base") or data.get("video_path_raw")
        if not video_path or not os.path.exists(video_path):
            await message.answer("❌ Video file not found — please regenerate first.")
            return

        edit_prompt = message.text or ""
        await message.answer("⏳ Uploading video & applying edit — takes ~3-5 min…")

        kling = container.inject(KlingService)
        edited_path = await kling.edit_video(video_path, edit_prompt)

        if not os.path.exists(edited_path):
            raise FileNotFoundError(f"Edited video not found: {edited_path}")

        await message.answer_video(FSInputFile(edited_path), caption="video")
        # Replace base video with edited version; keep subtitles state
        await state.update_data(
            video_path=edited_path,
            video_path_base=edited_path,
            video_path_raw=edited_path,
            video_path_subbed=None,
            subtitles_on=False,
        )
        await message.answer("✅ Edit applied!\nContinue?", reply_markup=get_video_keyboard(False))
    except Exception as e:
        logger.error(f"Video-Edit error: {e}", exc_info=True)
        await message.answer(f"Error: {e}\nTry again", reply_markup=get_video_keyboard(subtitles_on))
    finally:
        await state.update_data(generation_in_progress=False)
        await state.set_state(GenerationState.CONFIRM_VIDEO)
