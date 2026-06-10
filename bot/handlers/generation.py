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
from aiogram.types import CallbackQuery, FSInputFile, Message
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
from utils import container
from utils.consts import allowed_users, generate_hashtags_prompt
from utils.tg import send_long_message
from bot.handlers.common import (
    DEFAULT_TARGET_DURATION,
    _build_image_prompt,
    _build_video_prompt,
    _format_video_prompt_message,
    _is_generating,
    _post_process_video,
    _show_video_prompt,
    _split_video_prompt,
    generate_video_for_model,
)

router = Router()
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# ENTRY POINTS
# ─────────────────────────────────────────────

async def _begin_generation(message: Message, state: FSMContext):
    """Validate settings and start the generation flow."""
    if await _is_generating(state):
        await message.answer("⚠️ Generation in progress — please wait until it finishes.")
        return

    db = container.inject(DBService)
    settings = await db.get_settings(message.from_user.id, message.chat.id)

    if settings.get("text_model") is None:
        await message.answer("You must select a LLM to generate prompts\nTap ⚙️ Settings → 🧠 Text model")
        return
    if settings.get("image_model") is None:
        await message.answer("You must select an image model\nTap ⚙️ Settings → 🖼 Image model")
        return
    if settings.get("video_model") is None:
        await message.answer("You must select a video model\nTap ⚙️ Settings → 🎬 Video model")
        return
    if settings.get("system_plot_prompt") is None:
        await message.answer("A system prompt must be set for idea/plot\nTap ⚙️ Settings → 📝 Plot prompt")
        return
    if settings.get("system_image_prompt") is None:
        await message.answer("A system prompt must be set for image\nTap ⚙️ Settings → 🖼 Image prompt")
        return

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
        enhance_prompt = await gemini.generate_text(
            message.text,
            settings.get("system_plot_prompt") or "",
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
    """Prompt OK → generate image (Imagen / Gemini image / FLUX) and show for review."""
    await callback.answer()
    await callback.message.edit_reply_markup(reply_markup=None)
    await callback.message.answer("⏳ Generating image…")
    try:
        gemini = container.inject(GeminiService)
        imagegen = container.inject(ImageGenService)
        db = container.inject(DBService)
        settings = await db.get_settings(callback.from_user.id, callback.message.chat.id)
        data = await state.get_data()
        enhance_prompt = data.get("enhance_prompt", "")

        # Build the image prompt (overridden to strict portrait for lip-sync).
        image_prompt = await _build_image_prompt(enhance_prompt, settings, gemini)

        image_path = await imagegen.generate(
            prompt=image_prompt,
            model=settings.get("image_model") or "imagen-4.0-fast-generate-001",
            video_model=settings.get("video_model") or "seedance",
        )

        await state.update_data(image_path=image_path, image_prompt=image_prompt)
        await callback.message.answer_photo(FSInputFile(image_path), caption="image")
        await callback.message.answer("Image created.\nContinue?", reply_markup=get_image_keyboard())
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
        enhance_prompt = await gemini.generate_text(
            raw_prompt,
            settings.get("system_plot_prompt") or "",
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
    await callback.message.answer("Creating image...")
    try:
        gemini = container.inject(GeminiService)
        imagegen = container.inject(ImageGenService)
        db = container.inject(DBService)
        settings = await db.get_settings(callback.from_user.id, callback.message.chat.id)
        data = await state.get_data()
        enhance_prompt = data.get("enhance_prompt", "")
        image_prompt = await _build_image_prompt(enhance_prompt, settings, gemini)
        image_path = await imagegen.generate(
            prompt=image_prompt,
            model=settings.get("image_model") or "imagen-4.0-fast-generate-001",
            video_model=settings.get("video_model") or "seedance",
        )
        await state.update_data(image_path=image_path, image_prompt=image_prompt)
        await callback.message.answer_photo(FSInputFile(image_path), caption="image")
        await callback.message.answer("Image created.\nContinue?", reply_markup=get_image_keyboard())
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
        await state.update_data(
            video_scene=scene,
            video_voiceover=voiceover,
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
        await state.update_data(video_scene=new_scene)
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
        await state.update_data(video_voiceover=new_voiceover)
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
    await state.update_data(video_scene=message.text)
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
        image_path = data.get("image_path")
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
            image_path=image_path,
            notify=callback.message.answer,
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
