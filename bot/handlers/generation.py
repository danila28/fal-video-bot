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
    get_idea_entry_keyboard,
    get_image_keyboard,
    get_persistent_keyboard,
    get_prompt_keyboard,
    get_publish_keyboard,
    get_ref_image_mode_keyboard,
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
    _STRICT_IMAGE_SYS,
    _T2V_VIDEO_MODELS,
    _build_image_prompt,
    _build_ref_image_prompts,
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


def _is_reference_model(video_model: str) -> bool:
    """Check if the video model is a reference-to-video model."""
    video_model = (video_model or "").lower()
    return video_model.endswith("_ref")


async def _send_image_preview(message, image_paths: list[str], captions: list[str] | None = None) -> None:
    """Send generated images as a media group (album) or single photo.

    captions: optional list of captions to use instead of default numbering.
    """
    valid = [p for p in image_paths if os.path.exists(p)]
    if not valid:
        return
    captions = captions or []
    if len(valid) == 1:
        caption = captions[0] if captions else "image"
        await message.answer_photo(FSInputFile(valid[0]), caption=caption)
        return
    media = [
        InputMediaPhoto(
            media=FSInputFile(p),
            caption=captions[i] if i < len(captions) else (f"photo {i + 1}/{len(valid)}" if i == 0 else None),
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
    await message.answer(
        "Send your idea 💡\n\nOr use your own ready-made script — it will be "
        "taken as-is, with no plot rewriting:",
        reply_markup=get_idea_entry_keyboard(),
    )
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
        await state.update_data(raw_prompt=message.text, enhance_prompt=enhance_prompt, own_script=False)
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


async def _proceed_to_media(message: Message, state: FSMContext, user_id: int, chat_id: int):
    """Shared continuation once the concept text is final (normal flow or
    own-script bypass): T2V models jump straight to the video prompt, I2V
    models generate reference image(s) for review first."""
    db = container.inject(DBService)
    settings = await db.get_settings(user_id, chat_id)
    video_model = settings.get("video_model") or ""
    data = await state.get_data()
    enhance_prompt = data.get("enhance_prompt", "")
    strict = bool(data.get("own_script"))

    if video_model in _T2V_VIDEO_MODELS:
        # ── T2V: no image needed — jump straight to video prompt ──────────────
        await state.update_data(image_paths=[], image_path=None, image_prompt="")
        await state.update_data(generation_in_progress=True, generation_started_at=time.time())
        await message.answer("⏳ Generating video prompt…")
        try:
            gemini = container.inject(GeminiService)
            text_model = settings.get("text_model") or ""
            video_prompt, hashtags = await asyncio.gather(
                _build_video_prompt(enhance_prompt, settings, gemini, strict_script=strict),
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
                message, text, keyboard=get_video_prompt_keyboard(), parse_mode="HTML"
            )
            await state.set_state(GenerationState.CONFIRM_VIDEO_PROMPT)
        except Exception as e:
            logger.error(f"Video prompt generation failed (T2V): {e}")
            await message.answer(
                f"❌ Error generating prompt: {e}\nTry again.",
                reply_markup=get_prompt_keyboard(),
            )
            await state.set_state(GenerationState.ENHANCE_PROMPT)
        finally:
            await state.update_data(generation_in_progress=False)
        return

    # ── I2V: generate reference image(s) and show for review ────────────────
    # Flag guards against double-taps — without it two parallel image
    # pipelines run and the user gets (and pays for) duplicate images.
    image_count = int(settings.get("image_count", 1))
    await state.update_data(generation_in_progress=True, generation_started_at=time.time())
    await message.answer(
        f"⏳ Generating {image_count} image{'s' if image_count > 1 else ''}…"
    )
    try:
        gemini = container.inject(GeminiService)
        imagegen = container.inject(ImageGenService)

        image_prompt = await _build_image_prompt(enhance_prompt, settings, gemini, strict_script=strict)

        image_paths = await imagegen.generate_many(
            prompt=image_prompt,
            model=settings.get("image_model") or "",  # empty → ImageGenService default
            video_model=video_model or "seedance",
            count=image_count,
            notify=message.answer,
        )

        await state.update_data(image_paths=image_paths, image_prompt=image_prompt)
        await _send_image_preview(message, image_paths)
        await message.answer("Image(s) created.\nContinue?", reply_markup=get_image_keyboard())
        await state.set_state(GenerationState.CONFIRM_IMAGE)
    except Exception as e:
        logger.error(f"Image generation failed: {e}")
        await message.answer(
            f"❌ Image generation failed: {e}\nTry again or change the image model.",
            reply_markup=get_prompt_keyboard(),
        )
        await state.set_state(GenerationState.ENHANCE_PROMPT)
    finally:
        await state.update_data(generation_in_progress=False)


@router.callback_query(GenerationState.ENHANCE_PROMPT, lambda c: c.data == "prompt_ok", IsAllowed(allowed_users))
async def handle_prompt_ok(callback: CallbackQuery, state: FSMContext):
    """Prompt OK → for T2V models skip image and go straight to video prompt;
    for reference models show mode selection; for I2V models generate image."""
    if await _is_generating(state):
        await callback.answer("Already generating — please wait.", show_alert=False)
        return
    await callback.answer()
    await callback.message.edit_reply_markup(reply_markup=None)

    db = container.inject(DBService)
    settings = await db.get_settings(callback.from_user.id, callback.message.chat.id)
    video_model = settings.get("video_model") or ""

    # Reference models: show mode selection before image generation
    if _is_reference_model(video_model):
        image_count = int(settings.get("image_count", 1))
        if image_count > 1:
            await callback.message.answer(
                f"You selected {image_count} reference photos.\n\nHow would you like to describe them?",
                reply_markup=get_ref_image_mode_keyboard(),
            )
            await state.set_state(GenerationState.SELECT_REF_MODE)
            return

    await _proceed_to_media(callback.message, state, callback.from_user.id, callback.message.chat.id)


# ─────────────────────────────────────────────
# REFERENCE IMAGE MODES (manual + auto for reference models)
# ─────────────────────────────────────────────

@router.callback_query(GenerationState.SELECT_REF_MODE, lambda c: c.data.startswith("ref_mode:"), IsAllowed(allowed_users))
async def handle_ref_mode_select(callback: CallbackQuery, state: FSMContext):
    """Handle selection between manual descriptions or auto mode."""
    mode = callback.data.split(":", 1)[1]  # 'manual' or 'auto'
    await callback.answer()
    await callback.message.edit_reply_markup(reply_markup=None)

    db = container.inject(DBService)
    settings = await db.get_settings(callback.from_user.id, callback.message.chat.id)
    image_count = int(settings.get("image_count", 1))

    if mode == "manual":
        await state.update_data(
            ref_mode="manual",
            ref_descriptions=[],
            ref_current_index=0,
        )
        await callback.message.answer(f"📝 Photo 1 of {image_count}: describe it (min. 15 chars)")
        await state.set_state(GenerationState.ENTER_REF_IMAGES)
    else:  # auto mode
        await state.update_data(ref_mode="auto")
        await callback.message.answer("🤖 Generating descriptions...")
        await _proceed_to_ref_auto(callback.message, state, callback.from_user.id, callback.message.chat.id)


@router.message(GenerationState.ENTER_REF_IMAGES, IsAllowed(allowed_users))
async def handle_ref_image_description(message: Message, state: FSMContext):
    """Collect sequential image descriptions for manual mode.

    Special: /cancel to start over, /skip to go back to mode selection.
    """
    text = (message.text or "").strip()

    # Allow cancellation/reset
    if text.lower() in ("/cancel", "cancel"):
        await state.update_data(ref_descriptions=[], ref_current_index=0, ref_mode=None)
        data = await state.get_data()
        db = container.inject(DBService)
        settings = await db.get_settings(message.from_user.id, message.chat.id)
        image_count = int(settings.get("image_count", 1))
        await message.answer(
            f"Cancelled. Start over?\n\nHow would you like to describe {image_count} photo{'s' if image_count > 1 else ''}?",
            reply_markup=get_ref_image_mode_keyboard(),
        )
        await state.set_state(GenerationState.SELECT_REF_MODE)
        return

    data = await state.get_data()
    descriptions = data.get("ref_descriptions", [])
    current_idx = data.get("ref_current_index", 0)

    db = container.inject(DBService)
    settings = await db.get_settings(message.from_user.id, message.chat.id)
    image_count = int(settings.get("image_count", 1))

    if len(text) < 15:
        await message.answer("❌ Description too short — send at least 15 characters.\n(or type /cancel to start over)")
        return

    descriptions.append(text)
    current_idx += 1

    if current_idx >= image_count:
        # All descriptions collected
        await state.update_data(ref_descriptions=descriptions, ref_current_index=current_idx)
        await message.answer("⏳ Generating images…")
        await _proceed_to_ref_manual(message, state, message.from_user.id, message.chat.id)
    else:
        # Ask for next description
        await state.update_data(ref_descriptions=descriptions, ref_current_index=current_idx)
        await message.answer(f"📝 Photo {current_idx + 1} of {image_count}: describe it (min. 15 chars)\n(or type /cancel to start over)")


async def _proceed_to_ref_manual(message: Message, state: FSMContext, user_id: int, chat_id: int):
    """Generate images from manual descriptions.

    Each raw description is first passed through the niche's system_image_prompt
    (same styling pipeline as the single-image path) so reference photos match
    the selected preset's visual style instead of using the user's raw text as-is.
    """
    db = container.inject(DBService)
    settings = await db.get_settings(user_id, chat_id)
    data = await state.get_data()
    descriptions = data.get("ref_descriptions", [])
    video_model = settings.get("video_model") or ""
    strict = bool(data.get("own_script"))

    if not descriptions:
        await message.answer("❌ No descriptions provided.")
        await state.set_state(GenerationState.ENHANCE_PROMPT)
        return

    await state.update_data(generation_in_progress=True, generation_started_at=time.time())

    try:
        gemini = container.inject(GeminiService)
        imagegen = container.inject(ImageGenService)

        styled_prompts = await _build_ref_image_prompts(descriptions, settings, gemini, strict_script=strict)

        image_paths = await imagegen.generate_from_prompts(
            prompts=styled_prompts,
            model=settings.get("image_model") or "",
            video_model=video_model or "seedance",
            notify=message.answer,
        )

        captions = [f"Photo {i + 1}/{len(descriptions)}" for i in range(len(descriptions))]
        await state.update_data(
            image_paths=image_paths,
            image_prompt="",
            ref_mode="manual",
            ref_descriptions=descriptions,
        )
        await _send_image_preview(message, image_paths, captions=captions)
        await message.answer("Images created. Continue?", reply_markup=get_image_keyboard())
        await state.set_state(GenerationState.CONFIRM_IMAGE)
    except Exception as e:
        logger.error(f"Reference image generation failed (manual): {e}")
        await message.answer(
            f"❌ Image generation failed: {e}\nTry again.",
            reply_markup=get_prompt_keyboard(),
        )
        await state.set_state(GenerationState.ENHANCE_PROMPT)
    finally:
        await state.update_data(generation_in_progress=False)


async def _proceed_to_ref_auto(message: Message, state: FSMContext, user_id: int, chat_id: int):
    """Generate reference images via LLM auto mode.

    The niche's system_image_prompt (same style guide used by the single-image
    path) is folded into the meta-prompt so the LLM-authored image_prompts
    match the selected preset's visual style instead of a generic look.
    """
    db = container.inject(DBService)
    settings = await db.get_settings(user_id, chat_id)
    data = await state.get_data()
    enhance_prompt = data.get("enhance_prompt", "")
    image_count = int(settings.get("image_count", 1))
    video_model = settings.get("video_model") or ""
    strict = bool(data.get("own_script"))

    await state.update_data(generation_in_progress=True, generation_started_at=time.time())

    try:
        gemini = container.inject(GeminiService)
        from utils.presets import DEFAULT_IMAGE_PROMPT, get_image_tag_template, parse_ref_images_auto_json

        tag_template = get_image_tag_template(video_model)
        tag_examples = ", ".join([tag_template.format(i=i+1) for i in range(min(image_count, 2))])

        style_sys = _STRICT_IMAGE_SYS if strict else (settings.get("system_image_prompt") or DEFAULT_IMAGE_PROMPT)
        style_block = f"VISUAL STYLE TO FOLLOW FOR EVERY image_prompt:\n{style_sys}\n\n"

        llm_prompt = (
            f"Based on this concept: {enhance_prompt}\n\n"
            f"{style_block}"
            f"Generate {image_count} complementary image prompts for reference-to-video generation.\n"
            f"The images will be used together in a video, so create different but complementary roles/elements "
            f"(e.g. hero character, location, prop, second character) — every image_prompt must still follow the "
            f"visual style above (lighting, color palette, mood).\n\n"
            f"Return ONLY valid JSON (no markdown fences):\n"
            '{\n'
            '  "roles": [\n'
            '    {"index": 1, "role": "Main subject", "description": "..."},\n'
            "    ...\n"
            "  ],\n"
            f'  "image_prompts": ["prompt 1", "prompt 2", ...],\n'
            '  "combination_plan": "How these elements should appear across shots..."\n'
            "}\n\n"
            f"- image_prompts array MUST have EXACTLY {image_count} entries, one per role, in the same order as roles\n"
            "- Each image_prompt: English, ONE single still frame, no text/captions/logos/watermarks, "
            "9:16 vertical composition, G-rated content only\n"
            f"- Image tags available in scene prompts: {tag_examples}, etc."
        )

        text_model = settings.get("text_model") or ""
        response = await gemini.generate_text(
            llm_prompt,
            "You are a creative director. Generate image descriptions that will work well together as reference images.",
            text_model,
        )

        parsed = parse_ref_images_auto_json(response)
        if parsed is None:
            await message.answer(
                "❌ Failed to parse auto mode response. Try manual mode instead.",
                reply_markup=get_prompt_keyboard(),
            )
            await state.set_state(GenerationState.ENHANCE_PROMPT)
            return

        imagegen = container.inject(ImageGenService)
        image_prompts = parsed.get("image_prompts", [])[:image_count]
        image_paths = await imagegen.generate_from_prompts(
            prompts=image_prompts,
            model=settings.get("image_model") or "",
            video_model=video_model or "seedance",
            notify=message.answer,
        )

        roles = parsed.get("roles", [])
        captions = [
            f"{role.get('role', f'Photo {i+1}')}"
            for i, role in enumerate(roles[:len(image_paths)])
        ]
        for i in range(len(image_paths) - len(captions)):
            captions.append(f"Photo {len(captions) + 1}")

        await state.update_data(
            image_paths=image_paths,
            image_prompt="",
            ref_mode="auto",
            ref_roles=parsed.get("roles", []),
            ref_combination_plan=parsed.get("combination_plan", ""),
            ref_image_prompts=image_prompts,
        )
        await _send_image_preview(message, image_paths, captions=captions)
        await message.answer("Images created. Continue?", reply_markup=get_image_keyboard())
        await state.set_state(GenerationState.CONFIRM_IMAGE)
    except Exception as e:
        logger.error(f"Reference image generation failed (auto): {e}")
        await message.answer(
            f"❌ Auto mode failed: {e}\nTry manual mode instead.",
            reply_markup=get_prompt_keyboard(),
        )
        await state.set_state(GenerationState.ENHANCE_PROMPT)
    finally:
        await state.update_data(generation_in_progress=False)


# ─────────────────────────────────────────────
# OWN-SCRIPT MODE (hard bypass of the plot stage)
# ─────────────────────────────────────────────

@router.callback_query(lambda c: c.data == "own_script:start", IsAllowed(allowed_users))
async def handle_own_script_start(callback: CallbackQuery, state: FSMContext):
    await callback.answer()
    db = container.inject(DBService)
    settings = await db.get_settings(callback.from_user.id, callback.message.chat.id)
    td = settings.get("target_duration", DEFAULT_TARGET_DURATION)
    vo_on = settings.get("voiceover_enabled", True)
    speed = float(settings.get("video_speed", 1.0))
    lines = [
        "📝 <b>Own-script mode</b>",
        "",
        "Send your final script — it is used as-is: no plot rewriting, "
        "no invented characters or framing. The bot only splits it into shots "
        "for the video model.",
        "",
        "Technical settings come from ⚙️ Settings, not from the text:",
        f"• ⏱ Duration: <b>{td}s</b>",
        f"• 🗣 Voiceover: <b>{'ON' if vo_on else 'OFF'}</b>",
    ]
    if speed != 1.0:
        lines.append(f"• ⚡ Speed {speed:g}× — final length will be ~{round(td / speed)}s")
    lines += [
        "",
        "<i>Note: visual descriptions are translated to English for the video "
        "model, and wording may be slightly compressed to fit per-shot limits — "
        "content and order of events are preserved.</i>",
    ]
    await callback.message.answer("\n".join(lines), parse_mode="HTML")
    await state.set_state(GenerationState.OWN_SCRIPT)


@router.message(GenerationState.OWN_SCRIPT, IsAllowed(allowed_users))
async def handle_own_script_input(message: Message, state: FSMContext):
    if await _is_generating(state):
        await message.answer("⚠️ Generation in progress — please wait until it finishes.")
        return
    script = (message.text or "").strip()
    if len(script) < 20:
        await message.answer("❌ The script is too short — send the full scene description.")
        return
    await state.update_data(raw_prompt=script, enhance_prompt=script, own_script=True)
    await _proceed_to_media(message, state, message.from_user.id, message.chat.id)


@router.callback_query(GenerationState.ENHANCE_PROMPT, lambda c: c.data == "prompt_regenerate", IsAllowed(allowed_users))
async def handle_prompt_regenerate(callback: CallbackQuery, state: FSMContext):
    # Guard against double-taps — prevents duplicate parallel LLM calls.
    if await _is_generating(state):
        await callback.answer("Already generating — please wait.", show_alert=False)
        return
    await callback.answer()
    await callback.message.edit_reply_markup(reply_markup=None)
    data = await state.get_data()
    if data.get("own_script"):
        # Own-script mode: the text is final by definition — never re-run the
        # plot stage on it, that would reintroduce plot rewriting.
        await callback.message.answer(
            "📝 Own-script mode — the script is used as-is and is not regenerated.\n"
            "Send an edited version as a message, or click Continue.",
            reply_markup=get_prompt_keyboard(),
        )
        await state.set_state(GenerationState.ENHANCE_PROMPT)
        return
    await state.update_data(generation_in_progress=True, generation_started_at=time.time())
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
    finally:
        await state.update_data(generation_in_progress=False)
    await state.set_state(GenerationState.ENHANCE_PROMPT)


# ─────────────────────────────────────────────
# IMAGE STAGE
# ─────────────────────────────────────────────

@router.callback_query(GenerationState.CONFIRM_IMAGE, lambda c: c.data == "image_prompt_change", IsAllowed(allowed_users))
async def handle_image_prompt_change(callback: CallbackQuery, state: FSMContext):
    await callback.answer()
    await callback.message.edit_reply_markup(reply_markup=None)
    data = await state.get_data()

    # For reference models with manual/auto mode, return to mode selection
    if data.get("ref_mode") in ("manual", "auto"):
        db = container.inject(DBService)
        settings = await db.get_settings(callback.from_user.id, callback.message.chat.id)
        image_count = int(settings.get("image_count", 1))
        await callback.message.answer(
            f"Change {image_count} reference photos?\n\nHow would you like to describe them?",
            reply_markup=get_ref_image_mode_keyboard(),
        )
        await state.set_state(GenerationState.SELECT_REF_MODE)
        return

    enhance_prompt = data.get("enhance_prompt", "")
    await send_long_message(
        callback.message,
        f"Your prompt: {enhance_prompt}\nSend a corrected version or click Continue",
        keyboard=get_prompt_keyboard(),
    )
    await state.set_state(GenerationState.ENHANCE_PROMPT)


@router.callback_query(GenerationState.CONFIRM_IMAGE, lambda c: c.data == "image_regenerate", IsAllowed(allowed_users))
async def handle_image_regenerate(callback: CallbackQuery, state: FSMContext):
    # Guard against double-taps — otherwise duplicate images are generated.
    if await _is_generating(state):
        await callback.answer("Already generating — please wait.", show_alert=False)
        return
    await callback.answer()
    await callback.message.edit_reply_markup(reply_markup=None)

    data = await state.get_data()
    ref_mode = data.get("ref_mode")
    db = container.inject(DBService)
    settings = await db.get_settings(callback.from_user.id, callback.message.chat.id)

    # For reference models in manual/auto mode, redirect to re-enter mode selection
    if ref_mode in ("manual", "auto"):
        image_count = int(settings.get("image_count", 1))
        await callback.message.answer(
            f"Regenerate {image_count} reference photos?\n\nHow would you like to describe them?",
            reply_markup=get_ref_image_mode_keyboard(),
        )
        await state.set_state(GenerationState.SELECT_REF_MODE)
        return

    # Regular non-reference image regeneration
    await state.update_data(generation_in_progress=True, generation_started_at=time.time())
    try:
        gemini = container.inject(GeminiService)
        imagegen = container.inject(ImageGenService)
        image_count = int(settings.get("image_count", 1))
        await callback.message.answer(
            f"Creating {image_count} image{'s' if image_count > 1 else ''}…"
        )
        enhance_prompt = data.get("enhance_prompt", "")
        image_prompt = await _build_image_prompt(
            enhance_prompt, settings, gemini, strict_script=bool(data.get("own_script"))
        )
        image_paths = await imagegen.generate_many(
            prompt=image_prompt,
            model=settings.get("image_model") or "",
            video_model=settings.get("video_model") or "seedance",
            count=image_count,
            notify=callback.message.answer,
        )
        await state.update_data(image_paths=image_paths, image_prompt=image_prompt)
        await _send_image_preview(callback.message, image_paths)
        await callback.message.answer("Image(s) created.\nContinue?", reply_markup=get_image_keyboard())
    except Exception as e:
        await callback.message.answer(f"Error: {str(e)}\nTry again", reply_markup=get_image_keyboard())
    finally:
        await state.update_data(generation_in_progress=False)
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

        # Reference image context for system prompt
        ref_roles = data.get("ref_roles") if data.get("ref_mode") == "auto" else None
        ref_plan = data.get("ref_combination_plan", "") if data.get("ref_mode") == "auto" else ""
        ref_descriptions = data.get("ref_descriptions") if data.get("ref_mode") == "manual" else None

        await callback.message.answer("⏳ Generating video prompt…")
        text_model = settings.get("text_model") or ""
        video_prompt, hashtags = await asyncio.gather(
            _build_video_prompt(enhance_prompt, settings, gemini, ref_roles=ref_roles, ref_combination_plan=ref_plan, ref_descriptions=ref_descriptions),
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
        ref_roles = data.get("ref_roles") if data.get("ref_mode") == "auto" else None
        ref_plan = data.get("ref_combination_plan", "") if data.get("ref_mode") == "auto" else ""
        ref_descriptions = data.get("ref_descriptions") if data.get("ref_mode") == "manual" else None
        video_prompt = await _build_video_prompt(
            data.get("enhance_prompt", ""), settings, gemini,
            strict_script=bool(data.get("own_script")),
            ref_roles=ref_roles, ref_combination_plan=ref_plan, ref_descriptions=ref_descriptions,
        )
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
        ref_roles = data.get("ref_roles") if data.get("ref_mode") == "auto" else None
        ref_plan = data.get("ref_combination_plan", "") if data.get("ref_mode") == "auto" else ""
        ref_descriptions = data.get("ref_descriptions") if data.get("ref_mode") == "manual" else None
        video_prompt = await _build_video_prompt(
            data.get("enhance_prompt", ""), settings, gemini,
            strict_script=bool(data.get("own_script")),
            ref_roles=ref_roles, ref_combination_plan=ref_plan, ref_descriptions=ref_descriptions,
        )
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

        # 🗣 Voiceover OFF: hard programmatic guarantee — no TTS regardless of
        # what the script model produced. Also unlocks native model audio.
        if not settings.get("voiceover_enabled", True):
            if video_voiceover:
                await callback.message.answer(
                    "🗣 Voiceover is OFF in settings — generating without narration."
                )
            video_voiceover = ""

        # ⚡ Speed changes the final length after generation — warn up front so
        # the requested duration isn't a surprise.
        speed = float(settings.get("video_speed", 1.0))
        if speed != 1.0:
            td = settings.get("target_duration", DEFAULT_TARGET_DURATION)
            await callback.message.answer(
                f"⚡ Speed {speed:g}× is ON — the final video will be ~{round(td / speed)}s "
                f"(from {td}s generated). Set Speed to normal in ⚙️ for exact length."
            )

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
            has_native_audio=gen_result.get("has_native_audio", False),
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
            sfx_description=data.get("remix_sfx_description") or "",
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
        err = str(e)
        if "insufficient balance" in err.lower() or '"code":402' in err:
            err = (
                "💳 На Atlas Cloud закончился баланс — пополните его на atlascloud.ai "
                "и нажмите Regenerate."
            )
        await callback.message.answer(f"Error: {err}\nTry again", reply_markup=get_video_keyboard())
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

        # Re-apply the generation-time ⚡ speed factor: base/subbed are stored
        # unsped (they're needed for editing/re-burning), so without this the
        # toggle would silently show an unsped video and lose the speed effect.
        speed = float(data.get("applied_speed") or 1.0)
        if speed != 1.0:
            cache_key = "video_path_subbed_sped" if new_state else "video_path_base_sped"
            cached = data.get(cache_key)
            if cached and os.path.exists(cached):
                new_path = cached
            else:
                await callback.message.answer(f"⚡ Applying {speed:g}× speed…")
                gemini = container.inject(GeminiService)
                sped = await gemini.apply_speed(new_path, speed)
                await state.update_data(**{cache_key: sped})
                new_path = sped

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
        # Replace base video with edited version; keep subtitles state.
        # Speed caches belong to the PREVIOUS video — reset them, and the
        # edited video itself is unsped (applied_speed=1.0).
        await state.update_data(
            video_path=edited_path,
            video_path_base=edited_path,
            video_path_raw=edited_path,
            video_path_subbed=None,
            subtitles_on=False,
            applied_speed=1.0,
            video_path_base_sped=None,
            video_path_subbed_sped=None,
        )
        await message.answer("✅ Edit applied!\nContinue?", reply_markup=get_video_keyboard(False))
    except Exception as e:
        logger.error(f"Video-Edit error: {e}", exc_info=True)
        await message.answer(f"Error: {e}\nTry again", reply_markup=get_video_keyboard(subtitles_on))
    finally:
        await state.update_data(generation_in_progress=False)
        await state.set_state(GenerationState.CONFIRM_VIDEO)
