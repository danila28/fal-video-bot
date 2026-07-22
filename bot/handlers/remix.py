"""Remix flow: user sends a reference video file → Gemini extracts its formula
→ user confirms → the normal generation pipeline produces a NEW similar video.

The reference video is used ONLY as analysis input — it is never republished,
cut up, or fed into video-to-video restyling.
"""

import html
import logging
import math
import os
import time
import uuid

from aiogram import F, Router
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, Message

from bot.guard import IsAllowed
from bot.keyboards import get_remix_formula_keyboard
from bot.states import RemixState, GenerationState
from services.db import DBService
from services.gemini import GeminiService
from utils import container
from utils.consts import allowed_users
from utils.tg import send_long_message
from services.imagegen import ImageGenService
from bot.handlers.common import (
    DEFAULT_TARGET_DURATION,
    _T2V_VIDEO_MODELS,
    _build_image_prompt,
    _clip_duration_for_model,
    _is_generating,
    _show_video_prompt,
    _split_video_prompt,
)

router = Router()
logger = logging.getLogger(__name__)

# Telegram Bot API refuses to serve files larger than 20 MB to bots.
_TG_BOT_DOWNLOAD_LIMIT = 20 * 1024 * 1024


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

@router.callback_query(F.data == "remix:start", IsAllowed(allowed_users))
async def handle_remix_start(query: CallbackQuery, state: FSMContext):
    await query.answer()

    if await _is_generating(state):
        await query.message.answer("⚠️ Generation in progress — please wait until it finishes.")
        return

    db = container.inject(DBService)
    chat_accounts = await db.get_chat_accounts(query.message.chat.id)
    if not chat_accounts:
        await query.message.answer("No accounts configured for this chat\nTap ⚙️ Settings → 📤 Accounts")
        return

    await state.clear()
    await query.message.answer(
        "🔁 Send me the reference video as a FILE (up to 20 MB, up to 5 min).\n\n"
        "Download the video you like to your phone and forward it here — "
        "I'll analyze its formula (hook, pacing, structure) and generate "
        "a brand-new video with the same vibe.\n\n"
        "⚠️ Telegram limits bot downloads to 20 MB. If your video is bigger, "
        "trim or compress it first."
    )
    await state.set_state(RemixState.WAITING_LINK)


# ─────────────────────────────────────────────
# REFERENCE UPLOAD
# ─────────────────────────────────────────────

@router.message(RemixState.WAITING_LINK, F.video | F.document | F.video_note, IsAllowed(allowed_users))
async def handle_remix_upload(message: Message, state: FSMContext):
    """Receive the reference video file, download it and analyze."""
    media = message.video or message.video_note or message.document

    if message.document:
        mime = (message.document.mime_type or "").lower()
        if not mime.startswith("video/"):
            await message.answer("❌ This is not a video file. Send an mp4/mov video.")
            return

    file_size = getattr(media, "file_size", 0) or 0
    if file_size > _TG_BOT_DOWNLOAD_LIMIT:
        await message.answer(
            f"❌ File is too big: {file_size / (1024 * 1024):.1f} MB.\n"
            "Telegram allows bots to download at most 20 MB.\n\n"
            "Trim the video or compress it (any phone gallery app can do this) and send again."
        )
        return

    await state.set_state(RemixState.ANALYZING)
    status_msg = await message.answer("⏳ Downloading your video…")

    gemini = container.inject(GeminiService)
    video_path = os.path.join(gemini.static_dir, f"{uuid.uuid4()}_ref.mp4")

    try:
        os.makedirs(gemini.static_dir, exist_ok=True)
        await message.bot.download(media, destination=video_path)

        await status_msg.edit_text("⏳ Analyzing the video formula (this may take a minute)…")

        db = container.inject(DBService)
        settings = await db.get_settings(message.from_user.id, message.chat.id)
        target_duration = settings.get("target_duration", DEFAULT_TARGET_DURATION)
        video_model = (settings.get("video_model") or "seedance").lower()

        # Mirror the pipeline's own shot planning so the analysis output
        # matches what _normalize_shot_durations / the generators expect.
        clip_dur = _clip_duration_for_model(video_model)
        num_scenes = max(1, math.ceil(target_duration / clip_dur))
        min_shot = 3 if video_model.startswith("kling") else 4

        analysis = await gemini.analyze_reference_video(
            video_path,
            target_duration=target_duration,
            num_scenes=num_scenes,
            min_shot_seconds=min_shot,
            max_shot_seconds=clip_dur,
        )

        await state.update_data(remix_analysis=analysis)
        await _show_remix_formula(message, analysis, target_duration, video_model)
        await state.set_state(RemixState.CONFIRM_FORMULA)

    except Exception as e:
        logger.exception("Remix analysis failed")
        await status_msg.edit_text(
            f"❌ Failed to analyze video: {e}\n\nSend another video or tap 🎬 Generate video."
        )
        await state.set_state(RemixState.WAITING_LINK)
    finally:
        # The reference is only needed for analysis — delete it right away.
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
        except OSError:
            pass


@router.message(RemixState.WAITING_LINK, F.text, IsAllowed(allowed_users))
async def handle_remix_text(message: Message, state: FSMContext):
    """Links are no longer auto-downloaded — guide the user to upload the file."""
    if (message.text or "").strip().startswith(("http://", "https://")):
        await message.answer(
            "🔗 I don't download from links anymore — platforms block it.\n\n"
            "Save the video to your device and send it here as a file (≤20 MB)."
        )
    else:
        await message.answer("Send the reference video as a file (≤20 MB), or /start to cancel.")


# ─────────────────────────────────────────────
# FORMULA PREVIEW & CONFIRMATION
# ─────────────────────────────────────────────

async def _show_remix_formula(
    message: Message, analysis: dict, target_duration: int, video_model: str
) -> None:
    """Display the extracted video formula for review."""
    shots = analysis.get("shots", [])
    metadata = analysis.get("metadata", {}) if isinstance(analysis.get("metadata"), dict) else {}
    title = analysis.get("title", "")
    voiceover = analysis.get("voiceover", "")
    ref_dur = metadata.get("reference_duration_seconds")

    text = "📊 <b>Extracted Video Formula</b>\n\n"
    if title:
        text += f"🎯 <b>Hook:</b> {html.escape(str(title))}\n\n"
    if voiceover:
        vo_preview = voiceover[:300] + ("…" if len(voiceover) > 300 else "")
        text += f"🗣 <b>Voiceover:</b> {html.escape(vo_preview)}\n\n"

    text += f"🎭 Tone: {html.escape(str(metadata.get('detected_tone', '—')))}\n"
    text += f"⚡ Tempo: {html.escape(str(metadata.get('detected_tempo', '—')))}\n"
    text += f"🌐 Language: {html.escape(str(metadata.get('detected_language', '—')))}\n"
    if ref_dur:
        text += f"⏱ Reference: {ref_dur}s → your video: {target_duration}s (change in ⚙️ Settings)\n"
        if target_duration < float(ref_dur) * 0.6:
            text += (
                f"\n⚠️ Target ({target_duration}s) is much shorter than the reference ({ref_dur}s) — "
                "the whole formula gets squeezed into fewer scenes and loses its pacing. "
                f"Recommended: set ⏱ Duration to ~{max(15, round(float(ref_dur) / 5) * 5)}s in ⚙️ Settings, "
                "then re-send the video.\n"
            )
    text += f"\n<b>Structure ({len(shots)} scenes):</b>\n"

    for i, shot in enumerate(shots, 1):
        prompt = str(shot.get("scene_prompt", ""))
        dur = shot.get("duration_seconds", "?")
        trans = shot.get("transition", "cut")
        preview = prompt[:120] + ("…" if len(prompt) > 120 else "")
        text += f"{i}. {html.escape(preview)} ({dur}s, {trans})\n"

    if not video_model.endswith("_ref"):
        text += (
            "\n💡 Tip: reference videos usually feature the same person in every scene. "
            "For character consistency pick a 🎭 Ref model in ⚙️ Settings → 🎬 Video model."
        )

    text += "\n\n✅ Generate a new video based on this formula?"
    await send_long_message(message, text, keyboard=get_remix_formula_keyboard(), parse_mode="HTML")


@router.callback_query(F.data == "remix:confirm_formula", IsAllowed(allowed_users))
async def handle_confirm_formula(query: CallbackQuery, state: FSMContext):
    """Hand the extracted formula over to the normal generation pipeline."""
    if await _is_generating(state):
        await query.answer("Already generating — please wait.", show_alert=False)
        return
    await query.answer()

    data = await state.get_data()
    analysis = data.get("remix_analysis") or {}
    if not analysis.get("shots"):
        await query.message.answer("❌ Analysis data lost. Please start over.")
        await state.clear()
        return

    import json
    shots = analysis.get("shots", [])

    gemini = container.inject(GeminiService)

    # Rich concept text — fallback source for reference-image prompts when the
    # analysis didn't return a dedicated image_prompt.
    scenario_text = "\n".join(
        filter(None, [
            analysis.get("title", ""),
            analysis.get("voiceover", ""),
            "Scenes: " + " | ".join(str(s.get("scene_prompt", ""))[:150] for s in shots[:4]),
        ])
    ) or "Video based on reference formula"

    db = container.inject(DBService)
    settings = await db.get_settings(query.from_user.id, query.message.chat.id)
    video_model = (settings.get("video_model") or "seedance").lower()

    # I2V and _ref models anchor generation on reference image(s); the remix
    # flow skips the normal image stage, so produce them here. _ref models
    # hard-require images (RuntimeError in the pipeline otherwise).
    image_paths: list[str] = []
    if video_model not in _T2V_VIDEO_MODELS:
        image_count = int(settings.get("image_count", 1))
        await state.update_data(generation_in_progress=True, generation_started_at=time.time())
        await query.message.answer(
            f"⏳ Generating {image_count} reference image{'s' if image_count > 1 else ''} from the formula…"
        )
        try:
            imagegen = container.inject(ImageGenService)
            # The analysis returns a dedicated character+style description in
            # English — a far better photo source than the scenario mashup.
            image_source = analysis.get("image_prompt") or scenario_text
            image_prompt = await _build_image_prompt(
                image_source, settings, gemini, strict_script=True
            )
            image_paths = await imagegen.generate_many(
                prompt=image_prompt,
                model=settings.get("image_model") or "",
                video_model=video_model,
                count=image_count,
                notify=query.message.answer,
            )
        except Exception as e:
            logger.error(f"Remix reference image generation failed: {e}")
            if video_model.endswith("_ref"):
                await query.message.answer(
                    f"❌ Reference images are required for {video_model} but generation failed: {e}\n"
                    "Try again or switch the video model in ⚙️ Settings."
                )
                await state.update_data(generation_in_progress=False)
                return
            # Plain I2V has a no-image fallback in the pipeline — continue.
            await query.message.answer("⚠️ Image generation failed — continuing without reference images.")
        finally:
            await state.update_data(generation_in_progress=False)

    # Reference models route prompts through @ImageN tags — the analysis
    # doesn't know about them, so anchor every shot to the generated photos.
    if video_model.endswith("_ref") and image_paths:
        tags = ", ".join(f"@Image{i + 1}" for i in range(len(image_paths)))
        shots = [
            {**s, "scene_prompt": f"Use {tags} as the character/style reference. {s.get('scene_prompt', '')}"}
            for s in shots
        ]

    script_json = json.dumps(
        {
            "title": analysis.get("title", ""),
            "voiceover": analysis.get("voiceover", ""),
            "shots": shots,
        },
        ensure_ascii=False,
    )
    scene, voiceover = _split_video_prompt(script_json, gemini)

    # Fill exactly the state keys the vp_ok handler in generation.py consumes:
    # video_scene / video_voiceover / video_script_json (+ image_paths).
    await state.update_data(
        raw_prompt=scenario_text,
        enhance_prompt=scenario_text,
        own_script=True,
        image_paths=image_paths,
        image_path=image_paths[0] if image_paths else None,
        image_prompt="",
        video_scene=scene,
        video_voiceover=voiceover,
        video_script_json=script_json,
        remix_mode=True,
    )

    await _show_video_prompt(query.message, scene, voiceover)
    await state.set_state(GenerationState.CONFIRM_VIDEO_PROMPT)


@router.callback_query(F.data == "remix:edit_formula", IsAllowed(allowed_users))
async def handle_edit_formula(query: CallbackQuery, state: FSMContext):
    await query.answer()
    await query.message.answer(
        "После нажатия ✅ Generate вы сможете отредактировать сцены и войсовер "
        "на следующем шаге (кнопки ✏️ Edit scene / ✏️ Edit voiceover)."
    )


@router.callback_query(F.data == "remix:cancel", IsAllowed(allowed_users))
async def handle_remix_cancel(query: CallbackQuery, state: FSMContext):
    await query.answer()
    await query.message.answer("Cancelled. Tap 🎬 Generate video to start over.")
    await state.clear()
