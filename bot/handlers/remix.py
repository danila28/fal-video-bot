"""Remix flow: user sends a reference video file → Gemini extracts its formula
→ user reviews/edits the formula and the reference photos → the normal
generation pipeline produces a NEW similar video.

The reference video is used ONLY as analysis input — it is never republished,
cut up, or fed into video-to-video restyling.

Also hosts the formula library: confirmed formulas can be saved and re-used
later, optionally adapted to a new topic by a cheap Gemini text call.
"""

import html
import json
import logging
import math
import os
import time
import uuid

from aiogram import F, Router
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, Message

from bot.guard import IsAllowed
from bot.keyboards import (
    get_formula_library_keyboard,
    get_formula_topic_keyboard,
    get_remix_formula_keyboard,
    get_remix_images_keyboard,
)
from bot.states import RemixState, GenerationState
from services.db import DBService
from services.gemini import GeminiService
from services.imagegen import ImageGenService
from utils import container
from utils.consts import allowed_users
from utils.tg import send_long_message
from bot.handlers.common import (
    DEFAULT_TARGET_DURATION,
    _T2V_VIDEO_MODELS,
    _build_image_prompt,
    _clip_duration_for_model,
    _is_generating,
    _kling_min_clip_duration,
    _show_video_prompt,
    _split_video_prompt,
)

router = Router()
logger = logging.getLogger(__name__)

# Telegram Bot API refuses to serve files larger than 20 MB to bots.
_TG_BOT_DOWNLOAD_LIMIT = 20 * 1024 * 1024


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _plan_shots(settings: dict, target_duration: int | None = None):
    """Mirror the pipeline's own shot planning so analysis output matches
    what _normalize_shot_durations / the generators expect."""
    target = target_duration or settings.get("target_duration", DEFAULT_TARGET_DURATION)
    video_model = (settings.get("video_model") or "seedance").lower()
    clip_dur = _clip_duration_for_model(video_model)
    num_scenes = max(1, math.ceil(target / clip_dur))
    min_shot = _kling_min_clip_duration(video_model) if video_model.startswith("kling") else 4
    return target, video_model, clip_dur, num_scenes, min_shot


def _match_secs(analysis: dict, target_duration: int) -> int:
    """Return the 'match reference duration' value, or 0 when not worth showing."""
    metadata = analysis.get("metadata") if isinstance(analysis.get("metadata"), dict) else {}
    ref_dur = metadata.get("reference_duration_seconds")
    if not ref_dur:
        return 0
    match = max(10, min(90, int(round(float(ref_dur)))))
    if match == target_duration:
        return 0
    # Only offer when the difference is meaningful (>25%).
    if 0.75 * float(ref_dur) <= target_duration <= 1.25 * float(ref_dur):
        return 0
    return match


async def _show_remix_formula(
    message: Message,
    analysis: dict,
    target_duration: int,
    video_model: str,
    saved: bool = False,
) -> None:
    """Display the extracted video formula for review."""
    shots = analysis.get("shots", [])
    metadata = analysis.get("metadata", {}) if isinstance(analysis.get("metadata"), dict) else {}
    title = analysis.get("title", "")
    voiceover = analysis.get("voiceover", "")
    sfx = analysis.get("sfx_description", "")
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
    if sfx:
        text += f"🔊 Sound: {html.escape(str(sfx)[:100])}\n"
    if ref_dur:
        text += f"⏱ Reference: {ref_dur}s → your video: {target_duration}s\n"

    text += f"\n<b>Structure ({len(shots)} scenes):</b>\n"
    for i, shot in enumerate(shots, 1):
        prompt = str(shot.get("scene_prompt", ""))
        dur = shot.get("duration_seconds", "?")
        trans = shot.get("transition", "cut")
        preview = prompt[:120] + ("…" if len(prompt) > 120 else "")
        text += f"{i}. {html.escape(preview)} ({dur}s, {trans})\n"

    if not video_model.endswith("_ref"):
        text += (
            "\n💡 Tip: for character consistency across scenes pick a 🎭 Ref model "
            "in ⚙️ Settings → 🎬 Video model."
        )

    text += "\n\n✅ Generate a new video based on this formula?"
    await send_long_message(
        message,
        text,
        keyboard=get_remix_formula_keyboard(
            match_secs=_match_secs(analysis, target_duration), saved=saved
        ),
        parse_mode="HTML",
    )


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
# REFERENCE UPLOAD & ANALYSIS
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
        target, video_model, clip_dur, num_scenes, min_shot = _plan_shots(settings)

        analysis = await gemini.analyze_reference_video(
            video_path,
            target_duration=target,
            num_scenes=num_scenes,
            min_shot_seconds=min_shot,
            max_shot_seconds=clip_dur,
        )

        await state.update_data(remix_analysis=analysis, remix_saved=False)
        await _show_remix_formula(message, analysis, target, video_model)
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
    """Links are not auto-downloaded — guide the user to upload the file."""
    if (message.text or "").strip().startswith(("http://", "https://")):
        await message.answer(
            "🔗 I don't download from links — platforms block it.\n\n"
            "Save the video to your device and send it here as a file (≤20 MB)."
        )
    else:
        await message.answer("Send the reference video as a file (≤20 MB), or /start to cancel.")


# ─────────────────────────────────────────────
# FORMULA ACTIONS: match duration / edit / save
# ─────────────────────────────────────────────

@router.callback_query(F.data == "remix:match_duration", IsAllowed(allowed_users))
async def handle_match_duration(query: CallbackQuery, state: FSMContext):
    """One tap: set target duration to the reference's and re-time the formula."""
    data = await state.get_data()
    analysis = data.get("remix_analysis") or {}
    if not analysis.get("shots"):
        await query.answer("Formula lost — start over", show_alert=True)
        return

    db = container.inject(DBService)
    settings = await db.get_settings(query.from_user.id, query.message.chat.id)
    current_target = settings.get("target_duration", DEFAULT_TARGET_DURATION)
    new_target = _match_secs(analysis, current_target)
    if not new_target:
        await query.answer("Already matches the reference", show_alert=False)
        return

    await query.answer()
    await query.message.edit_reply_markup(reply_markup=None)
    await query.message.answer(f"⏳ Re-timing the formula to {new_target}s…")

    try:
        # Persist so _normalize_shot_durations and the generators use it too.
        await db.update_settings(
            query.from_user.id, query.message.chat.id, {"target_duration": new_target}
        )
        settings["target_duration"] = new_target
        target, video_model, clip_dur, num_scenes, min_shot = _plan_shots(settings, new_target)

        gemini = container.inject(GeminiService)
        refined = await gemini.refine_reference_formula(
            analysis,
            f"Re-time the formula to {new_target} seconds, preserving the proportional "
            "pacing and all content. Expand scene descriptions if more shots are needed.",
            target_duration=target,
            num_scenes=num_scenes,
            min_shot_seconds=min_shot,
            max_shot_seconds=clip_dur,
            model=settings.get("text_model") or "",
        )
        await state.update_data(remix_analysis=refined, remix_saved=False)
        await _show_remix_formula(query.message, refined, target, video_model)
        await state.set_state(RemixState.CONFIRM_FORMULA)
    except Exception as e:
        logger.exception("Formula re-timing failed")
        await query.message.answer(f"❌ Re-timing failed: {e}\nThe old formula is still active.")
        await _show_remix_formula(
            query.message, analysis, new_target,
            (settings.get("video_model") or "seedance").lower(),
        )


@router.callback_query(F.data == "remix:edit_formula", IsAllowed(allowed_users))
async def handle_edit_formula(query: CallbackQuery, state: FSMContext):
    await query.answer()
    await query.message.answer(
        "✏️ Send your edit as plain text, e.g.:\n"
        "• \"make the main character a woman, more serious tone\"\n"
        "• \"remove the second scene, add a close-up at the end\"\n"
        "• \"change setting to a cyberpunk city at night\""
    )
    await state.set_state(RemixState.EDIT_FORMULA)


@router.message(RemixState.EDIT_FORMULA, F.text, IsAllowed(allowed_users))
async def handle_edit_formula_text(message: Message, state: FSMContext):
    data = await state.get_data()
    analysis = data.get("remix_analysis") or {}
    if not analysis.get("shots"):
        await message.answer("❌ Formula lost. Start over with 🎬 Generate video.")
        await state.clear()
        return

    status = await message.answer("⏳ Applying your edit…")
    db = container.inject(DBService)
    settings = await db.get_settings(message.from_user.id, message.chat.id)
    target, video_model, clip_dur, num_scenes, min_shot = _plan_shots(settings)

    try:
        gemini = container.inject(GeminiService)
        refined = await gemini.refine_reference_formula(
            analysis,
            message.text,
            target_duration=target,
            num_scenes=num_scenes,
            min_shot_seconds=min_shot,
            max_shot_seconds=clip_dur,
            model=settings.get("text_model") or "",
        )
        await state.update_data(remix_analysis=refined, remix_saved=False)
        await status.edit_text("✅ Edit applied.")
        await _show_remix_formula(message, refined, target, video_model)
    except Exception as e:
        logger.exception("Formula edit failed")
        await status.edit_text(
            f"❌ Edit failed: {e}\nThe previous formula is still active — try rephrasing."
        )
        await _show_remix_formula(message, analysis, target, video_model)
    await state.set_state(RemixState.CONFIRM_FORMULA)


@router.callback_query(F.data == "remix:save_formula", IsAllowed(allowed_users))
async def handle_save_formula(query: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    analysis = data.get("remix_analysis") or {}
    if not analysis.get("shots"):
        await query.answer("Formula lost — nothing to save", show_alert=True)
        return

    db = container.inject(DBService)
    title = analysis.get("title") or (analysis.get("voiceover") or "")[:60] or "Untitled formula"
    formula_id = await db.save_formula(
        query.message.chat.id, query.from_user.id, title, analysis
    )
    await state.update_data(remix_saved=True)
    await query.answer(f"💾 Saved as #{formula_id}", show_alert=False)

    settings = await db.get_settings(query.from_user.id, query.message.chat.id)
    target = settings.get("target_duration", DEFAULT_TARGET_DURATION)
    try:
        await query.message.edit_reply_markup(
            reply_markup=get_remix_formula_keyboard(
                match_secs=_match_secs(analysis, target), saved=True
            )
        )
    except Exception:
        pass  # message may be too old to edit — the save itself succeeded


@router.callback_query(F.data == "remix:noop", IsAllowed(allowed_users))
async def handle_noop(query: CallbackQuery):
    await query.answer()


# ─────────────────────────────────────────────
# FORMULA LIBRARY
# ─────────────────────────────────────────────

@router.callback_query(F.data == "remix:lib", IsAllowed(allowed_users))
async def handle_library(query: CallbackQuery, state: FSMContext):
    await query.answer()

    db = container.inject(DBService)
    formulas = await db.list_formulas(query.message.chat.id)
    if not formulas:
        await query.message.answer(
            "📚 No saved formulas yet.\n\n"
            "Run 🔁 Remix from link and tap 💾 Save formula on one you like — "
            "it will appear here for reuse."
        )
        return
    await query.message.answer(
        "📚 <b>Your saved formulas</b> — tap one to reuse it:",
        reply_markup=get_formula_library_keyboard(formulas),
        parse_mode="HTML",
    )


@router.callback_query(F.data.startswith("remix:lib_use:"), IsAllowed(allowed_users))
async def handle_library_use(query: CallbackQuery, state: FSMContext):
    await query.answer()

    db = container.inject(DBService)
    chat_accounts = await db.get_chat_accounts(query.message.chat.id)
    if not chat_accounts:
        await query.message.answer("No accounts configured for this chat\nTap ⚙️ Settings → 📤 Accounts")
        return

    formula_id = int(query.data.rsplit(":", 1)[1])
    analysis = await db.get_formula(formula_id, query.message.chat.id)
    if not analysis or not analysis.get("shots"):
        await query.message.answer("❌ Formula not found (deleted?). Pick another one.")
        return

    await state.clear()
    await state.update_data(remix_analysis=analysis, remix_saved=True)
    await query.message.answer(
        f"📄 Formula #{formula_id} loaded.\n\n"
        "Send a NEW TOPIC as text (e.g. \"about coffee\", \"about morning workouts\") — "
        "I'll adapt the formula to it.\n\nOr use it exactly as saved:",
        reply_markup=get_formula_topic_keyboard(),
    )
    await state.set_state(RemixState.LIBRARY_TOPIC)


@router.message(RemixState.LIBRARY_TOPIC, F.text, IsAllowed(allowed_users))
async def handle_library_topic(message: Message, state: FSMContext):
    data = await state.get_data()
    analysis = data.get("remix_analysis") or {}
    if not analysis.get("shots"):
        await message.answer("❌ Formula lost. Open 📚 My formulas again.")
        await state.clear()
        return

    status = await message.answer("⏳ Adapting the formula to your topic…")
    db = container.inject(DBService)
    settings = await db.get_settings(message.from_user.id, message.chat.id)
    target, video_model, clip_dur, num_scenes, min_shot = _plan_shots(settings)

    try:
        gemini = container.inject(GeminiService)
        refined = await gemini.refine_reference_formula(
            analysis,
            f"Adapt this formula to a completely new topic: «{message.text.strip()}». "
            "Keep the structure, pacing, tone, hook style and emotional arc — replace the "
            "subject matter, voiceover content and scene visuals to fit the new topic.",
            target_duration=target,
            num_scenes=num_scenes,
            min_shot_seconds=min_shot,
            max_shot_seconds=clip_dur,
            model=settings.get("text_model") or "",
        )
        await state.update_data(remix_analysis=refined, remix_saved=False)
        await status.edit_text("✅ Formula adapted.")
        await _show_remix_formula(message, refined, target, video_model)
        await state.set_state(RemixState.CONFIRM_FORMULA)
    except Exception as e:
        logger.exception("Formula topic adaptation failed")
        await status.edit_text(f"❌ Adaptation failed: {e}\nTry a different wording.")


@router.callback_query(F.data == "remix:lib_asis", IsAllowed(allowed_users))
async def handle_library_asis(query: CallbackQuery, state: FSMContext):
    await query.answer()
    data = await state.get_data()
    analysis = data.get("remix_analysis") or {}
    if not analysis.get("shots"):
        await query.message.answer("❌ Formula lost. Open 📚 My formulas again.")
        await state.clear()
        return

    db = container.inject(DBService)
    settings = await db.get_settings(query.from_user.id, query.message.chat.id)
    target, video_model, *_ = _plan_shots(settings)
    await _show_remix_formula(query.message, analysis, target, video_model, saved=True)
    await state.set_state(RemixState.CONFIRM_FORMULA)


@router.callback_query(F.data.startswith("remix:lib_del:"), IsAllowed(allowed_users))
async def handle_library_delete(query: CallbackQuery, state: FSMContext):
    formula_id = int(query.data.rsplit(":", 1)[1])
    db = container.inject(DBService)
    deleted = await db.delete_formula(formula_id, query.message.chat.id)
    await query.answer("🗑 Deleted" if deleted else "Not found", show_alert=False)

    formulas = await db.list_formulas(query.message.chat.id)
    try:
        if formulas:
            await query.message.edit_reply_markup(
                reply_markup=get_formula_library_keyboard(formulas)
            )
        else:
            await query.message.edit_text("📚 No saved formulas left.")
    except Exception:
        pass


# ─────────────────────────────────────────────
# CONFIRM: photos preview → video prompt
# ─────────────────────────────────────────────

# Complementary roles so multi-photo references show DIFFERENT views of the
# same character instead of near-duplicates of one prompt.
_REF_PHOTO_ROLES = [
    "full-body shot: the character standing, entire outfit visible head to toe",
    "close-up portrait: face and shoulders, facial details clearly visible",
    "three-quarter view in a dynamic pose, showing the character from a different angle",
    "the character in the story's environment, medium shot with background context",
]


async def _generate_reference_images(
    message: Message, state: FSMContext, settings: dict, analysis: dict
) -> list[str]:
    """Generate reference photos from the formula's image_prompt. Raises on failure."""
    import asyncio

    gemini = container.inject(GeminiService)
    imagegen = container.inject(ImageGenService)
    video_model = (settings.get("video_model") or "seedance").lower()
    image_count = int(settings.get("image_count", 1))

    data = await state.get_data()
    image_prompt = data.get("remix_image_prompt", "")
    if not image_prompt:
        # The analysis returns a dedicated character+style description in
        # English — a far better photo source than a scenario mashup.
        image_source = analysis.get("image_prompt") or "\n".join(
            filter(None, [
                analysis.get("title", ""),
                analysis.get("voiceover", ""),
                " | ".join(str(s.get("scene_prompt", ""))[:150] for s in analysis.get("shots", [])[:4]),
            ])
        )
        image_prompt = await _build_image_prompt(image_source, settings, gemini, strict_script=True)
        await state.update_data(remix_image_prompt=image_prompt)

    if image_count <= 1:
        return await imagegen.generate_many(
            prompt=image_prompt,
            model=settings.get("image_model") or "",
            video_model=video_model,
            count=1,
            notify=message.answer,
        )

    # One prompt × N runs yields near-duplicates — give every photo its own
    # complementary role while pinning character and style.
    prompts = [
        (
            f"{image_prompt}\n\n"
            f"This photo specifically: {_REF_PHOTO_ROLES[i % len(_REF_PHOTO_ROLES)]}. "
            "Exactly the same character, outfit, colors and art style as the other photos in this set."
        )
        for i in range(image_count)
    ]
    model = settings.get("image_model") or ""
    return list(await asyncio.gather(*[
        imagegen.generate(p, model, video_model, message.answer) for p in prompts
    ]))


@router.callback_query(F.data == "remix:confirm_formula", IsAllowed(allowed_users))
async def handle_confirm_formula(query: CallbackQuery, state: FSMContext):
    """Formula confirmed → T2V goes straight to the video prompt; I2V/_ref
    models generate reference photos and show them for review first."""
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

    db = container.inject(DBService)
    settings = await db.get_settings(query.from_user.id, query.message.chat.id)
    video_model = (settings.get("video_model") or "seedance").lower()

    if video_model in _T2V_VIDEO_MODELS:
        await _finalize_remix(query.message, state, settings, analysis, image_paths=[])
        return

    image_count = int(settings.get("image_count", 1))
    await state.update_data(generation_in_progress=True, generation_started_at=time.time())
    await query.message.answer(
        f"⏳ Generating {image_count} reference image{'s' if image_count > 1 else ''} from the formula…"
    )
    try:
        image_paths = await _generate_reference_images(query.message, state, settings, analysis)
        await state.update_data(image_paths=image_paths)

        from bot.handlers.generation import _send_image_preview
        await _send_image_preview(query.message, image_paths)
        await query.message.answer(
            "Reference photo(s) ready. Continue to video generation?",
            reply_markup=get_remix_images_keyboard(),
        )
        await state.set_state(RemixState.CONFIRM_IMAGES)
    except Exception as e:
        logger.exception("Remix reference image generation failed")
        if video_model.endswith("_ref"):
            await query.message.answer(
                f"❌ Reference images are required for {video_model} but generation failed: {e}\n"
                "Try again or switch the video model in ⚙️ Settings."
            )
        else:
            await query.message.answer(
                f"⚠️ Image generation failed: {e}\nContinuing without reference images."
            )
            await _finalize_remix(query.message, state, settings, analysis, image_paths=[])
    finally:
        await state.update_data(generation_in_progress=False)


@router.callback_query(RemixState.CONFIRM_IMAGES, F.data == "remix:images_regen", IsAllowed(allowed_users))
async def handle_images_regen(query: CallbackQuery, state: FSMContext):
    if await _is_generating(state):
        await query.answer("Already generating — please wait.", show_alert=False)
        return
    await query.answer()

    data = await state.get_data()
    analysis = data.get("remix_analysis") or {}
    db = container.inject(DBService)
    settings = await db.get_settings(query.from_user.id, query.message.chat.id)

    await state.update_data(generation_in_progress=True, generation_started_at=time.time())
    await query.message.answer("🔄 Regenerating reference photo(s)…")
    try:
        image_paths = await _generate_reference_images(query.message, state, settings, analysis)
        await state.update_data(image_paths=image_paths)

        from bot.handlers.generation import _send_image_preview
        await _send_image_preview(query.message, image_paths)
        await query.message.answer(
            "New photo(s) ready. Continue?",
            reply_markup=get_remix_images_keyboard(),
        )
    except Exception as e:
        logger.exception("Remix image regeneration failed")
        await query.message.answer(f"❌ Regeneration failed: {e}\nTry again.")
    finally:
        await state.update_data(generation_in_progress=False)


@router.callback_query(RemixState.CONFIRM_IMAGES, F.data == "remix:images_ok", IsAllowed(allowed_users))
async def handle_images_ok(query: CallbackQuery, state: FSMContext):
    await query.answer()
    data = await state.get_data()
    analysis = data.get("remix_analysis") or {}
    if not analysis.get("shots"):
        await query.message.answer("❌ Analysis data lost. Please start over.")
        await state.clear()
        return

    db = container.inject(DBService)
    settings = await db.get_settings(query.from_user.id, query.message.chat.id)
    await _finalize_remix(
        query.message, state, settings, analysis,
        image_paths=data.get("image_paths") or [],
    )


async def _finalize_remix(
    message: Message,
    state: FSMContext,
    settings: dict,
    analysis: dict,
    image_paths: list[str],
) -> None:
    """Build the script from the formula and hand over to the normal pipeline."""
    gemini = container.inject(GeminiService)
    video_model = (settings.get("video_model") or "seedance").lower()
    shots = analysis.get("shots", [])

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

    scenario_text = "\n".join(
        filter(None, [
            analysis.get("title", ""),
            analysis.get("voiceover", ""),
        ])
    ) or "Video based on reference formula"

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
        remix_sfx_description=analysis.get("sfx_description", ""),
    )

    await _show_video_prompt(message, scene, voiceover)
    await state.set_state(GenerationState.CONFIRM_VIDEO_PROMPT)


@router.callback_query(F.data == "remix:cancel", IsAllowed(allowed_users))
async def handle_remix_cancel(query: CallbackQuery, state: FSMContext):
    await query.answer()
    await query.message.answer("Cancelled. Tap 🎬 Generate video to start over.")
    await state.clear()
