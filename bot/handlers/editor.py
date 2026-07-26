"""AI Editor flow: user uploads THEIR OWN video → describes an edit in text
(optionally attaching reference photos, e.g. a product) → Gemini Omni Flash
(task=edit via Atlas Cloud) transforms the footage while preserving everything
the instruction doesn't touch.

Successive edits chain manually: the previous result becomes the next source
video (Atlas doesn't expose Gemini's previous_interaction_id).
"""

import asyncio
import html
import logging
import os
import time
import uuid

from aiogram import F, Router
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, FSInputFile, Message

from bot.guard import IsAllowed
from bot.keyboards import (
    EDITOR_BUTTON_TEXT,
    GENERATE_BUTTON_TEXT,
    SETTINGS_BUTTON_TEXT,
    get_editor_confirm_keyboard,
    get_editor_result_keyboard,
    get_editor_templates_keyboard,
    get_publish_keyboard,
)
from bot.states import EditorState, GenerationState
from services.db import DBService
from services.gemini import GeminiService
from services.omni_edit import (
    MAX_REFERENCE_IMAGES,
    MAX_SOURCE_SECONDS,
    OmniEditService,
    estimate_price,
)
from utils import container
from utils.consts import allowed_users
from bot.handlers.common import _is_generating

router = Router()
logger = logging.getLogger(__name__)

# Telegram Bot API refuses to serve files larger than 20 MB to bots.
_TG_BOT_DOWNLOAD_LIMIT = 20 * 1024 * 1024

# Persistent-menu button texts must never be consumed as user input by the
# editor's text handlers — they fall through to their own entry handlers.
_MENU_TEXTS = {GENERATE_BUTTON_TEXT, SETTINGS_BUTTON_TEXT, EDITOR_BUTTON_TEXT}
_NOT_MENU_TEXT = F.text & ~F.text.in_(_MENU_TEXTS)

# Instruction templates for the quick-edit buttons. Each is a starting point
# the user completes with specifics in their own words.
_TEMPLATES: dict[str, str] = {
    "product": (
        "📦 Insert product — send your product photo as a FILE (not a gallery "
        "photo — Telegram compresses those and details get lost), then send "
        "the instruction, e.g.:\n\n"
        "<code>Insert the product from the reference photo into the scene — "
        "place it on the table, keep the original lighting</code>"
    ),
    "background": (
        "🖼 Change background — send the instruction, e.g.:\n\n"
        "<code>Replace the background with a sunny beach, keep the person "
        "and their movement untouched</code>"
    ),
    "weather": (
        "🌦 Weather / time of day — send the instruction, e.g.:\n\n"
        "<code>Make it night time with light rain, add matching ambient "
        "sound</code>"
    ),
    "style": (
        "🎨 Change art style — send the instruction, e.g.:\n\n"
        "<code>Restyle the whole video as a Pixar-like 3D animation, keep "
        "the motion and framing</code>"
    ),
    "remove": (
        "➖ Remove object — send the instruction, e.g.:\n\n"
        "<code>Remove the car in the background, fill in the street "
        "naturally</code>"
    ),
}


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

@router.message(F.text == EDITOR_BUTTON_TEXT, IsAllowed(allowed_users))
async def handle_editor_start(message: Message, state: FSMContext):
    if await _is_generating(state):
        await message.answer("⚠️ Generation in progress — please wait until it finishes.")
        return

    await state.clear()
    await message.answer(
        "🎨 <b>AI Editor</b> — edit a real video with a text instruction.\n\n"
        f"Send me YOUR video as a file (≤20 MB, ≤{MAX_SOURCE_SECONDS}s). "
        "I'll change only what you ask — everything else stays as filmed.\n\n"
        "⚠️ Own content only. Editing other people's videos or swapping in "
        "faces of real people is not supported.",
        parse_mode="HTML",
    )
    await state.set_state(EditorState.WAITING_VIDEO)


# ─────────────────────────────────────────────
# SOURCE VIDEO UPLOAD
# ─────────────────────────────────────────────

@router.message(EditorState.WAITING_VIDEO, F.video | F.document | F.video_note, IsAllowed(allowed_users))
async def handle_editor_video(message: Message, state: FSMContext):
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
            "Trim or compress the video and send it again."
        )
        return

    omni = container.inject(OmniEditService)
    os.makedirs(omni.static_dir, exist_ok=True)
    video_path = os.path.join(omni.static_dir, f"{uuid.uuid4()}_edit_src.mp4")

    status = await message.answer("⏳ Downloading your video…")
    try:
        await message.bot.download(media, destination=video_path)
        duration = await asyncio.to_thread(GeminiService._probe_duration, video_path)
    except Exception as e:
        logger.exception("Editor: source download failed")
        await status.edit_text(f"❌ Failed to download the video: {e}\nSend it again.")
        return

    if duration > MAX_SOURCE_SECONDS:
        try:
            os.remove(video_path)
        except OSError:
            pass
        await status.edit_text(
            f"❌ The video is {duration:.0f}s long — the editor accepts up to "
            f"{MAX_SOURCE_SECONDS}s.\nTrim it and send again."
        )
        return

    await state.update_data(
        editor_video_path=video_path,
        editor_video_duration=duration,
        editor_ref_paths=[],
    )
    await status.edit_text(f"✅ Video received ({duration:.0f}s).")
    await message.answer(
        "✏️ What should I change? Describe it in text.\n\n"
        "You can also attach up to "
        f"{MAX_REFERENCE_IMAGES} reference photos (e.g. your product) — send "
        "them as FILES so Telegram doesn't compress them. A caption on a "
        "photo works as the instruction too.",
        reply_markup=get_editor_templates_keyboard(),
    )
    await state.set_state(EditorState.WAITING_INSTRUCTION)


@router.message(EditorState.WAITING_VIDEO, _NOT_MENU_TEXT, IsAllowed(allowed_users))
async def handle_editor_video_text(message: Message, state: FSMContext):
    if (message.text or "").strip().startswith(("http://", "https://")):
        await message.answer(
            "🔗 I don't download from links. Save the video to your device "
            "and send it here as a file (≤20 MB)."
        )
    else:
        await message.answer(
            "Send the video you want to edit as a file (≤20 MB), "
            "or tap 🎬 Generate video to leave the editor."
        )


# ─────────────────────────────────────────────
# INSTRUCTION + REFERENCE PHOTOS
# ─────────────────────────────────────────────

@router.callback_query(
    EditorState.WAITING_INSTRUCTION, F.data.startswith("editor:tpl:"), IsAllowed(allowed_users)
)
async def handle_editor_template(query: CallbackQuery):
    await query.answer()
    hint = _TEMPLATES.get(query.data.rsplit(":", 1)[1])
    if hint:
        await query.message.answer(hint, parse_mode="HTML")


async def _show_confirm(message: Message, state: FSMContext) -> None:
    data = await state.get_data()
    instruction = data.get("editor_instruction", "")
    refs = data.get("editor_ref_paths") or []
    duration = float(data.get("editor_video_duration") or 0)
    price = estimate_price(duration)

    text = (
        f"✏️ <b>Edit:</b> {html.escape(instruction)}\n"
        + (f"📎 {len(refs)} reference photo(s)\n" if refs else "")
        + f"⏱ {duration:.0f}s source video\n\n"
        f"Model: Gemini Omni Flash Video Edit — changes only what you asked, "
        f"preserves the rest of the footage.\n"
        f"💰 Estimated cost: ~${price:.2f} (billed per second of source video)\n\n"
        "🚫 No face swaps onto real, recognizable people."
    )
    await message.answer(text, parse_mode="HTML", reply_markup=get_editor_confirm_keyboard(price))
    await state.set_state(EditorState.CONFIRM)


@router.message(EditorState.WAITING_INSTRUCTION, F.photo | F.document, IsAllowed(allowed_users))
async def handle_editor_reference(message: Message, state: FSMContext):
    """Collect a reference photo; its caption (if any) doubles as the instruction."""
    if message.document:
        mime = (message.document.mime_type or "").lower()
        if not mime.startswith("image/"):
            await message.answer("❌ Reference must be an image (PNG/JPEG/WebP).")
            return
        media = message.document
        ext = os.path.splitext(message.document.file_name or "")[1] or ".png"
    else:
        media = message.photo[-1]
        ext = ".jpg"

    data = await state.get_data()
    refs: list[str] = list(data.get("editor_ref_paths") or [])
    if len(refs) >= MAX_REFERENCE_IMAGES:
        await message.answer(
            f"⚠️ Maximum {MAX_REFERENCE_IMAGES} reference photos — this one is ignored."
        )
        return

    omni = container.inject(OmniEditService)
    ref_path = os.path.join(omni.static_dir, f"{uuid.uuid4()}_edit_ref{ext}")
    try:
        await message.bot.download(media, destination=ref_path)
    except Exception as e:
        logger.exception("Editor: reference download failed")
        await message.answer(f"❌ Failed to download the photo: {e}")
        return

    refs.append(ref_path)
    await state.update_data(editor_ref_paths=refs)

    caption = (message.caption or "").strip()
    if caption:
        await state.update_data(editor_instruction=caption)
        await _show_confirm(message, state)
    else:
        await message.answer(
            f"📎 Reference {len(refs)}/{MAX_REFERENCE_IMAGES} saved. "
            "Send more, or send the edit instruction as text."
        )


@router.message(EditorState.WAITING_INSTRUCTION, _NOT_MENU_TEXT, IsAllowed(allowed_users))
async def handle_editor_instruction(message: Message, state: FSMContext):
    instruction = (message.text or "").strip()
    if len(instruction) < 3:
        await message.answer("Describe the edit in a bit more detail.")
        return
    await state.update_data(editor_instruction=instruction)
    await _show_confirm(message, state)


@router.message(EditorState.CONFIRM, _NOT_MENU_TEXT, IsAllowed(allowed_users))
async def handle_editor_confirm_text(message: Message, state: FSMContext):
    """Text sent on the confirm screen replaces the instruction."""
    instruction = (message.text or "").strip()
    if len(instruction) < 3:
        return
    await state.update_data(editor_instruction=instruction)
    await _show_confirm(message, state)


# ─────────────────────────────────────────────
# CONFIRM → RUN
# ─────────────────────────────────────────────

@router.callback_query(EditorState.CONFIRM, F.data == "editor:retype", IsAllowed(allowed_users))
async def handle_editor_retype(query: CallbackQuery, state: FSMContext):
    await query.answer()
    await query.message.edit_reply_markup(reply_markup=None)
    await query.message.answer(
        "✏️ Send the new instruction (reference photos already attached are kept):",
        reply_markup=get_editor_templates_keyboard(),
    )
    await state.set_state(EditorState.WAITING_INSTRUCTION)


@router.callback_query(EditorState.CONFIRM, F.data == "editor:go", IsAllowed(allowed_users))
async def handle_editor_go(query: CallbackQuery, state: FSMContext):
    if await _is_generating(state):
        await query.answer("Already processing — please wait.", show_alert=False)
        return
    await query.answer()

    data = await state.get_data()
    video_path = data.get("editor_video_path") or ""
    instruction = data.get("editor_instruction") or ""
    refs = [p for p in (data.get("editor_ref_paths") or []) if os.path.exists(p)]

    if not video_path or not os.path.exists(video_path):
        await query.message.answer(
            "❌ The source video is gone (cleanup may have removed it). "
            "Start over with 🎨 AI Editor."
        )
        await state.clear()
        return
    if not instruction:
        await query.message.answer("❌ Instruction lost — send it again.")
        await state.set_state(EditorState.WAITING_INSTRUCTION)
        return

    await query.message.edit_reply_markup(reply_markup=None)
    await state.update_data(generation_in_progress=True, generation_started_at=time.time())
    await query.message.answer("⏳ Editing the video — this usually takes 2-4 minutes…")

    try:
        omni = container.inject(OmniEditService)
        result_path = await omni.edit_video(
            video_path, instruction, reference_image_paths=refs
        )

        await query.message.answer_video(FSInputFile(result_path), caption="edited video")
        await state.update_data(
            editor_result_path=result_path,
            # Bridge keys for the shared publish flow.
            video_path=result_path,
            enhance_prompt=instruction,
        )
        await query.message.answer(
            "Done! Continue?", reply_markup=get_editor_result_keyboard()
        )
        await state.set_state(EditorState.RESULT)
    except Exception as e:
        logger.exception("Omni edit failed")
        err = str(e)
        if "insufficient balance" in err.lower() or '"code":402' in err:
            err = (
                "💳 Atlas Cloud balance is empty — top it up at atlascloud.ai "
                "and try again."
            )
        await query.message.answer(f"❌ Edit failed: {err}")
        await _show_confirm(query.message, state)
    finally:
        await state.update_data(generation_in_progress=False)


# ─────────────────────────────────────────────
# RESULT: edit again / publish / done
# ─────────────────────────────────────────────

@router.callback_query(EditorState.RESULT, F.data == "editor:again", IsAllowed(allowed_users))
async def handle_editor_again(query: CallbackQuery, state: FSMContext):
    """Chain the next edit: the previous result becomes the new source."""
    await query.answer()
    data = await state.get_data()
    result_path = data.get("editor_result_path") or ""
    if not result_path or not os.path.exists(result_path):
        await query.message.answer("❌ Result file is gone. Start over with 🎨 AI Editor.")
        await state.clear()
        return

    duration = await asyncio.to_thread(GeminiService._probe_duration, result_path)
    await query.message.edit_reply_markup(reply_markup=None)
    await state.update_data(
        editor_video_path=result_path,
        editor_video_duration=duration,
        editor_ref_paths=[],
        editor_instruction="",
    )
    await query.message.answer(
        "🔁 Editing the LAST RESULT. What should I change now?",
        reply_markup=get_editor_templates_keyboard(),
    )
    await state.set_state(EditorState.WAITING_INSTRUCTION)


@router.callback_query(EditorState.RESULT, F.data == "editor:publish", IsAllowed(allowed_users))
async def handle_editor_publish(query: CallbackQuery, state: FSMContext):
    await query.answer()

    db = container.inject(DBService)
    chat_accounts = await db.get_chat_accounts(query.message.chat.id)
    if not chat_accounts:
        await query.message.answer(
            "No accounts configured for this chat\nTap ⚙️ Settings → 📤 Accounts"
        )
        return

    await query.message.edit_reply_markup(reply_markup=None)
    await query.message.answer("Do you want to publish?", reply_markup=get_publish_keyboard())
    await state.set_state(GenerationState.CONFIRM_PUBLISH)


@router.callback_query(EditorState.RESULT, F.data == "editor:done", IsAllowed(allowed_users))
async def handle_editor_done(query: CallbackQuery, state: FSMContext):
    await query.answer()
    await query.message.edit_reply_markup(reply_markup=None)
    await query.message.answer("✅ Done. Tap 🎨 AI Editor to edit another video.")
    await state.clear()


@router.callback_query(F.data == "editor:cancel", IsAllowed(allowed_users))
async def handle_editor_cancel(query: CallbackQuery, state: FSMContext):
    if await _is_generating(state):
        await query.answer("⚠️ Processing — cannot cancel now.", show_alert=True)
        return
    await query.answer()
    await query.message.answer("Cancelled. Tap 🎨 AI Editor to start over.")
    await state.clear()
