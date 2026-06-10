"""Publish flow: title → schedule/now → Blotato upload."""

import html
import logging
import os
from datetime import datetime, timedelta, timezone

from aiogram import Router
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, Message
from bot.guard import IsAllowed
from bot.keyboards import get_publish_keyboard, get_publish_time_keyboard
from bot.states import GenerationState
from services.blotato import BlotatoService
from services.db import DBService
from services.gemini import GeminiService
from utils import container
from utils.consts import allowed_users, get_hashtags_prompt_for_platforms
from bot.handlers.common import _is_generating

router = Router()
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# SHARED PUBLISH LOGIC
# ─────────────────────────────────────────────

async def _do_publish(msg: Message, user_id: int, state: FSMContext, scheduled_at: str = ""):
    """Shared publish logic used by both 'now' and 'scheduled' paths."""
    blotato = container.inject(BlotatoService)
    vertex = container.inject(GeminiService)
    db = container.inject(DBService)

    try:
        data = await state.get_data()

        if data.get("publishing_in_progress"):
            logger.warning("Double-publish blocked for user %s", user_id)
            return
        await state.update_data(publishing_in_progress=True)

        video_path = data.get("video_path")
        enhance_prompt = data.get("enhance_prompt", "")
        title = data.get("publish_title", "")

        if not video_path:
            raise Exception("Video not found in current session.")
        if not os.path.exists(video_path):
            raise FileNotFoundError(
                "Video file was deleted from disk (cleanup may have removed it). "
                "Please regenerate the video."
            )
        if not title:
            raise Exception("Video title is missing.")

        settings = await db.get_settings(user_id, msg.chat.id)
        text_model = settings.get("text_model") or ""

        chat_accounts = await db.get_chat_accounts(msg.chat.id)
        if not chat_accounts:
            raise Exception(
                "No accounts configured for this chat.\n"
                "Go to ⚙️ Settings → 📤 Accounts and add at least one account."
            )
        accounts = ",".join(
            f"{acc['platform']}:{acc['account_id']}" for acc in chat_accounts
        )

        if scheduled_at:
            await msg.answer(f"📅 Scheduling for {scheduled_at} UTC…")
        else:
            await msg.answer("📤 Publishing…")

        # Pick hashtag style based on actual publish platforms.
        publish_platforms = {acc["platform"] for acc in chat_accounts}
        hashtag_prompt = get_hashtags_prompt_for_platforms(publish_platforms)
        try:
            hashtags = await vertex.generate_text(enhance_prompt, hashtag_prompt, text_model)
        except Exception as ht_err:
            logger.warning(f"Hashtag generation failed, using cached: {ht_err}")
            hashtags = data.get("cached_hashtags") or ""

        await blotato.publish_video(
            video_path=video_path,
            title=f"{title} {hashtags}",
            accounts=accounts,
            scheduled_at=scheduled_at,
        )

        if scheduled_at:
            await msg.answer(f"✅ Scheduled for <b>{scheduled_at} UTC</b>!", parse_mode="HTML")
        else:
            await msg.answer("✅ Published successfully!")
        await state.clear()

    except Exception as e:
        await state.update_data(publishing_in_progress=False)
        await msg.answer(f"❌ Publish error: {str(e)}\nYou can try publishing again.")


# ─────────────────────────────────────────────
# PUBLISH HANDLERS
# ─────────────────────────────────────────────

@router.callback_query(GenerationState.CONFIRM_PUBLISH, lambda c: c.data == "publish", IsAllowed(allowed_users))
async def handle_publish(callback: CallbackQuery, state: FSMContext):
    await callback.answer()
    await callback.message.edit_reply_markup(reply_markup=None)
    await callback.message.answer("Send video title:")
    await state.set_state(GenerationState.SET_VIDEO_TITLE)


@router.message(GenerationState.SET_VIDEO_TITLE, IsAllowed(allowed_users))
async def handle_video_title(message: Message, state: FSMContext):
    await state.update_data(publish_title=message.text)
    await message.answer(
        f"Title: <b>{html.escape(message.text)}</b>\n\nWhen do you want to publish?",
        parse_mode="HTML",
        reply_markup=get_publish_time_keyboard(),
    )
    await state.set_state(GenerationState.SELECT_PUBLISH_TIME)


@router.callback_query(
    GenerationState.SELECT_PUBLISH_TIME,
    lambda c: c.data == "publish_time:now",
    IsAllowed(allowed_users),
)
async def handle_publish_now(callback: CallbackQuery, state: FSMContext):
    await callback.answer()
    await callback.message.edit_reply_markup(reply_markup=None)
    await _do_publish(callback.message, callback.from_user.id, state, scheduled_at="")


@router.callback_query(
    GenerationState.SELECT_PUBLISH_TIME,
    lambda c: c.data == "publish_time:schedule",
    IsAllowed(allowed_users),
)
async def handle_publish_schedule(callback: CallbackQuery, state: FSMContext):
    await callback.answer()
    await callback.message.edit_reply_markup(reply_markup=None)
    db = container.inject(DBService)
    settings = await db.get_settings(callback.from_user.id, callback.message.chat.id)
    utc_offset = settings.get("utc_offset_hours", 0)
    sign = "+" if utc_offset >= 0 else ""
    await callback.message.answer(
        f"📅 Send the publish date and time in your <b>local time (UTC{sign}{utc_offset})</b>:\n"
        "<code>YYYY-MM-DD HH:MM</code>\n\n"
        "Example: <code>2025-08-20 23:00</code>\n\n"
        "To change timezone: ⚙️ Settings → 🕐 Timezone",
        parse_mode="HTML",
    )
    await state.set_state(GenerationState.SET_PUBLISH_TIME)


@router.message(GenerationState.SET_PUBLISH_TIME, IsAllowed(allowed_users))
async def handle_set_publish_time(message: Message, state: FSMContext):
    raw = (message.text or "").strip().replace("T", " ")
    dt = None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            dt = datetime.strptime(raw, fmt)
            break
        except ValueError:
            continue
    if dt is None:
        await message.answer(
            "❌ Could not parse date/time.\n"
            "Use format: <code>YYYY-MM-DD HH:MM</code>\n"
            "Example: <code>2025-08-20 23:00</code>",
            parse_mode="HTML",
        )
        return

    db = container.inject(DBService)
    settings = await db.get_settings(message.from_user.id, message.chat.id)
    utc_offset = settings.get("utc_offset_hours", 0)
    dt_utc = dt - timedelta(hours=utc_offset)
    dt_utc = dt_utc.replace(tzinfo=timezone.utc)
    scheduled_at = dt_utc.strftime("%Y-%m-%dT%H:%M:%S+00:00")

    sign = "+" if utc_offset >= 0 else ""
    await message.answer(
        f"🕐 Local time (UTC{sign}{utc_offset}): <b>{dt.strftime('%Y-%m-%d %H:%M')}</b>\n"
        f"🌐 UTC time: <b>{dt_utc.strftime('%Y-%m-%d %H:%M')}</b>",
        parse_mode="HTML",
    )
    await _do_publish(message, message.from_user.id, state, scheduled_at=scheduled_at)


@router.callback_query(GenerationState.CONFIRM_PUBLISH, lambda c: c.data == "cancel", IsAllowed(allowed_users))
async def handle_cancel(callback: CallbackQuery, state: FSMContext):
    if await _is_generating(state):
        await callback.answer("⚠️ Generation in progress — cannot cancel now.", show_alert=True)
        return
    await callback.answer()
    await callback.message.edit_reply_markup(reply_markup=None)
    await callback.message.answer("Canceled.")
    await state.clear()
