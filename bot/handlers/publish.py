"""Publish flow: title → schedule/now → Blotato upload."""

import asyncio
import html
import logging
import os
from datetime import datetime, timedelta, timezone

from aiogram import Router
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, Message
from bot.guard import IsAllowed
from bot.keyboards import get_account_picker_keyboard, get_publish_time_keyboard
from bot.states import GenerationState
from services.blotato import BlotatoService
from services.db import DBService
from services.llm import LLMService
from utils import container
from utils.consts import allowed_users, get_hashtags_prompt_for_platforms
from bot.handlers.common import _is_generating

router = Router()
logger = logging.getLogger(__name__)

# Blotato reports 'published' only after the platform accepts the post, which
# can take minutes. Poll for the public link in the background instead of
# blocking the publish flow.
_PUB_POLL_INTERVAL = 20   # seconds between status checks
_PUB_POLL_ATTEMPTS = 30   # ~10 minutes total
_pub_tasks: set[asyncio.Task] = set()


def _account_label(acc: dict) -> str:
    return acc.get("label") or acc["account_id"]


async def _record_publications(
    msg: Message, user_id: int, chat_accounts: list[dict], results: list, scheduled_at: str,
) -> list[str]:
    """Store one publications row per selected account and build the summary.

    Blotato's result for each account is tagged with 'account' = "platform:id"
    (see BlotatoService.publish_video). The post itself already exists at this
    point, so a bookkeeping failure is logged and never reported as a failed post.
    """
    db = container.inject(DBService)
    by_pair = {f"{a['platform']}:{a['account_id']}": a for a in chat_accounts}
    lines: list[str] = []
    for res in results:
        acc = by_pair.get(res.get("account"))
        if acc is None:
            continue
        name = html.escape(f"{acc['platform']} · {_account_label(acc)}")
        if "error" in res:
            lines.append(f"❌ {name}: {html.escape(str(res['error'])[:200])}")
            try:
                await db.add_publication(
                    msg.chat.id, user_id, acc["id"], acc["platform"], None, "failed", res["error"],
                )
            except Exception:
                logger.exception("Could not record failed publication")
            continue

        sub_id = res.get("postSubmissionId")
        status = "scheduled" if scheduled_at else "submitted"
        try:
            pub_id = await db.add_publication(
                msg.chat.id, user_id, acc["id"], acc["platform"], sub_id, status,
            )
        except Exception:
            logger.exception("Could not record publication")
            pub_id = None

        lines.append(f"✅ {name}" + (" — scheduled" if scheduled_at else " — sent, waiting for link"))
        if sub_id and pub_id and not scheduled_at:
            task = asyncio.create_task(
                _track_publication(msg, pub_id, sub_id, acc["platform"])
            )
            _pub_tasks.add(task)
            task.add_done_callback(_pub_tasks.discard)
    return lines


async def _track_publication(msg: Message, pub_id: int, sub_id: str, platform: str) -> None:
    """Poll Blotato until the post is published or failed, then tell the chat the link."""
    blotato = container.inject(BlotatoService)
    db = container.inject(DBService)
    last_error = ""
    for _ in range(_PUB_POLL_ATTEMPTS):
        await asyncio.sleep(_PUB_POLL_INTERVAL)
        try:
            info = await blotato.get_post_status(sub_id)
        except Exception as e:
            last_error = str(e)
            logger.warning(f"Blotato status poll failed ({sub_id}): {e}")
            continue

        status = info.get("status")
        if status == "published":
            url = info.get("publicUrl")
            await db.update_publication(pub_id, "published" if url else "url_missing", public_url=url)
            if url:
                await msg.answer(f"🔗 {html.escape(platform)}: {html.escape(url)}")
            else:
                await msg.answer(f"✅ {html.escape(platform)} published (Blotato returned no link).")
            return
        if status == "failed":
            err = info.get("errorMessage") or "publish failed"
            await db.update_publication(pub_id, "failed", error_text=err)
            await msg.answer(f"❌ {html.escape(platform)}: {html.escape(err[:300])}")
            return

    await db.update_publication(
        pub_id, "url_missing", error_text=last_error or "still in progress after polling window",
    )
    await msg.answer(
        f"⏳ {html.escape(platform)}: no link yet after 10 minutes. "
        "The post may still be processing."
    )


# ─────────────────────────────────────────────
# SHARED PUBLISH LOGIC
# ─────────────────────────────────────────────

async def _do_publish(msg: Message, user_id: int, state: FSMContext, scheduled_at: str = ""):
    """Shared publish logic used by both 'now' and 'scheduled' paths."""
    blotato = container.inject(BlotatoService)
    llm = container.inject(LLMService)
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

        # Publish only to accounts the user explicitly picked (SELECT_PUBLISH_ACCOUNTS).
        selected_ids = set(data.get("publish_account_ids") or [])
        chat_accounts = [
            acc for acc in await db.get_chat_accounts(msg.chat.id)
            if acc["id"] in selected_ids
        ]
        if not chat_accounts:
            raise Exception("No accounts selected for publishing.")
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
            hashtags = await llm.generate_text(enhance_prompt, hashtag_prompt, text_model)
        except Exception as ht_err:
            logger.warning(f"Hashtag generation failed, using cached: {ht_err}")
            hashtags = data.get("cached_hashtags") or ""

        results = await blotato.publish_video(
            video_path=video_path,
            title=f"{title} {hashtags}",
            accounts=accounts,
            scheduled_at=scheduled_at,
        )

        lines = await _record_publications(msg, user_id, chat_accounts, results, scheduled_at)
        if not lines:
            # Blotato answered, but none of its results matched a selected account.
            # The post request was sent, so do not report it as a failure.
            lines = ["ℹ️ Publish request sent, but no per-account result came back."]
        if scheduled_at:
            lines.insert(0, f"📅 Scheduled for <b>{scheduled_at} UTC</b>:")
        await msg.answer("\n".join(lines), parse_mode="HTML")
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
    db = container.inject(DBService)
    accounts = await db.get_chat_accounts(callback.message.chat.id)
    if not accounts:
        await callback.message.answer(
            "No accounts configured for this chat.\n"
            "Go to ⚙️ Settings → 📤 Accounts and add at least one account."
        )
        return
    await state.update_data(publish_account_ids=[])
    await callback.message.answer(
        "Select the accounts to publish to:",
        reply_markup=get_account_picker_keyboard(accounts, set()),
    )
    await state.set_state(GenerationState.SELECT_PUBLISH_ACCOUNTS)


@router.callback_query(
    GenerationState.SELECT_PUBLISH_ACCOUNTS,
    lambda c: c.data.startswith("pubacc:toggle:"),
    IsAllowed(allowed_users),
)
async def handle_publish_account_toggle(callback: CallbackQuery, state: FSMContext):
    db = container.inject(DBService)
    accounts = await db.get_chat_accounts(callback.message.chat.id)
    row_id = int(callback.data.rsplit(":", 1)[1])
    if row_id not in {acc["id"] for acc in accounts}:
        await callback.answer("This account is not configured for this chat.", show_alert=True)
        return
    data = await state.get_data()
    selected = set(data.get("publish_account_ids") or [])
    selected ^= {row_id}
    await state.update_data(publish_account_ids=sorted(selected))
    await callback.answer()
    await callback.message.edit_reply_markup(
        reply_markup=get_account_picker_keyboard(accounts, selected)
    )


@router.callback_query(
    GenerationState.SELECT_PUBLISH_ACCOUNTS,
    lambda c: c.data == "pubacc:next",
    IsAllowed(allowed_users),
)
async def handle_publish_accounts_next(callback: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    if not data.get("publish_account_ids"):
        await callback.answer("Select at least one account.", show_alert=True)
        return
    await callback.answer()
    await callback.message.edit_reply_markup(reply_markup=None)
    await callback.message.answer("Send video title:")
    await state.set_state(GenerationState.SET_VIDEO_TITLE)


@router.callback_query(
    GenerationState.SELECT_PUBLISH_ACCOUNTS,
    lambda c: c.data == "pubacc:cancel",
    IsAllowed(allowed_users),
)
async def handle_publish_accounts_cancel(callback: CallbackQuery, state: FSMContext):
    await callback.answer()
    await callback.message.edit_reply_markup(reply_markup=None)
    await callback.message.answer("Canceled.")
    await state.clear()


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
