"""Startup recovery for generation jobs that were in flight during a restart.

Runs once, before polling starts. At that moment no job of this bot can really
be in flight, so every row still marked `submitted` belongs to a previous
process that died (typically a deploy).

For each affected user we reset the `generation_in_progress` FSM flag (so the
bot does not answer "please wait" for up to 30 minutes) and, if the job is
recent enough to matter, tell them the generation was interrupted.
"""

import logging

from aiogram import Bot
from aiogram.fsm.storage.base import StorageKey

from bot.storage import PostgresStorage
from services.db import DBService

logger = logging.getLogger(__name__)

# Jobs older than this are marked `expired` and the user is not notified.
MAX_NOTIFY_AGE_MINUTES = 30

_INTERRUPTED_TEXT = (
    "⚠️ The bot was restarted while your generation was running, so it was "
    "interrupted. Please start it again."
)


async def recover_interrupted_jobs(
    bot: Bot, db: DBService, storage: PostgresStorage
) -> None:
    """Never raises: a recovery failure must not prevent the bot from starting."""
    try:
        jobs = await db.interrupt_inflight_jobs(MAX_NOTIFY_AGE_MINUTES)
        if not jobs:
            return
        logger.info(f"recovery: {len(jobs)} in-flight job(s) from previous run")

        owners = {(j["chat_id"], j["user_id"]) for j in jobs}
        to_notify = {
            (j["chat_id"], j["user_id"]) for j in jobs if j["status"] == "interrupted"
        }

        for chat_id, user_id in owners:
            key = StorageKey(bot_id=bot.id, chat_id=chat_id, user_id=user_id)
            try:
                data = await storage.get_data(key)
                if data.get("generation_in_progress"):
                    await storage.update_data(
                        key, {"generation_in_progress": False, "generation_started_at": 0}
                    )
            except Exception:
                logger.warning(
                    f"recovery: cannot reset FSM flag for chat={chat_id} user={user_id}",
                    exc_info=True,
                )

        for chat_id, _user_id in to_notify:
            try:
                await bot.send_message(chat_id, _INTERRUPTED_TEXT)
            except Exception as e:
                # User blocked the bot / chat gone — nothing more to do.
                logger.warning(f"recovery: cannot notify chat={chat_id}: {e}")
    except Exception:
        logger.error("recovery: failed (bot continues to start)", exc_info=True)
