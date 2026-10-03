import json

from asyncpg import Pool


class DBService:
    """DB service"""

    def __init__(self, pool: Pool, bot_id: str):
        self.pool = pool
        self.bot_id = bot_id

    # ─────────────────────────────────────────────
    # USERS / CHATS (parents of every chat_id / user_id foreign key)
    # ─────────────────────────────────────────────

    async def ensure_identity(self, user_id: int, chat_id: int, chat_type: str) -> None:
        """Register the user and chat of the current update (idempotent).

        One round-trip, no write when both rows already exist. Must run before
        any handler writes to a table that references users / chats.
        """
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                WITH u AS (
                    INSERT INTO users (user_id) VALUES ($1) ON CONFLICT DO NOTHING
                )
                INSERT INTO chats (chat_id, chat_type) VALUES ($2, $3)
                ON CONFLICT (chat_id) DO UPDATE SET chat_type = EXCLUDED.chat_type
                WHERE chats.chat_type IS DISTINCT FROM EXCLUDED.chat_type
                """,
                user_id,
                chat_id,
                chat_type,
            )

    # ─────────────────────────────────────────────
    # USER SETTINGS (per user × chat × bot)
    # ─────────────────────────────────────────────

    async def get_settings(self, user_id: int, chat_id: int):
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT data FROM user_settings WHERE chat_id=$1 AND user_id=$2 AND bot_id=$3",
                chat_id,
                user_id,
                self.bot_id,
            )
            return json.loads(row["data"]) if row else {}

    async def update_settings(self, user_id: int, chat_id: int, data: dict):
        current = await self.get_settings(user_id, chat_id)
        current.update(data)
        await self.set_settings(user_id, chat_id, current)
        return current

    async def set_settings(self, user_id: int, chat_id: int, data: dict):
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO user_settings (chat_id, user_id, bot_id, data)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (chat_id, user_id, bot_id)
                DO UPDATE SET data = $4
                """,
                chat_id,
                user_id,
                self.bot_id,
                json.dumps(data),
            )

    # ─────────────────────────────────────────────
    # CHAT ACCOUNTS (per chat — platform + account_id + label)
    # ─────────────────────────────────────────────

    async def get_chat_accounts(self, chat_id: int) -> list[dict]:
        """Return all accounts configured for this chat, ordered by id."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, platform, account_id, label "
                "FROM chat_accounts WHERE chat_id=$1 ORDER BY id",
                chat_id,
            )
            return [dict(r) for r in rows]

    async def add_chat_account(
        self, chat_id: int, platform: str, account_id: str, label: str
    ) -> None:
        """Insert or update (upsert) a single account for this chat."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO chat_accounts (chat_id, platform, account_id, label)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (chat_id, platform, account_id)
                DO UPDATE SET label = EXCLUDED.label
                """,
                chat_id,
                platform.lower(),
                account_id,
                label,
            )

    async def remove_chat_account(self, row_id: int, chat_id: int) -> bool:
        """Delete account by primary key (also checks chat_id for safety).
        Returns True if a row was actually deleted."""
        async with self.pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM chat_accounts WHERE id=$1 AND chat_id=$2",
                row_id,
                chat_id,
            )
            # asyncpg returns e.g. "DELETE 1" or "DELETE 0"
            return result == "DELETE 1"

    # ─────────────────────────────────────────────
    # GENERATION LOG
    # ─────────────────────────────────────────────

    async def log_generation(
        self,
        user_id: int,
        chat_id: int,
        gen_type: str,
        video_model: str,
        target_duration: int = 0,
        success: bool = True,
        error_text: str | None = None,
    ) -> None:
        """Record one Veo generation attempt to generation_log.

        gen_type — 'initial' | 'regen' | 'extend'
        Errors are silently swallowed so a DB hiccup never interrupts a video job.
        """
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO generation_log
                        (user_id, chat_id, gen_type, video_model, target_duration, success, error_text)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    """,
                    user_id,
                    chat_id,
                    gen_type,
                    video_model or "",
                    target_duration,
                    success,
                    error_text,
                )
        except Exception:
            import logging
            logging.getLogger(__name__).warning(
                "log_generation failed (non-fatal)", exc_info=True
            )

    # ─────────────────────────────────────────────
    # VIDEO HISTORY
    # ─────────────────────────────────────────────

    async def add_video_history(
        self,
        user_id: int,
        chat_id: int,
        file_id: str,
        title: str = "",
        gen_type: str = "initial",
    ) -> None:
        """Save a Telegram file_id to the video history (non-fatal on error)."""
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO video_history (user_id, chat_id, file_id, title, gen_type)
                    VALUES ($1, $2, $3, $4, $5)
                    """,
                    user_id,
                    chat_id,
                    file_id,
                    title or "",
                    gen_type,
                )
        except Exception:
            import logging
            logging.getLogger(__name__).warning(
                "add_video_history failed (non-fatal)", exc_info=True
            )

    async def get_video_history(
        self, user_id: int, chat_id: int, limit: int = 10
    ) -> list[dict]:
        """Return the last `limit` videos for this user × chat, newest first."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, file_id, title, gen_type, created_at
                FROM video_history
                WHERE user_id=$1 AND chat_id=$2
                ORDER BY created_at DESC
                LIMIT $3
                """,
                user_id,
                chat_id,
                limit,
            )
            return [dict(r) for r in rows]

    # ─────────────────────────────────────────────
    # JOBS (Atlas predictions — see migrations/009_jobs.sql)
    # ─────────────────────────────────────────────

    async def create_job(
        self, pred_id: str, user_id: int, chat_id: int, model: str
    ) -> None:
        """Record a freshly submitted Atlas job (non-fatal on error)."""
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO jobs (bot_id, pred_id, user_id, chat_id, model)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (bot_id, pred_id) DO NOTHING
                    """,
                    self.bot_id,
                    pred_id,
                    user_id,
                    chat_id,
                    model,
                )
        except Exception:
            import logging
            logging.getLogger(__name__).warning(
                "create_job failed (non-fatal)", exc_info=True
            )

    async def finish_job(
        self, pred_id: str, status: str, error_text: str | None = None
    ) -> None:
        """Move an in-flight job to a final status (non-fatal on error).

        Only touches rows still in 'submitted', so a late update can never
        overwrite 'interrupted' / 'expired' set by startup recovery.
        """
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE jobs
                    SET status = $3, error_text = $4, updated_at = NOW()
                    WHERE bot_id = $1 AND pred_id = $2 AND status = 'submitted'
                    """,
                    self.bot_id,
                    pred_id,
                    status,
                    error_text[:500] if error_text else None,
                )
        except Exception:
            import logging
            logging.getLogger(__name__).warning(
                "finish_job failed (non-fatal)", exc_info=True
            )

    async def interrupt_inflight_jobs(self, max_notify_age_minutes: int) -> list[dict]:
        """Close every job still 'submitted' for this bot; return the closed rows.

        Call once at startup, before polling begins: nothing can be in flight
        then, so all such rows are leftovers of a dead process. Recent ones
        become 'interrupted' (worth telling the user), older ones 'expired'.
        A single UPDATE ... RETURNING keeps this atomic.
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                UPDATE jobs
                SET status = CASE
                        WHEN created_at > NOW() - make_interval(mins => $2)
                        THEN 'interrupted' ELSE 'expired' END,
                    updated_at = NOW()
                WHERE bot_id = $1 AND status = 'submitted'
                RETURNING pred_id, user_id, chat_id, model, status
                """,
                self.bot_id,
                max_notify_age_minutes,
            )
            return [dict(r) for r in rows]

    async def delete_old_jobs(self, older_than_days: int) -> int:
        """Retention sweep. Returns number of deleted rows."""
        async with self.pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM jobs WHERE bot_id = $1 "
                "AND created_at < NOW() - make_interval(days => $2)",
                self.bot_id,
                older_than_days,
            )
            # asyncpg returns e.g. "DELETE 5"
            return int(result.split()[-1])
