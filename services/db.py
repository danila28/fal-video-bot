import json

from asyncpg import Pool


class DBService:
    """DB service"""

    def __init__(self, pool: Pool, bot_id: str):
        self.pool = pool
        self.bot_id = bot_id

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
