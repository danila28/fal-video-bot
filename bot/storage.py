from aiogram.fsm.storage.base import BaseStorage, StorageKey
import asyncpg
import json


class PostgresStorage(BaseStorage):
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def close(self) -> None:
        pass

    async def set_state(self, key: StorageKey, state: str | None = None):
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO fsm_states (chat_id, user_id, state)
                VALUES ($1, $2, $3)
                ON CONFLICT (chat_id, user_id)
                DO UPDATE SET state = $3
                """,
                key.chat_id,
                key.user_id,
                state.state if state else None,
            )

    async def get_state(self, key: StorageKey):
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT state FROM fsm_states WHERE chat_id=$1 AND user_id=$2",
                key.chat_id,
                key.user_id,
            )
            return row["state"] if row else None

    async def set_data(self, key: StorageKey, data: dict):
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO fsm_data (chat_id, user_id, data)
                VALUES ($1, $2, $3)
                ON CONFLICT (chat_id, user_id)
                DO UPDATE SET data = $3
                """,
                key.chat_id,
                key.user_id,
                json.dumps(data),
            )

    async def get_data(self, key: StorageKey):
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT data FROM fsm_data WHERE chat_id=$1 AND user_id=$2",
                key.chat_id,
                key.user_id,
            )
            return json.loads(row["data"]) if row else {}

    async def update_data(self, key: StorageKey, data: dict):
        current = await self.get_data(key)
        current.update(data)
        await self.set_data(key, current)
        return current

    async def clear(self, key: StorageKey):
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "DELETE FROM fsm_states WHERE chat_id=$1 AND user_id=$2",
                    key.chat_id,
                    key.user_id,
                )
                await conn.execute(
                    "DELETE FROM fsm_data WHERE chat_id=$1 AND user_id=$2",
                    key.chat_id,
                    key.user_id,
                )
