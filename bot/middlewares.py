from typing import Any, Awaitable, Callable

from aiogram import BaseMiddleware
from aiogram.types import TelegramObject

from services.db import DBService
from utils import container
from utils.consts import allowed_users
from utils.job_context import JobOwner, current_owner


class JobOwnerMiddleware(BaseMiddleware):
    """Expose (chat_id, user_id) of the current update to non-handler code.

    Register as an *outer* update middleware so it runs for every update type.
    `event_from_user` / `event_chat` are filled in by aiogram's built-in
    UserContextMiddleware, which is registered before user middlewares.
    """

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        user = data.get("event_from_user")
        chat = data.get("event_chat")
        if user is None or chat is None:
            return await handler(event, data)

        token = current_owner.set(JobOwner(chat_id=chat.id, user_id=user.id))
        try:
            return await handler(event, data)
        finally:
            current_owner.reset(token)


class EnsureIdentityMiddleware(BaseMiddleware):
    """Make sure the sender and the chat exist in `users` / `chats`.

    Every table that stores chat_id / user_id references those two tables
    (migration 010), so the rows must exist before any handler writes.
    Register as an outer update middleware. Only allowed users are registered:
    all handlers sit behind IsAllowed, so strangers never write anything and
    should not leave rows behind.
    """

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        user = data.get("event_from_user")
        chat = data.get("event_chat")
        if user is not None and chat is not None and user.id in allowed_users:
            await container.inject(DBService).ensure_identity(user.id, chat.id, chat.type)
        return await handler(event, data)
