from aiogram.filters import BaseFilter
from aiogram.types import Message, CallbackQuery, User
from typing import Optional


class IsAllowed(BaseFilter):
    """Allow only users whose Telegram ID is in the allowed_users set.

    In aiogram 3 the filter's __call__ receives the event object as a positional
    argument, followed by DI-injected keyword arguments (event_from_user, message,
    callback_query, …).  We use *_ to absorb the positional event arg so that the
    named parameters become keyword-only and can be injected without triggering
    "multiple values for argument" conflicts.

    Priority order:
      1. event_from_user — injected by UserContextMiddleware for every event type.
      2. callback_query  — the person who clicked the button.
      3. message         — the person who sent the message.

    callback_query is checked BEFORE message because in callback context aiogram
    also injects message (the original message the button was on), whose from_user
    would be the bot itself, not the user who clicked.
    """

    def __init__(self, allowed_users: set[int]):
        self.allowed_users = allowed_users

    async def __call__(
        self,
        *_,                                      # absorbs the positional event arg;
                                                 # makes all named params keyword-only
        event_from_user: Optional[User] = None,
        callback_query: Optional[CallbackQuery] = None,
        message: Optional[Message] = None,
    ) -> bool:
        # Priority 1: most reliable — available for every event type.
        if event_from_user is not None:
            return event_from_user.id in self.allowed_users

        # Priority 2: who clicked the button.
        if callback_query is not None:
            user = callback_query.from_user
            if user is not None:
                return user.id in self.allowed_users

        # Priority 3: who sent the message.
        if message is not None:
            user = message.from_user
            if user is not None:
                return user.id in self.allowed_users

        return False
