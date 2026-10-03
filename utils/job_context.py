"""Who is the current generation job being run for?

AtlasClient is shared by every generation service and knows nothing about
Telegram. To record `chat_id` / `user_id` for a job without threading them
through ~30 call sites, a middleware stores them in a ContextVar at the start
of each update; AtlasClient reads them when it submits a job.

ContextVars are copied into child tasks, so this also works inside
`asyncio.gather(...)` and `asyncio.to_thread(...)`.
"""

from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True)
class JobOwner:
    chat_id: int
    user_id: int


current_owner: ContextVar[JobOwner | None] = ContextVar("current_job_owner", default=None)
