"""Periodic cleanup of the `static/` working directory.

Every generation creates a handful of temp artefacts (raw mp4, graded mp4,
mp3, muxed mp4, ass, sub mp4) that nothing else cleans up. Without a sweep
they grow linearly until the disk fills. This module runs a background
asyncio task that deletes anything older than `MAX_AGE_HOURS`.

Files we keep tracked by Telegram are already uploaded to TG by then, so
deleting local copies is safe. Active in-flight generations are protected
because their files are minutes old, far below the threshold.
"""

import asyncio
import logging
import os
import time

logger = logging.getLogger(__name__)

# Files older than this are sweep candidates. 24h is a safe default — a
# generation pipeline finishes in ~3 minutes worst case, so nothing in
# active use will be near 24h.
MAX_AGE_HOURS = 24
SWEEP_INTERVAL_SECONDS = 60 * 60  # once an hour


def _sweep_once(directory: str, max_age_seconds: float) -> int:
    """Delete files older than `max_age_seconds`. Returns count removed."""
    if not os.path.isdir(directory):
        return 0
    now = time.time()
    removed = 0
    for name in os.listdir(directory):
        path = os.path.join(directory, name)
        try:
            if not os.path.isfile(path):
                continue
            age = now - os.path.getmtime(path)
            if age > max_age_seconds:
                os.remove(path)
                removed += 1
        except OSError as e:
            logger.warning(f"cleanup: cannot remove {path}: {e}")
    return removed


async def run_cleanup_loop(static_dir: str):
    """Async task — runs forever, sweeping `static_dir` once per interval."""
    max_age = MAX_AGE_HOURS * 3600
    logger.info(
        f"cleanup: started (dir={static_dir}, max_age={MAX_AGE_HOURS}h, "
        f"interval={SWEEP_INTERVAL_SECONDS}s)"
    )
    while True:
        try:
            removed = await asyncio.to_thread(_sweep_once, static_dir, max_age)
            if removed:
                logger.info(f"cleanup: removed {removed} stale files")
        except Exception as e:
            # Never let a sweep crash kill the loop — log and try again.
            logger.error(f"cleanup: sweep failed: {e}")
        await asyncio.sleep(SWEEP_INTERVAL_SECONDS)
