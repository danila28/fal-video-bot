"""Entry point for the atlas-video-bot Telegram bot."""

import asyncio
import logging
import os
import sys

import asyncpg
from bot.storage import PostgresStorage
from services.blotato import BlotatoService
from services.db import DBService
from services.elevenlabs import ElevenLabsService
from services.gemini import GeminiService
from services.imagegen import ImageGenService
from services.kling import KlingService
from services.minimax import MiniMaxService
from services.omni_edit import OmniEditService
from services.seedance import SeedanceService
from services.wan import WanEditService
from utils import container
from utils.config import config
from aiogram import Bot, Dispatcher
from bot.handlers import router
from bot.middlewares import EnsureIdentityMiddleware, JobOwnerMiddleware
from utils.cleanup import run_cleanup_loop, run_jobs_cleanup_loop
from utils.recovery import recover_interrupted_jobs
from utils.migrations import run_migrations


async def main():
    """Main function to start the bot."""

    # Server locale may default stdout/stderr to ASCII, which crashes logging
    # as soon as any log line contains non-ASCII text (Cyrillic error messages,
    # emoji, or non-English content pulled from Gemini responses).
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Starting atlas-video-bot...")

    bot = Bot(token=config["TELEGRAM_TOKEN"])

    # DB_PORT in docker-compose is the *external* host port; inside the network
    # postgres always listens on 5432, so we keep the DSN port-less.
    dsn = (
        f"postgresql://{config['DB_USERNAME']}:{config['DB_PASSWORD']}"
        f"@{config['DB_HOST']}/{config['CORE_DB_NAME']}"
    )
    pool = await asyncpg.create_pool(dsn)

    await run_migrations(pool)

    storage = PostgresStorage(pool=pool)

    db = DBService(pool=pool, bot_id="fvb")
    container.instance(db)

    container.instance(GeminiService(api_key=config["GEMINI_API_KEY"]))

    container.instance(
        ImageGenService(
            api_key=config["GEMINI_API_KEY"],
            atlas_api_key=config["ATLAS_API_KEY"],
        )
    )
    container.instance(KlingService(api_key=config["ATLAS_API_KEY"]))
    container.instance(SeedanceService(api_key=config["ATLAS_API_KEY"]))
    container.instance(MiniMaxService(api_key=config["ATLAS_API_KEY"]))
    container.instance(OmniEditService(api_key=config["ATLAS_API_KEY"]))
    container.instance(WanEditService(api_key=config["ATLAS_API_KEY"]))

    container.instance(
        BlotatoService(
            base_url=config["BLOTATO_BASE_URL"],
            api_key=config["BLOTATO_API_KEY"],
        )
    )
    container.instance(
        ElevenLabsService(
            api_key=config.get("ELEVENLABS_API_KEY") or "",
            default_voice_id=config.get("ELEVENLABS_DEFAULT_VOICE_ID") or "",
        )
    )

    dp = Dispatcher(storage=storage)
    # Registers the sender / chat in `users` / `chats` before any handler runs
    # (foreign keys, migration 010).
    dp.update.outer_middleware(EnsureIdentityMiddleware())
    # Lets AtlasClient record which chat/user a job belongs to (jobs table).
    dp.update.outer_middleware(JobOwnerMiddleware())
    dp.include_router(router)

    await bot.delete_my_commands()

    # Close jobs left in flight by the previous process and tell their users.
    # Must run before polling starts, while nothing of ours can be in flight.
    await recover_interrupted_jobs(bot, db, storage)

    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
    # Keep references: the event loop only holds weak refs to tasks, so an
    # unreferenced background task can be garbage-collected mid-run.
    background_tasks = {
        asyncio.create_task(run_cleanup_loop(static_dir)),
        asyncio.create_task(run_jobs_cleanup_loop(db)),
    }

    logger.info("Starting polling...")
    await dp.start_polling(bot)
    logger.info("Bot stopped")


if __name__ == "__main__":
    asyncio.run(main())
