import logging
import os
import asyncpg

logger = logging.getLogger(__name__)

# Resolve migrations dir relative to this file so the path is always correct
# regardless of the working directory at startup.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_MIGRATIONS_PATH = os.path.join(_PROJECT_ROOT, "migrations")

# DDL for the migrations tracking table. Created before any migration runs so
# it exists even on a completely fresh database.
_CREATE_TRACKING_TABLE = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    filename TEXT PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""


async def run_migrations(pool: asyncpg.Pool, migrations_path: str = _DEFAULT_MIGRATIONS_PATH):
    """Run pending SQL migrations in filename order.

    Each file is executed at most once: after a successful run its name is
    recorded in `schema_migrations`. On the next startup that file is skipped.
    This makes migrations safe to add over time without worrying about
    idempotency — each file only ever runs once.
    """
    if not os.path.isdir(migrations_path):
        raise RuntimeError(
            f"Migrations directory not found: {migrations_path}\n"
            "Make sure the 'migrations/' folder exists in the project root."
        )
    files = sorted(f for f in os.listdir(migrations_path) if f.endswith(".sql"))
    if not files:
        logger.warning(f"No .sql migration files found in {migrations_path}")
        return

    async with pool.acquire() as conn:
        # Ensure the tracking table exists before querying it.
        await conn.execute(_CREATE_TRACKING_TABLE)

        # Fetch already-applied migrations in one round-trip.
        applied = {
            row["filename"]
            for row in await conn.fetch("SELECT filename FROM schema_migrations")
        }

        for filename in files:
            if filename in applied:
                logger.debug(f"Migration already applied, skipping: {filename}")
                continue

            path = os.path.join(migrations_path, filename)
            with open(path, "r", encoding="utf-8") as f:
                sql = f.read()

            logger.info(f"Applying migration: {filename}")
            # Run migration + tracking insert in a single transaction so a
            # partial failure doesn't leave the file marked as applied.
            async with conn.transaction():
                await conn.execute(sql)
                await conn.execute(
                    "INSERT INTO schema_migrations (filename) VALUES ($1)", filename
                )
            logger.info(f"Migration applied: {filename}")
