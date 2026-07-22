import os
from dotenv import load_dotenv

load_dotenv()

config = {
    "TELEGRAM_TOKEN": os.getenv("TELEGRAM_TOKEN"),

    # Gemini Developer API key — used for text generation and Imagen 4 Fast.
    # `GEMINI_API_KEY` is preferred; `VERTEX_API_KEY` kept as alias so .env
    # files copied from video-gen-bot keep working without edits.
    "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY") or os.getenv("VERTEX_API_KEY"),

    # Atlas Cloud API key — used by all video generation and FLUX/Ideogram image models.
    "ATLAS_API_KEY": os.getenv("ATLAS_API_KEY"),

    # PostgreSQL — shared with video-gen-bot. user_settings / chat_accounts /
    # video_history / generation_log / fsm_states tables are co-tenant by
    # (chat_id, user_id), so two bots can safely share the same database as
    # long as each chat is dedicated to a single bot.
    "CORE_DB_NAME": os.getenv("CORE_DB_NAME"),
    "DB_USERNAME": os.getenv("DB_USERNAME"),
    "DB_PASSWORD": os.getenv("DB_PASSWORD"),
    "DB_HOST": os.getenv("DB_HOST", "postgres"),
    "DB_PORT": os.getenv("DB_PORT", "5432"),

    "BLOTATO_BASE_URL": os.getenv("BLOTATO_BASE_URL"),
    "BLOTATO_API_KEY": os.getenv("BLOTATO_API_KEY"),

    "ELEVENLABS_API_KEY": os.getenv("ELEVENLABS_API_KEY"),
    "ELEVENLABS_DEFAULT_VOICE_ID": os.getenv("ELEVENLABS_DEFAULT_VOICE_ID", ""),

    # Optional outro clip appended after every generated video.
    "OUTRO_VIDEO_PATH": os.getenv(
        "OUTRO_VIDEO_PATH",
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "subscribe_video.mp4",
        ),
    ),
}

_REQUIRED = [
    "TELEGRAM_TOKEN",
    "GEMINI_API_KEY",
    "ATLAS_API_KEY",
    "CORE_DB_NAME",
    "DB_USERNAME",
    "DB_PASSWORD",
]

_missing = [k for k in _REQUIRED if not config.get(k)]
if _missing:
    raise RuntimeError(
        f"Missing required environment variables: {', '.join(_missing)}\n"
        "Check your .env file."
    )

# API keys and tokens travel through .env files edited by hand; a copy-paste
# through a messenger or rich-text editor can silently replace Latin letters
# with Unicode lookalikes. Such a key *looks* fine but crashes deep inside the
# HTTP stack ("'ascii' codec can't encode…" — headers are ASCII-only), which
# is nearly impossible to trace back to the .env. Fail fast with a clear
# message instead.
_ASCII_ONLY = [
    "TELEGRAM_TOKEN",
    "GEMINI_API_KEY",
    "ATLAS_API_KEY",
    "BLOTATO_API_KEY",
    "ELEVENLABS_API_KEY",
]

for _key in _ASCII_ONLY:
    _val = config.get(_key) or ""
    if not _val.isascii():
        _bad = sorted({c for c in _val if not c.isascii()})
        raise RuntimeError(
            f"{_key} contains non-ASCII characters {_bad} — the value was likely "
            "corrupted by a copy-paste (Cyrillic lookalikes / smart quotes). "
            "Re-paste it into .env directly from the provider's dashboard."
        )
