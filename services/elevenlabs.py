"""ElevenLabs TTS client.

Wraps the SDK to expose a single async method that returns both the synthesised
audio file and word-level timings. The timings drive karaoke-style ASS
subtitles (each word lights up exactly when it's spoken), which is the only
reason we need ElevenLabs over Veo's built-in audio.
"""

import asyncio
import logging
import os
import uuid
from typing import List, Tuple

from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

logger = logging.getLogger(__name__)


# ── Custom exceptions ─────────────────────────────────────────────────────────

class _NoVoiceConfiguredError(Exception):
    """Raised when no ElevenLabs voice ID has been set."""


class _PaymentRequiredError(Exception):
    """Raised when ElevenLabs returns HTTP 402 (plan upgrade needed)."""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _check_payment_required(exc: Exception) -> None:
    """Convert a 402 SDK error into a user-friendly _PaymentRequiredError.

    The ElevenLabs SDK raises different exception types across versions; we
    detect 402 by inspecting the string representation which always contains
    the status code and/or the 'payment_required' code word.

    For any other exception we re-raise it unchanged so the retry decorator
    can handle transient failures (429, 5xx, network blips) as usual.
    """
    msg = str(exc)
    if "402" in msg or "payment_required" in msg or "paid_plan_required" in msg:
        raise _PaymentRequiredError(
            "The selected ElevenLabs voice requires a paid subscription.\n"
            "Fix: go to ⚙️ Settings → 🎙 Voice and enter a voice ID from "
            "your own ElevenLabs account (Voices → My Voices → copy Voice ID).\n"
            "Free-tier tip: create a free 'Instant Voice Clone' at elevenlabs.io "
            "and use that voice ID instead."
        ) from exc
    raise exc  # not a 402 — let tenacity decide whether to retry


# Retry only for transient failures (429, 5xx, network).
# Never retry our own plan/config errors — they won't recover on their own.
_retry_synth = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_not_exception_type((_NoVoiceConfiguredError, _PaymentRequiredError)),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING),
)

# (word, start_seconds, end_seconds) — what we hand to the karaoke ASS builder
WordTiming = Tuple[str, float, float]


# ── Service ───────────────────────────────────────────────────────────────────

class ElevenLabsService:
    DEFAULT_MODEL = "eleven_turbo_v2_5"  # fast + multilingual + supports timings

    def __init__(self, api_key: str, default_voice_id: str):
        self.api_key = api_key
        self.default_voice_id = default_voice_id
        # Lazy import — keeps the module importable even if the dependency is
        # missing locally (e.g. running tests for unrelated modules).
        from elevenlabs.client import ElevenLabs
        self._client = ElevenLabs(api_key=api_key)
        self.static_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static"
        )

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    @_retry_synth
    async def synthesize(
        self,
        text: str,
        voice_id: str = "",
        model: str = "",
    ) -> Tuple[str, List[WordTiming]]:
        """Generate speech and return (audio_path, word_timings).

        Empty `voice_id` falls back to the configured default. Word timings come
        from the `with_timestamps` endpoint, which returns per-character marks;
        we collapse those into per-word `(word, start, end)` tuples so the
        subtitle renderer doesn't have to know about ElevenLabs internals.

        Returned audio is mp3 — ffmpeg muxes it into the final mp4 just fine.
        """
        if not self.is_configured:
            raise Exception("ElevenLabs API key is not configured")
        if not text or not text.strip():
            raise Exception("Empty text passed to ElevenLabs")

        # Resolve voice — raise immediately (no retry) when nothing is set.
        effective_voice = voice_id or self.default_voice_id
        if not effective_voice:
            raise _NoVoiceConfiguredError(
                "No ElevenLabs voice ID configured.\n"
                "Go to ⚙️ Settings → 🎙 Voice and paste your ElevenLabs Voice ID.\n"
                "Free-tier users must create their own voice at elevenlabs.io — "
                "library/premade voices (e.g. Rachel) require a paid plan."
            )

        os.makedirs(self.static_dir, exist_ok=True)
        model_id = model or self.DEFAULT_MODEL

        # SDK is sync — push to a thread to keep the bot loop responsive.
        try:
            result = await asyncio.to_thread(
                self._client.text_to_speech.convert_with_timestamps,
                voice_id=effective_voice,
                text=text,
                model_id=model_id,
                output_format="mp3_44100_128",
            )
        except Exception as exc:
            _check_payment_required(exc)  # raises _PaymentRequiredError or re-raises exc

        # SDK returns either an object with .audio_base64/.alignment or a dict.
        # Use getattr first; fall back to dict access only if result is a dict,
        # to avoid AttributeError when calling .get() on a non-dict object.
        _as_dict = result if isinstance(result, dict) else {}
        audio_b64 = getattr(result, "audio_base64", None) or _as_dict.get("audio_base64")
        alignment = getattr(result, "alignment", None) or _as_dict.get("alignment")
        if not audio_b64 or not alignment:
            raise Exception(f"Unexpected ElevenLabs response shape: {result}")

        import base64
        audio_bytes = base64.b64decode(audio_b64)
        audio_path = os.path.join(self.static_dir, f"{uuid.uuid4()}.mp3")
        with open(audio_path, "wb") as f:
            f.write(audio_bytes)

        word_timings = self._chars_to_words(alignment)
        logger.info(
            f"ElevenLabs: synthesised {len(text)} chars → "
            f"{audio_path}, {len(word_timings)} words"
        )
        return audio_path, word_timings

    async def synthesize_sfx(
        self,
        description: str,
        duration_seconds: float = 8.0,
    ) -> str:
        """Generate ambient sound effects via ElevenLabs Sound Effects API.

        Returns path to the saved mp3 file, or empty string on any failure
        so the caller can treat SFX as strictly optional.

        duration_seconds is clamped to ElevenLabs limits (0.5–22s).
        """
        if not self.is_configured:
            return ""
        if not description or not description.strip():
            return ""

        duration_seconds = max(0.5, min(22.0, duration_seconds))

        try:
            os.makedirs(self.static_dir, exist_ok=True)

            result = await asyncio.to_thread(
                self._client.text_to_sound_effects.convert,
                text=description.strip(),
                duration_seconds=duration_seconds,
                prompt_influence=0.35,
            )

            # SDK may return an iterator or bytes directly
            if hasattr(result, "__iter__") and not isinstance(result, (bytes, bytearray)):
                audio_bytes = b"".join(result)
            else:
                audio_bytes = bytes(result)

            if not audio_bytes:
                logger.warning("ElevenLabs SFX returned empty audio")
                return ""

            sfx_path = os.path.join(self.static_dir, f"{uuid.uuid4()}_sfx.mp3")
            with open(sfx_path, "wb") as f:
                f.write(audio_bytes)

            logger.info(f"ElevenLabs SFX generated ({duration_seconds:.1f}s) → {sfx_path}")
            return sfx_path

        except Exception as e:
            logger.warning(f"ElevenLabs SFX failed (non-fatal): {e}")
            return ""

    @staticmethod
    def _chars_to_words(alignment) -> List[WordTiming]:
        """Collapse per-character timings into per-word timings.

        The endpoint returns three parallel arrays:
          characters / character_start_times_seconds / character_end_times_seconds
        We split on whitespace boundaries; word start = first char start, word
        end = last char end. Empty words are skipped (e.g. consecutive spaces).
        """
        _d = alignment if isinstance(alignment, dict) else {}
        chars = (
            getattr(alignment, "characters", None)
            or _d.get("characters", [])
        )
        starts = (
            getattr(alignment, "character_start_times_seconds", None)
            or _d.get("character_start_times_seconds", [])
        )
        ends = (
            getattr(alignment, "character_end_times_seconds", None)
            or _d.get("character_end_times_seconds", [])
        )

        words: List[WordTiming] = []
        cur_word = ""
        cur_start: float | None = None
        cur_end: float | None = None

        for ch, s, e in zip(chars, starts, ends):
            if ch.isspace():
                if cur_word:
                    words.append((cur_word, cur_start, cur_end))
                    cur_word = ""
                    cur_start = None
                continue
            if cur_start is None:
                cur_start = float(s)
            cur_word += ch
            cur_end = float(e)

        if cur_word and cur_start is not None and cur_end is not None:
            words.append((cur_word, cur_start, cur_end))

        return words
