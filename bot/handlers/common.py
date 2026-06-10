"""Shared helpers, constants and pipeline routers for fal-video-bot."""

import asyncio
import html
import logging
import os
import random
import re
import time

from aiogram.fsm.context import FSMContext
from bot.keyboards import get_video_prompt_keyboard
from services.elevenlabs import ElevenLabsService
from services.gemini import GeminiService
from services.imagegen import ImageGenService
from services.kling import KlingService
from services.omnihuman import OmniHumanService
from services.seedance import SeedanceService
from utils import container
from utils.config import config
from utils.tg import send_long_message

# Max chars of scene description passed to ElevenLabs SFX as ambient sound context.
_SFX_SCENE_CHARS = 200

logger = logging.getLogger(__name__)


# ── Duration config ──────────────────────────────────────────────────────────

# Supported target durations. Used directly for Seedance (clip count = target/10).
# Kling / OmniHuman duration is driven by the TTS audio length (model decides
# how many seconds based on the supplied voice track).
DURATION_SEGMENTS: dict[int, tuple[int, int, int]] = {
    15: (8, 1, 7),
    30: (8, 3, 7),
    45: (8, 5, 7),
    60: (8, 7, 7),
}
DEFAULT_TARGET_DURATION = 15

# Models that produce lip-sync video (already contain TTS audio inside the file).
LIP_SYNC_MODELS = {"kling", "omnihuman"}
# Models that produce silent scene clips (TTS is added in post-processing).
SCENE_MODELS = {"seedance"}


# ── Avatar style for lip-sync portraits ──────────────────────────────────────

_AVATAR_STYLE_HINT = (
    "Ultra-realistic portrait, clean minimal background, smart casual outfit, "
    "confident engaging expression, soft natural lighting, sharp focus on face, "
    "shallow depth of field, photorealistic skin, no artificial AI look. "
    "Vertical portrait composition suitable for talking-head lip-sync video."
)


# ── State helpers ────────────────────────────────────────────────────────────

async def _is_generating(state: FSMContext) -> bool:
    """True if a generation is actively running.

    Auto-clears the flag when set more than 30 minutes ago so a crash can't
    leave the bot permanently locked.
    """
    data = await state.get_data()
    if not data.get("generation_in_progress"):
        return False
    started_at = data.get("generation_started_at", 0)
    if time.time() - started_at > 1800:
        logger.warning("Auto-clearing stale generation_in_progress flag (>30 min)")
        await state.update_data(generation_in_progress=False, generation_started_at=0)
        return False
    return True


# ── Post-processing pipeline ─────────────────────────────────────────────────

async def _post_process_video(
    gemini: GeminiService,
    raw_video_path: str,
    video_prompt: str,
    subs_default_on: bool,
    *,
    video_model: str,
    grade_enabled: bool = False,
    grade_params: dict | None = None,
    voice_id: str = "",
    notify=None,
    voiceover_text: str = "",
    word_timings: list | None = None,
    tts_audio_path: str = "",
    sfx_enabled: bool = False,
    video_speed: float = 1.0,
    music_path: str = "",
    music_volume: float = 0.18,
):
    """Pipeline: raw → grade → SFX → audio mux → karaoke subs → speed.

    Behaviour depends on `video_model`:

    • Lip-sync (kling/omnihuman): the raw video ALREADY contains the TTS
      voice. We do NOT replace it — we mix optional music/SFX over the top
      and keep the model's voice track.

    • Scene clips (seedance): the raw video is silent — we mux TTS audio,
      optional music and SFX into the file.

    `voiceover_text` and `word_timings` are usually generated BEFORE this
    function is called (lip-sync needs the audio file first to feed into
    the model). They are passed back in so karaoke subtitles use the same
    timings as the embedded voice.

    Returns the same dict shape video-gen-bot uses, so handlers don't care
    which model produced the video.
    """
    word_timings = word_timings or []
    is_lip_sync = video_model in LIP_SYNC_MODELS

    # Stage 1: colour grade (audio stream preserved via acodec=copy).
    base_silent = raw_video_path
    if grade_enabled:
        base_silent = await gemini.apply_grade(raw_video_path, grade_params)

    # Stage 1.5: ElevenLabs SFX (ambient sounds).
    sfx_path = ""
    if sfx_enabled:
        try:
            eleven = container.inject(ElevenLabsService)
            if eleven.is_configured:
                video_dur = await asyncio.to_thread(GeminiService._probe_duration, base_silent)
                sfx_desc = _sfx_description_from_scene(video_prompt)
                sfx_path = await eleven.synthesize_sfx(
                    sfx_desc, duration_seconds=min(video_dur, 22.0)
                )
        except Exception as e:
            logger.warning(f"SFX generation failed (non-fatal): {e}")

    # Stage 2: audio mux.
    base_path = base_silent
    has_music = bool(music_path and os.path.exists(music_path))
    has_tts_track = bool(tts_audio_path and os.path.exists(tts_audio_path))

    if is_lip_sync:
        # Lip-sync videos already contain the voice — only re-encode if we
        # actually have music or SFX to add. Otherwise keep raw file.
        if has_music or sfx_path:
            base_path = await gemini.mux_audio(
                base_silent,
                audio_path="",                # no TTS to add — voice is in video
                sfx_path=sfx_path,
                music_path=music_path,
                music_volume=music_volume,
                replace_existing_audio=False,  # KEEP original voice
            )
    else:
        # Scene clips are silent — always mux TTS (+ optional music / SFX)
        # when we have a voiceover to put on top.
        if has_tts_track:
            try:
                base_path = await gemini.mux_audio(
                    base_silent,
                    audio_path=tts_audio_path,
                    sfx_path=sfx_path,
                    music_path=music_path,
                    music_volume=music_volume,
                    replace_existing_audio=True,
                )
            except Exception as e:
                logger.error(f"TTS pipeline failed, falling back to silent video: {e}")
                base_path = base_silent
                if notify:
                    reason = str(e).splitlines()[0][:200]
                    await notify(f"⚠️ Audio mux failed — video will be silent.\nReason: {reason}")
        elif sfx_path or has_music:
            base_path = await gemini.mux_audio(
                base_silent,
                audio_path="",
                sfx_path=sfx_path,
                music_path=music_path,
                music_volume=music_volume,
                replace_existing_audio=True,
            )

    # Stage 3: karaoke subtitles.
    if subs_default_on and word_timings:
        subbed = await gemini.burn_karaoke_subtitles(base_path, word_timings)
        final = await gemini.apply_speed(subbed, video_speed) if video_speed != 1.0 else subbed
        return {
            "video_path": final,
            "video_path_raw": raw_video_path,
            "video_path_base": base_path,
            "video_path_subbed": subbed,
            "voiceover_text": voiceover_text,
            "word_timings": word_timings,
            "subtitles_on": True,
            "grade_params": grade_params,
            "voice_id": voice_id,
        }
    final = await gemini.apply_speed(base_path, video_speed) if video_speed != 1.0 else base_path
    return {
        "video_path": final,
        "video_path_raw": raw_video_path,
        "video_path_base": base_path,
        "video_path_subbed": None,
        "voiceover_text": voiceover_text,
        "word_timings": word_timings,
        "subtitles_on": False,
        "grade_params": grade_params,
        "voice_id": voice_id,
    }


def _sfx_description_from_scene(video_prompt: str) -> str:
    """Strip the Voiceover line and use the first chunk as ambient-sound context."""
    lines = [
        ln for ln in video_prompt.splitlines()
        if not ln.lower().lstrip().startswith("voiceover")
    ]
    scene = " ".join(ln.strip() for ln in lines if ln.strip())
    snippet = scene[:_SFX_SCENE_CHARS].strip()
    return f"Ambient ASMR sounds for: {snippet}"


# ── Video generation routers ─────────────────────────────────────────────────

async def generate_video_for_model(
    *,
    gemini: GeminiService,
    settings: dict,
    video_prompt: str,
    voiceover_text: str,
    image_path: str | None,
    notify,
) -> dict:
    """Generate a raw video for the configured model.

    Returns dict with:
      - raw_video_path: path to the unprocessed mp4
      - tts_audio_path: path to TTS mp3 (or "" if model produced audio internally)
      - word_timings: list[(word, start, end)] for karaoke
      - voiceover_text: final voiceover string used
    """
    model = (settings.get("video_model") or "seedance").lower()

    if model in LIP_SYNC_MODELS:
        return await _generate_lipsync(
            model=model,
            gemini=gemini,
            settings=settings,
            voiceover_text=voiceover_text,
            image_path=image_path,
            notify=notify,
        )
    # Default — scene model (Seedance).
    return await _generate_seedance(
        gemini=gemini,
        settings=settings,
        video_prompt=video_prompt,
        voiceover_text=voiceover_text,
        image_path=image_path,
        notify=notify,
    )


async def _generate_lipsync(
    *,
    model: str,
    gemini: GeminiService,
    settings: dict,
    voiceover_text: str,
    image_path: str | None,
    notify,
) -> dict:
    """Kling / OmniHuman flow: TTS first → fal.ai(photo + audio) → video."""
    if not image_path or not os.path.exists(image_path):
        raise Exception("Lip-sync models require a reference photo (image_path).")
    if not voiceover_text or not voiceover_text.strip():
        raise Exception("Lip-sync models require a non-empty voiceover_text.")

    eleven = container.inject(ElevenLabsService)
    if not eleven.is_configured:
        raise Exception(
            "ElevenLabs API key is not configured — lip-sync requires TTS.\n"
            "Set ELEVENLABS_API_KEY in .env and a voice ID in ⚙️ Settings → 🎙 Voice."
        )

    await notify("🔊 Synthesizing voice via ElevenLabs…")
    audio_path, word_timings = await eleven.synthesize(
        text=voiceover_text,
        voice_id=settings.get("voice_id") or "",
    )
    await notify("✅ Voice ready")

    label = "Kling Avatar v2" if model == "kling" else "OmniHuman-1"
    await notify(
        f"⏱ Generating lip-sync via <b>{label}</b> — this takes <b>~7-10 minutes</b>, please wait…"
    )

    if model == "kling":
        svc = container.inject(KlingService)
    else:
        svc = container.inject(OmniHumanService)

    raw_video = await svc.generate_avatar_video(
        photo_path=image_path,
        audio_path=audio_path,
        prompt="",
    )
    await notify(f"✅ {label} video ready")

    return {
        "raw_video_path": raw_video,
        "tts_audio_path": "",            # voice is already inside the video
        "word_timings": word_timings,    # used only for karaoke subtitles
        "voiceover_text": voiceover_text,
    }


async def _generate_seedance(
    *,
    gemini: GeminiService,
    settings: dict,
    video_prompt: str,
    voiceover_text: str,
    image_path: str | None,
    notify,
) -> dict:
    """Seedance flow: split voiceover into scenes → generate clips → concat."""
    target_duration = settings.get("target_duration", DEFAULT_TARGET_DURATION)
    n_clips = max(1, target_duration // 10)

    scenes = _split_voiceover_into_scenes(voiceover_text or video_prompt, n_clips)
    await notify(
        f"⏱ Generating <b>Seedance 2.0</b> ({len(scenes)} clip(s) × 10s) — "
        "each clip takes ~3-5 min…"
    )

    seedance = container.inject(SeedanceService)

    if image_path and os.path.exists(image_path):
        anchor_url = await seedance.upload_photo(image_path)
    else:
        anchor_url = ""

    if anchor_url:
        anchor_urls = [anchor_url] * len(scenes)
        clips = await seedance.generate_clips(
            scene_prompts=scenes,
            anchor_photo_urls=anchor_urls,
            clip_duration=10,
        )
    else:
        # No reference photo — fall back to text-to-video.
        clips = []
        for i, prompt in enumerate(scenes):
            await notify(f"🎞 Seedance clip {i + 1}/{len(scenes)}…")
            clip = await seedance.generate_clip(prompt=prompt, duration=10)
            clips.append(clip)

    raw_video = await gemini.concat_videos(clips) if len(clips) > 1 else clips[0]
    await notify("✅ Seedance clips ready")

    # Synthesise voiceover NOW (after video) so we can mux + karaoke in post.
    tts_audio_path = ""
    word_timings: list = []
    if voiceover_text and voiceover_text.strip():
        try:
            eleven = container.inject(ElevenLabsService)
            if eleven.is_configured:
                await notify("🔊 Synthesizing voice via ElevenLabs…")
                tts_audio_path, word_timings = await eleven.synthesize(
                    text=voiceover_text,
                    voice_id=settings.get("voice_id") or "",
                )
        except Exception as e:
            logger.warning(f"TTS for Seedance failed (non-fatal): {e}")
            await notify(f"⚠️ Voice synthesis failed: {str(e).splitlines()[0][:160]}")

    return {
        "raw_video_path": raw_video,
        "tts_audio_path": tts_audio_path,
        "word_timings": word_timings,
        "voiceover_text": voiceover_text,
    }


def _split_voiceover_into_scenes(voiceover: str, target_count: int) -> list[str]:
    """Split voiceover text into scene prompts for multi-clip Seedance.

    Splits on sentence boundaries, then groups so the final scene count
    matches `target_count`. Used directly as image-to-video prompts.
    """
    sentences = re.split(r'(?<=[.!?])\s+', voiceover.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return [voiceover or "scene continues"]
    if len(sentences) <= target_count:
        return sentences

    per_group = max(1, len(sentences) // target_count)
    scenes: list[str] = []
    for i in range(0, len(sentences), per_group):
        group = sentences[i:i + per_group]
        scenes.append(" ".join(group))
        if len(scenes) == target_count:
            remainder = sentences[i + per_group:]
            if remainder:
                scenes[-1] += " " + " ".join(remainder)
            break
    return scenes


# ── Prompt builders ─────────────────────────────────────────────────────────

async def _build_video_prompt(enhance_prompt: str, settings: dict, gemini: GeminiService) -> str:
    """Generate a video_prompt from the enhanced idea, tuned to the chosen model.

    For lip-sync models we ask the LLM for a clean voiceover script only
    (no shot list — the model doesn't render scenes).

    For scene models (Seedance) we ask for both a multi-shot scene and a
    voiceover, as in video-gen-bot.
    """
    base_sys = settings.get("system_video_prompt") or ""
    target_duration = settings.get("target_duration", DEFAULT_TARGET_DURATION)
    target_words = max(20, int(target_duration * 2.3))
    model = (settings.get("video_model") or "seedance").lower()

    if model in LIP_SYNC_MODELS:
        _FORMAT_SUFFIX = (
            f"\n\n--- OUTPUT FORMAT (mandatory) ---\n"
            f"Write a single voiceover monologue, written in the FIRST PERSON, "
            f"approximately {target_words} words ({target_duration} seconds at "
            f"natural talking pace). The speaker is an on-camera presenter "
            f"directly addressing the viewer.\n\n"
            f'Output ONLY the line:\nVoiceover: "<narration text>"\n\n'
            f"Rules:\n"
            f"- Do NOT use markdown\n"
            f"- The word Voiceover: appears ONLY ONCE\n"
            f"- Use straight double quotes\n"
            f"- Conversational, engaging, G-rated content"
        )
    else:
        _FORMAT_SUFFIX = (
            f"\n\n--- OUTPUT FORMAT (mandatory) ---\n"
            f"Write TWO sections only:\n\n"
            f"Scene: Describe a sequence of 4-5 camera shots separated by 'cut to:'. "
            f"Each shot = shot size + subject action + camera movement. "
            f"Use this exact pattern:\n"
            f"Wide shot of [subject+setting], camera slowly pushes in. "
            f"Cut to: close-up of [detail+action], rack focus. "
            f"Cut to: macro shot of [texture/ASMR detail], static. "
            f"Cut to: medium shot of [character reaction], dolly back. "
            f"Cut to: overhead shot of [final moment], crane up.\n"
            f"Shot sizes to use: wide shot, medium shot, close-up, extreme close-up, "
            f"macro shot, overhead shot, low angle shot.\n"
            f"Camera moves to use: slow push in, dolly back, rack focus, pan left/right, "
            f"crane up, handheld, static.\n"
            f"Max 120 words for the Scene section.\n\n"
            f'Voiceover: "narration text of approximately {target_words} words '
            f'({target_duration} seconds at natural talking pace)"\n\n'
            f"Rules:\n"
            f"- Scene must depict EXACTLY the topic from the input\n"
            f"- Do NOT use markdown (>, **, *)\n"
            f"- The word Voiceover: appears ONLY ONCE\n"
            f"- Use straight double quotes around the narration\n"
            f"- G-rated, child-safe content only"
        )

    sys_with_hint = (base_sys + _FORMAT_SUFFIX) if base_sys else _FORMAT_SUFFIX
    return await gemini.generate_text(
        enhance_prompt,
        sys_with_hint,
        settings.get("text_model") or "",
    )


async def _build_image_prompt(
    enhance_prompt: str,
    settings: dict,
    gemini: GeminiService,
) -> str:
    """Generate the image prompt that drives Imagen.

    For lip-sync models we override the user's system prompt with a strict
    portrait template so the photo is always front-facing (Kling / OmniHuman
    quality drops fast on non-portrait shots).
    """
    model = (settings.get("video_model") or "seedance").lower()
    base_sys = settings.get("system_image_prompt") or ""

    if model in LIP_SYNC_MODELS:
        base_sys = (
            (base_sys + "\n\n") if base_sys else ""
        ) + (
            "Output a description of a single person suitable for a "
            "front-facing studio portrait. Describe age, gender, ethnicity, "
            "hair, clothing, expression and background in plain prose. "
            "Do NOT mention multiple frames, multiple poses, or sequences — "
            "the goal is ONE photo.\n" + _AVATAR_STYLE_HINT
        )

    return await gemini.generate_text(
        enhance_prompt,
        base_sys,
        settings.get("text_model") or "",
    )


# ── Helpers reused from video-gen-bot ───────────────────────────────────────

def _extract_first_scene(video_prompt: str, gemini) -> str:
    """Extract the opening visual beat from a video_prompt."""
    scene, _ = _split_video_prompt(video_prompt, gemini)
    if not scene:
        return ""

    lines = scene.splitlines()
    block: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if block:
                break
            continue

        low = stripped.lower()
        if low.startswith(("style:", "camera:", "audio:", "sfx:", "---")):
            break

        is_shot_header = stripped.startswith("[") and (
            bool(re.match(r"\[\d{1,2}:\d{2}", stripped))
            or "s]" in low
            or "shot" in low
        )

        if is_shot_header and block:
            break

        block.append(stripped)
        if len(block) >= 4:
            break

    return " ".join(block)


def _split_video_prompt(video_prompt: str, gemini) -> tuple[str, str]:
    """Split full video_prompt into (scene, voiceover)."""
    voiceover = gemini.extract_voiceover(video_prompt) or ""
    if not voiceover:
        return video_prompt.strip(), ""
    lines = video_prompt.splitlines()
    scene_lines = [
        ln for ln in lines
        if not ln.lower().lstrip().startswith("voiceover")
    ]
    scene = "\n".join(scene_lines).strip()
    return scene, voiceover


def _format_video_prompt_message(scene: str, voiceover: str) -> str:
    parts = []
    if scene:
        parts.append(f"<b>🎬 Scene / Hook:</b>\n<code>{html.escape(scene)}</code>")
    if voiceover:
        parts.append(f"\n<b>🎙 Voiceover:</b>\n<code>{html.escape(voiceover)}</code>")
    else:
        parts.append("\n<i>⚠️ Voiceover not found — video will have no narration.</i>")
    return "\n".join(parts)


async def _show_video_prompt(target, scene: str, voiceover: str):
    text = _format_video_prompt_message(scene, voiceover)
    await send_long_message(target, text, keyboard=get_video_prompt_keyboard(), parse_mode="HTML")


async def _apply_outro(gemini: GeminiService, result: dict) -> dict:
    """Append the subscribe outro clip to every video variant in result."""
    outro_path = config.get("OUTRO_VIDEO_PATH", "")
    if not outro_path or not os.path.exists(outro_path):
        return result

    base_with_outro = await gemini.append_outro(result["video_path_base"], outro_path)
    result["video_path_base"] = base_with_outro

    subbed_with_outro = None
    if result.get("video_path_subbed"):
        subbed_with_outro = await gemini.append_outro(result["video_path_subbed"], outro_path)
        result["video_path_subbed"] = subbed_with_outro

    if result.get("subtitles_on") and subbed_with_outro:
        result["video_path"] = subbed_with_outro
    else:
        result["video_path"] = base_with_outro

    return result
