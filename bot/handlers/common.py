"""Shared helpers, constants and pipeline routers for atlas-video-bot."""

import asyncio
import html
import logging
import math
import os
import re

from aiogram.fsm.context import FSMContext
from bot.keyboards import get_video_prompt_keyboard
from services.elevenlabs import ElevenLabsService
from services.gemini import GeminiService
from services.happyhorse import HappyHorseService
from services.imagegen import ImageGenService
from services.kling import KlingService
from services.pixverse import PixVerseService
from services.seedance import SeedanceService
from utils import container
from utils.config import config
from utils.tg import send_long_message

_SFX_SCENE_CHARS = 200

logger = logging.getLogger(__name__)

# Models that produce native audio — TTS must be layered on top, not replace the track.
_NATIVE_AUDIO_MODELS = {"happy_horse"}


# ── Duration config ──────────────────────────────────────────────────────────

DURATION_SEGMENTS: dict[int, tuple[int, int, int]] = {
    20: (2, 0, 0),
    40: (4, 0, 0),
    60: (6, 0, 0),
}
DEFAULT_TARGET_DURATION = 20


# ── State helpers ────────────────────────────────────────────────────────────

async def _is_generating(state: FSMContext) -> bool:
    import time
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

    Models in _NATIVE_AUDIO_MODELS already have audio baked in — TTS is
    layered on top (replace_existing_audio=False) instead of replacing silence.
    """
    word_timings = word_timings or []
    has_native_audio = video_model in _NATIVE_AUDIO_MODELS

    # Stage 1: colour grade.
    base_silent = raw_video_path
    if grade_enabled:
        base_silent = await gemini.apply_grade(raw_video_path, grade_params)

    # Stage 1.5: ElevenLabs SFX (only for models that produce silent video).
    sfx_path = ""
    if sfx_enabled and not has_native_audio:
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

    if has_tts_track or sfx_path or has_music:
        try:
            base_path = await gemini.mux_audio(
                base_silent,
                audio_path=tts_audio_path,
                sfx_path=sfx_path,
                music_path=music_path,
                music_volume=music_volume,
                # Native-audio models already have a track — preserve it, don't replace.
                replace_existing_audio=not has_native_audio,
            )
        except Exception as e:
            logger.error(f"Audio mux failed (non-fatal): {e}")
            if notify:
                reason = str(e).splitlines()[0][:200]
                await notify(f"⚠️ Audio mux failed — video will be silent.\nReason: {reason}")

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
    """Route to the correct generation function based on `settings['video_model']`.

    Returns dict with:
      - raw_video_path: path to the unprocessed mp4
      - tts_audio_path: path to TTS mp3 (may be empty)
      - word_timings: list[(word, start, end)] for karaoke
      - voiceover_text: final voiceover string used
      - has_native_audio: True if the video already contains audio
    """
    model = (settings.get("video_model") or "seedance").lower()

    if model in {"kling", "kling_v3_std", "kling_o3_pro", "kling_o3_std",
                 "kling_o3_pro_ref", "kling_o3_std_ref"}:
        return await _generate_kling(
            gemini=gemini, settings=settings,
            video_prompt=video_prompt, voiceover_text=voiceover_text,
            image_path=image_path, notify=notify,
        )
    if model == "happy_horse":
        return await _generate_happyhorse(
            gemini=gemini, settings=settings,
            video_prompt=video_prompt, voiceover_text=voiceover_text,
            image_path=image_path, notify=notify,
        )
    if model == "pixverse":
        return await _generate_pixverse(
            gemini=gemini, settings=settings,
            video_prompt=video_prompt, voiceover_text=voiceover_text,
            image_path=image_path, notify=notify,
        )
    # Seedance variants (default)
    return await _generate_seedance(
        gemini=gemini, settings=settings,
        video_prompt=video_prompt, voiceover_text=voiceover_text,
        image_path=image_path, notify=notify,
    )


# ── Per-model generation functions ───────────────────────────────────────────

async def _generate_seedance(
    *,
    gemini: GeminiService,
    settings: dict,
    video_prompt: str,
    voiceover_text: str,
    image_path: str | None,
    notify,
) -> dict:
    from services.seedance import MODEL_IDS as _SEEDANCE_IDS, MODEL_LABELS as _SEEDANCE_LABELS
    model = (settings.get("video_model") or "seedance").lower()
    atlas_model_id = _SEEDANCE_IDS.get(model, _SEEDANCE_IDS["seedance"])
    label = _SEEDANCE_LABELS.get(model, "Seedance 2.0")

    target_duration = settings.get("target_duration", DEFAULT_TARGET_DURATION)
    n_clips = max(1, math.ceil(target_duration / 10))

    scenes = _split_scene_into_shots(video_prompt, n_clips)
    await notify(
        f"⏱ Generating <b>{label}</b> ({len(scenes)} clip(s) × 10s) — "
        "each clip takes ~3-5 min…"
    )

    seedance = container.inject(SeedanceService)

    if image_path and os.path.exists(image_path):
        anchor_url = await seedance.upload_photo(image_path)
        anchor_urls = [anchor_url] * len(scenes)
        clips = await seedance.generate_clips(
            scene_prompts=scenes,
            anchor_photo_urls=anchor_urls,
            clip_duration=10,
            model_id=atlas_model_id,
        )
    else:
        clips = []
        for i, prompt in enumerate(scenes):
            await notify(f"🎞 {label} clip {i + 1}/{len(scenes)}…")
            clip = await seedance.generate_clip(prompt=prompt, duration=10, model_id=atlas_model_id)
            clips.append(clip)

    raw_video = await gemini.concat_videos(clips) if len(clips) > 1 else clips[0]
    await notify(f"✅ {label} clips ready")

    tts_audio_path, word_timings = await _synthesize_tts(voiceover_text, settings, notify)

    return {
        "raw_video_path": raw_video,
        "tts_audio_path": tts_audio_path,
        "word_timings": word_timings,
        "voiceover_text": voiceover_text,
        "has_native_audio": False,
    }


async def _generate_kling(
    *,
    gemini: GeminiService,
    settings: dict,
    video_prompt: str,
    voiceover_text: str,
    image_path: str | None,
    notify,
) -> dict:
    from services.kling import MODEL_IDS as _KLING_IDS, MODEL_LABELS as _KLING_LABELS
    model = (settings.get("video_model") or "kling").lower()
    atlas_model_id = _KLING_IDS.get(model, _KLING_IDS["kling"])
    label = _KLING_LABELS.get(model, "Kling")

    target_duration = settings.get("target_duration", DEFAULT_TARGET_DURATION)
    n_clips = max(1, math.ceil(target_duration / 10))
    negative_prompt = settings.get("negative_prompt") or ""

    scenes = _split_scene_into_shots(video_prompt, n_clips)
    await notify(
        f"⏱ Generating <b>{label}</b> ({len(scenes)} clip(s) × 10s) — "
        "each clip takes ~2-4 min…"
    )

    kling = container.inject(KlingService)

    if image_path and os.path.exists(image_path):
        anchor_url = await kling.upload_photo(image_path)
        anchor_urls = [anchor_url] * len(scenes)
        clips = await kling.generate_clips(
            scene_prompts=scenes,
            anchor_photo_urls=anchor_urls,
            clip_duration=10,
            negative_prompt=negative_prompt,
            model_id=atlas_model_id,
        )
    else:
        clips = []
        for i, prompt in enumerate(scenes):
            await notify(f"🎞 {label} clip {i + 1}/{len(scenes)}…")
            clip = await kling.generate_clip(
                prompt=prompt, duration=10,
                negative_prompt=negative_prompt,
                model_id=atlas_model_id,
            )
            clips.append(clip)

    raw_video = await gemini.concat_videos(clips) if len(clips) > 1 else clips[0]
    await notify(f"✅ {label} clips ready")

    tts_audio_path, word_timings = await _synthesize_tts(voiceover_text, settings, notify)

    return {
        "raw_video_path": raw_video,
        "tts_audio_path": tts_audio_path,
        "word_timings": word_timings,
        "voiceover_text": voiceover_text,
        "has_native_audio": False,
    }


async def _generate_happyhorse(
    *,
    gemini: GeminiService,
    settings: dict,
    video_prompt: str,
    voiceover_text: str,
    image_path: str | None,
    notify,
) -> dict:
    """Happy Horse generates one video (≤15s) with built-in Foley audio.

    A reference photo is required — generation is skipped with an error if missing.
    TTS is synthesized separately and layered on top in post-processing.
    """
    target_duration = settings.get("target_duration", DEFAULT_TARGET_DURATION)
    clip_duration = max(3, min(15, target_duration))
    resolution = settings.get("video_resolution", "720p")

    if not image_path or not os.path.exists(image_path):
        raise ValueError(
            "Happy Horse requires a reference photo. "
            "Go back and make sure an image was generated."
        )

    await notify(
        f"⏱ Generating <b>Happy Horse</b> ({clip_duration}s, native audio) — "
        "takes ~2-4 min…"
    )

    horse = container.inject(HappyHorseService)
    anchor_url = await horse.upload_photo(image_path)

    # Use the scene description (without voiceover) as the animation prompt.
    scene_only = _strip_voiceover(video_prompt)
    raw_video = await horse.generate_clip(
        prompt=scene_only or video_prompt,
        image_url=anchor_url,
        duration=clip_duration,
        resolution=resolution,
    )
    await notify("✅ Happy Horse clip ready")

    tts_audio_path, word_timings = await _synthesize_tts(voiceover_text, settings, notify)

    return {
        "raw_video_path": raw_video,
        "tts_audio_path": tts_audio_path,
        "word_timings": word_timings,
        "voiceover_text": voiceover_text,
        "has_native_audio": True,  # Foley audio is baked into the video
    }


async def _generate_pixverse(
    *,
    gemini: GeminiService,
    settings: dict,
    video_prompt: str,
    voiceover_text: str,
    image_path: str | None,
    notify,
) -> dict:
    target_duration = settings.get("target_duration", DEFAULT_TARGET_DURATION)
    n_clips = max(1, math.ceil(target_duration / 5))

    scenes = _split_scene_into_shots(video_prompt, n_clips)
    await notify(
        f"⏱ Generating <b>PixVerse V4.5</b> ({len(scenes)} clip(s) × 5s) — "
        "each clip takes ~1-3 min…"
    )

    pixverse = container.inject(PixVerseService)

    if image_path and os.path.exists(image_path):
        anchor_url = await pixverse.upload_photo(image_path)
        anchor_urls = [anchor_url] * len(scenes)
        clips = await pixverse.generate_clips(
            scene_prompts=scenes,
            anchor_photo_urls=anchor_urls,
            clip_duration=5,
        )
    else:
        clips = []
        for i, prompt in enumerate(scenes):
            await notify(f"🎞 PixVerse clip {i + 1}/{len(scenes)}…")
            clip = await pixverse.generate_clip(prompt=prompt, duration=5)
            clips.append(clip)

    raw_video = await gemini.concat_videos(clips) if len(clips) > 1 else clips[0]
    await notify("✅ PixVerse clips ready")

    tts_audio_path, word_timings = await _synthesize_tts(voiceover_text, settings, notify)

    return {
        "raw_video_path": raw_video,
        "tts_audio_path": tts_audio_path,
        "word_timings": word_timings,
        "voiceover_text": voiceover_text,
        "has_native_audio": False,
    }


# ── Shared TTS helper ────────────────────────────────────────────────────────

async def _synthesize_tts(
    voiceover_text: str,
    settings: dict,
    notify,
) -> tuple[str, list]:
    """Synthesize TTS via ElevenLabs. Returns (audio_path, word_timings)."""
    if not voiceover_text or not voiceover_text.strip():
        return "", []
    try:
        eleven = container.inject(ElevenLabsService)
        if eleven.is_configured:
            await notify("🔊 Synthesizing voice via ElevenLabs…")
            audio_path, word_timings = await eleven.synthesize(
                text=voiceover_text,
                voice_id=settings.get("voice_id") or "",
            )
            return audio_path, word_timings
    except Exception as e:
        logger.warning(f"TTS failed (non-fatal): {e}")
        await notify(f"⚠️ Voice synthesis failed: {str(e).splitlines()[0][:160]}")
    return "", []


# ── Scene splitting helpers ───────────────────────────────────────────────────

def _split_scene_into_shots(video_prompt: str, target_count: int) -> list[str]:
    """Split Scene section by 'Cut to:' into per-clip visual prompts.

    Falls back to voiceover sentence splitting when the scene has no cuts.
    """
    scene = _strip_voiceover(video_prompt)
    # Remove leading "Scene:" label
    scene = re.sub(r'^scene\s*:\s*', '', scene.strip(), flags=re.IGNORECASE)

    shots = [s.strip() for s in re.split(r'\bcut\s+to\s*:\s*', scene, flags=re.IGNORECASE) if s.strip()]

    if not shots:
        return _split_voiceover_into_scenes(video_prompt, target_count)

    if len(shots) >= target_count:
        # Merge excess shots into the last bucket
        result = shots[:target_count - 1]
        result.append(" Cut to: ".join(shots[target_count - 1:]))
        return result

    # Too few shots — pad by repeating the last one
    while len(shots) < target_count:
        shots.append(shots[-1])
    return shots


def _split_voiceover_into_scenes(voiceover: str, target_count: int) -> list[str]:
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


def _strip_voiceover(video_prompt: str) -> str:
    """Return only the scene part, without the Voiceover: line."""
    lines = [
        ln for ln in video_prompt.splitlines()
        if not ln.lower().lstrip().startswith("voiceover")
    ]
    return "\n".join(lines).strip()


# ── Prompt builders ─────────────────────────────────────────────────────────

async def _build_video_prompt(enhance_prompt: str, settings: dict, gemini: GeminiService) -> str:
    base_sys = settings.get("system_video_prompt") or ""
    target_duration = settings.get("target_duration", DEFAULT_TARGET_DURATION)
    target_words = max(20, int(target_duration * 2.3))

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
    base_sys = settings.get("system_image_prompt") or ""
    return await gemini.generate_text(
        enhance_prompt,
        base_sys,
        settings.get("text_model") or "",
    )


# ── Helpers reused across handlers ──────────────────────────────────────────

def _extract_first_scene(video_prompt: str, gemini) -> str:
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
