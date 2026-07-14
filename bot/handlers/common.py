"""Shared helpers, constants and pipeline routers for atlas-video-bot."""

import asyncio
import html
import json as _json
import logging
import math
import os
import re

from aiogram.fsm.context import FSMContext
from bot.keyboards import get_video_prompt_keyboard
from services.elevenlabs import ElevenLabsService
from services.gemini import GeminiService
from services.imagegen import ImageGenService
from services.kling import KlingService
from services.seedance import SeedanceService
from utils import container
from utils.config import config
from utils.tg import send_long_message

_SFX_SCENE_CHARS = 200

logger = logging.getLogger(__name__)

# Models that generate video from text only — no reference image needed or used.
_T2V_VIDEO_MODELS = {"seedance_t2v", "seedance_mini_t2v", "kling_t2v", "kling_turbo_t2v"}


# ── Duration config ──────────────────────────────────────────────────────────

DURATION_SEGMENTS: dict[int, tuple[int, int, int]] = {
    15: (1, 0, 0),
    30: (2, 0, 0),
    45: (3, 0, 0),
    60: (4, 0, 0),
}
DEFAULT_TARGET_DURATION = 30


def _substitute_prompt_vars(prompt: str, settings: dict) -> str:
    """Substitute {DURATION}, {NUM_SCENES}, {SCENE_DURATION} from settings.

    Uses str.replace (not str.format) so arbitrary braces in user-entered or
    AI-generated prompt text can never crash the pipeline.
    """
    target_duration = settings.get("target_duration", DEFAULT_TARGET_DURATION)
    video_model = (settings.get("video_model") or "seedance").lower()

    # Use the same duration logic as the video generation pipeline
    clip_duration = _clip_duration_for_model(video_model)

    # Calculate number of scenes and per-scene duration
    num_scenes = max(1, math.ceil(target_duration / clip_duration))
    scene_duration = target_duration // num_scenes if num_scenes > 0 else clip_duration

    return (
        prompt.replace("{DURATION}", str(target_duration))
        .replace("{NUM_SCENES}", str(num_scenes))
        .replace("{SCENE_DURATION}", str(scene_duration))
    )


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
    has_native_audio: bool = False,
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

    has_native_audio comes from the generation result: when the model produced
    its own audio track (e.g. Kling multi-shot sound, Seedance ASMR ambient),
    TTS/music are layered on top (replace_existing_audio=False) instead of
    replacing it, and the separate SFX stage is skipped.
    """
    word_timings = word_timings or []

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
    image_paths: list[str] | None,
    notify,
    script_json: str = "",
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

    shots: list[dict] | None = None
    if script_json:
        parsed = GeminiService.parse_script_json(script_json)
        if parsed is not None:
            shots = parsed.get("shots") or None

    # Safety net: LLM-produced shot durations may not sum to the requested
    # target — renormalize in code so the video length matches the setting.
    if shots:
        if model.startswith("kling"):
            min_dur, max_dur = 3, 10
        else:
            min_dur, max_dur = 4, 15
        target = settings.get("target_duration", DEFAULT_TARGET_DURATION)
        shots = _normalize_shot_durations(shots, target, min_dur, max_dur)

    # Prefix routing: any kling_* goes to Kling (unknown/removed variants fall
    # back to Kling v3 Pro inside), everything else — to Seedance (default).
    if model.startswith("kling"):
        return await _generate_kling(
            gemini=gemini, settings=settings,
            video_prompt=video_prompt, voiceover_text=voiceover_text,
            image_paths=image_paths, notify=notify, shots=shots,
        )
    # Seedance variants (default)
    return await _generate_seedance(
        gemini=gemini, settings=settings,
        video_prompt=video_prompt, voiceover_text=voiceover_text,
        image_paths=image_paths, notify=notify, shots=shots,
    )


# ── Per-model generation functions ───────────────────────────────────────────

async def _generate_seedance(
    *,
    gemini: GeminiService,
    settings: dict,
    video_prompt: str,
    voiceover_text: str,
    image_paths: list[str] | None,
    notify,
    shots: list[dict] | None = None,
) -> dict:
    from services.seedance import MODEL_IDS as _SEEDANCE_IDS, MODEL_LABELS as _SEEDANCE_LABELS
    model = (settings.get("video_model") or "seedance").lower()
    atlas_model_id = _SEEDANCE_IDS.get(model, _SEEDANCE_IDS["seedance"])
    label = _SEEDANCE_LABELS.get(model, "Seedance 2.0")
    resolution = settings.get("video_resolution", "720p")

    seedance = container.inject(SeedanceService)
    valid_paths = [p for p in (image_paths or []) if os.path.exists(p)]

    # ── Seedance I2V / T2V ────────────────────────────────────────────────────
    target_duration = settings.get("target_duration", DEFAULT_TARGET_DURATION)
    clip_dur = _clip_duration_for_model(model)

    if shots:
        scenes = [_build_shot_prompt(s) for s in shots]
        durations = [int(max(4, min(15, s.get("duration_seconds", clip_dur)))) for s in shots]
        transitions = [s.get("transition", "cut") for s in shots]
    else:
        durations = _plan_clip_durations(target_duration, min_dur=4, max_dur=clip_dur)
        scenes = _split_scene_into_shots(video_prompt, len(durations))
        transitions = None

    is_t2v = model in _T2V_VIDEO_MODELS

    # Native model audio: for the ASMR niche the model's own synchronized sound
    # IS the content (TTS mixed quietly on top); and when there is no voiceover
    # at all (e.g. 🗣 Voiceover OFF), ambient sound beats dead silence —
    # mirrors the Kling multi-shot behaviour.
    keep_native = (
        settings.get("content_preset") == "asmr"
        or not bool(voiceover_text and voiceover_text.strip())
    )

    if not is_t2v and valid_paths:
        uploaded_urls = list(await asyncio.gather(*[seedance.upload_photo(p) for p in valid_paths]))
        total_dur = sum(durations)

        if total_dur <= 15:
            # Fits one Atlas call (hard 15s per-request limit) — render all
            # scenes continuously in a single multi-scene request.
            await notify(
                f"⏱ Generating <b>{label}</b> ({len(scenes)} scene(s)"
                + f", {total_dur}s total"
                + (", native audio" if keep_native else "")
                + ") — takes ~3-5 min…"
            )
            raw_video = await seedance.generate_multi_scene_clip(
                scene_prompts=scenes,
                image_url=uploaded_urls[0],
                total_duration=total_dur,
                resolution=resolution,
                model_id=atlas_model_id,
                keep_native_audio=keep_native,
            )
        else:
            # Longer than one call allows — clip-by-clip with last-frame
            # stitching for continuity, then FFmpeg concat.
            durations = _add_crossfade_padding(durations, joints=len(scenes) - 1, max_dur=15)
            total_dur = sum(durations)
            await notify(
                f"⏱ Generating <b>{label}</b> ({len(scenes)} clip(s)"
                + f", {total_dur}s total"
                + (", native audio" if keep_native else "")
                + ") — each clip takes ~3-5 min…"
            )
            n = len(uploaded_urls)
            anchor_urls = [uploaded_urls[i % n] for i in range(len(scenes))]
            clips = await seedance.generate_clips(
                scene_prompts=scenes,
                anchor_photo_urls=anchor_urls,
                clip_duration=durations,
                resolution=resolution,
                model_id=atlas_model_id,
                keep_native_audio=keep_native,
            )
            concat_transitions = transitions[:-1] if transitions else None
            raw_video = (
                await gemini.concat_videos(clips, crossfade=0.5, transitions=concat_transitions)
                if len(clips) > 1 else clips[0]
            )
    else:
        # No reference images — fallback to T2V or clip-by-clip I2V with last-frame stitching
        durations = _add_crossfade_padding(durations, joints=len(scenes) - 1, max_dur=15)
        await notify(
            f"⏱ Generating <b>{label}</b> ({len(scenes)} clip(s)) — each clip takes ~3-5 min…"
        )
        clips = []
        anchor_image_url = ""
        for i, (scene, dur) in enumerate(zip(scenes, durations)):
            await notify(f"🎞 {label} clip {i + 1}/{len(scenes)}…")
            effective = (
                "Seamlessly continuing from previous scene — "
                "same characters, same lighting, smooth flow. " + scene
                if i > 0 else scene
            )
            clip = await seedance.generate_clip(
                prompt=effective,
                image_url=anchor_image_url,  # Empty string first iteration (triggers T2V), then last-frame
                duration=dur,
                resolution=resolution,
                model_id=atlas_model_id,
                keep_native_audio=keep_native,
            )
            clips.append(clip)
            # Extract last frame for next clip's anchor (continuity stitching)
            if i < len(scenes) - 1:
                frame_path = await gemini.extract_last_frame(clip)
                if frame_path:
                    try:
                        new_url = await seedance.upload_photo(frame_path)
                        if new_url:
                            anchor_image_url = new_url
                    finally:
                        try:
                            os.remove(frame_path)
                        except OSError:
                            pass
        concat_transitions = transitions[:-1] if transitions else None
        raw_video = await gemini.concat_videos(clips, crossfade=0.5, transitions=concat_transitions) if len(clips) > 1 else clips[0]

    await notify(f"✅ {label} clips ready")

    tts_audio_path, word_timings = await _synthesize_tts(voiceover_text, settings, notify)

    return {
        "raw_video_path": raw_video,
        "tts_audio_path": tts_audio_path,
        "word_timings": word_timings,
        "voiceover_text": voiceover_text,
        "has_native_audio": keep_native,
    }


async def _generate_kling(
    *,
    gemini: GeminiService,
    settings: dict,
    video_prompt: str,
    voiceover_text: str,
    image_paths: list[str] | None,
    notify,
    shots: list[dict] | None = None,
) -> dict:
    from services.kling import MODEL_IDS as _KLING_IDS, MODEL_LABELS as _KLING_LABELS
    model = (settings.get("video_model") or "kling").lower()
    atlas_model_id = _KLING_IDS.get(model, _KLING_IDS["kling"])
    label = _KLING_LABELS.get(model, "Kling")

    kling = container.inject(KlingService)
    valid_paths = [p for p in (image_paths or []) if os.path.exists(p)]

    from services.kling import MULTIFRAME_SETTINGS_KEYS as _KLING_MF_KEYS
    target_duration = settings.get("target_duration", DEFAULT_TARGET_DURATION)
    clip_dur = _clip_duration_for_model(model)
    negative_prompt = settings.get("negative_prompt") or ""

    if shots:
        scenes = [_build_shot_prompt(s) for s in shots]
        shot_durations_list = [int(max(3, min(10, s.get("duration_seconds", clip_dur)))) for s in shots]
        shot_transitions = [s.get("transition", "cut") for s in shots]
    else:
        shot_durations_list = _plan_clip_durations(target_duration, min_dur=3, max_dur=10)
        scenes = _split_scene_into_shots(video_prompt, len(shot_durations_list))
        shot_transitions = None

    is_t2v = model in _T2V_VIDEO_MODELS

    # Model-generated audio: always for the ASMR niche (sound IS the content),
    # otherwise only when there is no voiceover to lay on top of silence.
    use_native_audio = (
        settings.get("content_preset") == "asmr"
        or not bool(voiceover_text and voiceover_text.strip())
    )
    # Only the multi-shot path can actually generate sound; the regular
    # clip-by-clip paths below always render silent (sound=False).
    native_audio_generated = False

    # ── Multi-frame path (kling, kling_v3_std, kling_t2v) ────────────────────
    if model in _KLING_MF_KEYS:
        native_audio_generated = use_native_audio
        # Atlas caps one multi_shot call at 15s total (and 6 shots max) — group
        # shots by cumulative duration, not by a fixed shot count, so a batch
        # of long shots can't silently exceed the per-call limit.
        _BATCH_MAX_DURATION = 15
        _BATCH_MAX_SHOTS = 6
        batch_indices = _batch_shot_indices(shot_durations_list, _BATCH_MAX_DURATION, _BATCH_MAX_SHOTS)
        if len(batch_indices) > 1:
            # Multiple batches will be concat'ed with 0.5s crossfades — pad the
            # shot durations to compensate the overlap loss, then re-batch.
            shot_durations_list = _add_crossfade_padding(
                shot_durations_list, joints=len(batch_indices) - 1, max_dur=10
            )
            batch_indices = _batch_shot_indices(shot_durations_list, _BATCH_MAX_DURATION, _BATCH_MAX_SHOTS)
        batches = [[scenes[i] for i in idxs] for idxs in batch_indices]
        batch_dur_lists = [[shot_durations_list[i] for i in idxs] for idxs in batch_indices]
        n_batches = len(batches)

        batch_transitions: list[str] | None = None
        if shot_transitions and n_batches > 1:
            batch_transitions = []
            for idxs in batch_indices[:-1]:
                last_idx = idxs[-1]
                t = shot_transitions[last_idx] if last_idx < len(shot_transitions) else "cut"
                batch_transitions.append(t)

        ref_url = ""
        if not is_t2v and valid_paths:
            ref_url = await kling.upload_photo(valid_paths[0])

        await notify(
            f"⏱ Generating <b>{label}</b> multi-frame"
            f" ({len(scenes)} shots → {n_batches} batch(es)"
            + (f", ref photo" if ref_url else "")
            + (f", native audio" if use_native_audio else "")
            + ") — ~2-4 min per batch…"
        )

        clips = []
        current_ref_url = ref_url

        for i, (batch, b_durs) in enumerate(zip(batches, batch_dur_lists)):
            await notify(f"🎞 {label} batch {i + 1}/{n_batches} ({len(batch)} shots)…")
            clip = await kling.generate_multiframe_clip(
                scene_prompts=batch,
                shot_duration=clip_dur,
                shot_durations=b_durs,
                image_reference_url=current_ref_url,
                motion_has_audio=use_native_audio,
                negative_prompt=negative_prompt,
                model_id=atlas_model_id,
            )
            clips.append(clip)

            if i < n_batches - 1:
                frame_path = await gemini.extract_last_frame(clip)
                if frame_path:
                    try:
                        new_url = await kling.upload_photo(frame_path)
                        if new_url:
                            current_ref_url = new_url
                    finally:
                        try:
                            os.remove(frame_path)
                        except OSError:
                            pass

        raw_video = (
            await gemini.concat_videos(clips, crossfade=0.5, transitions=batch_transitions)
            if len(clips) > 1 else clips[0]
        )

    # ── Regular Kling I2V path (Turbo, O3) ───────────────────────────────────
    elif not is_t2v and valid_paths:
        uploaded_urls = list(await asyncio.gather(*[kling.upload_photo(p) for p in valid_paths]))
        shot_durations_list = _add_crossfade_padding(
            shot_durations_list, joints=len(scenes) - 1, max_dur=10
        )
        await notify(
            f"⏱ Generating <b>{label}</b> ({len(scenes)} clip(s)"
            + (f", {len(valid_paths)} ref photo(s)" if valid_paths else "")
            + ") — each clip takes ~2-4 min…"
        )

        if shots:
            clips = []
            current_img_url = uploaded_urls[0]
            for i, (scene, dur) in enumerate(zip(scenes, shot_durations_list)):
                clip = await kling.generate_clip(
                    prompt=scene,
                    image_url=current_img_url,
                    duration=dur,
                    negative_prompt=negative_prompt,
                    model_id=atlas_model_id,
                )
                if i < len(scenes) - 1:
                    frame_path = await gemini.extract_last_frame(clip)
                    if frame_path:
                        try:
                            new_url = await kling.upload_photo(frame_path)
                            if new_url:
                                current_img_url = new_url
                        finally:
                            try:
                                os.remove(frame_path)
                            except OSError:
                                pass
                clips.append(clip)
        else:
            n = len(uploaded_urls)
            anchor_urls = [uploaded_urls[i % n] for i in range(len(scenes))]
            clips = await kling.generate_clips(
                scene_prompts=scenes,
                anchor_photo_urls=anchor_urls,
                clip_duration=shot_durations_list,
                negative_prompt=negative_prompt,
                model_id=atlas_model_id,
            )

        concat_transitions = shot_transitions[:-1] if shot_transitions else None
        raw_video = (
            await gemini.concat_videos(clips, crossfade=0.5, transitions=concat_transitions)
            if len(clips) > 1 else clips[0]
        )

    # ── T2V path (no reference images) ──────────────────────────────────────────
    else:
        shot_durations_list = _add_crossfade_padding(
            shot_durations_list, joints=len(scenes) - 1, max_dur=10
        )
        await notify(
            f"⏱ Generating <b>{label}</b> ({len(scenes)} clip(s)) — each clip takes ~2-4 min…"
        )
        clips = []
        anchor_image_url = ""
        for i, (scene, dur) in enumerate(zip(scenes, shot_durations_list)):
            await notify(f"🎞 {label} clip {i + 1}/{len(scenes)}…")
            effective = (
                "Seamlessly continuing from previous scene — "
                "same characters, same lighting, smooth flow. " + scene
                if i > 0 else scene
            )
            clip = await kling.generate_clip(
                prompt=effective,
                image_url=anchor_image_url,  # Empty first iteration (T2V), then last-frame
                duration=dur,
                negative_prompt=negative_prompt,
                model_id=atlas_model_id,
            )
            clips.append(clip)
            # Extract last frame for next clip's anchor (continuity stitching)
            if i < len(scenes) - 1:
                frame_path = await gemini.extract_last_frame(clip)
                if frame_path:
                    try:
                        new_url = await kling.upload_photo(frame_path)
                        if new_url:
                            anchor_image_url = new_url
                    finally:
                        try:
                            os.remove(frame_path)
                        except OSError:
                            pass

        concat_transitions = shot_transitions[:-1] if shot_transitions else None
        raw_video = (
            await gemini.concat_videos(clips, crossfade=0.5, transitions=concat_transitions)
            if len(clips) > 1 else clips[0]
        )

    await notify(f"✅ {label} clips ready")

    tts_audio_path, word_timings = await _synthesize_tts(voiceover_text, settings, notify)

    return {
        "raw_video_path": raw_video,
        "tts_audio_path": tts_audio_path,
        "word_timings": word_timings,
        "voiceover_text": voiceover_text,
        "has_native_audio": native_audio_generated,
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

def _clip_duration_for_model(video_model: str) -> int:
    """Return the per-scene duration (seconds) for the given video model."""
    if video_model.startswith("kling"):
        return 15  # All Kling models support up to 15 seconds
    if video_model.startswith("seedance"):
        return 15
    return 15      # Default for all other models


def _plan_clip_durations(target_duration: int, min_dur: int, max_dur: int) -> list[int]:
    """Split target_duration into the fewest clips of length min_dur..max_dur
    that sum to target_duration exactly, instead of always rounding up to a
    multiple of max_dur (e.g. 20s target with max_dur=15 → [10, 10], not [15, 15])."""
    n_clips = max(1, math.ceil(target_duration / max_dur))
    base = target_duration // n_clips
    remainder = target_duration - base * n_clips
    durations = [base + (1 if i < remainder else 0) for i in range(n_clips)]
    return [max(min_dur, min(max_dur, d)) for d in durations]


def _normalize_shot_durations(
    shots: list[dict], target_duration: int, min_dur: int, max_dur: int
) -> list[dict]:
    """Rescale LLM-produced shot durations so they sum to target_duration.

    LLMs are unreliable at arithmetic — a 30s request may come back as 27s or
    34s of shots. Proportionally rescale, clamp to the model's per-shot range,
    then nudge ±1s until the sum is exact (or as close as the clamps allow)."""
    durs = [int(s.get("duration_seconds") or min_dur) for s in shots]
    total = sum(durs)
    if total <= 0:
        durs = [min_dur] * len(shots)
        total = sum(durs)
    # Clamp FIRST — even when the sum happens to match the target, individual
    # values may violate the model's per-shot range (e.g. 15+15=30 on Kling
    # whose max is 10) and would be rejected by Atlas.
    scaled = [
        max(min_dur, min(max_dur, round(d * target_duration / total)))
        for d in durs
    ]
    diff = target_duration - sum(scaled)
    i = 0
    guard = 4 * len(scaled) + 8
    while diff != 0 and guard > 0:
        idx = i % len(scaled)
        step = 1 if diff > 0 else -1
        candidate = scaled[idx] + step
        if min_dur <= candidate <= max_dur:
            scaled[idx] = candidate
            diff -= step
        i += 1
        guard -= 1
    if sum(scaled) != target_duration:
        logger.warning(
            f"Shot durations clamped to {sum(scaled)}s (target {target_duration}s, "
            f"per-shot range {min_dur}-{max_dur}s, {len(scaled)} shots)"
        )
    for s, d in zip(shots, scaled):
        s["duration_seconds"] = d
    return shots


def _add_crossfade_padding(
    durations: list[int], joints: int, max_dur: int, fade: float = 0.5
) -> list[int]:
    """Compensate concat crossfades: each joint overlaps `fade` seconds, so the
    final file comes out ~fade*joints shorter than the sum of clip lengths.
    Add whole seconds to clips (respecting max_dur) to cover the loss."""
    extra = round(fade * joints)
    i = 0
    attempts = 2 * len(durations)
    while extra > 0 and attempts > 0:
        idx = i % len(durations)
        if durations[idx] < max_dur:
            durations[idx] += 1
            extra -= 1
        i += 1
        attempts -= 1
    return durations


def _batch_shot_indices(durations: list[int], max_batch_duration: int, max_shots_per_batch: int) -> list[list[int]]:
    """Group shot indices into batches whose durations sum to at most
    max_batch_duration (Atlas Cloud's per-call limit for Kling's multi_shot
    API), instead of a fixed shot count — a batch may hold few long shots or
    several short ones, as long as the total fits in one Atlas call."""
    batches: list[list[int]] = []
    current: list[int] = []
    current_dur = 0
    for i, dur in enumerate(durations):
        if current and (current_dur + dur > max_batch_duration or len(current) >= max_shots_per_batch):
            batches.append(current)
            current = []
            current_dur = 0
        current.append(i)
        current_dur += dur
    if current:
        batches.append(current)
    return batches


# Hard, non-editable preamble for the own-script mode: the user's text is a
# final approved script — structure it, never reinvent it. Niche presets and
# the user's system_video_prompt do NOT apply in this mode.
_STRICT_SCRIPT_SYS = (
    "OWN-SCRIPT MODE — the user's message is a FINAL, approved video script.\n"
    "- Do NOT invent new characters, narrators, framing devices, locations or "
    "events that are not in the script.\n"
    "- Preserve the script's content, subjects and order of events exactly; "
    "your only job is to split it into shots.\n"
    "- scene_prompt must be in English: translate the script's visual "
    "descriptions faithfully if they are in another language.\n"
    "- Compress wording only as much as needed to fit shot limits — never add "
    "new creative material while doing so.\n"
    "- spoken_text: derive it strictly from the script itself, in the script's "
    "original language, staying faithful to its wording."
)


async def _build_video_prompt(
    enhance_prompt: str, settings: dict, gemini: GeminiService, strict_script: bool = False
) -> str:
    """Generate video script as structured JSON. Falls back to text format on parse failure.

    strict_script=True (own-script mode): the user's text is treated as a final
    script — the niche preset's video prompt is ignored and a hard preamble
    forbids inventing new content; the model only structures the text into shots.
    """
    base_sys = (
        _STRICT_SCRIPT_SYS if strict_script
        else _substitute_prompt_vars(settings.get("system_video_prompt") or "", settings)
    )
    target_duration = settings.get("target_duration", DEFAULT_TARGET_DURATION)
    video_model = (settings.get("video_model") or "seedance").lower()
    clip_duration = _clip_duration_for_model(video_model)

    # Per-model duration constraints
    if video_model.startswith("seedance"):
        min_dur, max_dur = 4, 15
    elif "kling" in video_model:
        min_dur, max_dur = 3, 10
    else:
        min_dur, max_dur = 4, clip_duration

    # Plan exact per-shot durations that sum to target_duration (fewest clips
    # of length min_dur..max_dur), instead of always rounding up to a multiple
    # of the model's max clip length (e.g. 20s target → [10, 10], not [15, 15]).
    planned_durations = _plan_clip_durations(target_duration, min_dur, max_dur)
    n_clips = len(planned_durations)
    actual_duration = sum(planned_durations)
    target_words = max(20, int(actual_duration * 2.5))

    shot_mode_str = "multi"
    shots_count_rule = f"shots array has EXACTLY {n_clips} elements"
    transition_rule  = "'cut' for action/pace/energy, 'dissolve' for mood shift/scene change, 'fade' for the FINAL shot ONLY"
    scene_prompt_rule = "scene_prompt: English ONLY, describe VISUALS only — no camera directions here, no spoken text. Max 55 words per shot"
    camera_motion_rule = "camera_motion: 3-8 words, camera movement for THIS shot only (e.g. 'slow push-in', 'pan left', 'static wide', 'crane up', 'orbit right')"

    _JSON_SYS = (
        "Output ONLY valid JSON. No prose. No markdown fences. Schema:\n"
        "{\n"
        f'  "spoken_text": "full voiceover ~{target_words} words (~{actual_duration}s at 2.5 words/sec)",\n'
        f'  "caption": "social-media caption with 3-5 hashtags (max 150 chars)",\n'
        f'  "shot_mode": "{shot_mode_str}",\n'
        '  "shots": [\n'
        '    {\n'
        '      "index": 0,\n'
        f'      "duration_seconds": {planned_durations[0]},\n'
        '      "scene_prompt": "subject action + setting + lighting (English ONLY, visuals ONLY)",\n'
        '      "camera_motion": "camera movement description",\n'
        '      "transition": "cut | dissolve | fade"\n'
        '    }\n'
        '  ]\n'
        '}\n'
        "Rules:\n"
        f"- {shots_count_rule}\n"
        f"- duration_seconds per shot IN ORDER MUST be exactly: {planned_durations} "
        f"(these already sum to {actual_duration}s — the user's requested video length. Do not change them)\n"
        f"- {scene_prompt_rule}\n"
        f"- {camera_motion_rule}\n"
        "- spoken_text: MUST be in the same language as the user input\n"
        "- caption: same language as spoken_text\n"
        f"- transition: {transition_rule}\n"
        "- scene_prompt must depict EXACTLY the topic from the input\n"
        "- G-rated, child-safe content only\n"
        f"- Total: {n_clips} shot(s), EXACTLY {actual_duration}s\n"
    )

    sys_prompt = (base_sys + "\n\n" + _JSON_SYS) if base_sys else _JSON_SYS
    raw = await gemini.generate_text(
        enhance_prompt,
        sys_prompt,
        settings.get("text_model") or "",
    )

    if GeminiService.parse_script_json(raw) is not None:
        return raw  # valid JSON ✓

    logger.warning("Gemini returned invalid JSON for video script — falling back to text format")
    return await _build_video_prompt_text(enhance_prompt, settings, gemini, strict_script=strict_script)


async def _build_video_prompt_text(
    enhance_prompt: str, settings: dict, gemini: GeminiService, strict_script: bool = False
) -> str:
    """Text-format fallback (original implementation)."""
    base_sys = (
        _STRICT_SCRIPT_SYS if strict_script
        else _substitute_prompt_vars(settings.get("system_video_prompt") or "", settings)
    )
    target_duration = settings.get("target_duration", DEFAULT_TARGET_DURATION)
    video_model = (settings.get("video_model") or "seedance").lower()
    clip_duration = _clip_duration_for_model(video_model)

    if video_model.startswith("seedance"):
        min_dur, max_dur = 4, 15
    elif "kling" in video_model:
        min_dur, max_dur = 3, 10
    else:
        min_dur, max_dur = 4, clip_duration

    planned_durations = _plan_clip_durations(target_duration, min_dur, max_dur)
    n_clips = len(planned_durations)
    actual_duration = sum(planned_durations)
    target_words = max(20, int(actual_duration * 2.3))
    scene_word_limit = n_clips * 50

    scene_instruction = (
        f"Scene: Describe EXACTLY {n_clips} camera shot(s) separated by 'Cut to:'. "
        f"Each shot covers ~{round(actual_duration / n_clips)} seconds of the story. "
        f"Format each shot as: shot size + character action + camera movement.\n"
        f"Shot sizes: wide shot, medium shot, close-up, extreme close-up, overhead shot, low angle shot.\n"
        f"Camera moves: slow push in, dolly back, rack focus, pan left/right, crane up, handheld, static.\n"
        f"Max {scene_word_limit} words for the Scene section."
    )
    shots_rule = f"- Write EXACTLY {n_clips} shots in the Scene — no more, no less"

    _FORMAT_SUFFIX = (
        f"\n\n--- OUTPUT FORMAT (mandatory) ---\n"
        f"Write THREE sections only:\n\n"
        f"{scene_instruction}\n\n"
        f'Voiceover: "narration text of approximately {target_words} words '
        f'({actual_duration} seconds at natural talking pace)"\n\n'
        f'Caption: "engaging social-media caption with 3-5 relevant hashtags (max 150 chars)"\n\n'
        f"Rules:\n"
        f"{shots_rule}\n"
        f"- Scene description MUST be written in English regardless of the input language\n"
        f"- Voiceover MUST be written in the same language as the input\n"
        f"- Caption MUST be in the same language as the Voiceover\n"
        f"- Each shot must sync with what is spoken in the Voiceover at that moment\n"
        f"- Scene must depict EXACTLY the topic from the input\n"
        f"- Do NOT use markdown (>, **, *)\n"
        f"- The words Voiceover: and Caption: each appear ONLY ONCE\n"
        f"- Use straight double quotes around the narration and caption\n"
        f"- G-rated, child-safe content only"
    )

    sys_with_hint = (base_sys + _FORMAT_SUFFIX) if base_sys else _FORMAT_SUFFIX
    return await gemini.generate_text(
        enhance_prompt,
        sys_with_hint,
        settings.get("text_model") or "",
    )


_STRICT_IMAGE_SYS = (
    "OWN-SCRIPT MODE — the text you receive is a final, approved video script. "
    "Create ONE image-generation prompt that faithfully depicts the OPENING "
    "moment of this script: same subjects, same setting, same mood. Do not add "
    "characters, objects or framing devices that are not in the script."
)


async def _build_image_prompt(
    enhance_prompt: str,
    settings: dict,
    gemini: GeminiService,
    strict_script: bool = False,
) -> str:
    from utils.presets import DEFAULT_IMAGE_PROMPT, IMAGE_SYS_SUFFIX
    # User/preset text sets the style; the hard output requirements (single
    # frame, English, 9:16, no text/watermarks) are always enforced in code —
    # same pattern as _JSON_SYS for the video script.
    if strict_script:
        base_sys = _STRICT_IMAGE_SYS
    else:
        base_sys = settings.get("system_image_prompt") or DEFAULT_IMAGE_PROMPT
    return await gemini.generate_text(
        enhance_prompt,
        base_sys + IMAGE_SYS_SUFFIX,
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


def _build_shot_prompt(shot: dict) -> str:
    """Combine scene_prompt + camera_motion into one prompt string for video models."""
    scene = shot.get("scene_prompt", "")
    cam = shot.get("camera_motion", "")
    return f"{scene}. Camera: {cam}." if cam else scene


def _split_video_prompt(video_prompt: str, gemini) -> tuple[str, str]:
    # JSON format from Gemini structured output
    parsed = GeminiService.parse_script_json(video_prompt)
    if parsed is not None:
        voiceover = parsed.get("spoken_text", "")
        shots = parsed.get("shots", [])
        if shots:
            parts = []
            for i, shot in enumerate(shots):
                dur = shot.get("duration_seconds", "?")
                trans = shot.get("transition", "cut")
                prompt = shot.get("scene_prompt", "")
                cam = shot.get("camera_motion", "")
                cam_str = f" | {cam}" if cam else ""
                if len(shots) == 1:
                    parts.append(f"Shot 1 ({dur}s{cam_str}): {prompt}")
                else:
                    parts.append(f"[Shot {i + 1}, {dur}s, {trans}{cam_str}] {prompt}")
            scene = "\n".join(parts)
        else:
            scene = ""
        return scene, voiceover

    # Text format fallback
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
