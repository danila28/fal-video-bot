"""Seedance video generation via Atlas Cloud.

Supported models (set via settings video_model):
  seedance          → Seedance 2.0       Image-to-Video  (current default)
  seedance_fast     → Seedance 2.0 Fast  Image-to-Video
  seedance_mini     → Seedance 2.0 Mini  Image-to-Video
  seedance_t2v      → Seedance 2.0       Text-to-Video
  seedance_mini_t2v → Seedance 2.0 Mini  Text-to-Video

Clip duration: up to 15 s per Atlas call for all Seedance variants.
"""

import asyncio
import logging
import os
import uuid

from services.atlas import AtlasClient

logger = logging.getLogger(__name__)

# ── Atlas Cloud model IDs ─────────────────────────────────────────────────────

_I2V       = "bytedance/seedance-2.0/image-to-video"
_T2V       = "bytedance/seedance-2.0/text-to-video"
_FAST_I2V  = "bytedance/seedance-2.0-fast/image-to-video"
_MINI_I2V  = "bytedance/seedance-2.0-mini/image-to-video"
_MINI_T2V  = "bytedance/seedance-2.0-mini/text-to-video"

# Backwards-compat aliases
_MODEL_IMAGE_TO_VIDEO = _I2V
_MODEL_TEXT_TO_VIDEO  = _T2V

# Atlas hard limit: one Seedance request renders at most 15 seconds.
_MAX_CALL_DURATION = 15


def _normalize_resolution(model_id: str, resolution: str) -> str:
    """Mini models don't accept plain '1080p' — only the '-SR' (super-resolution)
    variant. Other tiers accept 720p/1080p as-is."""
    if "mini" in model_id and resolution == "1080p":
        return "1080p-SR"
    return resolution

# Map from settings model name → Atlas model ID
MODEL_IDS: dict[str, str] = {
    "seedance":           _I2V,
    "seedance_t2v":       _T2V,
    "seedance_fast":      _FAST_I2V,
    "seedance_mini":      _MINI_I2V,
    "seedance_mini_t2v":  _MINI_T2V,
}

# Human-readable labels used in notify messages
MODEL_LABELS: dict[str, str] = {
    "seedance":           "Seedance 2.0",
    "seedance_t2v":       "Seedance 2.0 T2V",
    "seedance_fast":      "Seedance 2.0 Fast",
    "seedance_mini":      "Seedance 2.0 Mini",
    "seedance_mini_t2v":  "Seedance 2.0 Mini T2V",
}


class SeedanceService:
    def __init__(self, api_key: str, static_dir: str = ""):
        self._atlas = AtlasClient(api_key, static_dir)
        self.static_dir = self._atlas.static_dir

    async def upload_photo(self, photo_path: str) -> str:
        url = await self._atlas.upload_file(photo_path)
        logger.info(f"Seedance: uploaded photo {photo_path} → {url}")
        return url

    async def generate_clip(
        self,
        prompt: str,
        image_url: str = "",
        image_urls: list[str] | None = None,
        duration: int = 10,
        aspect_ratio: str = "9:16",
        resolution: str = "720p",
        model_id: str = _I2V,
        keep_native_audio: bool = False,
    ) -> str:
        """Generate one clip. Returns local path to downloaded MP4.

        With an image: I2V using `image_url`. Without: falls back to the T2V
        sibling of `model_id` (same tier — Mini stays Mini, Fast stays base T2V).
        keep_native_audio: let Seedance generate its own synchronized ambient
        audio (footsteps, texture sounds, etc). Off by default — that audio is
        normally replaced by our own TTS mix in post-processing anyway, so
        generating it would just waste render time/cost. Turn on for niches
        where the model's own sound IS the point (e.g. ASMR) — post-processing
        must then mix TTS on top instead of replacing (see has_native_audio).
        """
        os.makedirs(self.static_dir, exist_ok=True)
        duration = max(4, min(_MAX_CALL_DURATION, duration))
        resolution = _normalize_resolution(model_id, resolution)

        # image_urls takes priority over image_url
        effective_urls = image_urls if image_urls else ([image_url] if image_url else [])

        if effective_urls:
            model = model_id
            params = {
                "prompt": prompt,
                "image_url": effective_urls[0],
                "duration": duration,
                "ratio": aspect_ratio,
                "resolution": resolution,
                "generate_audio": keep_native_audio,
            }
        else:
            model = (
                model_id if "text-to-video" in model_id
                else model_id.replace("/image-to-video", "/text-to-video")
            )
            params = {
                "prompt": prompt,
                "duration": duration,
                "ratio": aspect_ratio,
                "resolution": resolution,
                "generate_audio": keep_native_audio,
                "watermark": False,
            }

        logger.info(
            f"Seedance generating clip | model={model} | {duration}s"
            f" | images={len(effective_urls)}"
        )
        video_url = await self._atlas.generate_video(model, params)
        return await self._atlas.download(video_url, ext="mp4")

    async def generate_multi_scene_clip(
        self,
        scene_prompts: list[str],
        image_url: str = "",
        clip_duration: int = 15,
        total_duration: int | None = None,
        model_id: str = "",
        aspect_ratio: str = "9:16",
        resolution: str = "720p",
        keep_native_audio: bool = False,
    ) -> str:
        """Single Atlas API call with [Scene1]...[SceneN] markers.

        All scenes are rendered by the model in one shot, preserving visual
        continuity without the need for last-frame stitching.
        total_duration overrides the clip_duration * len(scene_prompts) default —
        pass it to hit an exact target duration instead of a per-scene multiple.
        keep_native_audio: see generate_clip — lets Seedance generate its own
        ambient audio track instead of rendering silent.
        """
        os.makedirs(self.static_dir, exist_ok=True)

        combined_prompt = " ".join(
            f"[Scene{i + 1}] {p}" for i, p in enumerate(scene_prompts)
        )
        if total_duration is None:
            total_duration = clip_duration * len(scene_prompts)
        if total_duration > _MAX_CALL_DURATION:
            # One Atlas call renders at most 15s — the caller must split longer
            # targets into several clips (generate_clips) instead.
            logger.warning(
                f"Seedance multi-scene {total_duration}s exceeds {_MAX_CALL_DURATION}s "
                f"per-call limit — clamping (use generate_clips for longer videos)"
            )
            total_duration = _MAX_CALL_DURATION
        resolution = _normalize_resolution(model_id or "", resolution)

        atlas_model = model_id or (_I2V if image_url else _T2V)
        params: dict = {
            "prompt": combined_prompt,
            "duration": total_duration,
            "ratio": aspect_ratio,
            "resolution": resolution,
            "generate_audio": keep_native_audio,
            "watermark": False,
        }
        if image_url:
            params["image_url"] = image_url

        logger.info(
            f"Seedance multi-scene | model={atlas_model} | scenes={len(scene_prompts)}"
            f" | total_duration={total_duration}s"
        )
        video_url = await self._atlas.generate_video(atlas_model, params)
        return await self._atlas.download(video_url, ext="mp4")

    async def generate_clips(
        self,
        scene_prompts: list[str],
        anchor_photo_urls: list[str],
        clip_duration: int | list[int] = 10,
        resolution: str = "720p",
        model_id: str = _I2V,
        keep_native_audio: bool = False,
    ) -> list[str]:
        """Generate multiple clips.

        Extracts the last frame after each clip and uses it as the anchor for
        the next clip (visual continuity). `anchor_photo_urls` cycles through
        multiple images when provided.
        clip_duration: single value for every clip, or a per-clip list.
        keep_native_audio: see generate_clip.
        """
        clips: list[str] = []
        durations = (
            clip_duration if isinstance(clip_duration, list)
            else [clip_duration] * len(scene_prompts)
        )

        for i, (prompt, photo_url) in enumerate(zip(scene_prompts, anchor_photo_urls)):
            logger.info(f"Generating Seedance clip {i + 1}/{len(scene_prompts)}")
            dur = durations[i] if i < len(durations) else durations[-1]

            clip_path = await self.generate_clip(
                prompt=prompt,
                image_url=photo_url,
                duration=dur,
                resolution=resolution,
                model_id=model_id,
                keep_native_audio=keep_native_audio,
            )
            clips.append(clip_path)

            # Last-frame continuity (visual flow between consecutive clips)
            if i < len(scene_prompts) - 1:
                frame_path = await self._extract_last_frame(clip_path)
                if frame_path:
                    frame_url = await self.upload_photo(frame_path)
                    anchor_photo_urls[i + 1] = frame_url
                    try:
                        os.remove(frame_path)
                    except OSError:
                        pass

        return clips

    async def _extract_last_frame(self, video_path: str) -> str:
        try:
            import ffmpeg
            duration = await asyncio.to_thread(self._probe_duration, video_path)
            seek = max(0.0, duration - 0.1)
            out_path = os.path.join(self.static_dir, f"{uuid.uuid4()}_frame.png")
            await asyncio.to_thread(
                lambda: (
                    ffmpeg.input(video_path, ss=seek)
                    .output(out_path, vframes=1)
                    .overwrite_output()
                    .run(quiet=True)
                )
            )
            return out_path
        except Exception as e:
            logger.warning(f"extract_last_frame failed (non-fatal): {e}")
            return ""

    @staticmethod
    def _probe_duration(video_path: str) -> float:
        try:
            import ffmpeg
            info = ffmpeg.probe(video_path)
            for stream in info.get("streams", []):
                if stream.get("codec_type") == "video" and stream.get("duration"):
                    return float(stream["duration"])
            if info.get("format", {}).get("duration"):
                return float(info["format"]["duration"])
        except Exception as e:
            logger.warning(f"ffprobe failed: {e}")
        return 10.0
