"""Seedance video generation via Atlas Cloud.

Supported models (set via settings video_model):
  seedance          → Seedance 2.0       Image-to-Video  (current default)
  seedance_fast     → Seedance 2.0 Fast  Image-to-Video
  seedance_ref      → Seedance 2.0       Reference-to-Video
  seedance_fast_ref → Seedance 2.0 Fast  Reference-to-Video

Reference-to-Video models use `image_urls` (array) instead of `image_url`,
and the prompt must reference the image with `@Image1`. Last-frame continuity
is skipped for them (reference defines character identity, not starting frame).

Clip duration: 10 s (standard unit for all Seedance variants).
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
_FAST_I2V  = "bytedance/seedance-2.0/fast/image-to-video"
_REF       = "bytedance/seedance-2.0/reference-to-video"
_FAST_REF  = "bytedance/seedance-2.0/fast/reference-to-video"

# Backwards-compat aliases
_MODEL_IMAGE_TO_VIDEO = _I2V
_MODEL_TEXT_TO_VIDEO  = _T2V

# Map from settings model name → Atlas model ID
MODEL_IDS: dict[str, str] = {
    "seedance":          _I2V,
    "seedance_fast":     _FAST_I2V,
    "seedance_ref":      _REF,
    "seedance_fast_ref": _FAST_REF,
}

# Human-readable labels used in notify messages
MODEL_LABELS: dict[str, str] = {
    "seedance":          "Seedance 2.0",
    "seedance_fast":     "Seedance 2.0 Fast",
    "seedance_ref":      "Seedance 2.0 Reference",
    "seedance_fast_ref": "Seedance 2.0 Fast Reference",
}

_REFERENCE_MODELS = {_REF, _FAST_REF}


class SeedanceService:
    def __init__(self, api_key: str, static_dir: str = ""):
        self._atlas = AtlasClient(api_key, static_dir)
        self.static_dir = self._atlas.static_dir

    async def upload_photo(self, photo_path: str) -> str:
        """Upload a local photo to Atlas Cloud storage. Returns public URL."""
        url = await self._atlas.upload_file(photo_path)
        logger.info(f"Seedance: uploaded {photo_path} → {url}")
        return url

    async def generate_clip(
        self,
        prompt: str,
        image_url: str = "",
        duration: int = 10,
        aspect_ratio: str = "9:16",
        resolution: str = "720p",
        model_id: str = _I2V,
    ) -> str:
        """Generate one clip. Returns local path to downloaded MP4.

        Reference models use `image_urls` (array) + `@Image1` in prompt.
        Image-to-Video models use `image_url` (single string) + `ratio` param.
        """
        os.makedirs(self.static_dir, exist_ok=True)
        is_reference = model_id in _REFERENCE_MODELS

        if image_url:
            model = model_id
            if is_reference:
                ref_prompt = f"@Image1 {prompt}" if not prompt.startswith("@Image1") else prompt
                params = {
                    "prompt": ref_prompt,
                    "image_urls": [image_url],
                    "duration": duration,
                    "aspect_ratio": aspect_ratio,
                    "resolution": resolution,
                    "generate_audio": False,
                    "watermark": False,
                }
            else:
                params = {
                    "prompt": prompt,
                    "image_url": image_url,
                    "duration": duration,
                    "ratio": aspect_ratio,
                    "resolution": resolution,
                    "generate_audio": False,
                    "watermark": False,
                }
        else:
            model = _T2V
            params = {
                "prompt": prompt,
                "duration": duration,
                "ratio": aspect_ratio,
                "resolution": resolution,
                "generate_audio": False,
                "watermark": False,
            }

        logger.info(f"Seedance generating clip | model={model} | {duration}s")
        video_url = await self._atlas.generate_video(model, params)
        return await self._atlas.download(video_url, ext="mp4")

    async def generate_clips(
        self,
        scene_prompts: list[str],
        anchor_photo_urls: list[str],
        clip_duration: int = 10,
        model_id: str = _I2V,
    ) -> list[str]:
        """Generate multiple clips.

        For Image-to-Video models: extracts last frame after each clip and uses
        it as the anchor for the next clip (visual continuity).
        For Reference-to-Video models: always uses the original anchor photo
        because the reference defines character identity, not the starting frame.
        """
        is_reference = model_id in _REFERENCE_MODELS
        clips: list[str] = []

        for i, (prompt, photo_url) in enumerate(zip(scene_prompts, anchor_photo_urls)):
            logger.info(f"Generating Seedance clip {i + 1}/{len(scene_prompts)}")
            clip_path = await self.generate_clip(
                prompt=prompt,
                image_url=photo_url,
                duration=clip_duration,
                model_id=model_id,
            )
            clips.append(clip_path)

            # Last-frame continuity only for Image-to-Video models
            if not is_reference and i < len(scene_prompts) - 1:
                same_photo = anchor_photo_urls[i] == anchor_photo_urls[i + 1]
                if same_photo:
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
