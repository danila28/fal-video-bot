"""Kling video generation via Atlas Cloud.

Supported models (set via settings video_model):
  kling             → Kling v3.0 Pro  Image-to-Video  (current default)
  kling_v3_std      → Kling v3.0 Std  Image-to-Video
  kling_o3_pro      → Kling O3 Pro    Image-to-Video
  kling_o3_std      → Kling O3 Std    Image-to-Video
  kling_o3_pro_ref  → Kling O3 Pro    Reference-to-Video
  kling_o3_std_ref  → Kling O3 Std    Reference-to-Video

Reference-to-Video models use an `images` array (not `image`) and do NOT
support negative_prompt. Last-frame continuity is skipped for them because
the reference defines character identity, not the starting frame.
"""

import asyncio
import logging
import os
import uuid

from services.atlas import AtlasClient

logger = logging.getLogger(__name__)

# ── Atlas Cloud model IDs ─────────────────────────────────────────────────────

_V3_PRO_I2V   = "kwaivgi/kling-v3.0-pro/image-to-video"
_V3_PRO_T2V   = "kwaivgi/kling-v3.0-pro/text-to-video"
_V3_STD_I2V   = "kwaivgi/kling-v3.0-std/image-to-video"
_O3_PRO_I2V   = "kwaivgi/kling-video-o3-pro/image-to-video"
_O3_STD_I2V   = "kwaivgi/kling-video-o3-std/image-to-video"
_O3_PRO_REF   = "kwaivgi/kling-video-o3-pro/reference-to-video"
_O3_STD_REF   = "kwaivgi/kling-video-o3-std/reference-to-video"

# Backwards-compat aliases used by existing code
_MODEL_IMAGE_TO_VIDEO = _V3_PRO_I2V
_MODEL_TEXT_TO_VIDEO  = _V3_PRO_T2V

# Map from settings model name → Atlas model ID
MODEL_IDS: dict[str, str] = {
    "kling":            _V3_PRO_I2V,
    "kling_t2v":        _V3_PRO_T2V,
    "kling_v3_std":     _V3_STD_I2V,
    "kling_o3_pro":     _O3_PRO_I2V,
    "kling_o3_std":     _O3_STD_I2V,
    "kling_o3_pro_ref": _O3_PRO_REF,
    "kling_o3_std_ref": _O3_STD_REF,
}

# Human-readable labels used in notify messages
MODEL_LABELS: dict[str, str] = {
    "kling":            "Kling v3 Pro",
    "kling_t2v":        "Kling v3 Pro T2V",
    "kling_v3_std":     "Kling v3 Std",
    "kling_o3_pro":     "Kling O3 Pro",
    "kling_o3_std":     "Kling O3 Std",
    "kling_o3_pro_ref": "Kling O3 Pro Reference",
    "kling_o3_std_ref": "Kling O3 Std Reference",
}

_REFERENCE_MODELS = {_O3_PRO_REF, _O3_STD_REF}


class KlingService:
    def __init__(self, api_key: str, static_dir: str = ""):
        self._atlas = AtlasClient(api_key, static_dir)
        self.static_dir = self._atlas.static_dir

    async def upload_photo(self, photo_path: str) -> str:
        """Upload a local photo to Atlas Cloud storage. Returns public URL."""
        url = await self._atlas.upload_file(photo_path)
        logger.info(f"Kling: uploaded {photo_path} → {url}")
        return url

    async def generate_clip(
        self,
        prompt: str,
        image_url: str = "",
        duration: int = 10,
        aspect_ratio: str = "9:16",
        negative_prompt: str = "",
        model_id: str = _V3_PRO_I2V,
    ) -> str:
        """Generate one clip. Returns local path to downloaded MP4.

        Reference models use `images` (array) instead of `image` and ignore
        negative_prompt (not supported by the Reference API).
        """
        os.makedirs(self.static_dir, exist_ok=True)
        is_reference = model_id in _REFERENCE_MODELS

        if image_url:
            model = model_id
            if is_reference:
                params: dict = {
                    "prompt": prompt,
                    "images": [image_url],
                    "duration": duration,
                    "aspect_ratio": aspect_ratio,
                    "sound": False,
                }
            else:
                params = {
                    "prompt": prompt,
                    "image": image_url,
                    "duration": duration,
                }
                if negative_prompt:
                    params["negative_prompt"] = negative_prompt
        else:
            # Text-to-video fallback always uses v3.0 Pro
            model = _V3_PRO_T2V
            params = {
                "prompt": prompt,
                "duration": duration,
                "aspect_ratio": aspect_ratio,
            }

        logger.info(f"Kling generating clip | model={model} | {duration}s")
        video_url = await self._atlas.generate_video(model, params)
        return await self._atlas.download(video_url, ext="mp4")

    async def generate_clips(
        self,
        scene_prompts: list[str],
        anchor_photo_urls: list[str],
        clip_duration: int = 10,
        negative_prompt: str = "",
        model_id: str = _V3_PRO_I2V,
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
            logger.info(f"Kling clip {i + 1}/{len(scene_prompts)} | model={model_id}")

            # For Reference models: add continuation hint to clips 2+ so the model
            # matches the lighting, background and mood of the previous clip.
            if is_reference and i > 0:
                effective_prompt = (
                    "Seamlessly continuing from previous scene — "
                    "same lighting, same background, same camera angle, smooth action flow. "
                    + prompt
                )
            else:
                effective_prompt = prompt

            clip_path = await self.generate_clip(
                prompt=effective_prompt,
                image_url=photo_url,
                duration=clip_duration,
                negative_prompt=negative_prompt,
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
