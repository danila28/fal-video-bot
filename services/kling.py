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

_V3_PRO_I2V    = "kwaivgi/kling-v3.0-pro/image-to-video"
_V3_PRO_T2V    = "kwaivgi/kling-v3.0-pro/text-to-video"
_V3_STD_I2V    = "kwaivgi/kling-v3.0-std/image-to-video"
_V3_TURBO_I2V  = "kwaivgi/kling-v3.0-turbo/image-to-video"
_V3_TURBO_T2V  = "kwaivgi/kling-v3.0-turbo/text-to-video"
_O3_PRO_I2V    = "kwaivgi/kling-video-o3-pro/image-to-video"
_O3_STD_I2V    = "kwaivgi/kling-video-o3-std/image-to-video"
_O3_PRO_REF    = "kwaivgi/kling-video-o3-pro/reference-to-video"
_O3_STD_REF    = "kwaivgi/kling-video-o3-std/reference-to-video"
_O3_PRO_EDIT   = "kwaivgi/kling-video-o3-pro/video-edit"
_O3_STD_EDIT   = "kwaivgi/kling-video-o3-std/video-edit"
_V3_OMNI_I2V   = "kwaivgi/kling-video-o3-pro/image-to-video"
# Multi-frame guidances endpoint (up to 6 shots × 5s = 15s per call)
_V3_MULTI      = "kling-v3"

# Backwards-compat aliases used by existing code
_MODEL_IMAGE_TO_VIDEO = _V3_PRO_I2V
_MODEL_TEXT_TO_VIDEO  = _V3_PRO_T2V

# Map from settings model name → Atlas model ID
MODEL_IDS: dict[str, str] = {
    "kling":            _V3_PRO_I2V,
    "kling_t2v":        _V3_PRO_T2V,
    "kling_v3_std":     _V3_STD_I2V,
    "kling_turbo":      _V3_TURBO_I2V,
    "kling_turbo_t2v":  _V3_TURBO_T2V,
    "kling_o3_pro":     _O3_PRO_I2V,
    "kling_o3_std":     _O3_STD_I2V,
    "kling_o3_pro_ref": _O3_PRO_REF,
    "kling_o3_std_ref": _O3_STD_REF,
    "kling_omni":       _V3_OMNI_I2V,
}

# Human-readable labels used in notify messages
MODEL_LABELS: dict[str, str] = {
    "kling":            "Kling v3 Pro",
    "kling_t2v":        "Kling v3 Pro T2V",
    "kling_v3_std":     "Kling v3 Std",
    "kling_turbo":      "Kling v3 Turbo",
    "kling_turbo_t2v":  "Kling v3 Turbo T2V",
    "kling_o3_pro":     "Kling O3 Pro",
    "kling_o3_std":     "Kling O3 Std",
    "kling_o3_pro_ref": "Kling O3 Pro Reference",
    "kling_o3_std_ref": "Kling O3 Std Reference",
    "kling_omni":       "Kling v3 Omni",
}

_REFERENCE_MODELS = {_O3_PRO_REF, _O3_STD_REF}
_OMNI_MODELS      = {_V3_OMNI_I2V}

# Settings keys whose generation uses the guidances/multi-frame API
MULTIFRAME_SETTINGS_KEYS: frozenset[str] = frozenset({"kling", "kling_v3_std", "kling_t2v"})


class KlingService:
    def __init__(self, api_key: str, static_dir: str = ""):
        self._atlas = AtlasClient(api_key, static_dir)
        self.static_dir = self._atlas.static_dir

    async def upload_photo(self, photo_path: str) -> str:
        url = await self._atlas.upload_file(photo_path)
        logger.info(f"Kling: uploaded photo {photo_path} → {url}")
        return url

    async def upload_audio(self, audio_path: str) -> str:
        url = await self._atlas.upload_file(audio_path)
        logger.info(f"Kling: uploaded audio {audio_path} → {url}")
        return url

    async def generate_clip(
        self,
        prompt: str,
        image_url: str = "",
        image_urls: list[str] | None = None,
        voice_element_audio_url: str = "",
        duration: int = 10,
        aspect_ratio: str = "9:16",
        negative_prompt: str = "",
        model_id: str = _V3_PRO_I2V,
    ) -> str:
        """Generate one clip. Returns local path to downloaded MP4.

        Omni models use `image_urls` + `voice_element_audio` for native lip-sync.
        Reference models use `images` (array) and ignore negative_prompt.
        I2V models use `image` (single string).
        """
        os.makedirs(self.static_dir, exist_ok=True)
        is_reference = model_id in _REFERENCE_MODELS
        is_omni      = model_id in _OMNI_MODELS

        # image_urls takes priority over image_url
        effective_urls = image_urls if image_urls else ([image_url] if image_url else [])

        if effective_urls:
            model = model_id
            if is_omni:
                params: dict = {
                    "prompt": prompt,
                    "image_urls": effective_urls,
                    "duration": duration,
                    "aspect_ratio": aspect_ratio,
                    "mode": "pro",
                }
                if voice_element_audio_url:
                    params["voice_element_audio"] = voice_element_audio_url
            elif is_reference:
                params = {
                    "prompt": prompt,
                    "images": effective_urls,
                    "duration": duration,
                    "aspect_ratio": aspect_ratio,
                    "sound": False,
                }
            else:
                params = {
                    "prompt": prompt,
                    "image": effective_urls[0],
                    "duration": duration,
                }
                if negative_prompt:
                    params["negative_prompt"] = negative_prompt
        else:
            model = _V3_PRO_T2V
            params = {
                "prompt": prompt,
                "duration": duration,
                "aspect_ratio": aspect_ratio,
            }

        logger.info(
            f"Kling generating clip | model={model} | {duration}s"
            f" | images={len(effective_urls)} | audio={'yes' if voice_element_audio_url else 'no'}"
        )
        video_url = await self._atlas.generate_video(model, params)
        return await self._atlas.download(video_url, ext="mp4")

    async def upload_video(self, video_path: str) -> str:
        """Upload a local video to Atlas Cloud storage. Returns public URL."""
        url = await self._atlas.upload_file(video_path)
        logger.info(f"Kling: uploaded video {video_path} → {url}")
        return url

    async def edit_video(
        self,
        video_path: str,
        prompt: str,
        image_urls: list[str] | None = None,
        keep_original_sound: bool = True,
        use_pro: bool = True,
    ) -> str:
        """Edit an existing video via text prompt (V2V). Returns local path to MP4."""
        os.makedirs(self.static_dir, exist_ok=True)
        video_url = await self.upload_video(video_path)
        model = _O3_PRO_EDIT if use_pro else _O3_STD_EDIT
        params: dict = {
            "video": video_url,
            "prompt": prompt,
            "keep_original_sound": keep_original_sound,
        }
        if image_urls:
            params["images"] = image_urls[:4]
        logger.info(f"Kling Video-Edit | model={model} | prompt={prompt[:80]}…")
        result_url = await self._atlas.generate_video(model, params)
        return await self._atlas.download(result_url, ext="mp4")

    async def generate_clips(
        self,
        scene_prompts: list[str],
        anchor_photo_urls: list[str],
        clip_duration: int = 10,
        negative_prompt: str = "",
        model_id: str = _V3_PRO_I2V,
        all_reference_urls: list[str] | None = None,
    ) -> list[str]:
        """Generate multiple clips.

        For I2V models: extracts last frame after each clip and uses it as the
        anchor for the next clip. `anchor_photo_urls` cycles through multiple
        images when provided.
        For Reference models: passes `all_reference_urls` (all uploaded images) to
        every clip so the model uses all of them for character consistency.
        """
        is_reference = model_id in _REFERENCE_MODELS
        clips: list[str] = []

        for i, (prompt, photo_url) in enumerate(zip(scene_prompts, anchor_photo_urls)):
            logger.info(f"Kling clip {i + 1}/{len(scene_prompts)} | model={model_id}")

            if is_reference and i > 0:
                effective_prompt = (
                    "Seamlessly continuing from previous scene — "
                    "same lighting, same background, same camera angle, smooth action flow. "
                    + prompt
                )
            else:
                effective_prompt = prompt

            if is_reference and all_reference_urls:
                clip_path = await self.generate_clip(
                    prompt=effective_prompt,
                    image_urls=all_reference_urls,
                    duration=clip_duration,
                    model_id=model_id,
                )
            else:
                clip_path = await self.generate_clip(
                    prompt=effective_prompt,
                    image_url=photo_url,
                    duration=clip_duration,
                    negative_prompt=negative_prompt,
                    model_id=model_id,
                )
            clips.append(clip_path)

            # Last-frame continuity for all I2V models (regardless of photo count)
            if not is_reference and i < len(scene_prompts) - 1:
                frame_path = await self._extract_last_frame(clip_path)
                if frame_path:
                    frame_url = await self.upload_photo(frame_path)
                    anchor_photo_urls[i + 1] = frame_url
                    try:
                        os.remove(frame_path)
                    except OSError:
                        pass

        return clips

    async def generate_multiframe_clip(
        self,
        scene_prompts: list[str],
        shot_duration: int = 5,
        shot_durations: list[int] | None = None,
        image_reference_url: str = "",
        motion_has_audio: bool = False,
        face_consistency: bool = True,
        negative_prompt: str = "",
    ) -> str:
        """Generate one clip via the guidances (multi-frame) API.

        scene_prompts: up to 6 shot descriptions (max 6 × 5s = 15s per call).
        shot_durations: per-shot durations; falls back to shot_duration if absent.
        image_reference_url: single character anchor image for I2V; omit for T2V.
        motion_has_audio: let Kling generate its own background audio.
        face_consistency: stabilise facial features across shots.
        """
        os.makedirs(self.static_dir, exist_ok=True)
        effective_durs = (
            shot_durations
            if shot_durations and len(shot_durations) == len(scene_prompts)
            else [shot_duration] * len(scene_prompts)
        )
        params: dict = {
            "guidances": [
                {"index": i, "prompt": p, "duration": effective_durs[i]}
                for i, p in enumerate(scene_prompts)
            ],
            "motion_has_audio": motion_has_audio,
            "face_consistency": face_consistency,
        }
        if image_reference_url:
            params["image_reference"] = image_reference_url
        if negative_prompt:
            params["negative_prompt"] = negative_prompt

        logger.info(
            f"Kling multi-frame | shots={len(scene_prompts)} × {shot_duration}s"
            f" | total={len(scene_prompts) * shot_duration}s"
            f" | ref={'yes' if image_reference_url else 'no'}"
        )
        video_url = await self._atlas.generate_video(_V3_MULTI, params)
        return await self._atlas.download(video_url, ext="mp4")

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
