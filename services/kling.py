"""Kling video generation via Atlas Cloud.

Supported models (set via settings video_model):
  kling             → Kling v3.0 Pro    Image-to-Video  (multi-shot batches)
  kling_v3_std      → Kling v3.0 Std    Image-to-Video  (multi-shot batches)
  kling_t2v         → Kling v3.0 Pro    Text-to-Video   (multi-shot batches)
  kling_turbo       → Kling v3.0 Turbo  Image-to-Video
  kling_turbo_t2v   → Kling v3.0 Turbo  Text-to-Video
  kling_o3_pro      → Kling O3 Pro      Image-to-Video
  kling_o3_std      → Kling O3 Std      Image-to-Video
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
_O3_PRO_EDIT   = "kwaivgi/kling-video-o3-pro/video-edit"
_O3_STD_EDIT   = "kwaivgi/kling-video-o3-std/video-edit"

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
}

# Settings keys whose generation uses the multi_shot storyboard API
# (same model IDs as normal generation, with multi_shot=True + multi_prompt=[...])
MULTIFRAME_SETTINGS_KEYS: frozenset[str] = frozenset({"kling", "kling_v3_std", "kling_t2v"})

# Atlas hard limit for each multi_prompt[].prompt entry (ret:1201 above it)
_MULTI_PROMPT_MAX_CHARS = 512


def _trim_shot_prompt(prompt: str, limit: int = _MULTI_PROMPT_MAX_CHARS) -> str:
    """Trim a shot prompt to Atlas's per-entry limit, cutting at a word boundary."""
    prompt = prompt.strip()
    if len(prompt) <= limit:
        return prompt
    cut = prompt[:limit]
    last_space = cut.rfind(" ")
    if last_space > limit // 2:
        cut = cut[:last_space]
    logger.warning(
        f"Kling multi-shot prompt trimmed {len(prompt)} → {len(cut)} chars (Atlas 512 limit)"
    )
    return cut.rstrip(",.;:— ")


class KlingService:
    def __init__(self, api_key: str, static_dir: str = ""):
        self._atlas = AtlasClient(api_key, static_dir)
        self.static_dir = self._atlas.static_dir

    async def upload_photo(self, photo_path: str) -> str:
        url = await self._atlas.upload_file(photo_path)
        logger.info(f"Kling: uploaded photo {photo_path} → {url}")
        return url

    async def generate_clip(
        self,
        prompt: str,
        image_url: str = "",
        image_urls: list[str] | None = None,
        duration: int = 10,
        aspect_ratio: str = "9:16",
        negative_prompt: str = "",
        model_id: str = _V3_PRO_I2V,
    ) -> str:
        """Generate one clip. Returns local path to downloaded MP4.

        With an image: I2V using `image`. Without: falls back to the T2V
        sibling of `model_id` (same tier — Turbo stays Turbo, Pro stays Pro).
        """
        os.makedirs(self.static_dir, exist_ok=True)

        # image_urls takes priority over image_url
        effective_urls = image_urls if image_urls else ([image_url] if image_url else [])

        if effective_urls:
            model = model_id
            params: dict = {
                "prompt": prompt,
                "image": effective_urls[0],
                "duration": duration,
                "sound": False,  # Base clip stays silent — our own TTS mix replaces it in post-processing
            }
            if negative_prompt:
                params["negative_prompt"] = negative_prompt
        else:
            model = (
                model_id if "text-to-video" in model_id
                else model_id.replace("/image-to-video", "/text-to-video")
            )
            params = {
                "prompt": prompt,
                "duration": duration,
                "aspect_ratio": aspect_ratio,
                "sound": False,  # Disable Kling's built-in foley audio for T2V
            }

        logger.info(
            f"Kling generating clip | model={model} | {duration}s"
            f" | images={len(effective_urls)}"
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
        clip_duration: int | list[int] = 10,
        negative_prompt: str = "",
        model_id: str = _V3_PRO_I2V,
    ) -> list[str]:
        """Generate multiple clips.

        Extracts the last frame after each clip and uses it as the anchor for
        the next clip. `anchor_photo_urls` cycles through multiple images when
        provided. clip_duration: single value for every clip, or a per-clip list.
        """
        clips: list[str] = []
        durations = (
            clip_duration if isinstance(clip_duration, list)
            else [clip_duration] * len(scene_prompts)
        )

        for i, (prompt, photo_url) in enumerate(zip(scene_prompts, anchor_photo_urls)):
            logger.info(f"Kling clip {i + 1}/{len(scene_prompts)} | model={model_id}")
            dur = durations[i] if i < len(durations) else durations[-1]

            clip_path = await self.generate_clip(
                prompt=prompt,
                image_url=photo_url,
                duration=dur,
                negative_prompt=negative_prompt,
                model_id=model_id,
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

    async def generate_multiframe_clip(
        self,
        scene_prompts: list[str],
        shot_duration: int = 5,
        shot_durations: list[int] | None = None,
        image_reference_url: str = "",
        motion_has_audio: bool = False,
        negative_prompt: str = "",
        model_id: str = _V3_PRO_T2V,
        aspect_ratio: str = "9:16",
    ) -> str:
        """Generate one clip via Kling's multi_shot storyboard mode.

        Uses the model's normal generateVideo model ID with multi_shot=True
        and a multi_prompt array — NOT a separate "guidances" endpoint/model
        (that field/model ID doesn't exist on Atlas Cloud and returns HTTP 400).

        scene_prompts: up to 6 shot descriptions (max 6 × 5s = 15s per call).
        shot_durations: per-shot durations; falls back to shot_duration if absent.
        image_reference_url: single character anchor image for I2V; omit for T2V.
        motion_has_audio: let Kling generate its own synchronized audio ("sound").
        model_id: I2V or T2V Kling model ID (kling/kling_v3_std/kling_t2v tier).
        """
        os.makedirs(self.static_dir, exist_ok=True)
        effective_durs = (
            shot_durations
            if shot_durations and len(shot_durations) == len(scene_prompts)
            else [shot_duration] * len(scene_prompts)
        )
        # Atlas rejects multi_prompt entries over 512 chars (ret:1201) — trim at
        # a word boundary. The single-prompt limit (2500) doesn't apply here.
        trimmed = [_trim_shot_prompt(p) for p in scene_prompts]
        params: dict = {
            "duration": sum(effective_durs),
            "multi_shot": True,
            "shot_type": "customize",
            "multi_prompt": [
                {"index": i + 1, "prompt": p, "duration": effective_durs[i]}
                for i, p in enumerate(trimmed)
            ],
            "sound": motion_has_audio,
        }
        if image_reference_url:
            params["image"] = image_reference_url
        else:
            params["aspect_ratio"] = aspect_ratio
        if negative_prompt:
            params["negative_prompt"] = negative_prompt

        logger.info(
            f"Kling multi-shot | model={model_id} | shots={len(scene_prompts)}"
            f" | total={sum(effective_durs)}s"
            f" | ref={'yes' if image_reference_url else 'no'}"
        )
        video_url = await self._atlas.generate_video(model_id, params)
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
