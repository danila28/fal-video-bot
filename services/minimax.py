"""MiniMax H3 video generation via Atlas Cloud.

Supported models (set via settings video_model):
  minimax_h3       → MiniMax H3   Image-to-Video
  minimax_h3_t2v   → MiniMax H3   Text-to-Video
  minimax_h3_ref   → MiniMax H3   Reference-to-Video

Modeled directly on SeedanceService: same Atlas submit/poll/download
pattern via AtlasClient, same @Image1..N reference-tagging convention,
single tier (no fast/mini variants exist for h3). Atlas lists
"minimax/h3/{task}" under the identical provider/model/task URL shape as
"bytedance/seedance-2.0/{task}". Param field names carried over from
Seedance (image_url/image_urls/ratio/duration/generate_audio/watermark)
are STILL UNVERIFIED beyond `resolution` — that one field name is
confirmed correct because Atlas's first live 400 evaluated its value
against an enum instead of rejecting the field as unknown.

Resolution is CONFIRMED (from that same live error) to use its own
vocabulary, not Seedance/Kling's "480p/720p/1080p":
    supported: 480P, 768P, 2K
The bot's shared video_resolution setting only offers 480p/720p/1080p —
_normalize_resolution() maps those to the nearest MiniMax tier.

Clip duration: assumed up to 15s per Atlas call, same as every other Atlas
video model seen so far — UNVERIFIED, adjust _MAX_CALL_DURATION if Atlas
reports a different cap.
"""

import asyncio
import logging
import os
import uuid

from services.atlas import AtlasClient

logger = logging.getLogger(__name__)

# ── Atlas Cloud model IDs ─────────────────────────────────────────────────────

_I2V = "minimax/h3/image-to-video"
_T2V = "minimax/h3/text-to-video"
_REF = "minimax/h3/reference-to-video"

# Atlas hard limit (assumed, see module docstring): one request renders at
# most 15 seconds.
_MAX_CALL_DURATION = 15

# MiniMax H3's own resolution tiers (confirmed via live Atlas 400 error) —
# unrelated to Seedance/Kling's 480p/720p/1080p vocabulary. The bot's shared
# settings UI only offers the latter, so map to the nearest MiniMax tier:
# 720p has no exact match, 768P is the closest higher tier; 1080p maps up to
# the ceiling tier 2K (no 1080p-equivalent option exists).
_RESOLUTION_MAP: dict[str, str] = {
    "480p":  "480P",
    "720p":  "768P",
    "1080p": "2K",
}


def _normalize_resolution(resolution: str) -> str:
    return _RESOLUTION_MAP.get((resolution or "").lower(), resolution)

# Map from settings model name → Atlas model ID
MODEL_IDS: dict[str, str] = {
    "minimax_h3":      _I2V,
    "minimax_h3_t2v":  _T2V,
    "minimax_h3_ref":  _REF,
}

# Human-readable labels used in notify messages
MODEL_LABELS: dict[str, str] = {
    "minimax_h3":      "MiniMax H3",
    "minimax_h3_t2v":  "MiniMax H3 T2V",
    "minimax_h3_ref":  "MiniMax H3 Reference",
}

_REFERENCE_MODELS = {_REF}


class MiniMaxService:
    def __init__(self, api_key: str, static_dir: str = ""):
        self._atlas = AtlasClient(api_key, static_dir)
        self.static_dir = self._atlas.static_dir

    async def upload_photo(self, photo_path: str) -> str:
        url = await self._atlas.upload_file(photo_path)
        logger.info(f"MiniMax: uploaded photo {photo_path} → {url}")
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
        sibling of `model_id`.
        """
        os.makedirs(self.static_dir, exist_ok=True)
        duration = max(4, min(_MAX_CALL_DURATION, duration))
        resolution = _normalize_resolution(resolution)
        is_reference = model_id in _REFERENCE_MODELS

        # image_urls takes priority over image_url
        effective_urls = image_urls if image_urls else ([image_url] if image_url else [])

        if effective_urls:
            model = model_id
            if is_reference:
                # Reference mode: array of images + @ImageN tags in the prompt
                tags = " ".join(f"@Image{i + 1}" for i in range(len(effective_urls)))
                ref_prompt = (
                    f"{tags} {prompt}"
                    if not any(f"@Image{i + 1}" in prompt for i in range(len(effective_urls)))
                    else prompt
                )
                params = {
                    "prompt": ref_prompt,
                    "image_urls": effective_urls,
                    "duration": duration,
                    "ratio": aspect_ratio,
                    "resolution": resolution,
                    "generate_audio": keep_native_audio,
                }
            else:
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
                else model_id
                .replace("/image-to-video", "/text-to-video")
                .replace("/reference-to-video", "/text-to-video")
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
            f"MiniMax generating clip | model={model} | {duration}s"
            f" | images={len(effective_urls)} | resolution={params.get('resolution', 'n/a')}"
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

        total_duration overrides the clip_duration * len(scene_prompts) default —
        pass it to hit an exact target duration instead of a per-scene multiple.
        """
        os.makedirs(self.static_dir, exist_ok=True)
        resolution = _normalize_resolution(resolution)

        combined_prompt = " ".join(
            f"[Scene{i + 1}] {p}" for i, p in enumerate(scene_prompts)
        )
        if total_duration is None:
            total_duration = clip_duration * len(scene_prompts)
        if total_duration > _MAX_CALL_DURATION:
            logger.warning(
                f"MiniMax multi-scene {total_duration}s exceeds {_MAX_CALL_DURATION}s "
                f"per-call limit — clamping (use generate_clips for longer videos)"
            )
            total_duration = _MAX_CALL_DURATION

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
            f"MiniMax multi-scene | model={atlas_model} | scenes={len(scene_prompts)}"
            f" | total_duration={total_duration}s | resolution={resolution}"
        )
        video_url = await self._atlas.generate_video(atlas_model, params)
        return await self._atlas.download(video_url, ext="mp4")

    async def generate_clips(
        self,
        scene_prompts: list[str],
        anchor_photo_urls: list[str],
        clip_duration: int | list[int] = 10,
        resolution: str = "720p",
        aspect_ratio: str = "9:16",
        model_id: str = _I2V,
        keep_native_audio: bool = False,
        all_reference_urls: list[str] | None = None,
    ) -> list[str]:
        """Generate multiple clips.

        I2V models: extracts the last frame after each clip and uses it as the
        anchor for the next clip (visual continuity). `anchor_photo_urls`
        cycles through multiple images when provided.
        Reference models: pass `all_reference_urls` — every clip receives ALL
        reference images; last-frame stitching is skipped.
        """
        is_reference = model_id in _REFERENCE_MODELS
        clips: list[str] = []
        durations = (
            clip_duration if isinstance(clip_duration, list)
            else [clip_duration] * len(scene_prompts)
        )

        for i, (prompt, photo_url) in enumerate(zip(scene_prompts, anchor_photo_urls)):
            logger.info(f"Generating MiniMax clip {i + 1}/{len(scene_prompts)}")
            dur = durations[i] if i < len(durations) else durations[-1]

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
                    duration=dur,
                    aspect_ratio=aspect_ratio,
                    resolution=resolution,
                    model_id=model_id,
                    keep_native_audio=keep_native_audio,
                )
            else:
                clip_path = await self.generate_clip(
                    prompt=effective_prompt,
                    image_url=photo_url,
                    duration=dur,
                    aspect_ratio=aspect_ratio,
                    resolution=resolution,
                    model_id=model_id,
                    keep_native_audio=keep_native_audio,
                )
            clips.append(clip_path)

            # Verify Atlas actually honored the requested duration — if it
            # silently rounds/clamps, the final video would run longer/shorter
            # than the user's target_duration setting.
            try:
                actual_dur = await asyncio.to_thread(self._probe_duration, clip_path)
                if actual_dur > 0 and abs(actual_dur - dur) > 1.0:
                    logger.warning(
                        f"MiniMax clip {i + 1}/{len(scene_prompts)} duration mismatch: "
                        f"requested {dur}s, Atlas returned {actual_dur:.1f}s (model={model_id})"
                    )
            except Exception as e:
                logger.debug(f"Duration probe failed (non-fatal): {e}")

            # Last-frame continuity (I2V only — references anchor identity instead)
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
