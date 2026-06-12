"""Seedance 2.0 video generation via Atlas Cloud.

Flow (identical to Kling / PixVerse services):
  1. Upload reference photo to Atlas Cloud storage
  2. Generate clips via image-to-video (or text-to-video if no photo)
  3. Return local MP4 paths for concatenation in post-processing

Clip duration: 10 s (standard Seedance unit).
"""

import asyncio
import logging
import os
import uuid

from services.atlas import AtlasClient

logger = logging.getLogger(__name__)

_MODEL_IMAGE_TO_VIDEO = "bytedance/seedance-2.0/image-to-video"
_MODEL_TEXT_TO_VIDEO  = "bytedance/seedance-2.0/text-to-video"


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
    ) -> str:
        """Generate one clip. Returns local path to downloaded MP4."""
        os.makedirs(self.static_dir, exist_ok=True)

        if image_url:
            model = _MODEL_IMAGE_TO_VIDEO
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
            model = _MODEL_TEXT_TO_VIDEO
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
    ) -> list[str]:
        """Generate multiple clips. Each clip uses its own anchor photo.

        After each clip the last frame is extracted and used as the image_url
        for the next clip to maintain visual continuity — but only if the
        clip's dedicated photo is the same as the previous one.
        """
        clips: list[str] = []

        for i, (prompt, photo_url) in enumerate(zip(scene_prompts, anchor_photo_urls)):
            logger.info(f"Generating Seedance clip {i + 1}/{len(scene_prompts)}")
            clip_path = await self.generate_clip(
                prompt=prompt,
                image_url=photo_url,
                duration=clip_duration,
            )
            clips.append(clip_path)

            if i < len(scene_prompts) - 1:
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
