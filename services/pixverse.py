"""PixVerse V4.5 scene video generation via Atlas Cloud.

Multi-clip flow identical to SeedanceService.
Good for stylized content, creative effects, and character animation.

Default clip duration is 5 s (PixVerse's standard unit).
"""

import asyncio
import logging
import os
import uuid

from services.atlas import AtlasClient

logger = logging.getLogger(__name__)

_MODEL_IMAGE_TO_VIDEO = "pixverse/pixverse-v4.5-i2v"
_MODEL_TEXT_TO_VIDEO  = "pixverse/pixverse-v4.5-t2v"


class PixVerseService:
    def __init__(self, api_key: str, static_dir: str = ""):
        self._atlas = AtlasClient(api_key, static_dir)
        self.static_dir = self._atlas.static_dir

    async def upload_photo(self, photo_path: str) -> str:
        """Upload a local photo to Atlas Cloud storage. Returns public URL."""
        url = await self._atlas.upload_file(photo_path)
        logger.info(f"PixVerse: uploaded {photo_path} → {url}")
        return url

    async def generate_clip(
        self,
        prompt: str,
        image_url: str = "",
        duration: int = 5,
        aspect_ratio: str = "9:16",
        resolution: str = "720p",
    ) -> str:
        """Generate one clip. Returns local path to downloaded MP4."""
        os.makedirs(self.static_dir, exist_ok=True)

        if image_url:
            model = _MODEL_IMAGE_TO_VIDEO
            params: dict = {
                "prompt": prompt,
                "image_url": image_url,
                "duration": duration,
                "aspect_ratio": aspect_ratio,
                "quality": resolution,
                "generate_audio": False,
            }
        else:
            model = _MODEL_TEXT_TO_VIDEO
            params = {
                "prompt": prompt,
                "duration": duration,
                "aspect_ratio": aspect_ratio,
                "quality": resolution,
                "generate_audio": False,
            }

        logger.info(f"PixVerse V4.5 generating clip | model={model} | {duration}s")
        video_url = await self._atlas.generate_video(model, params)
        return await self._atlas.download(video_url, ext="mp4")

    async def generate_clips(
        self,
        scene_prompts: list[str],
        anchor_photo_urls: list[str],
        clip_duration: int = 5,
    ) -> list[str]:
        """Generate multiple clips with last-frame continuity anchoring."""
        clips: list[str] = []

        for i, (prompt, photo_url) in enumerate(zip(scene_prompts, anchor_photo_urls)):
            logger.info(f"PixVerse clip {i + 1}/{len(scene_prompts)}")
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
        return 5.0
