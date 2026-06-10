"""Kling v2.1 scene video generation via fal.ai.

Flow (identical to SeedanceService):
  1. Upload reference photo to fal.ai storage
  2. Generate clips via image-to-video (or text-to-video if no photo)
  3. Return local MP4 paths for concatenation in post-processing

Kling v2.1 supports 5s and 10s clip durations.
"""

import asyncio
import logging
import os
import uuid

import aiohttp
import fal_client

logger = logging.getLogger(__name__)

_MODEL_IMAGE_TO_VIDEO = "fal-ai/kling-video/v2.1/pro/image-to-video"
_MODEL_TEXT_TO_VIDEO  = "fal-ai/kling-video/v2.1/pro/text-to-video"


class KlingService:
    def __init__(self, api_key: str, static_dir: str = ""):
        os.environ["FAL_KEY"] = api_key
        if not static_dir:
            static_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static"
            )
        self.static_dir = static_dir
        os.makedirs(static_dir, exist_ok=True)

    async def upload_photo(self, photo_path: str) -> str:
        """Upload a local photo to fal.ai storage. Returns public URL."""
        url = await asyncio.to_thread(fal_client.upload_file, photo_path)
        logger.info(f"Uploaded {photo_path} → {url}")
        return url

    async def generate_clip(
        self,
        prompt: str,
        image_url: str = "",
        duration: int = 10,
        aspect_ratio: str = "9:16",
        negative_prompt: str = "",
    ) -> str:
        """Generate one clip. Returns local path to downloaded MP4."""
        os.makedirs(self.static_dir, exist_ok=True)

        if image_url:
            model = _MODEL_IMAGE_TO_VIDEO
            arguments: dict = {
                "prompt": prompt,
                "image_url": image_url,
                "duration": str(duration),
                "aspect_ratio": aspect_ratio,
            }
        else:
            model = _MODEL_TEXT_TO_VIDEO
            arguments = {
                "prompt": prompt,
                "duration": str(duration),
                "aspect_ratio": aspect_ratio,
            }

        if negative_prompt:
            arguments["negative_prompt"] = negative_prompt

        logger.info(f"Kling generating clip | model={model} | {duration}s")

        result = await asyncio.to_thread(
            fal_client.subscribe,
            model,
            arguments=arguments,
        )

        video_url = result["video"]["url"]
        return await self._download(video_url)

    async def generate_clips(
        self,
        scene_prompts: list[str],
        anchor_photo_urls: list[str],
        clip_duration: int = 10,
        negative_prompt: str = "",
    ) -> list[str]:
        """Generate multiple clips with last-frame continuity anchoring."""
        clips: list[str] = []

        for i, (prompt, photo_url) in enumerate(zip(scene_prompts, anchor_photo_urls)):
            logger.info(f"Kling clip {i + 1}/{len(scene_prompts)}")
            clip_path = await self.generate_clip(
                prompt=prompt,
                image_url=photo_url,
                duration=clip_duration,
                negative_prompt=negative_prompt,
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

    async def _download(self, url: str) -> str:
        filename = f"{uuid.uuid4()}.mp4"
        dest = os.path.join(self.static_dir, filename)
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    raise Exception(f"Failed to download video: HTTP {resp.status}")
                with open(dest, "wb") as f:
                    async for chunk in resp.content.iter_chunked(65536):
                        f.write(chunk)
        logger.info(f"Downloaded Kling clip → {dest}")
        return dest

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
