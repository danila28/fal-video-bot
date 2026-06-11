"""Happy Horse (Alibaba) — image-to-video with native Foley audio and lip-sync.

Single-clip generation: 1 photo → 1 video (3-15s) with built-in ambient sound.
No multi-clip needed — the whole video is generated in one API call.

Because the output already contains native audio, post-processing must NOT
replace the audio track — it should layer TTS on top instead.
"""

import asyncio
import logging
import os
import uuid

import aiohttp
import fal_client

logger = logging.getLogger(__name__)

_MODEL = "alibaba/happy-horse/image-to-video"


class HappyHorseService:
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
        image_url: str,
        duration: int = 10,
        resolution: str = "720p",
    ) -> str:
        """Generate one clip with native audio. image_url is required.

        Returns local path to the downloaded MP4.
        The video already contains Foley audio — post-processing should
        layer TTS on top rather than replace the audio track.
        """
        os.makedirs(self.static_dir, exist_ok=True)
        duration = max(3, min(15, duration))

        logger.info(f"HappyHorse generating clip | {duration}s | {resolution}")

        result = await asyncio.to_thread(
            fal_client.subscribe,
            _MODEL,
            arguments={
                "image_url": image_url,
                "prompt": prompt,
                "duration": duration,
                "resolution": resolution,
            },
        )

        video_url = result["video"]["url"]
        return await self._download(video_url)

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
        logger.info(f"Downloaded HappyHorse clip → {dest}")
        return dest
