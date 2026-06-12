"""Happy Horse 1.0 (Alibaba) — image-to-video with native Foley audio.

Single-clip generation: 1 photo → 1 video (3–15 s) with built-in ambient sound.
No multi-clip needed — the whole video is generated in one API call.

Because the output already contains native audio, post-processing must NOT
replace the audio track — it should layer TTS on top instead.
"""

import logging
import os

from services.atlas import AtlasClient

logger = logging.getLogger(__name__)

_MODEL = "alibaba/happyhorse-1.0/image-to-video"


class HappyHorseService:
    def __init__(self, api_key: str, static_dir: str = ""):
        self._atlas = AtlasClient(api_key, static_dir)
        self.static_dir = self._atlas.static_dir

    async def upload_photo(self, photo_path: str) -> str:
        """Upload a local photo to Atlas Cloud storage. Returns public URL."""
        url = await self._atlas.upload_file(photo_path)
        logger.info(f"HappyHorse: uploaded {photo_path} → {url}")
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

        params = {
            "image_url": image_url,
            "prompt": prompt,
            "duration": duration,
            "resolution": resolution,
        }

        video_url = await self._atlas.generate_video(_MODEL, params)
        return await self._atlas.download(video_url, ext="mp4")
