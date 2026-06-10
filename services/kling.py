"""Kling Avatar v2 video generation via fal.ai.

Flow:
  1. Upload reference photo to fal.ai storage (parallel with audio upload)
  2. Upload TTS audio to fal.ai storage
  3. Call Kling Avatar v2 (image + audio → lip-sync talking head video)
  4. Download resulting MP4
"""

import asyncio
import logging
import os
import uuid

import aiohttp
import fal_client

logger = logging.getLogger(__name__)

_MODEL_STANDARD = "fal-ai/kling-video/ai-avatar/v2/standard"
_MODEL_PRO      = "fal-ai/kling-video/ai-avatar/v2/pro"


class KlingService:
    def __init__(self, api_key: str, static_dir: str = "", use_pro: bool = False):
        os.environ["FAL_KEY"] = api_key
        if not static_dir:
            static_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static"
            )
        self.static_dir = static_dir
        self.model = _MODEL_PRO if use_pro else _MODEL_STANDARD
        os.makedirs(static_dir, exist_ok=True)

    async def upload_file(self, file_path: str) -> str:
        """Upload any local file to fal.ai storage. Returns public URL."""
        url = await asyncio.to_thread(fal_client.upload_file, file_path)
        logger.info(f"Uploaded {file_path} → {url}")
        return url

    async def generate_avatar_video(
        self,
        photo_path: str,
        audio_path: str,
        prompt: str = "",
    ) -> str:
        """Generate a lip-sync talking head video from photo + audio.

        Uploads both files to fal.ai in parallel, then calls Kling Avatar v2.
        Returns local path to downloaded MP4.
        """
        logger.info("Uploading photo and audio to fal.ai (parallel)…")
        photo_url, audio_url = await asyncio.gather(
            self.upload_file(photo_path),
            self.upload_file(audio_path),
        )

        logger.info(f"Generating Kling Avatar | model={self.model}")
        arguments: dict = {
            "image_url": photo_url,
            "audio_url": audio_url,
        }
        if prompt:
            arguments["prompt"] = prompt

        result = await asyncio.wait_for(
            asyncio.to_thread(
                fal_client.subscribe,
                self.model,
                arguments=arguments,
            ),
            timeout=600,  # 10 minutes max
        )

        video_url = result["video"]["url"]
        return await self._download(video_url)

    async def _download(self, url: str) -> str:
        """Download video from URL to local static dir. Returns local path."""
        filename = f"{uuid.uuid4()}.mp4"
        dest = os.path.join(self.static_dir, filename)
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    raise Exception(f"Failed to download video: HTTP {resp.status}")
                with open(dest, "wb") as f:
                    async for chunk in resp.content.iter_chunked(65536):
                        f.write(chunk)
        logger.info(f"Downloaded avatar video → {dest}")
        return dest
