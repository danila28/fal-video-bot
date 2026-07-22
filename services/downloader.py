"""Video download service using yt-dlp."""

import asyncio
import logging
import os
import uuid
from typing import Optional

logger = logging.getLogger(__name__)


class DownloaderService:
    """Download videos from TikTok, Instagram Reels, YouTube Shorts, etc."""

    def __init__(self, static_dir: str = ""):
        if not static_dir:
            static_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static"
            )
        self.static_dir = static_dir
        os.makedirs(static_dir, exist_ok=True)

    async def download(self, url: str, max_size_mb: int = 500) -> Optional[str]:
        """Download video from URL. Returns local path or None on failure.

        Supports TikTok, Instagram Reels, YouTube Shorts, Twitter/X, etc.
        max_size_mb: maximum video size in MB (default 500)
        """
        try:
            import yt_dlp
        except ImportError:
            logger.error("yt-dlp not installed. Run: pip install yt-dlp")
            raise Exception("yt-dlp library is not installed. Please ensure it's installed: pip install yt-dlp")

        try:
            filename = f"{uuid.uuid4()}.mp4"
            output_path = os.path.join(self.static_dir, filename)

            ydl_opts = {
                "format": "best[ext=mp4]",
                "outtmpl": output_path.replace(".mp4", ""),
                "quiet": True,
                "no_warnings": True,
                "socket_timeout": 30,
                "retries": 3,
                "file_access_retries": 3,
                "http_headers": {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
                },
                "postprocessors": [{
                    "key": "FFmpegVideoConvertor",
                    "preferedformat": "mp4",
                }],
                # YouTube требует cookies для некоторых видео
                "cookies_from_browser": None,
                "extractor_args": {
                    "youtube": {
                        "skip": ["dash", "translated_subs"],
                    }
                },
            }

            def _download():
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    logger.info(f"Starting download from: {url}")
                    info = ydl.extract_info(url, download=True)
                    if not info:
                        raise Exception("Failed to extract video info")
                    actual_path = ydl.prepare_filename(info)
                    logger.info(f"Downloaded to: {actual_path}")
                    return actual_path

            actual_path = await asyncio.to_thread(_download)

            # yt-dlp may output different extension, rename to .mp4 if needed
            if actual_path and os.path.exists(actual_path):
                if not actual_path.endswith(".mp4"):
                    if os.path.exists(output_path):
                        os.remove(output_path)
                    os.rename(actual_path, output_path)
                    actual_path = output_path
            else:
                # Try to find the file by checking the directory
                files = [f for f in os.listdir(self.static_dir)
                        if f.startswith(filename.replace('.mp4', '')) and f != filename]
                if files:
                    found_path = os.path.join(self.static_dir, files[0])
                    if not found_path.endswith(".mp4"):
                        os.rename(found_path, output_path)
                        actual_path = output_path
                    else:
                        actual_path = found_path

            if os.path.exists(actual_path if actual_path else output_path):
                final_path = actual_path if actual_path else output_path
                size_mb = os.path.getsize(final_path) / (1024 * 1024)

                if size_mb > max_size_mb:
                    os.remove(final_path)
                    raise Exception(f"Video too large: {size_mb:.1f}MB (max {max_size_mb}MB)")

                logger.info(f"Downloaded {url} → {final_path} ({size_mb:.1f} MB)")
                return final_path

            raise Exception(f"Video file not found after download")

        except Exception as e:
            logger.error(f"Download failed for {url}: {e}")
            raise Exception(f"Failed to download video: {str(e)}")
