"""Wan 2.7 video editing via Atlas Cloud — alternative engine to Gemini Omni
Flash, offered as an explicit user choice (not an automatic fallback) since
its behavior on videos containing real people has not been verified against
our own footage.

Model: alibaba/wan-2.7/video-edit. Takes a source video + text prompt (+
optional reference images) and edits it, preserving what the prompt doesn't
touch.

Pricing: $0.10 per second of SOURCE video (Atlas listing, July 2026). The
billed-seconds clamp is NOT confirmed the way Omni's is — verify against the
first real invoice and adjust BILL_MIN_SECONDS/BILL_MAX_SECONDS if it's off.
"""

import logging

from services.atlas import AtlasClient

logger = logging.getLogger(__name__)

MODEL_ID = "alibaba/wan-2.7/video-edit"

PRICE_PER_SECOND = 0.10
# Unconfirmed — mirrors Omni's clamp as a placeholder until the real billing
# behavior is observed on an invoice.
BILL_MIN_SECONDS = 3
BILL_MAX_SECONDS = 30

MAX_REFERENCE_IMAGES = 5


def estimate_price(duration_seconds: float) -> float:
    """Cost of one edit run: source duration clamped to 3-30s x $0.10."""
    billed = max(BILL_MIN_SECONDS, min(BILL_MAX_SECONDS, duration_seconds))
    return billed * PRICE_PER_SECOND


class WanEditService:
    def __init__(self, api_key: str, static_dir: str = ""):
        self._atlas = AtlasClient(api_key, static_dir)
        self.static_dir = self._atlas.static_dir

    async def edit_video(
        self,
        video_path: str,
        prompt: str,
        reference_image_paths: list[str] | None = None,
    ) -> str:
        """Edit a local video per the prompt. Returns local path to the result.

        reference_image_paths: up to 5 images guiding the edit. Signature
        matches OmniEditService.edit_video so callers can pick either engine
        without branching on parameters.
        """
        video_url = await self._atlas.upload_file(video_path)

        params: dict = {
            "prompt": prompt,
            "video": video_url,
        }

        refs = (reference_image_paths or [])[:MAX_REFERENCE_IMAGES]
        if refs:
            image_urls = [await self._atlas.upload_file(p) for p in refs]
            params["images"] = image_urls

        logger.info(
            f"Wan edit | prompt={prompt[:80]!r} | refs={len(refs)}"
        )
        result_url = await self._atlas.generate_video(MODEL_ID, params)
        return await self._atlas.download(result_url, ext="mp4")
