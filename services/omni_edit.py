"""Gemini Omni Flash video editing via Atlas Cloud.

Model: google/gemini-omni-flash/video-edit (verified against the Atlas model
page, July 2026). Takes a source video + text prompt (+ up to 5 optional
reference images) and returns an edited video that preserves everything the
prompt doesn't touch. Output duration/aspect follow the source; 720p only.

Limits (Atlas): source video ≤100 MB and ≤30 s.
Pricing: clamp(source_duration, 3, 30) × $0.14 per second of SOURCE video.

Atlas does NOT expose Gemini's previous_interaction_id chaining — successive
edits are chained manually by feeding the previous result back as the new
source video (the handler does this).
"""

import logging

from services.atlas import AtlasClient

logger = logging.getLogger(__name__)

MODEL_ID = "google/gemini-omni-flash/video-edit"

# Verified on the Atlas model page — keep in sync with atlascloud.ai pricing.
PRICE_PER_SECOND = 0.14
BILL_MIN_SECONDS = 3
BILL_MAX_SECONDS = 30

MAX_SOURCE_SECONDS = 30
MAX_REFERENCE_IMAGES = 5


def estimate_price(duration_seconds: float) -> float:
    """Cost of one edit run: source duration clamped to 3–30 s × $0.14."""
    billed = max(BILL_MIN_SECONDS, min(BILL_MAX_SECONDS, duration_seconds))
    return billed * PRICE_PER_SECOND


class OmniEditService:
    def __init__(self, api_key: str, static_dir: str = ""):
        self._atlas = AtlasClient(api_key, static_dir)
        self.static_dir = self._atlas.static_dir

    async def edit_video(
        self,
        video_path: str,
        prompt: str,
        reference_image_paths: list[str] | None = None,
        thinking_level: str = "default",
    ) -> str:
        """Edit a local video per the prompt. Returns local path to the result.

        reference_image_paths: up to 5 images guiding the edit (e.g. a product
        to insert). thinking_level: default | high | low — latency vs quality
        on complex edits.
        """
        video_url = await self._atlas.upload_file(video_path)

        params: dict = {
            "prompt": prompt,
            "video": video_url,
            "resolution": "720p",
        }
        if thinking_level != "default":
            params["thinking_level"] = thinking_level

        refs = (reference_image_paths or [])[:MAX_REFERENCE_IMAGES]
        if refs:
            image_urls = [await self._atlas.upload_file(p) for p in refs]
            params["images"] = image_urls

        logger.info(
            f"Omni edit | prompt={prompt[:80]!r} | refs={len(refs)} "
            f"| thinking={thinking_level}"
        )
        result_url = await self._atlas.generate_video(MODEL_ID, params)
        return await self._atlas.download(result_url, ext="mp4")
