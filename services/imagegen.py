"""Photo generation for video pipelines.

All models are hosted on Atlas Cloud and produce 9:16 vertical output.

  • gemini-3.1-flash-image — Nano Banana 2 (fast, affordable)
  • gemini-3-pro-image      — Nano Banana Pro (highest quality)
"""

import asyncio
import logging
import os
import uuid

from services.atlas import AtlasClient

logger = logging.getLogger(__name__)

_NANO_BANANA_2_MODEL = "nano-banana-2"
_NANO_BANANA_PRO_MODEL = "nano-banana-pro"

# All Nano Banana models are Atlas-hosted image generation models
_IMAGE_MODELS = {_NANO_BANANA_2_MODEL, _NANO_BANANA_PRO_MODEL}


class ImageGenService:
    def __init__(self, api_key: str = "", atlas_api_key: str = "", static_dir: str = ""):
        self._atlas = AtlasClient(atlas_api_key, static_dir) if atlas_api_key else None
        if not static_dir:
            static_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static"
            )
        self.static_dir = static_dir
        os.makedirs(static_dir, exist_ok=True)

    @staticmethod
    def _aspect_for_video_model(video_model: str) -> str:
        return "9:16"

    def _require_atlas(self) -> AtlasClient:
        if self._atlas is None:
            raise RuntimeError("ATLAS_API_KEY is required for image generation")
        return self._atlas

    # ── Atlas image generation ────────────────────────────────────────────

    async def _generate_via_atlas(self, prompt: str, model: str, aspect_ratio: str) -> str:
        """Generate image via Atlas Cloud Nano Banana models."""
        atlas = self._require_atlas()
        # Try minimal params first — some models may not support aspect_ratio/output_format
        output_url = await atlas.generate_image(
            model,
            {"prompt": prompt},
        )
        return await atlas.download(output_url, ext="jpg")

    # ── Public API ────────────────────────────────────────────────────────

    _DEFAULT_MODEL = _NANO_BANANA_2_MODEL

    async def generate_many(
        self,
        prompt: str,
        model: str = "",
        video_model: str = "seedance",
        count: int = 1,
        notify=None,
    ) -> list[str]:
        """Generate `count` images in parallel. Returns list of local file paths."""
        count = max(1, min(4, count))
        model = model or self._DEFAULT_MODEL
        tasks = [self.generate(prompt, model, video_model, notify) for _ in range(count)]
        return list(await asyncio.gather(*tasks))

    async def generate(
        self,
        prompt: str,
        model: str = "",
        video_model: str = "seedance",
        notify=None,
    ) -> str:
        """Generate one image. Raises on failure."""
        model = model or self._DEFAULT_MODEL
        aspect_ratio = self._aspect_for_video_model(video_model)
        logger.info(f"Image generation | model={model} | aspect={aspect_ratio} | prompt={prompt[:80]}…")

        if model in _IMAGE_MODELS:
            return await self._generate_via_atlas(prompt, model, aspect_ratio)
        raise ValueError(f"Unknown image model '{model}'. Go to ⚙️ Settings → 🖼 Image model and reselect.")
