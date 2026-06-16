"""Photo generation for video pipelines — Atlas Cloud only.

Backends chosen via `model` parameter:
  • black-forest-labs/flux-2-pro/text-to-image       — FLUX 2 Pro (default, universal)
  • black-forest-labs/flux-kontext-pro-text-to-image — FLUX Kontext (character consistency)
  • ideogram/ideogram-v3/text-to-image               — Ideogram V3 (stylised art)

All models produce 9:16 vertical output via Atlas Cloud.
On failure, automatically falls back to FLUX 2 Pro.
"""

import logging
import os
import uuid

from services.atlas import AtlasClient

logger = logging.getLogger(__name__)

_FLUX_MODEL         = "black-forest-labs/flux-2-pro/text-to-image"
_FLUX_KONTEXT_MODEL = "black-forest-labs/flux-kontext-pro-text-to-image"
_IDEOGRAM_MODEL     = "ideogram/ideogram-v3/text-to-image"


class ImageGenService:
    def __init__(self, api_key: str = "", atlas_api_key: str = "", static_dir: str = ""):
        # api_key kept for backwards-compatible instantiation; not used.
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

    # ── Atlas backends ────────────────────────────────────────────────────

    async def _generate_via_flux(self, prompt: str, aspect_ratio: str) -> str:
        atlas = self._require_atlas()
        output_url = await atlas.generate_image(
            _FLUX_MODEL,
            {
                "prompt": prompt,
                "aspect_ratio": aspect_ratio,
                "num_inference_steps": 28,
                "output_format": "jpeg",
            },
        )
        return await atlas.download(output_url, ext="jpg")

    async def _generate_via_kontext(self, prompt: str, aspect_ratio: str) -> str:
        atlas = self._require_atlas()
        output_url = await atlas.generate_image(
            _FLUX_KONTEXT_MODEL,
            {
                "prompt": prompt,
                "aspect_ratio": aspect_ratio,
                "num_inference_steps": 28,
                "output_format": "jpeg",
            },
        )
        return await atlas.download(output_url, ext="jpg")

    async def _generate_via_ideogram(self, prompt: str, aspect_ratio: str) -> str:
        atlas = self._require_atlas()
        output_url = await atlas.generate_image(
            _IDEOGRAM_MODEL,
            {
                "prompt": prompt,
                "aspect_ratio": aspect_ratio,
                "style_type": "REALISTIC",
            },
        )
        return await atlas.download(output_url, ext="jpg")

    # ── Public API ────────────────────────────────────────────────────────

    async def generate(
        self,
        prompt: str,
        model: str = _FLUX_MODEL,
        video_model: str = "seedance",
        notify=None,
    ) -> str:
        """Generate one image. Falls back to FLUX 2 Pro on any error."""
        aspect_ratio = self._aspect_for_video_model(video_model)
        logger.info(f"Image generation | model={model} | aspect={aspect_ratio} | prompt={prompt[:80]}…")

        try:
            if model == _FLUX_KONTEXT_MODEL:
                return await self._generate_via_kontext(prompt, aspect_ratio)
            if model == _IDEOGRAM_MODEL:
                return await self._generate_via_ideogram(prompt, aspect_ratio)
            return await self._generate_via_flux(prompt, aspect_ratio)
        except Exception as e:
            if model != _FLUX_MODEL:
                logger.warning(f"Image model {model} failed ({e}); falling back to FLUX 2 Pro…")
                if notify is not None:
                    await notify("⚠️ Image model unavailable, switched to FLUX 2 Pro automatically.")
                return await self._generate_via_flux(prompt, aspect_ratio)
            raise
