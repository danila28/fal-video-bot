"""Photo generation for video pipelines.

Backends:
  Atlas Cloud (atlas_api_key):
  • google/nano-banana-2/text-to-image   — Nano Banana 2 (fast, affordable)
  • google/nano-banana-pro/text-to-image — Nano Banana Pro (highest quality)

  Gemini Developer API (api_key):
  • gemini-3.1-flash-image — Nano Banana 2 (fast, affordable)
  • gemini-3-pro-image     — Nano Banana Pro (highest quality)

All models produce 9:16 vertical output.
"""

import asyncio
import logging
import os
import uuid

from google import genai
from google.genai import types

from services.atlas import AtlasClient

logger = logging.getLogger(__name__)

# ── Atlas models ──────────────────────────────────────────────────────────
_ATLAS_NANO_BANANA_2 = "google/nano-banana-2/text-to-image"
_ATLAS_NANO_BANANA_PRO = "google/nano-banana-pro/text-to-image"
_ATLAS_MODELS = {_ATLAS_NANO_BANANA_2, _ATLAS_NANO_BANANA_PRO}

# ── Gemini Developer API models ───────────────────────────────────────────
_GEMINI_NANO_BANANA_2 = "gemini-3.1-flash-image"
_GEMINI_NANO_BANANA_PRO = "gemini-3-pro-image"
_GEMINI_MODELS = {_GEMINI_NANO_BANANA_2, _GEMINI_NANO_BANANA_PRO}

_ALL_MODELS = _ATLAS_MODELS | _GEMINI_MODELS


class ImageGenService:
    def __init__(self, api_key: str = "", atlas_api_key: str = "", static_dir: str = ""):
        self._atlas = AtlasClient(atlas_api_key, static_dir) if atlas_api_key else None
        # Gemini Developer API client (standard API key, no vertexai=True)
        self._gemini = genai.Client(api_key=api_key) if api_key else None
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
            raise RuntimeError("ATLAS_API_KEY is required for Atlas image generation")
        return self._atlas

    def _require_gemini(self) -> genai.Client:
        if self._gemini is None:
            raise RuntimeError("GEMINI_API_KEY is required for Gemini image generation")
        return self._gemini

    # ── Atlas image generation ────────────────────────────────────────────

    async def _generate_via_atlas(self, prompt: str, model: str, aspect_ratio: str) -> str:
        """Generate image via Atlas Cloud Nano Banana models."""
        atlas = self._require_atlas()
        output_url = await atlas.generate_image(
            model,
            {
                "prompt": prompt,
                "aspect_ratio": aspect_ratio,
            },
        )
        return await atlas.download(output_url, ext="jpg")

    # ── Gemini Developer API image generation ─────────────────────────────

    async def _generate_via_gemini(self, prompt: str, model: str) -> str:
        """Generate image via Gemini Developer API Nano Banana models."""
        client = self._require_gemini()
        os.makedirs(self.static_dir, exist_ok=True)

        contents = [types.Content(role="user", parts=[types.Part(text=prompt)])]
        config = types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=types.ImageConfig(aspect_ratio="9:16"),
        )

        response = await asyncio.to_thread(
            client.models.generate_content,
            model=model,
            contents=contents,
            config=config,
        )

        pf = getattr(response, "prompt_feedback", None)
        if pf and getattr(pf, "block_reason", None):
            raise RuntimeError(f"Gemini image blocked: {pf.block_reason}")

        for cand in getattr(response, "candidates", []) or []:
            for part in getattr(getattr(cand, "content", None), "parts", []) or []:
                inline = getattr(part, "inline_data", None)
                if inline and getattr(inline, "data", None):
                    path = os.path.join(self.static_dir, f"{uuid.uuid4()}.png")
                    with open(path, "wb") as f:
                        f.write(inline.data)
                    logger.info(f"Gemini image saved: {path}")
                    return path

        raise RuntimeError(f"Gemini model {model} returned no image data")

    # ── Public API ────────────────────────────────────────────────────────

    _DEFAULT_MODEL = _ATLAS_NANO_BANANA_2

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

    async def generate_from_prompts(
        self,
        prompts: list[str],
        model: str = "",
        video_model: str = "seedance",
        notify=None,
    ) -> list[str]:
        """Generate different images from different prompts in parallel.

        Returns list of local file paths (same order as input prompts).
        """
        if not prompts:
            return []
        prompts = prompts[:4]  # Max 4 images
        model = model or self._DEFAULT_MODEL
        tasks = [self.generate(p, model, video_model, notify) for p in prompts]
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

        if model in _ATLAS_MODELS:
            return await self._generate_via_atlas(prompt, model, aspect_ratio)
        if model in _GEMINI_MODELS:
            return await self._generate_via_gemini(prompt, model)
        raise ValueError(f"Unknown image model '{model}'. Go to ⚙️ Settings → 🖼 Image model and reselect.")
