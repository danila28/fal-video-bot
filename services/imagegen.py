"""Photo generation for video pipelines.

Backends chosen via `model` parameter:

  Vertex AI (Gemini image models):
  • gemini-3.1-flash-image  — Nano Banana 2  (fast, affordable)
  • gemini-3-pro-image       — Nano Banana Pro (highest quality)

  Atlas Cloud:
  • black-forest-labs/flux-2-pro/text-to-image       — FLUX 2 Pro (universal)
  • black-forest-labs/flux-kontext-pro-text-to-image — FLUX Kontext (character consistency)
  • ideogram/ideogram-v3/text-to-image               — Ideogram V3 (stylised art)

All models produce 9:16 vertical output.
On failure, automatically falls back to FLUX 2 Pro (Atlas).
"""

import asyncio
import logging
import os
import uuid

from google import genai
from google.genai import types

from services.atlas import AtlasClient

logger = logging.getLogger(__name__)

_FLUX_MODEL         = "black-forest-labs/flux-2-pro/text-to-image"
_FLUX_KONTEXT_MODEL = "black-forest-labs/flux-kontext-pro-text-to-image"
_IDEOGRAM_MODEL     = "ideogram/ideogram-v3/text-to-image"

_VERTEX_IMAGE_MODELS = {"gemini-3.1-flash-image", "gemini-3-pro-image"}


class ImageGenService:
    def __init__(self, api_key: str = "", atlas_api_key: str = "", static_dir: str = ""):
        self._atlas = AtlasClient(atlas_api_key, static_dir) if atlas_api_key else None
        # Vertex client — same API key, vertexai=True routes to Google Cloud
        self._vertex = genai.Client(api_key=api_key, vertexai=True) if api_key else None
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

    def _require_vertex(self) -> genai.Client:
        if self._vertex is None:
            raise RuntimeError("GEMINI_API_KEY is required for Vertex image generation")
        return self._vertex

    # ── Vertex backends ───────────────────────────────────────────────────

    async def _generate_via_vertex(self, prompt: str, model: str) -> str:
        client = self._require_vertex()
        os.makedirs(self.static_dir, exist_ok=True)

        contents = [types.Content(role="user", parts=[types.Part(text=prompt)])]
        config = types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=types.ImageConfig(aspect_ratio="9:16", output_mime_type="image/png"),
        )

        response = await asyncio.to_thread(
            client.models.generate_content,
            model=model,
            contents=contents,
            config=config,
        )

        if getattr(response, "prompt_feedback", None) is not None:
            raise RuntimeError(f"Vertex image blocked: {response.prompt_feedback}")

        for cand in getattr(response, "candidates", []) or []:
            for part in getattr(getattr(cand, "content", None), "parts", []) or []:
                inline = getattr(part, "inline_data", None)
                if inline and getattr(inline, "data", None):
                    path = os.path.join(self.static_dir, f"{uuid.uuid4()}.png")
                    with open(path, "wb") as f:
                        f.write(inline.data)
                    logger.info(f"Vertex image saved: {path}")
                    return path

        raise RuntimeError(f"Vertex model {model} returned no image data")

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

    async def generate_many(
        self,
        prompt: str,
        model: str = _FLUX_MODEL,
        video_model: str = "seedance",
        count: int = 1,
        notify=None,
    ) -> list[str]:
        """Generate `count` images in parallel. Returns list of local file paths."""
        count = max(1, min(4, count))
        tasks = [self.generate(prompt, model, video_model, notify) for _ in range(count)]
        return list(await asyncio.gather(*tasks))

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
            if model in _VERTEX_IMAGE_MODELS:
                return await self._generate_via_vertex(prompt, model)
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
