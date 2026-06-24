"""Photo generation for video pipelines.

Backends chosen via `model` parameter:

  Vertex AI (api_key + vertexai=True — same key as VERTEX_API_KEY / GEMINI_API_KEY):
  • gemini-2.0-flash-preview-image-generation  — fast, affordable
  • imagen-3.0-generate-001                    — Imagen 3, highest quality

  Atlas Cloud (atlas_api_key):
  • black-forest-labs/flux-2-pro/text-to-image       — FLUX 2 Pro (universal)
  • black-forest-labs/flux-kontext-pro-text-to-image — FLUX Kontext (character consistency)
  • ideogram/ideogram-v3/text-to-image               — Ideogram V3 (stylised art)

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

_FLUX_MODEL         = "black-forest-labs/flux-2-pro/text-to-image"
_FLUX_KONTEXT_MODEL = "black-forest-labs/flux-kontext-pro-text-to-image"
_IDEOGRAM_MODEL     = "ideogram/ideogram-v3/text-to-image"

# Gemini image generation via generate_content + IMAGE modality
_GEMINI_IMAGE_MODELS = {"gemini-2.0-flash-preview-image-generation"}
# Imagen 3 via generate_images()
_IMAGEN_MODELS = {"imagen-3.0-generate-001"}


class ImageGenService:
    def __init__(self, api_key: str = "", atlas_api_key: str = "", static_dir: str = ""):
        self._atlas = AtlasClient(atlas_api_key, static_dir) if atlas_api_key else None
        # Vertex AI client — same pattern as video-gen-bot's VertexService.client.
        # Routes Gemini image models and Imagen 3 through the Vertex AI endpoint.
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
            raise RuntimeError("ATLAS_API_KEY is required for Atlas image generation")
        return self._atlas

    def _require_vertex(self) -> genai.Client:
        if self._vertex is None:
            raise RuntimeError("GEMINI_API_KEY / VERTEX_API_KEY is required for Google image generation")
        return self._vertex

    # ── Google / Vertex backends ──────────────────────────────────────────

    async def _generate_via_gemini(self, prompt: str, model: str) -> str:
        """Gemini image generation via generate_content + IMAGE modality (Vertex AI)."""
        client = self._require_vertex()
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

    async def _generate_via_imagen(self, prompt: str, model: str) -> str:
        """Imagen 3 via generate_images() (Vertex AI)."""
        client = self._require_vertex()
        os.makedirs(self.static_dir, exist_ok=True)

        config = types.GenerateImagesConfig(
            number_of_images=1,
            aspect_ratio="9:16",
            output_mime_type="image/png",
        )
        response = await asyncio.to_thread(
            client.models.generate_images,
            model=model,
            prompt=prompt,
            config=config,
        )
        for img in getattr(response, "generated_images", []) or []:
            image_bytes = getattr(getattr(img, "image", None), "image_bytes", None)
            if image_bytes:
                path = os.path.join(self.static_dir, f"{uuid.uuid4()}.png")
                with open(path, "wb") as f:
                    f.write(image_bytes)
                logger.info(f"Imagen image saved: {path}")
                return path
        raise RuntimeError(f"Imagen model {model} returned no image data")

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

    _DEFAULT_MODEL = "gemini-2.0-flash-preview-image-generation"

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

        if model in _GEMINI_IMAGE_MODELS:
            return await self._generate_via_gemini(prompt, model)
        if model in _IMAGEN_MODELS:
            return await self._generate_via_imagen(prompt, model)
        if model == _FLUX_MODEL:
            return await self._generate_via_flux(prompt, aspect_ratio)
        if model == _FLUX_KONTEXT_MODEL:
            return await self._generate_via_kontext(prompt, aspect_ratio)
        if model == _IDEOGRAM_MODEL:
            return await self._generate_via_ideogram(prompt, aspect_ratio)
        raise ValueError(f"Unknown image model '{model}'. Go to ⚙️ Settings → 🖼 Image model and reselect.")
