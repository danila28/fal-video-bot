"""Photo generation for video pipelines.

Backends — chosen via `model` parameter:
  • imagen-4.0-fast-generate-001        — Google Imagen 4 Fast (Gemini API)
  • gemini-2.5-flash-image              — Gemini image (multimodal, fast, cheap)
  • black-forest-labs/flux-2-pro/text-to-image    — FLUX 2 Pro (fallback)
  • black-forest-labs/flux-kontext-pro-text-to-image — FLUX Kontext: character consistency
  • ideogram/ideogram-v3/text-to-image  — Ideogram V3: stylised characters, creative art

If the primary backend fails with a server-side 5xx, the call automatically
falls back to FLUX 2 Pro so the pipeline does not break on transient outages.

All scene models use 9:16 vertical output.
"""

import asyncio
import logging
import os
import uuid

import aiohttp
from google import genai
from google.genai import types
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

from services.atlas import AtlasClient

logger = logging.getLogger(__name__)

_IMAGEN_MODEL_DEFAULT  = "imagen-4.0-fast-generate-001"
_GEMINI_IMAGE_MODEL    = "gemini-2.5-flash-image"
_FLUX_MODEL            = "black-forest-labs/flux-2-pro/text-to-image"
_FLUX_KONTEXT_MODEL    = "black-forest-labs/flux-kontext-pro-text-to-image"
_IDEOGRAM_MODEL        = "ideogram/ideogram-v3/text-to-image"


def _is_server_error(exc: BaseException) -> bool:
    msg = str(exc)
    return "503" in msg or "502" in msg or "500" in msg or "UNAVAILABLE" in msg


_imagen_retry = retry(
    retry=retry_if_exception(_is_server_error),
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=3, max=10),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING),
)


class ImageGenService:
    def __init__(self, api_key: str, atlas_api_key: str = "", static_dir: str = ""):
        self._gemini = genai.Client(api_key=api_key)
        self._atlas = AtlasClient(atlas_api_key, static_dir) if atlas_api_key else None
        if not static_dir:
            static_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static"
            )
        self.static_dir = static_dir
        os.makedirs(static_dir, exist_ok=True)

    # ── Aspect ratio helper ───────────────────────────────────────────────

    @staticmethod
    def _aspect_for_video_model(video_model: str) -> str:
        """9:16 vertical for all scene models (Kling, Seedance, PixVerse)."""
        return "9:16"

    # ── Google Imagen ─────────────────────────────────────────────────────

    @_imagen_retry
    def _call_imagen(self, prompt: str, model: str, aspect_ratio: str):
        return self._gemini.models.generate_images(
            model=model,
            prompt=prompt,
            config=types.GenerateImagesConfig(
                number_of_images=1,
                aspect_ratio=aspect_ratio,
                safety_filter_level="BLOCK_LOW_AND_ABOVE",
                person_generation="ALLOW_ADULT",
            ),
        )

    async def _generate_via_imagen(self, prompt: str, model: str, aspect_ratio: str) -> str:
        result = await asyncio.to_thread(self._call_imagen, prompt, model, aspect_ratio)
        images = result.generated_images
        if not images:
            raise RuntimeError("Imagen returned no images")
        image_bytes = images[0].image.image_bytes
        filename = f"{uuid.uuid4()}.jpg"
        dest = os.path.join(self.static_dir, filename)
        with open(dest, "wb") as f:
            f.write(image_bytes)
        logger.info(f"Imagen → {dest}")
        return dest

    # ── Gemini multimodal image ───────────────────────────────────────────

    async def _generate_via_gemini_image(self, prompt: str, aspect_ratio: str) -> str:
        contents = [types.Content(role="user", parts=[types.Part(text=prompt)])]
        config = types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=types.ImageConfig(
                aspect_ratio=aspect_ratio, output_mime_type="image/png"
            ),
        )
        response = await asyncio.to_thread(
            self._gemini.models.generate_content,
            model=_GEMINI_IMAGE_MODEL,
            contents=contents,
            config=config,
        )

        if response.prompt_feedback is not None:
            raise Exception(f"Prompt blocked: {response.prompt_feedback}")

        candidates = getattr(response, "candidates", None) or []
        for cand in candidates:
            content = getattr(cand, "content", None)
            parts = getattr(content, "parts", None) or []
            for part in parts:
                inline = getattr(part, "inline_data", None)
                if inline and getattr(inline, "data", None):
                    filename = f"{uuid.uuid4()}.png"
                    dest = os.path.join(self.static_dir, filename)
                    with open(dest, "wb") as f:
                        f.write(inline.data)
                    logger.info(f"Gemini image → {dest}")
                    return dest

        finish_reasons = [getattr(c, "finish_reason", None) for c in candidates]
        raise Exception(f"Gemini image returned no data. finish_reasons={finish_reasons}")

    # ── Atlas Cloud FLUX 2 Pro ────────────────────────────────────────────

    def _require_atlas(self) -> AtlasClient:
        if self._atlas is None:
            raise RuntimeError(
                "ATLAS_API_KEY is required for FLUX / Ideogram image generation"
            )
        return self._atlas

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

    # ── Atlas Cloud FLUX Kontext Pro ──────────────────────────────────────

    async def _generate_via_kontext(self, prompt: str, aspect_ratio: str) -> str:
        """FLUX Kontext Pro: text-to-image with strong character consistency."""
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

    # ── Atlas Cloud Ideogram V3 ───────────────────────────────────────────

    async def _generate_via_ideogram(self, prompt: str, aspect_ratio: str) -> str:
        """Ideogram V3: stylised characters, creative art, strong text rendering."""
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
        model: str = _IMAGEN_MODEL_DEFAULT,
        video_model: str = "seedance",
    ) -> str:
        """Generate ONE photo from a prompt.

        `model` picks the primary backend; FLUX 2 Pro is used as automatic
        fallback when the primary backend fails with a server error.
        `video_model` selects the output aspect ratio.
        """
        aspect_ratio = self._aspect_for_video_model(video_model)
        logger.info(
            f"Image generation start | model={model} | "
            f"video_model={video_model} | aspect={aspect_ratio} | "
            f"prompt={prompt[:80]}…"
        )

        try:
            if model.startswith("imagen-"):
                return await self._generate_via_imagen(prompt, model, aspect_ratio)
            if model == _GEMINI_IMAGE_MODEL:
                return await self._generate_via_gemini_image(prompt, aspect_ratio)
            if model == _FLUX_KONTEXT_MODEL:
                return await self._generate_via_kontext(prompt, aspect_ratio)
            if model == _IDEOGRAM_MODEL:
                return await self._generate_via_ideogram(prompt, aspect_ratio)
            if model == _FLUX_MODEL or model.startswith("black-forest-labs/"):
                return await self._generate_via_flux(prompt, aspect_ratio)
            # Unknown model — default to Imagen Fast
            return await self._generate_via_imagen(prompt, _IMAGEN_MODEL_DEFAULT, aspect_ratio)
        except Exception as e:
            logger.warning(f"Primary image backend failed ({e}); falling back to FLUX 2 Pro…")
            return await self._generate_via_flux(prompt, aspect_ratio)
