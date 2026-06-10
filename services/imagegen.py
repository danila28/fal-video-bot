"""Photo generation for fal.ai video pipelines.

Three backends — chosen via `model` parameter:
  • imagen-4.0-fast-generate-001  — Google Imagen 4 Fast ("банано", Gemini API)
  • gemini-2.5-flash-image        — Gemini image (multimodal, fast, cheap)
  • fal-ai/flux-pro/v1.1          — fal.ai FLUX Pro 1.1 (third-party)

If the primary backend fails with a server-side 5xx, the call automatically
falls back to FLUX Pro so the pipeline doesn't break on transient outages.

Output shape depends on the downstream video model:
  • kling / omnihuman → 3:4 portrait (lip-sync wants front-facing face)
  • seedance          → 9:16 vertical (scene anchor for vertical clips)
"""

import asyncio
import logging
import os
import uuid

import aiohttp
import fal_client
from google import genai
from google.genai import types
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

logger = logging.getLogger(__name__)

_IMAGEN_MODEL_DEFAULT = "imagen-4.0-fast-generate-001"
_GEMINI_IMAGE_MODEL   = "gemini-2.5-flash-image"
_FLUX_MODEL           = "fal-ai/flux-pro/v1.1"


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
    def __init__(self, api_key: str, fal_api_key: str = "", static_dir: str = ""):
        self._gemini = genai.Client(api_key=api_key)
        if fal_api_key:
            os.environ["FAL_KEY"] = fal_api_key
        # Default to <project>/static so the directory lines up with the rest
        # of the pipeline (TTS / GeminiService / cleanup loop).
        if not static_dir:
            static_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static"
            )
        self.static_dir = static_dir
        os.makedirs(static_dir, exist_ok=True)

    # ── Aspect ratio helper ───────────────────────────────────────────────

    @staticmethod
    def _aspect_for_video_model(video_model: str) -> str:
        """3:4 portrait for lip-sync; 9:16 vertical for scene clips."""
        if video_model in ("kling", "omnihuman"):
            return "3:4"
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

    # ── fal.ai FLUX Pro ───────────────────────────────────────────────────

    async def _generate_via_flux(self, prompt: str, aspect_ratio: str) -> str:
        flux_size = "portrait_4_3" if aspect_ratio == "3:4" else "portrait_16_9"
        result = await asyncio.to_thread(
            fal_client.subscribe,
            _FLUX_MODEL,
            arguments={
                "prompt": prompt,
                "image_size": flux_size,
                "num_inference_steps": 28,
                "safety_tolerance": "2",
                "output_format": "jpeg",
            },
        )
        images = result.get("images") or []
        if not images:
            raise RuntimeError(f"FLUX returned no images")
        url = images[0]["url"]
        return await self._download(url)

    async def _download(self, url: str) -> str:
        filename = f"{uuid.uuid4()}.jpg"
        dest = os.path.join(self.static_dir, filename)
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"Download failed: HTTP {resp.status}")
                with open(dest, "wb") as f:
                    async for chunk in resp.content.iter_chunked(65536):
                        f.write(chunk)
        logger.info(f"FLUX → {dest}")
        return dest

    # ── Public API ────────────────────────────────────────────────────────

    async def generate(
        self,
        prompt: str,
        model: str = _IMAGEN_MODEL_DEFAULT,
        video_model: str = "seedance",
    ) -> str:
        """Generate ONE photo from a prompt.

        `model` picks the primary backend; FLUX Pro is used as automatic
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
            if model.startswith("fal-ai/"):
                return await self._generate_via_flux(prompt, aspect_ratio)
            # Unknown model — default to Imagen Fast
            return await self._generate_via_imagen(prompt, _IMAGEN_MODEL_DEFAULT, aspect_ratio)
        except Exception as e:
            logger.warning(f"Primary image backend failed ({e}); falling back to FLUX Pro…")
            return await self._generate_via_flux(prompt, aspect_ratio)
