"""LLM service: multi-provider router for text generation.

Supports OpenAI (luna, terra, sol) and Gemini providers.
Handles images, retries, and graceful degradation.
"""

import asyncio
import base64
import logging
import os
import re
from typing import Protocol

from google import genai
from google.genai import types
from openai import AsyncOpenAI, RateLimitError, APIConnectionError, APITimeoutError, InternalServerError, BadRequestError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
    retry_if_exception,
)

logger = logging.getLogger(__name__)


# ── Custom Exceptions ──────────────────────────────────────────────────────

class LLMError(Exception):
    """Base LLM error."""
    pass


class LLMEmptyResponse(LLMError):
    """Model returned empty content."""
    def __init__(self, model: str, reason: str = "empty"):
        self.model = model
        self.reason = reason
        super().__init__(f"LLM empty response ({model}): {reason}")


class LLMRefusal(LLMError):
    """Model refused the request."""
    def __init__(self, model: str, reason: str):
        self.model = model
        self.reason = reason
        super().__init__(f"LLM refusal ({model}): {reason}")


# ── Retry logic for OpenAI (temporary errors only) ─────────────────────────

def _should_retry_openai(exc: Exception) -> bool:
    """Retry on temporary errors, not on 4xx (except 429)."""
    return isinstance(exc, (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError))


_retry_openai = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception(_should_retry_openai),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING),
)

_retry_cheap = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING),
)


# ── Provider Protocol ──────────────────────────────────────────────────────

class TextProvider(Protocol):
    """Abstract interface for text generation providers."""
    name: str  # "openai" or "gemini"

    def catalog(self) -> list[dict]:
        """Return list of available models: [{"name": "...", "price": "...", "vision": bool}, ...]"""
        ...

    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        model: str = "",
        image_paths: list[str] | None = None,
    ) -> str:
        """Generate text. Raises LLMEmptyResponse, LLMRefusal, or LLMError on failure."""
        ...


# ── OpenAI Provider ────────────────────────────────────────────────────────

class OpenAITextProvider:
    """OpenAI text generation provider (luna, terra, sol)."""

    name = "openai"

    _CATALOG = [
        {"name": "luna", "price": "💰 ~$0.30/1M tok", "vision": True},
        {"name": "terra", "price": "⚖️  ~$0.75/1M tok", "vision": True},
        {"name": "sol", "price": "🏆 ~$3.00/1M tok", "vision": True},
    ]

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._client = None

    @property
    def client(self) -> AsyncOpenAI:
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            self._client = AsyncOpenAI(api_key=self.api_key, max_retries=0, timeout=30)
        return self._client

    def catalog(self) -> list[dict]:
        return self._CATALOG.copy()

    @_retry_openai
    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        model: str = "",
        image_paths: list[str] | None = None,
    ) -> str:
        model = model or "terra"

        try:
            # Build message content with optional images
            content = []

            # Add images as image_url (base64 data URI)
            if image_paths:
                for path in image_paths:
                    data_uri = self._load_image_as_data_uri(path)
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": data_uri}
                    })

            # Add text prompt
            content.append({"type": "text", "text": prompt})

            # Build messages
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": content if len(content) > 1 else content[0]})

            # Call OpenAI
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
            )

            # Extract text
            if not response.choices or not response.choices[0].message:
                raise LLMEmptyResponse(model, reason="no choices")

            message = response.choices[0].message

            # Check refusal
            if message.refusal:
                raise LLMRefusal(model, reason=message.refusal)

            # Check content
            text = message.content or ""
            if not text or not text.strip():
                raise LLMEmptyResponse(model, reason="empty content")

            return text

        except (LLMEmptyResponse, LLMRefusal):
            raise
        except BadRequestError as e:
            logger.error(f"OpenAI BadRequest (model={model}): {e}")
            raise LLMError(f"Bad request to OpenAI: {str(e)}")
        except Exception as e:
            logger.error(f"OpenAI error (model={model}): {e}")
            raise LLMError(f"Failed to generate text via OpenAI: {str(e)}")

    @staticmethod
    def _load_image_as_data_uri(image_path: str) -> str:
        """Load image and encode as base64 data URI."""
        mime_types = {
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png", ".webp": "image/webp",
        }
        ext = os.path.splitext(image_path)[1].lower()
        mime = mime_types.get(ext, "image/jpeg")

        try:
            with open(image_path, "rb") as f:
                data = base64.b64encode(f.read()).decode("utf-8")
            return f"data:{mime};base64,{data}"
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            raise LLMError(f"Failed to load image: {str(e)}")


# ── Gemini Provider ────────────────────────────────────────────────────────

class GeminiTextProvider:
    """Gemini text generation provider."""

    name = "gemini"

    _CATALOG = [
        {"name": "gemini-3.1-flash-lite", "price": "💰 ~$0.10/1M tok", "vision": True},
        {"name": "gemini-3.5-flash", "price": "⚖️  ~$0.15/1M tok", "vision": True},
        {"name": "gemini-3.1-pro", "price": "🏆 ~$1.25/1M tok", "vision": True},
    ]

    _IMAGE_MIME_BY_EXT = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".webp": "image/webp",
    }

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._client = None

    @property
    def client(self) -> genai.Client:
        """Lazy initialization of Gemini client."""
        if self._client is None:
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def catalog(self) -> list[dict]:
        return self._CATALOG.copy()

    @_retry_cheap
    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        model: str = "",
        image_paths: list[str] | None = None,
    ) -> str:
        model = model or "gemini-3.1-flash-lite"

        try:
            # Build parts (images + text)
            parts = []
            for p in (image_paths or []):
                mime = self._IMAGE_MIME_BY_EXT.get(os.path.splitext(p)[1].lower(), "image/jpeg")
                with open(p, "rb") as f:
                    parts.append(types.Part.from_bytes(data=f.read(), mime_type=mime))
            parts.append(types.Part(text=prompt))

            # Build request
            contents = [types.Content(role="user", parts=parts)]
            config = types.GenerateContentConfig(system_instruction=system_prompt)

            # Call Gemini
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=model,
                contents=contents,
                config=config,
            )

            # Extract text
            text = getattr(response, "text", None)
            if not text or not text.strip():
                candidates = getattr(response, "candidates", None) or []
                finish_reasons = [getattr(c, "finish_reason", None) for c in candidates]
                feedback = getattr(response, "prompt_feedback", None)
                raise LLMEmptyResponse(
                    model,
                    reason=f"finish_reasons={finish_reasons}, prompt_feedback={feedback}"
                )

            return text

        except LLMEmptyResponse:
            raise
        except Exception as e:
            logger.error(f"Gemini error (model={model}): {e}")
            raise LLMError(f"Failed to generate text via Gemini: {str(e)}")


# ── LLM Router ────────────────────────────────────────────────────────────

class LLMService:
    """Multi-provider LLM router.

    Routes requests to OpenAI or Gemini based on model name.
    Automatically selects default model if none specified.
    Handles image vision fallback gracefully.
    """

    def __init__(self, providers: list, default_model: str = "terra"):
        self.providers_map = {p.name: p for p in providers}
        self.default_model = default_model

        if not self.providers_map:
            raise RuntimeError("At least one LLM provider must be configured")

    def get_text_models(self) -> list[dict]:
        """Collect menu from all providers with their catalogs."""
        models = []
        for provider in self.providers_map.values():
            for model in provider.catalog():
                models.append(model)
        return models

    def _resolve_model(self, model: str) -> tuple[str, str]:
        """Resolve model name to (provider_name, actual_model).

        Returns (provider, model) or raises LLMError if model not found.
        Empty model uses default.
        Unknown model logs warning and uses default.
        """
        if not model or not model.strip():
            model = self.default_model

        # Find provider for this model
        for provider in self.providers_map.values():
            for cat in provider.catalog():
                if cat["name"] == model:
                    return (provider.name, model)

        # Unknown model: log warning and use default
        logger.warning(f"Unknown LLM model '{model}', falling back to '{self.default_model}'")
        return (self.providers_map["openai" if "openai" in self.providers_map else list(self.providers_map.keys())[0]].name, self.default_model)

    async def generate_text(
        self,
        prompt: str,
        system_prompt: str = "",
        model: str = "",
        image_paths: list[str] | None = None,
    ) -> str:
        """Generate text with optional images.

        Routes to appropriate provider based on model name.
        Falls back to vision-capable model if needed.
        """
        provider_name, resolved_model = self._resolve_model(model)
        provider = self.providers_map[provider_name]

        # Check if model supports images
        catalog = {m["name"]: m for m in provider.catalog()}
        if image_paths and resolved_model in catalog and not catalog[resolved_model].get("vision", False):
            logger.warning(
                f"Model '{resolved_model}' does not support images. "
                f"Falling back to a vision-capable model."
            )
            # Find first vision-capable model in provider
            for cat in provider.catalog():
                if cat.get("vision", False):
                    resolved_model = cat["name"]
                    break

        return await provider.generate(prompt, system_prompt, resolved_model, image_paths)
