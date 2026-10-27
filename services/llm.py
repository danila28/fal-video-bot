"""LLM service: multi-provider router for text generation.

Supports OpenAI (luna, terra, sol) and Gemini providers.
Handles images, retries, and graceful degradation.
"""

import anthropic
import asyncio
import base64
import logging
import os
from typing import Protocol

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from openai import AsyncOpenAI, RateLimitError, APIConnectionError, APITimeoutError, InternalServerError, BadRequestError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
    retry_if_exception,
    retry_if_exception_type,
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


class LLMTransientError(LLMError):
    """A failure worth retrying: 5xx, network trouble, or a rate limit that has quota."""


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

def _is_transient_gemini_error(exc: Exception) -> bool:
    """5xx and network errors are retried; so is a 429 that still has quota.
    A 429 with "limit: 0" means this key's tier has no quota for the model, so
    retrying only makes the user wait. 4xx (404 unknown model, 400, 403) never retry."""
    code = getattr(exc, "code", None)
    text = str(exc).lower()
    if isinstance(exc, genai_errors.ServerError) or code in (500, 502, 503, 504):
        return True
    if code == 429:
        return "limit: 0" not in text
    return isinstance(exc, (ConnectionError, TimeoutError, asyncio.TimeoutError))


_retry_gemini = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(LLMTransientError),
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
        # The API id is "gemini-3.1-pro-preview"; plain "gemini-3.1-pro" returns 404.
        {"name": "gemini-3.1-pro-preview", "price": "🏆 ~$1.25/1M tok", "vision": True},
        # Listed by the Gemini API for our key. Needs a paid Google tier (free-tier
        # quota is 0). Price unknown; vision support not yet verified.
        {"name": "gemini-4-argon", "price": "🧪 new · paid Google tier · price unverified", "vision": True},
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

    @_retry_gemini
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
            error_cls = LLMTransientError if _is_transient_gemini_error(e) else LLMError
            raise error_cls(f"Failed to generate text via Gemini: {str(e)}")


# ── Anthropic Provider ─────────────────────────────────────────────────────

def _should_retry_anthropic(exc: Exception) -> bool:
    """Retry only temporary failures (rate limit, network, 5xx), never 4xx."""
    return isinstance(exc, (
        anthropic.RateLimitError,
        anthropic.APIConnectionError,  # includes APITimeoutError
        anthropic.InternalServerError,
    ))


_retry_anthropic = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception(_should_retry_anthropic),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING),
)

_IMAGE_MEDIA_TYPES = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".png": "image/png", ".webp": "image/webp",
}


class AnthropicTextProvider:
    """Claude text provider. Used for the final script review (Gemini drafts,
    Claude polishes). Only registered when ANTHROPIC_API_KEY is set."""

    name = "anthropic"
    # Claude is used only for the final script review, so it is kept out of the
    # text-model menu: choosing it there makes every plot/hashtag call a paid
    # Claude call. Saved choices still resolve (see LLMService._resolve_model).
    in_menu = False

    _CATALOG = [
        {"name": "claude-opus-5-5", "price": "🧠 Claude Opus 5.5 · price unverified", "vision": True},
    ]

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._client = None

    @property
    def client(self) -> anthropic.AsyncAnthropic:
        """Lazy client. max_retries=0: retries are handled by tenacity above."""
        if self._client is None:
            self._client = anthropic.AsyncAnthropic(api_key=self.api_key, max_retries=0, timeout=60)
        return self._client

    def catalog(self) -> list[dict]:
        return self._CATALOG.copy()

    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        model: str = "",
        image_paths: list[str] | None = None,
    ) -> str:
        text, _usage = await self.generate_with_usage(prompt, system_prompt, model, image_paths)
        return text

    async def generate_with_usage(
        self,
        prompt: str,
        system_prompt: str = "",
        model: str = "",
        image_paths: list[str] | None = None,
    ) -> tuple[str, dict]:
        """Like generate(), but also returns {'input_tokens', 'output_tokens'}
        so the caller can write them to the cost ledger."""
        model = model or "claude-opus-5-5"
        return await self._call(model, prompt, system_prompt, image_paths)

    @_retry_anthropic
    async def _call(self, model: str, prompt: str, system_prompt: str, image_paths) -> tuple[str, dict]:
        content: list[dict] = []
        for path in image_paths or []:
            media_type = _IMAGE_MEDIA_TYPES.get(os.path.splitext(path)[1].lower(), "image/jpeg")
            with open(path, "rb") as f:
                data = base64.b64encode(f.read()).decode("utf-8")
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": data},
            })
        content.append({"type": "text", "text": prompt})

        kwargs = {
            "model": model,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": content}],
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        try:
            response = await self.client.messages.create(**kwargs)
        except (anthropic.RateLimitError, anthropic.APIConnectionError,
                anthropic.APITimeoutError, anthropic.InternalServerError):
            raise  # transient: tenacity retries
        except anthropic.APIError as e:
            logger.error(f"Anthropic error (model={model}): {e}")
            raise LLMError(f"Anthropic request failed: {e}")

        if getattr(response, "stop_reason", None) == "refusal":
            raise LLMRefusal(model, reason="model refused")

        text = "".join(
            block.text for block in (response.content or [])
            if getattr(block, "type", "") == "text"
        )
        if not text.strip():
            raise LLMEmptyResponse(model, reason=f"empty content (stop_reason={response.stop_reason})")

        usage = {
            "input_tokens": getattr(response.usage, "input_tokens", None),
            "output_tokens": getattr(response.usage, "output_tokens", None),
        }
        return text, usage


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

    # Names that were saved in user settings under an id the API does not accept.
    _MODEL_ALIASES = {"gemini-3.1-pro": "gemini-3.1-pro-preview"}

    def get_text_models(self) -> list[dict]:
        """Menu entries from every provider that wants to be listed (Claude, which
        is used only for the final script review, is not)."""
        models = []
        for provider in self.providers_map.values():
            if getattr(provider, "in_menu", True) is False:
                continue
            models.extend(provider.catalog())
        return models

    def _find_model(self, model: str) -> tuple[str, str] | None:
        for provider in self.providers_map.values():
            for cat in provider.catalog():
                if cat["name"] == model:
                    return (provider.name, model)
        return None

    def _resolve_model(self, model: str) -> tuple[str, str]:
        """Resolve a model name to (provider_name, model).

        Empty name -> default model. Old/renamed ids are mapped through
        _MODEL_ALIASES. An unknown name falls back to the default model, and when
        the default's provider is not configured, to the first model of the
        first configured provider, so a stale setting never sends one provider
        another provider's model name.
        """
        model = (model or "").strip() or self.default_model
        model = self._MODEL_ALIASES.get(model, model)

        found = self._find_model(model)
        if found:
            return found

        logger.warning(f"Unknown LLM model '{model}', falling back to '{self.default_model}'")
        found = self._find_model(self.default_model)
        if found:
            return found
        for provider in self.providers_map.values():
            if getattr(provider, "in_menu", True) is False:
                continue  # never fall back to the review-only provider
            catalog = provider.catalog()
            if catalog:
                return (provider.name, catalog[0]["name"])
        raise LLMError("No LLM model is available")

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
