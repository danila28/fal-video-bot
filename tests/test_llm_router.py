"""Tests for LLM service router."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from services.llm import LLMService, OpenAITextProvider, GeminiTextProvider
from services.llm import LLMError, LLMEmptyResponse, LLMRefusal


class MockProvider:
    """Mock LLM provider for testing."""

    def __init__(self, name: str, catalog: list = None):
        self.name = name
        self._catalog = catalog or [
            {"name": "test-model-1", "price": "cheap", "vision": False},
            {"name": "test-model-2", "price": "expensive", "vision": True},
        ]

    def catalog(self) -> list:
        return self._catalog.copy()

    async def generate(self, prompt: str, system_prompt: str = "", model: str = "", image_paths=None) -> str:
        return f"Response from {self.name}"


def test_router_init():
    """Test LLMService initialization."""
    provider1 = MockProvider("provider1")
    provider2 = MockProvider("provider2")

    router = LLMService(providers=[provider1, provider2], default_model="test-model-1")
    assert router.default_model == "test-model-1"
    assert "provider1" in router.providers_map
    assert "provider2" in router.providers_map


def test_router_init_no_providers():
    """Test that LLMService requires at least one provider."""
    with pytest.raises(RuntimeError, match="At least one LLM provider"):
        LLMService(providers=[])


def test_get_text_models():
    """Test that router collects models from all providers."""
    provider1 = MockProvider("provider1", [
        {"name": "model-1a", "price": "a", "vision": False},
        {"name": "model-1b", "price": "b", "vision": True},
    ])
    provider2 = MockProvider("provider2", [
        {"name": "model-2a", "price": "c", "vision": False},
    ])

    router = LLMService(providers=[provider1, provider2])
    models = router.get_text_models()

    assert len(models) == 3
    model_names = {m["name"] for m in models}
    assert model_names == {"model-1a", "model-1b", "model-2a"}


def test_resolve_model_empty():
    """Test that empty model name uses default."""
    provider = MockProvider("test")
    router = LLMService(providers=[provider], default_model="test-model-1")

    provider_name, model = router._resolve_model("")
    assert model == "test-model-1"
    assert provider_name == "test"


def test_resolve_model_unknown():
    """Test that unknown model name uses default (with warning)."""
    provider = MockProvider("test")
    router = LLMService(providers=[provider], default_model="test-model-1")

    with patch("services.llm.logger.warning") as mock_warn:
        provider_name, model = router._resolve_model("unknown-model")
        assert model == "test-model-1"
        mock_warn.assert_called_once()


@pytest.mark.asyncio
async def test_generate_text_routing():
    """Test that generate_text routes to correct provider."""
    provider1 = AsyncMock(spec=MockProvider)
    provider1.name = "provider1"
    provider1.catalog.return_value = [
        {"name": "model-1", "price": "a", "vision": False},
    ]
    provider1.generate.return_value = "Response from provider1"

    router = LLMService(providers=[provider1])

    result = await router.generate_text("test prompt", model="model-1")
    assert result == "Response from provider1"
    provider1.generate.assert_called_once()


@pytest.mark.asyncio
async def test_generate_text_vision_fallback():
    """Test fallback to vision-capable model when images provided."""
    provider = MockProvider("test", [
        {"name": "no-vision", "price": "a", "vision": False},
        {"name": "has-vision", "price": "b", "vision": True},
    ])

    # Mock the generate method
    provider.generate = AsyncMock(return_value="response")

    router = LLMService(providers=[provider])

    result = await router.generate_text(
        "test prompt",
        model="no-vision",
        image_paths=["/fake/path.jpg"],
    )

    # Should have called with "has-vision" model, not "no-vision"
    call_args = provider.generate.call_args
    assert call_args[1]["model"] == "has-vision"
