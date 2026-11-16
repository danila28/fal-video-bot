"""gemini-4-hydrogen: selectable text model and the working default."""

import asyncio
import pathlib

from services.llm import GeminiTextProvider, LLMService, OpenAITextProvider


def _router(*providers, default="gemini-4-hydrogen"):
    return LLMService(providers=list(providers), default_model=default)


def test_hydrogen_is_in_the_text_model_menu_with_vision():
    gemini = GeminiTextProvider(api_key="k")
    entry = next(m for m in gemini.catalog() if m["name"] == "gemini-4-hydrogen")
    assert entry["vision"] is True and not entry.get("hidden")
    assert "gemini-4-hydrogen" in [m["name"] for m in _router(gemini).get_text_models()]


def test_hydrogen_routes_to_gemini():
    assert _router(GeminiTextProvider(api_key="k"))._resolve_model("gemini-4-hydrogen") == (
        "gemini", "gemini-4-hydrogen",
    )


def test_empty_or_unknown_saved_model_uses_hydrogen():
    router = _router(OpenAITextProvider(api_key="k"), GeminiTextProvider(api_key="k"))
    assert router._resolve_model("") == ("gemini", "gemini-4-hydrogen")
    assert router._resolve_model("a-model-that-was-removed") == ("gemini", "gemini-4-hydrogen")


def test_without_a_gemini_key_the_router_still_resolves_something():
    router = _router(OpenAITextProvider(api_key="k"))
    provider, model = router._resolve_model("")
    assert provider == "openai" and model


def test_the_sent_model_is_hydrogen_when_it_is_selected(monkeypatch):
    sent = {}
    provider = GeminiTextProvider(api_key="k")

    async def fake_generate(prompt, system_prompt="", model="", image_paths=None):
        sent["model"] = model
        return "ok"

    monkeypatch.setattr(provider, "generate", fake_generate)
    asyncio.run(_router(provider).generate_text("p", model="gemini-4-hydrogen"))
    assert sent["model"] == "gemini-4-hydrogen"


def test_main_uses_hydrogen_as_the_default():
    source = pathlib.Path(__file__).resolve().parent.parent.joinpath("main.py").read_text(encoding="utf-8")
    assert 'default_model="gemini-4-hydrogen"' in source and 'default_model="terra"' not in source
