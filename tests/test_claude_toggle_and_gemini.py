"""Claude polish switch (settings button + handler) and the Gemini provider:
the new gemini-4-argon model and retries only for temporary errors."""

import asyncio
from types import SimpleNamespace

import pytest
from google.genai import errors as genai_errors
from tenacity import wait_none

from bot.handlers import settings as st
from bot.handlers.common import _explain_provider_error
from bot.keyboards import get_settings_keyboard
from services.db import DBService
from services.llm import GeminiTextProvider, LLMError, LLMService, LLMTransientError
from utils import container
from utils.config import config


def _labels(markup):
    return [(b.text, b.callback_data) for row in markup.inline_keyboard for b in row]


# ── keyboard ────────────────────────────────────────────────────────────────

def test_toggle_button_shows_each_state():
    assert ("✨ Claude polish: OFF", "settings:claude_toggle") in _labels(get_settings_keyboard(False, False))
    assert ("✨ Claude polish: ON ✅", "settings:claude_toggle") in _labels(get_settings_keyboard(False, True))
    assert ("✨ Claude polish: not configured", "noop") in _labels(get_settings_keyboard(False, None))


def test_existing_call_style_still_works():
    labels = [t for t, _ in _labels(get_settings_keyboard(True))]   # old signature: is_t2v only
    assert "🧠 Text model" in labels and "🖼 Image model" not in labels


# ── handler ─────────────────────────────────────────────────────────────────

class FakeMessage:
    def __init__(self):
        self.chat = SimpleNamespace(id=-100)
        self.markups = []

    async def edit_reply_markup(self, reply_markup=None):
        self.markups.append(reply_markup)


class FakeCallback:
    def __init__(self):
        self.from_user = SimpleNamespace(id=42)
        self.message = FakeMessage()
        self.toasts = []
        self.alerts = []

    async def answer(self, text=None, show_alert=False):
        (self.alerts if show_alert else self.toasts).append(text)


class FakeDB:
    def __init__(self, data=None):
        self.data = dict(data or {})

    async def get_settings(self, user_id, chat_id):
        return dict(self.data)

    async def update_settings(self, user_id, chat_id, values):
        self.data.update(values)
        return dict(self.data)


def test_toggle_flips_the_saved_flag_and_refreshes_the_button(monkeypatch):
    db = FakeDB({"video_model": "kling"})
    monkeypatch.setitem(container.container, DBService, db)
    monkeypatch.setitem(config, "ANTHROPIC_API_KEY", "sk-ant-test")
    cb = FakeCallback()

    asyncio.run(st.settings_claude_toggle(cb))
    assert db.data["claude_enrichment"] is True
    assert ("✨ Claude polish: ON ✅", "settings:claude_toggle") in _labels(cb.message.markups[-1])

    asyncio.run(st.settings_claude_toggle(cb))
    assert db.data["claude_enrichment"] is False
    assert ("✨ Claude polish: OFF", "settings:claude_toggle") in _labels(cb.message.markups[-1])


def test_toggle_is_refused_without_a_key(monkeypatch):
    db = FakeDB()
    monkeypatch.setitem(container.container, DBService, db)
    monkeypatch.setitem(config, "ANTHROPIC_API_KEY", None)
    cb = FakeCallback()
    asyncio.run(st.settings_claude_toggle(cb))
    assert "claude_enrichment" not in db.data
    assert cb.alerts


def test_settings_page_reflects_saved_state(monkeypatch):
    monkeypatch.setitem(container.container, DBService, FakeDB({"claude_enrichment": True}))
    monkeypatch.setitem(config, "ANTHROPIC_API_KEY", "sk-ant-test")
    kb = asyncio.run(st._settings_keyboard(42, -100))
    assert ("✨ Claude polish: ON ✅", "settings:claude_toggle") in _labels(kb)


# ── gemini-4-argon in the catalogue ─────────────────────────────────────────

def test_gemini_4_argon_is_selectable_and_routes_to_gemini():
    router = LLMService(providers=[GeminiTextProvider(api_key="k")], default_model="gemini-3.5-flash")
    assert "gemini-4-argon" in [m["name"] for m in router.get_text_models()]
    assert router._resolve_model("gemini-4-argon") == ("gemini", "gemini-4-argon")


# ── Gemini retries: temporary errors only ───────────────────────────────────

def _client_error(code, message):
    return genai_errors.ClientError(code, {"error": {"code": code, "message": message, "status": "X"}})


def _server_error(code):
    return genai_errors.ServerError(code, {"error": {"code": code, "message": "busy", "status": "UNAVAILABLE"}})


def _provider(outcomes, monkeypatch):
    calls = []

    def generate_content(model, contents, config):
        calls.append(model)
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(text=outcome)

    provider = GeminiTextProvider(api_key="k")
    provider._client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    monkeypatch.setattr(GeminiTextProvider.generate.retry, "wait", wait_none())
    return provider, calls


def test_no_quota_error_fails_fast_without_retries(monkeypatch):
    err = _client_error(
        429, "Quota exceeded for metric: generate_content_free_tier_requests, limit: 0, model: gemini-4-argon",
    )
    provider, calls = _provider([err], monkeypatch)
    with pytest.raises(LLMError) as exc:
        asyncio.run(provider.generate("p", "", "gemini-4-argon"))
    assert not isinstance(exc.value, LLMTransientError)
    assert len(calls) == 1


def test_unknown_model_404_fails_fast(monkeypatch):
    provider, calls = _provider([_client_error(404, "models/x is not found")], monkeypatch)
    with pytest.raises(LLMError):
        asyncio.run(provider.generate("p", "", "x"))
    assert len(calls) == 1


def test_server_error_is_retried_then_succeeds(monkeypatch):
    provider, calls = _provider([_server_error(503), _server_error(503), "hello"], monkeypatch)
    assert asyncio.run(provider.generate("p", "", "gemini-3.5-flash")) == "hello"
    assert len(calls) == 3


def test_rate_limit_with_quota_is_retried(monkeypatch):
    provider, calls = _provider([_client_error(429, "Too many requests, retry later"), "ok"], monkeypatch)
    assert asyncio.run(provider.generate("p", "", "gemini-3.5-flash")) == "ok"
    assert len(calls) == 2


def test_retries_stop_after_three_attempts(monkeypatch):
    provider, calls = _provider([_server_error(503)] * 3, monkeypatch)
    with pytest.raises(LLMTransientError):
        asyncio.run(provider.generate("p", "", "gemini-3.5-flash"))
    assert len(calls) == 3


# ── hint wording per kind of model ──────────────────────────────────────────

def test_free_tier_hint_for_text_model_points_to_text_settings():
    err = "429 RESOURCE_EXHAUSTED. Quota exceeded for metric: x_free_tier_requests, limit: 0, model: gemini-4-argon"
    hint = _explain_provider_error(Exception(err))
    assert "gemini-4-argon" in hint and "Text model" in hint and "Image model" not in hint
