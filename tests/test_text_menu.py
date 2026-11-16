"""Two-level text-model menu (service -> models) and gemini-4-hydrogen end to end.

Only the network client of the Gemini provider is replaced; routing, request
building, the menu handlers and the script pipeline are the real code.
"""

import asyncio
import json
from types import SimpleNamespace

import pytest

from bot.handlers import common, settings as st
from bot.keyboards import get_text_models_keyboard, get_text_service_keyboard
from services.db import DBService
from services.llm import (
    AnthropicTextProvider,
    GeminiTextProvider,
    LLMEmptyResponse,
    LLMService,
    OpenAITextProvider,
)
from utils import container


def _full_router():
    return LLMService(
        providers=[GeminiTextProvider(api_key="k"), OpenAITextProvider(api_key="k"),
                   AnthropicTextProvider(api_key="k")],
        default_model="gemini-4-hydrogen",
    )


def _gemini_only_router():
    return LLMService(providers=[GeminiTextProvider(api_key="k")], default_model="gemini-4-hydrogen")


# ── grouping ────────────────────────────────────────────────────────────────

def test_models_are_grouped_by_provider_without_claude_or_hidden_entries():
    menu = _full_router().get_text_models_by_provider()
    assert list(menu) == ["gemini", "openai"]                       # provider order, no anthropic
    names = [m["name"] for m in menu["gemini"]]
    assert "gemini-4-hydrogen" in names and "gemini-4-argon" in names
    assert "gemini-3.1-pro-preview" not in names                    # hidden for testing
    assert all(not m.get("hidden") for entries in menu.values() for m in entries)


def test_flat_list_is_the_concatenation_of_the_groups():
    router = _full_router()
    flat = [m["name"] for m in router.get_text_models()]
    grouped = [m["name"] for entries in router.get_text_models_by_provider().values() for m in entries]
    assert flat == grouped


def test_a_provider_with_nothing_to_offer_is_left_out():
    class Empty:
        name = "empty"

        def catalog(self):
            return [{"name": "x", "price": "p", "vision": True, "hidden": True, "replacement": "y"}]

    router = LLMService(providers=[GeminiTextProvider(api_key="k"), Empty()], default_model="gemini-4-hydrogen")
    assert "empty" not in router.get_text_models_by_provider()


# ── keyboards ───────────────────────────────────────────────────────────────

def _rows(markup):
    return [(b.text, b.callback_data) for row in markup.inline_keyboard for b in row]


def test_service_keyboard_marks_the_current_service_and_counts_models():
    rows = _rows(get_text_service_keyboard("gemini", {"gemini": 5, "openai": 3}))
    assert rows[0] == ("✅ 🔷 Google Gemini · 5 models", "tsvc:gemini")
    assert rows[1] == ("🟢 OpenAI · 3 models", "tsvc:openai")


def test_service_without_a_key_is_an_inert_row():
    rows = _rows(get_text_service_keyboard("gemini", {"gemini": 5}))
    assert rows[1] == ("🟢 OpenAI · not configured", "noop")


def test_model_keyboard_marks_current_and_goes_back():
    entries = _gemini_only_router().get_text_models_by_provider()["gemini"]
    rows = _rows(get_text_models_keyboard(entries, "gemini-4-hydrogen"))
    hydrogen = next(r for r in rows if r[1] == "text-model:gemini-4-hydrogen")
    assert hydrogen[0].startswith("✅ gemini-4-hydrogen")
    assert rows[-1] == ("⬅️ Back to services", "tsvc:back")
    assert all(len(cb.encode()) <= 64 for _, cb in rows)


# ── handlers ────────────────────────────────────────────────────────────────

class FakeMessage:
    def __init__(self):
        self.chat = SimpleNamespace(id=-100)
        self.sent = []
        self.edits = []
        self.markups = []

    async def answer(self, text, parse_mode=None, reply_markup=None):
        self.sent.append(text)
        self.markups.append(reply_markup)

    async def edit_text(self, text, parse_mode=None, reply_markup=None):
        self.edits.append(text)
        self.markups.append(reply_markup)


class FakeCallback:
    def __init__(self, data):
        self.data = data
        self.from_user = SimpleNamespace(id=42)
        self.message = FakeMessage()
        self.alerts = []

    async def answer(self, text=None, show_alert=False):
        if show_alert and text:
            self.alerts.append(text)


class FakeDB:
    def __init__(self, data=None, fail=False):
        self.data = dict(data or {})
        self.updates = []
        self.fail = fail

    async def get_settings(self, user_id, chat_id):
        return dict(self.data)

    async def update_settings(self, user_id, chat_id, values):
        if self.fail:
            raise RuntimeError("db down")
        self.updates.append(values)
        self.data.update(values)
        return dict(self.data)


def _env(monkeypatch, router=None, settings=None, fail=False):
    db = FakeDB(settings, fail=fail)
    monkeypatch.setitem(container.container, DBService, db)
    monkeypatch.setitem(container.container, LLMService, router or _full_router())
    return db


def test_text_model_button_shows_services_with_the_default_when_nothing_is_saved(monkeypatch):
    _env(monkeypatch)
    cb = FakeCallback("settings:text_model")
    asyncio.run(st.settings_text_model(cb))
    text = cb.message.sent[-1]
    assert "gemini-4-hydrogen" in text and "Google Gemini" in text and "Choose a service" in text
    callbacks = [c for _, c in _rows(cb.message.markups[-1])]
    assert callbacks == ["tsvc:gemini", "tsvc:openai"]


def test_a_saved_model_that_is_gone_is_shown_with_its_replacement(monkeypatch):
    _env(monkeypatch, settings={"text_model": "gemini-2.5-flash-lite"})   # a real saved value from the DB
    cb = FakeCallback("settings:text_model")
    asyncio.run(st.settings_text_model(cb))
    text = cb.message.sent[-1]
    assert "gemini-2.5-flash-lite" in text and "not available" in text and "gemini-4-hydrogen" in text


def test_the_old_pro_id_is_shown_as_its_replacement(monkeypatch):
    _env(monkeypatch, settings={"text_model": "gemini-3.1-pro"})
    cb = FakeCallback("settings:text_model")
    asyncio.run(st.settings_text_model(cb))
    assert "gemini-3.5-flash" in cb.message.sent[-1]


def test_service_click_lists_its_models_in_place(monkeypatch):
    _env(monkeypatch, settings={"text_model": "gemini-4-hydrogen"})
    cb = FakeCallback("tsvc:gemini")
    asyncio.run(st.settings_text_service(cb))
    assert "Pick a model" in cb.message.edits[-1]
    rows = _rows(cb.message.markups[-1])
    assert ("✅ gemini-4-hydrogen  |  🚀 new · fast · price unverified", "text-model:gemini-4-hydrogen") in rows


def test_unconfigured_service_click_is_refused(monkeypatch):
    _env(monkeypatch, router=_gemini_only_router())
    cb = FakeCallback("tsvc:openai")
    asyncio.run(st.settings_text_service(cb))
    assert cb.alerts and not cb.message.edits


def test_back_returns_to_the_service_list(monkeypatch):
    _env(monkeypatch)
    cb = FakeCallback("tsvc:back")
    asyncio.run(st.settings_text_back(cb))
    assert "Choose a service" in cb.message.edits[-1]


def test_choosing_hydrogen_saves_it_and_confirms_the_service(monkeypatch):
    db = _env(monkeypatch)
    cb = FakeCallback("text-model:gemini-4-hydrogen")
    asyncio.run(st.handle_text_model_selected(cb))
    assert db.updates == [{"text_model": "gemini-4-hydrogen"}]
    text = cb.message.edits[-1]
    assert "Selected" in text and "gemini-4-hydrogen" in text and "Google Gemini" in text
    assert any(t.startswith("✅ gemini-4-hydrogen") for t, _ in _rows(cb.message.markups[-1]))


def test_choosing_an_openai_model_names_openai(monkeypatch):
    db = _env(monkeypatch)
    cb = FakeCallback("text-model:gpt-test")
    # the catalogue has OpenAI models; pick the first one the menu really offers
    first = _full_router().get_text_models_by_provider()["openai"][0]["name"]
    cb = FakeCallback(f"text-model:{first}")
    asyncio.run(st.handle_text_model_selected(cb))
    assert db.updates == [{"text_model": first}] and "OpenAI" in cb.message.edits[-1]


@pytest.mark.parametrize("model", ["gemini-3.1-pro-preview", "claude-opus-5-5", "nonexistent"])
def test_hidden_review_only_and_unknown_models_cannot_be_chosen(monkeypatch, model):
    db = _env(monkeypatch)
    cb = FakeCallback(f"text-model:{model}")
    asyncio.run(st.handle_text_model_selected(cb))
    assert db.updates == [] and cb.alerts


def test_a_database_failure_is_reported_without_claiming_success(monkeypatch):
    _env(monkeypatch, fail=True)
    cb = FakeCallback("text-model:gemini-4-hydrogen")
    asyncio.run(st.handle_text_model_selected(cb))
    assert cb.message.edits == [] and any("db down" in t for t in cb.message.sent)


# ── gemini-4-hydrogen: request building ─────────────────────────────────────

def _provider(reply="ok", error=None):
    calls = []

    def generate_content(model, contents, config):
        calls.append({"model": model, "contents": contents, "config": config})
        if error:
            raise error
        return SimpleNamespace(text=reply, candidates=[], prompt_feedback=None)

    provider = GeminiTextProvider(api_key="k")
    provider._client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    return provider, calls


def test_hydrogen_request_carries_the_exact_model_system_prompt_and_text():
    provider, calls = _provider("hello")
    router = LLMService(providers=[provider], default_model="gemini-4-hydrogen")
    out = asyncio.run(router.generate_text("write a plot", "be brief", "gemini-4-hydrogen"))
    assert out == "hello"
    call = calls[0]
    assert call["model"] == "gemini-4-hydrogen"
    assert call["config"].system_instruction == "be brief"
    assert call["contents"][0].parts[-1].text == "write a plot"


def test_an_empty_model_name_is_sent_as_the_default_hydrogen():
    provider, calls = _provider("hello")
    router = LLMService(providers=[provider], default_model="gemini-4-hydrogen")
    asyncio.run(router.generate_text("p", "", ""))
    assert calls[0]["model"] == "gemini-4-hydrogen"


def test_hydrogen_gets_the_photos_as_image_parts_before_the_text(tmp_path):
    img = tmp_path / "a.png"
    img.write_bytes(b"\x89PNG-test")
    provider, calls = _provider("a red square")
    router = LLMService(providers=[provider], default_model="gemini-4-hydrogen")
    asyncio.run(router.generate_text("describe", "", "gemini-4-hydrogen", image_paths=[str(img)]))
    assert calls[0]["model"] == "gemini-4-hydrogen"          # no fallback to another model for images
    parts = calls[0]["contents"][0].parts
    assert parts[0].inline_data.mime_type == "image/png" and parts[0].inline_data.data == b"\x89PNG-test"
    assert parts[-1].text == "describe"


def test_cyrillic_text_survives_the_round_trip():
    reply = "Кот не хочет мыться: он прячется под диван."
    provider, calls = _provider(reply)
    router = LLMService(providers=[provider], default_model="gemini-4-hydrogen")
    assert asyncio.run(router.generate_text("Идея: кот не хочет мыться", "", "gemini-4-hydrogen")) == reply
    assert calls[0]["contents"][0].parts[-1].text == "Идея: кот не хочет мыться"


def test_an_empty_answer_is_an_error_not_an_empty_string():
    provider, _ = _provider(reply="   ")
    with pytest.raises(LLMEmptyResponse):
        asyncio.run(provider.generate("p", "", "gemini-4-hydrogen"))


# ── gemini-4-hydrogen: the real script pipeline ─────────────────────────────

def _script_json(durations):
    return json.dumps({
        "spoken_text": "Привет, это тест",
        "caption": "#тест",
        "shots": [{"index": i, "duration_seconds": d, "scene_prompt": f"shot {i}",
                   "camera_motion": "static", "transition": "cut"} for i, d in enumerate(durations)],
    }, ensure_ascii=False)


def test_script_pipeline_uses_hydrogen_end_to_end():
    provider, calls = _provider(_script_json([5, 5]))
    router = LLMService(providers=[provider], default_model="gemini-4-hydrogen")
    out = asyncio.run(common._build_video_prompt(
        "робот учится готовить",
        {"text_model": "gemini-4-hydrogen", "target_duration": 10, "video_model": "seedance"},
        router,
    ))
    parsed = common.GeminiService.parse_script_json(out)
    assert parsed is not None and len(parsed["shots"]) == 2
    assert parsed["spoken_text"] == "Привет, это тест"
    assert [c["model"] for c in calls] == ["gemini-4-hydrogen"]       # exactly one call, the right model


def test_script_pipeline_accepts_a_fenced_json_answer():
    fenced = "```json\n" + _script_json([5, 5]) + "\n```"
    provider, _ = _provider(fenced)
    router = LLMService(providers=[provider], default_model="gemini-4-hydrogen")
    out = asyncio.run(common._build_video_prompt(
        "idea", {"text_model": "gemini-4-hydrogen", "target_duration": 10, "video_model": "seedance"}, router,
    ))
    assert common.GeminiService.parse_script_json(out) is not None
