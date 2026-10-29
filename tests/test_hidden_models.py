"""Models hidden for testing: gone from the menus, still in the code, and a saved
choice is served by a working model instead of failing."""

import asyncio
from types import SimpleNamespace

import pytest

from bot.handlers import settings as st
from services import imagegen
from services.db import DBService
from services.gemini import GeminiService
from services.llm import GeminiTextProvider, LLMService
from utils import container


def _router():
    return LLMService(providers=[GeminiTextProvider(api_key="k")], default_model="gemini-3.5-flash")


# ── text: gemini-3.1-pro-preview ────────────────────────────────────────────

def test_pro_text_model_is_not_offered_in_the_menu():
    names = [m["name"] for m in _router().get_text_models()]
    assert "gemini-3.1-pro-preview" not in names
    assert {"gemini-3.5-flash", "gemini-3.1-flash-lite", "gemini-4-argon"} <= set(names)


def test_pro_text_model_is_kept_in_the_catalogue_not_deleted():
    entry = next(m for m in GeminiTextProvider(api_key="k").catalog() if m["name"] == "gemini-3.1-pro-preview")
    assert entry["hidden"] is True and entry["replacement"] == "gemini-3.5-flash"


def test_saved_pro_choice_is_served_by_the_replacement():
    router = _router()
    assert router._resolve_model("gemini-3.1-pro-preview") == ("gemini", "gemini-3.5-flash")
    assert router._resolve_model("gemini-3.1-pro") == ("gemini", "gemini-3.5-flash")   # the old saved id too


def test_source_carries_the_do_not_delete_warning():
    import pathlib
    for rel in ("services/llm.py", "services/gemini.py", "services/imagegen.py"):
        source = pathlib.Path(__file__).resolve().parent.parent.joinpath(rel).read_text(encoding="utf-8")
        assert "DO NOT DELETE" in source, rel


# ── image: gemini-3-pro-image ───────────────────────────────────────────────

class FakeMessage:
    def __init__(self):
        self.chat = SimpleNamespace(id=-100)
        self.sent = []
        self.cleared = False

    async def answer(self, text, reply_markup=None, **kwargs):
        self.sent.append((text, reply_markup))

    async def edit_reply_markup(self, reply_markup=None):
        self.cleared = True


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
    def __init__(self):
        self.updates = []

    async def update_settings(self, user_id, chat_id, values):
        self.updates.append(values)


@pytest.fixture
def env(monkeypatch):
    db = FakeDB()
    monkeypatch.setitem(container.container, DBService, db)
    monkeypatch.setitem(container.container, GeminiService, GeminiService(api_key=""))
    return db


def test_pro_image_model_is_kept_in_the_catalogue_but_flagged():
    entries = GeminiService(api_key="").get_image_models()
    pro = next(e for e in entries if e.get("name") == "gemini-3-pro-image")
    assert pro["hidden"] is True
    assert any(e.get("name") == "gemini-3.1-flash-image" and not e.get("hidden") for e in entries)


def test_image_menu_does_not_offer_the_pro_model(env):
    cb = FakeCallback("settings:image_model")
    asyncio.run(st.settings_image_model(cb))
    _, markup = cb.message.sent[-1]
    callbacks = [b.callback_data for row in markup.inline_keyboard for b in row]
    assert "image-model:gemini-3-pro-image" not in callbacks
    assert "image-model:google/nano-banana-2/text-to-image" in callbacks


def test_stale_keyboard_cannot_select_the_hidden_image_model(env):
    cb = FakeCallback("image-model:gemini-3-pro-image")
    asyncio.run(st.handle_image_model_selected(cb))
    assert env.updates == [] and cb.alerts


def test_visible_image_model_can_still_be_selected(env):
    cb = FakeCallback("image-model:google/nano-banana-2/text-to-image")
    asyncio.run(st.handle_image_model_selected(cb))
    assert env.updates == [{"image_model": "google/nano-banana-2/text-to-image"}]


def test_saved_pro_image_choice_goes_to_atlas_not_to_the_blocked_model(monkeypatch):
    service = imagegen.ImageGenService(api_key="", atlas_api_key="")
    routed = []

    async def via_atlas(prompt, model, aspect):
        routed.append(("atlas", model))
        return "out.png"

    async def via_gemini(prompt, model):
        routed.append(("gemini", model))
        return "out.png"

    monkeypatch.setattr(service, "_generate_via_atlas", via_atlas)
    monkeypatch.setattr(service, "_generate_via_gemini", via_gemini)
    asyncio.run(service.generate("p", model="gemini-3-pro-image"))
    assert routed == [("atlas", "google/nano-banana-2/text-to-image")]


def test_other_gemini_image_model_is_untouched(monkeypatch):
    service = imagegen.ImageGenService(api_key="", atlas_api_key="")
    routed = []

    async def via_gemini(prompt, model):
        routed.append(model)
        return "out.png"

    monkeypatch.setattr(service, "_generate_via_gemini", via_gemini)
    asyncio.run(service.generate("p", model="gemini-3.1-flash-image"))
    assert routed == ["gemini-3.1-flash-image"]
