"""Video menu callback handlers, called directly with fake Telegram objects."""

import asyncio
from types import SimpleNamespace

import pytest

from bot.handlers import settings as st
from services.db import DBService
from services.gemini import GeminiService
from utils import container
from utils.config import config


class FakeMessage:
    def __init__(self):
        self.chat = SimpleNamespace(id=-100)
        self.edits = []
        self.answers = []

    async def edit_text(self, text, parse_mode=None, reply_markup=None):
        self.edits.append(text)

    async def edit_reply_markup(self, reply_markup=None):
        pass

    async def answer(self, text, parse_mode=None, reply_markup=None):
        self.answers.append(text)


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
    def __init__(self, video_model=""):
        self.settings = {"video_model": video_model}
        self.updates = []

    async def get_settings(self, user_id, chat_id):
        return dict(self.settings)

    async def update_settings(self, user_id, chat_id, values):
        self.updates.append(values)
        self.settings.update(values)


@pytest.fixture
def db(monkeypatch):
    fake = FakeDB(video_model="seedance")
    monkeypatch.setitem(container.container, DBService, fake)
    # main.py registers GeminiService for the menu catalogue; tests do the same.
    monkeypatch.setitem(container.container, GeminiService, GeminiService(api_key=""))
    return fake


def test_selecting_atlas_model_saves_it_and_confirms_service(db, monkeypatch):
    monkeypatch.setitem(config, "HIGGSFIELD_API_KEY", None)
    cb = FakeCallback("video-model:kling")
    asyncio.run(st.handle_video_model_selected(cb))
    assert db.updates == [{"video_model": "kling"}]
    text = cb.message.edits[-1]
    assert "Selected" in text and "Kling v3 Pro" in text and "Atlas Cloud" in text


def test_selecting_higgsfield_without_key_is_refused(db, monkeypatch):
    monkeypatch.setitem(config, "HIGGSFIELD_API_KEY", None)
    cb = FakeCallback("video-model:higgsfield_kling_std")
    asyncio.run(st.handle_video_model_selected(cb))
    assert db.updates == []
    assert cb.alerts and "not configured" in cb.alerts[0]


def test_selecting_higgsfield_with_key_saves_and_names_the_service(db, monkeypatch):
    monkeypatch.setitem(config, "HIGGSFIELD_API_KEY", "hf-test")
    cb = FakeCallback("video-model:higgsfield_minimax_h3")
    asyncio.run(st.handle_video_model_selected(cb))
    assert db.updates == [{"video_model": "higgsfield_minimax_h3"}]
    assert "Higgsfield" in cb.message.edits[-1]


def test_unknown_model_is_refused_without_saving(db, monkeypatch):
    monkeypatch.setitem(config, "HIGGSFIELD_API_KEY", "hf-test")
    cb = FakeCallback("video-model:veo_removed_model")
    asyncio.run(st.handle_video_model_selected(cb))
    assert db.updates == []
    assert cb.alerts


def test_service_click_without_key_shows_alert_only(db, monkeypatch):
    monkeypatch.setitem(config, "HIGGSFIELD_API_KEY", None)
    cb = FakeCallback("vsvc:higgsfield")
    asyncio.run(st.settings_video_service(cb))
    assert cb.message.edits == [] and cb.alerts


def test_service_click_opens_the_models_of_that_service(db, monkeypatch):
    monkeypatch.setitem(config, "HIGGSFIELD_API_KEY", "hf-test")
    cb = FakeCallback("vsvc:higgsfield")
    asyncio.run(st.settings_video_service(cb))
    text = cb.message.edits[-1]
    assert "Higgsfield" in text and "Current:" in text


def test_current_model_line_for_unset_and_unknown(db):
    assert "not selected" in st._video_model_line("")
    assert "not in the menu" in st._video_model_line("old_removed_model")
