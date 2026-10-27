"""The real script-building path (_build_video_prompt) with the Claude switch,
using gemini-4-argon as the only text model and fakes for every provider."""

import asyncio
import json

from bot.handlers import common
from services.db import DBService
from services.llm import LLMService
from utils import container


def _script(durations, spoken):
    return json.dumps({
        "spoken_text": spoken,
        "caption": "#tag",
        "shots": [
            {"index": i + 1, "duration_seconds": d, "scene_prompt": f"shot {i + 1}",
             "camera_motion": "static", "transition": "cut"}
            for i, d in enumerate(durations)
        ],
    })


class FakeRouter:
    """Stands in for LLMService: records which model wrote the draft."""

    def __init__(self, draft, claude=None):
        self.draft = draft
        self.models_used = []
        self.providers_map = {"anthropic": claude} if claude else {}

    async def generate_text(self, prompt, system_prompt="", model="", image_paths=None):
        self.models_used.append(model)
        return self.draft


class FakeClaude:
    def __init__(self, reply):
        self.reply = reply
        self.calls = 0

    async def generate_with_usage(self, prompt, system, model, image_paths=None):
        self.calls += 1
        return self.reply, {"input_tokens": 1, "output_tokens": 2}


class FakeDB:
    async def log_cost(self, **kwargs):
        pass


def _build(monkeypatch, router, **settings):
    monkeypatch.setitem(container.container, LLMService, router)
    monkeypatch.setitem(container.container, DBService, FakeDB())
    base = {"text_model": "gemini-4-argon", "target_duration": 15, "video_model": "seedance"}
    base.update(settings)
    return asyncio.run(common._build_video_prompt("a robot learns to cook", base, router))


def test_argon_writes_the_draft_and_claude_polishes_when_switched_on(monkeypatch):
    claude = FakeClaude(_script([6, 5, 4], "sharper hook"))
    router = FakeRouter(_script([5, 5, 5], "plain hook"), claude)
    out = _build(monkeypatch, router, claude_enrichment=True)
    assert json.loads(out)["spoken_text"] == "sharper hook"
    assert router.models_used == ["gemini-4-argon"]      # the draft really came from argon
    assert claude.calls == 1


def test_switched_off_means_argon_only_and_no_claude_call(monkeypatch):
    claude = FakeClaude(_script([6, 5, 4], "sharper hook"))
    router = FakeRouter(_script([5, 5, 5], "plain hook"), claude)
    out = _build(monkeypatch, router, claude_enrichment=False)
    assert json.loads(out)["spoken_text"] == "plain hook"
    assert router.models_used == ["gemini-4-argon"]
    assert claude.calls == 0


def test_bad_claude_output_keeps_the_argon_draft(monkeypatch):
    claude = FakeClaude(_script([15, 15, 15], "far too long"))
    router = FakeRouter(_script([5, 5, 5], "plain hook"), claude)
    out = _build(monkeypatch, router, claude_enrichment=True)
    assert json.loads(out)["spoken_text"] == "plain hook"
    assert claude.calls == 1
