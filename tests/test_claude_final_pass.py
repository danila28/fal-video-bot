"""Claude final pass: provider behaviour and the hard rules applied to its output.

The Anthropic client is replaced by a fake; nothing is sent over the network.
"""

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

import anthropic

from bot.handlers import common
from services import llm
from services.llm import AnthropicTextProvider, LLMEmptyResponse, LLMError, LLMRefusal


def _script(durations, scene="a red car on a road", spoken="hello there"):
    return json.dumps({
        "spoken_text": spoken,
        "caption": "#car",
        "shots": [
            {"index": i + 1, "duration_seconds": d, "scene_prompt": scene,
             "camera_motion": "static", "transition": "cut"}
            for i, d in enumerate(durations)
        ],
    })


# ── provider ────────────────────────────────────────────────────────────────

class FakeMessages:
    def __init__(self, outcomes):
        self._outcomes = list(outcomes)
        self.kwargs = []

    async def create(self, **kwargs):
        self.kwargs.append(kwargs)
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _provider_with(outcomes):
    provider = AnthropicTextProvider(api_key="sk-ant-test")
    fake = FakeMessages(outcomes)
    provider._client = SimpleNamespace(messages=fake)
    return provider, fake


def _response(text="ok", stop="end_turn", content=None):
    blocks = content if content is not None else [SimpleNamespace(type="text", text=text)]
    return SimpleNamespace(
        content=blocks,
        stop_reason=stop,
        usage=SimpleNamespace(input_tokens=120, output_tokens=40),
    )


def test_success_returns_text_and_usage():
    provider, fake = _provider_with([_response("corrected")])
    text, usage = asyncio.run(provider.generate_with_usage("draft", "sys", ""))
    assert text == "corrected"
    assert usage == {"input_tokens": 120, "output_tokens": 40}
    sent = fake.kwargs[0]
    assert sent["model"] == "claude-opus-5-5"
    assert sent["system"] == "sys"
    assert sent["messages"][0]["content"][-1] == {"type": "text", "text": "draft"}


def test_refusal_is_reported_not_returned():
    provider, _ = _provider_with([_response(stop="refusal", content=[])])
    with pytest.raises(LLMRefusal):
        asyncio.run(provider.generate("x", "", ""))


def test_empty_output_is_reported():
    provider, _ = _provider_with([_response(content=[SimpleNamespace(type="text", text="  ")])])
    with pytest.raises(LLMEmptyResponse):
        asyncio.run(provider.generate("x", "", ""))


def test_client_error_is_not_retried():
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    err = anthropic.BadRequestError("bad", response=httpx.Response(400, request=request), body=None)
    provider, fake = _provider_with([err])
    with pytest.raises(LLMError):
        asyncio.run(provider.generate("x", "", ""))
    assert len(fake.kwargs) == 1  # exactly one attempt


def test_only_temporary_errors_are_retried():
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    rate = anthropic.RateLimitError("slow", response=httpx.Response(429, request=request), body=None)
    bad = anthropic.BadRequestError("bad", response=httpx.Response(400, request=request), body=None)
    assert llm._should_retry_anthropic(rate) is True
    assert llm._should_retry_anthropic(bad) is False


def test_image_is_sent_as_base64_block(tmp_path):
    img = tmp_path / "p.png"
    img.write_bytes(b"\x89PNG fake")
    provider, fake = _provider_with([_response("ok")])
    asyncio.run(provider.generate("x", "", "", image_paths=[str(img)]))
    block = fake.kwargs[0]["messages"][0]["content"][0]
    assert block["type"] == "image"
    assert block["source"]["type"] == "base64"
    assert block["source"]["media_type"] == "image/png"


def test_no_key_means_no_provider(monkeypatch):
    from services.llm import LLMService, OpenAITextProvider
    router = LLMService(providers=[OpenAITextProvider(api_key="sk-test")])
    assert "anthropic" not in router.providers_map


# ── hard rules on the polished script ───────────────────────────────────────

DRAFT = _script([5, 5, 5])  # 15 s total, 3 shots


def test_valid_polish_is_accepted():
    polished = _script([6, 5, 5], spoken="a sharper hook")
    out = common._validate_final_script(DRAFT, polished, 15)
    assert out is not None
    data = json.loads(out)
    assert data["spoken_text"] == "a sharper hook"
    assert [s["duration_seconds"] for s in data["shots"]] == [6, 5, 5]


def test_changed_shot_count_is_rejected():
    assert common._validate_final_script(DRAFT, _script([5, 5, 5, 5]), 20) is None
    assert common._validate_final_script(DRAFT, _script([15]), 15) is None


def test_duration_within_plus_minus_two_is_accepted():
    assert common._validate_final_script(DRAFT, _script([5, 5, 4]), 15) is not None  # 14 s, off by 1
    assert common._validate_final_script(DRAFT, _script([5, 5, 7]), 15) is not None  # 17 s, off by 2


def test_duration_outside_plus_minus_two_is_rejected():
    assert common._validate_final_script(DRAFT, _script([5, 5, 2]), 15) is None   # 12 s, off by 3
    assert common._validate_final_script(DRAFT, _script([6, 6, 6]), 15) is None   # 18 s, off by 3


def test_per_shot_duration_limits_are_enforced():
    assert common._validate_final_script(DRAFT, _script([3, 6, 6]), 15) is None   # 3 < 4
    assert common._validate_final_script(DRAFT, _script([16, 0, 0]), 16) is None  # 16 > 15


def test_empty_scene_prompt_is_rejected():
    empty = json.loads(_script([5, 5, 5]))
    empty["shots"][1]["scene_prompt"] = "   "
    assert common._validate_final_script(DRAFT, json.dumps(empty), 15) is None


def test_non_json_output_is_rejected():
    assert common._validate_final_script(DRAFT, "Sure! Here is the script...", 15) is None
    assert common._validate_final_script("not json", _script([5, 5, 5]), 15) is None


# ── integration: _final_pass_script with a fake router in the DI container ──

class _FakeRouter:
    def __init__(self, provider):
        self.providers_map = {"anthropic": provider} if provider is not None else {}


class _FakeClaude:
    def __init__(self, reply=None, error=None):
        self.reply = reply
        self.error = error
        self.calls = 0

    async def generate_with_usage(self, prompt, system, model, image_paths=None):
        self.calls += 1
        if self.error:
            raise self.error
        return self.reply, {"input_tokens": 10, "output_tokens": 20}


class _FakeDB:
    def __init__(self):
        self.logged = []

    async def log_cost(self, **kwargs):
        self.logged.append(kwargs)


def _run_final_pass(monkeypatch, provider, settings, draft=DRAFT):
    from utils import container
    from services.llm import LLMService
    monkeypatch.setitem(container.container, LLMService, _FakeRouter(provider))
    db = _FakeDB()
    monkeypatch.setitem(container.container, common.DBService, db)
    out = asyncio.run(common._final_pass_script(draft, settings))
    return out, db


def test_final_pass_uses_claude_output_when_valid(monkeypatch):
    claude = _FakeClaude(reply=_script([6, 5, 5], spoken="better"))
    out, db = _run_final_pass(monkeypatch, claude, {"text_model": "terra", "target_duration": 15, "claude_enrichment": True})
    assert json.loads(out)["spoken_text"] == "better"
    assert claude.calls == 1


def test_final_pass_keeps_draft_when_claude_is_not_configured(monkeypatch):
    out, _ = _run_final_pass(monkeypatch, None, {"text_model": "terra", "target_duration": 15, "claude_enrichment": True})
    assert out == DRAFT


def test_final_pass_keeps_draft_on_api_error(monkeypatch):
    claude = _FakeClaude(error=LLMError("upstream down"))
    out, _ = _run_final_pass(monkeypatch, claude, {"text_model": "terra", "target_duration": 15, "claude_enrichment": True})
    assert out == DRAFT


def test_final_pass_keeps_draft_when_rules_are_broken(monkeypatch):
    claude = _FakeClaude(reply=_script([15, 15, 15], spoken="too long"))
    out, _ = _run_final_pass(monkeypatch, claude, {"text_model": "terra", "target_duration": 15, "claude_enrichment": True})
    assert out == DRAFT


def test_final_pass_skipped_when_draft_already_came_from_claude(monkeypatch):
    claude = _FakeClaude(reply=_script([6, 5, 5]))
    out, _ = _run_final_pass(monkeypatch, claude, {"text_model": "claude-opus-5-5", "target_duration": 15, "claude_enrichment": True})
    assert out == DRAFT
    assert claude.calls == 0


def test_final_pass_logs_tokens_for_the_owner(monkeypatch):
    from utils.job_context import JobOwner, current_owner
    claude = _FakeClaude(reply=_script([6, 5, 5]))
    token = current_owner.set(JobOwner(chat_id=-100, user_id=42))
    try:
        _, db = _run_final_pass(monkeypatch, claude, {"text_model": "terra", "target_duration": 15, "claude_enrichment": True})
    finally:
        current_owner.reset(token)
    assert db.logged and db.logged[0]["provider"] == "anthropic"
    assert db.logged[0]["input_tokens"] == 10 and db.logged[0]["output_tokens"] == 20
    assert db.logged[0]["est_usd"] is None  # no verified price, so no estimate


def test_final_pass_is_off_by_default(monkeypatch):
    claude = _FakeClaude(reply=_script([6, 5, 5], spoken="better"))
    for settings in ({"text_model": "terra", "target_duration": 15},
                     {"text_model": "terra", "target_duration": 15, "claude_enrichment": False}):
        out, _ = _run_final_pass(monkeypatch, claude, settings)
        assert out == DRAFT
    assert claude.calls == 0   # no API call, no cost, when the switch is off
