"""Atlas/Higgsfield: submit retries for gateway errors, longer video timeout,
and the in-progress guard that must outlive a whole multi-clip video."""

import asyncio
import time
from types import SimpleNamespace

import aiohttp
import pytest

from bot.handlers.common import _explain_provider_error, _is_generating
from services import atlas, higgsfield as hf
from services.atlas import AtlasClient


class FakeResp:
    def __init__(self, status, payload=None, text=""):
        self.status = status
        self._payload = payload
        self._text = text

    async def text(self):
        return self._text

    async def json(self):
        return self._payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class FakeSession:
    """Plays a list of outcomes for post()/get(); an Exception outcome is raised."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.posts = 0
        self.gets = 0

    def _next(self):
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def post(self, url, **kwargs):
        self.posts += 1
        return self._next()

    def get(self, url, **kwargs):
        self.gets += 1
        return self._next()


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(atlas, "_SUBMIT_RETRY_DELAYS", (0, 0))

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr(atlas.asyncio, "sleep", no_sleep)
    return AtlasClient(api_key="k", static_dir=str(tmp_path))


OK = {"data": {"id": "pred-1"}}
GATEWAY_TEXT = "upstream connect error or disconnect/reset before headers. reset reason: connection timeout"


# ── submit retries ──────────────────────────────────────────────────────────

def test_gateway_503_is_retried_and_then_succeeds(client):
    session = FakeSession([FakeResp(503, text=GATEWAY_TEXT), FakeResp(503, text=GATEWAY_TEXT), FakeResp(200, OK)])
    data = asyncio.run(client._submit(session, "generateImage", {"model": "m"}))
    assert data == OK and session.posts == 3


def test_persistent_503_gives_up_after_three_attempts(client):
    session = FakeSession([FakeResp(503, text=GATEWAY_TEXT)] * 3)
    with pytest.raises(Exception, match="HTTP 503"):
        asyncio.run(client._submit(session, "generateImage", {"model": "m"}))
    assert session.posts == 3


@pytest.mark.parametrize("status", [400, 401, 402, 422])
def test_client_errors_are_never_retried(client, status):
    session = FakeSession([FakeResp(status, text="nope"), FakeResp(200, OK)])
    with pytest.raises(Exception, match=f"HTTP {status}"):
        asyncio.run(client._submit(session, "generateVideo", {"model": "m"}))
    assert session.posts == 1   # a paid job is never submitted twice after a real error


def test_failed_connection_is_retried(client):
    refused = aiohttp.ClientConnectorError(
        SimpleNamespace(host="api", port=443, is_ssl=True, ssl=True), OSError("refused"),
    )
    session = FakeSession([refused, FakeResp(200, OK)])
    assert asyncio.run(client._submit(session, "generateImage", {"model": "m"})) == OK
    assert session.posts == 2


# ── timeouts ────────────────────────────────────────────────────────────────

def test_video_timeout_is_longer_than_the_old_ten_minutes():
    assert atlas._POLL_TIMEOUT_VIDEO >= 1800
    assert atlas._POLL_TIMEOUT_IMAGE == 600


def test_env_override_is_parsed_safely(monkeypatch):
    monkeypatch.setenv("ATLAS_TEST_T", "2400")
    assert atlas._env_seconds("ATLAS_TEST_T", 1800) == 2400
    monkeypatch.setenv("ATLAS_TEST_T", "garbage")
    assert atlas._env_seconds("ATLAS_TEST_T", 1800) == 1800
    monkeypatch.setenv("ATLAS_TEST_T", "5")
    assert atlas._env_seconds("ATLAS_TEST_T", 1800) == 60   # floor, never a near-zero timeout


def test_poll_slows_down_after_two_minutes_then_times_out_with_job_id(client):
    processing = FakeResp(200, {"data": {"status": "processing"}})
    session = FakeSession([processing] * 100)
    with pytest.raises(Exception) as exc:
        asyncio.run(client._poll(session, "pred-9", "kling", timeout=130))
    assert session.gets == 25                       # 24 polls at 5 s, then one at 10 s
    message = str(exc.value)
    assert "timeout after 130s" in message and "pred-9" in message and "charged" in message


def test_poll_returns_the_output_when_it_finishes_late(client):
    session = FakeSession([FakeResp(200, {"data": {"status": "processing"}})] * 80
                          + [FakeResp(200, {"data": {"status": "completed", "outputs": ["https://v/x.mp4"]}})])
    # 80 polls = more than the old 10-minute cap would ever allow with 5 s steps and a 600 s limit
    assert asyncio.run(client._poll(session, "p", "m", timeout=1800)) == "https://v/x.mp4"


def test_video_and_image_jobs_use_their_own_timeouts(client, monkeypatch):
    seen = {}

    async def fake_submit(self, session, endpoint, payload):
        return OK

    async def fake_poll(self, session, pred_id, model, timeout=0):
        seen[model] = timeout
        return "https://out"

    async def noop(*args, **kwargs):
        return None

    monkeypatch.setattr(AtlasClient, "_submit", fake_submit)
    monkeypatch.setattr(AtlasClient, "_poll", fake_poll)
    monkeypatch.setattr(AtlasClient, "_track_start", staticmethod(noop))
    monkeypatch.setattr(AtlasClient, "_track_finish", staticmethod(noop))

    asyncio.run(client.generate_video("vid-model", {}))
    asyncio.run(client.generate_image("img-model", {}))
    assert seen["vid-model"] == atlas._POLL_TIMEOUT_VIDEO
    assert seen["img-model"] == atlas._POLL_TIMEOUT_IMAGE


def test_higgsfield_timeout_default_and_message(monkeypatch):
    assert hf._POLL_TIMEOUT >= 1800

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr(hf.asyncio, "sleep", no_sleep)
    monkeypatch.setattr(hf, "_POLL_TIMEOUT", 10)
    monkeypatch.setattr(hf, "_POLL_INTERVAL", 5)
    service = hf.HiggsfieldService(api_key="k", static_dir=".")
    session = FakeSession([FakeResp(200, {"status": "in_progress"})] * 5)
    with pytest.raises(hf.HiggsfieldError, match="req-7"):
        asyncio.run(service._wait(session, {"request_id": "req-7", "status_url": "https://s"}, "L"))


# ── user-facing hint ────────────────────────────────────────────────────────

def test_atlas_gateway_error_gets_a_plain_hint():
    err = f"Atlas generateImage submit failed HTTP 503: {GATEWAY_TEXT}"
    hint = _explain_provider_error(Exception(err))
    assert "temporarily unavailable" in hint and "upstream connect" not in hint


def test_other_atlas_errors_pass_through():
    err = "Atlas generateVideo submit failed HTTP 402: insufficient balance"
    assert _explain_provider_error(Exception(err)) == err


# ── in-progress guard ───────────────────────────────────────────────────────

class FakeState:
    def __init__(self, started_ago):
        self.data = {"generation_in_progress": True, "generation_started_at": time.time() - started_ago}

    async def get_data(self):
        return dict(self.data)

    async def update_data(self, **kwargs):
        self.data.update(kwargs)


def test_guard_still_holds_after_an_hour_but_clears_after_two():
    long_running = FakeState(started_ago=3600)       # a multi-clip video can take this long
    assert asyncio.run(_is_generating(long_running)) is True
    stuck = FakeState(started_ago=3 * 3600)
    assert asyncio.run(_is_generating(stuck)) is False
    assert stuck.data["generation_in_progress"] is False
