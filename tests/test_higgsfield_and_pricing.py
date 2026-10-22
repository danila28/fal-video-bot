"""Higgsfield request building, polling, and the video cost estimate.

No network: the HTTP session is replaced by a small fake.
"""

import asyncio

import pytest

from services import higgsfield as hf
from services.pricing import (
    estimate_video_usd,
    format_video_cost_line,
    video_provider,
)


# ── request bodies ───────────────────────────────────────────────────────────

def test_kling_body_fields_and_duration_clamp():
    body = hf.build_request_body(
        "higgsfield_kling_std", prompt="walk", duration=1,
        image_urls=["https://x/a.jpg"], sound=True,
    )
    assert body == {
        "prompt": "walk", "image_url": "https://x/a.jpg", "duration": 3, "sound": "on",
    }


def test_kling_prompt_is_truncated_to_documented_limit():
    body = hf.build_request_body(
        "higgsfield_kling_std", prompt="a" * 3000, duration=5, image_urls=["https://x/a.jpg"],
    )
    assert len(body["prompt"]) == 2500


def test_minimax_body_has_documented_fields_only():
    body = hf.build_request_body(
        "higgsfield_minimax_h3", prompt="scene", duration=30, image_urls=["https://x/a.jpg"],
    )
    assert body == {
        "prompt": "scene", "image_url": "https://x/a.jpg", "duration": 15,
        "resolution": "2K", "aspect_ratio": "9:16", "aigc_watermark": False,
    }


def test_seedance_ref_sends_all_images_and_valid_resolution():
    urls = [f"https://x/{i}.jpg" for i in range(12)]
    body = hf.build_request_body(
        "higgsfield_seedance_ref", prompt="p", duration=7, image_urls=urls,
        resolution="1080p", sound=False,
    )
    assert body["image_urls"] == urls[:9]          # documented maximum is 9
    assert body["resolution"] == "1080p"
    assert body["generate_audio"] is False
    assert body["duration"] == 7


def test_seedance_unknown_resolution_falls_back_to_720p():
    body = hf.build_request_body(
        "higgsfield_seedance_ref", prompt="p", duration=5,
        image_urls=["https://x/a.jpg"], resolution="8k",
    )
    assert body["resolution"] == "720p"


def test_empty_prompt_gets_a_non_empty_default():
    body = hf.build_request_body(
        "higgsfield_minimax_h3", prompt="   ", duration=5, image_urls=["https://x/a.jpg"],
    )
    assert body["prompt"]  # documented minimum length is 1


def test_unknown_model_and_missing_image_are_rejected():
    with pytest.raises(hf.HiggsfieldError):
        hf.build_request_body("higgsfield_nope", prompt="p", duration=5, image_urls=["u"])
    with pytest.raises(hf.HiggsfieldError):
        hf.build_request_body("higgsfield_kling_std", prompt="p", duration=5, image_urls=[])


def test_endpoints_match_documentation():
    assert hf.HF_MODELS["higgsfield_kling_std"].endpoint == "kling-video/v3.0/std/image-to-video"
    assert hf.HF_MODELS["higgsfield_minimax_h3"].endpoint == "minimax/h3/image-to-video"
    assert hf.HF_MODELS["higgsfield_seedance_ref"].endpoint == "bytedance/seedance-2.0/reference-to-video"
    assert hf.HF_MODELS["higgsfield_seedance_ref"].reference is True
    assert hf.HF_MODELS["higgsfield_kling_std"].reference is False


def test_extract_video_url():
    assert hf.extract_video_url({"status": "completed", "video": {"url": "https://v/x.mp4"}}) == "https://v/x.mp4"
    assert hf.extract_video_url({"status": "completed"}) is None
    assert hf.extract_video_url({"video": "not-a-dict"}) is None


# ── HTTP flow with a fake session ────────────────────────────────────────────

class FakeResp:
    def __init__(self, status: int, payload=None, text: str = ""):
        self.status = status
        self._payload = payload
        self._text = text or (str(payload) if payload is not None else "")

    async def text(self):
        return self._text

    async def json(self):
        return self._payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class FakeSession:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def _next(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return self._responses.pop(0)

    def post(self, url, **kwargs):
        return self._next("POST", url, **kwargs)

    def get(self, url, **kwargs):
        return self._next("GET", url, **kwargs)


@pytest.fixture
def service(tmp_path, monkeypatch):
    monkeypatch.setattr(hf, "_POLL_INTERVAL", 0)
    return hf.HiggsfieldService(api_key="hf-test", static_dir=str(tmp_path))


def test_headers_use_key_auth(service):
    headers = service._headers(json_body=True)
    assert headers["Authorization"] == "Key hf-test"
    assert headers["Idempotency-Key"]


def test_submit_returns_request_handle(service):
    session = FakeSession([FakeResp(200, {"status": "queued", "request_id": "r1", "status_url": "https://s"})])
    data = asyncio.run(service._submit(session, "minimax/h3/image-to-video", {"prompt": "p"}))
    assert data["request_id"] == "r1"
    method, url, _ = session.calls[0]
    assert method == "POST" and url == "https://api.higgsfield.ai/minimax/h3/image-to-video"


def test_submit_http_error_is_reported(service):
    session = FakeSession([FakeResp(402, text='{"detail":"insufficient balance"}')])
    with pytest.raises(hf.HiggsfieldError, match="HTTP 402"):
        asyncio.run(service._submit(session, "x", {}))


def test_wait_returns_url_after_queued_then_completed(service):
    session = FakeSession([
        FakeResp(200, {"status": "in_progress"}),
        FakeResp(200, {"status": "completed", "video": {"url": "https://v/out.mp4"}}),
    ])
    url = asyncio.run(service._wait(session, {"request_id": "r", "status_url": "https://s"}, "L"))
    assert url == "https://v/out.mp4"


def test_wait_recovers_from_transient_5xx(service):
    session = FakeSession([
        FakeResp(503, text="gateway"),
        FakeResp(200, {"status": "completed", "video": {"url": "https://v/out.mp4"}}),
    ])
    assert asyncio.run(service._wait(session, {"request_id": "r", "status_url": "https://s"}, "L")) == "https://v/out.mp4"


def test_wait_nsfw_is_a_clear_error(service):
    session = FakeSession([FakeResp(200, {"status": "nsfw"})])
    with pytest.raises(hf.HiggsfieldError, match="content filter"):
        asyncio.run(service._wait(session, {"request_id": "r", "status_url": "https://s"}, "L"))


def test_wait_failed_includes_error_text(service):
    session = FakeSession([FakeResp(200, {"status": "failed", "error": "bad image"})])
    with pytest.raises(hf.HiggsfieldError, match="bad image"):
        asyncio.run(service._wait(session, {"request_id": "r", "status_url": "https://s"}, "L"))


def test_wait_completed_without_video_is_an_error(service):
    session = FakeSession([FakeResp(200, {"status": "completed"})])
    with pytest.raises(hf.HiggsfieldError, match="no video url"):
        asyncio.run(service._wait(session, {"request_id": "r", "status_url": "https://s"}, "L"))


# ── cost estimate ────────────────────────────────────────────────────────────

def test_known_atlas_price_is_estimated_per_second():
    assert estimate_video_usd("seedance", 10) == pytest.approx(0.9)
    assert estimate_video_usd("kling", 30) == pytest.approx(2.85)


def test_unknown_price_is_none_not_zero():
    assert estimate_video_usd("higgsfield_kling_std", 10) is None
    assert estimate_video_usd("seedance", 0) is None


def test_cost_line_says_unverified_or_unavailable():
    line = format_video_cost_line("seedance", 15)
    assert "$1.35" in line and "unverified" in line
    assert "not available" in format_video_cost_line("higgsfield_minimax_h3", 15)


def test_video_provider_routing():
    assert video_provider("higgsfield_kling_std") == "higgsfield"
    assert video_provider("kling") == "atlas"
    assert video_provider("") == "atlas"
