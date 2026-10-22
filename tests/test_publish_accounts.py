"""Per-account publishing: Blotato result mapping, status polling, bookkeeping,
and the account picker keyboard. Blotato and the database are replaced by fakes.
"""

import asyncio
from types import SimpleNamespace

from bot.handlers import publish as pub
from bot.keyboards import get_account_picker_keyboard
from services import blotato as blotato_mod
from services.blotato import BlotatoService
from services.db import DBService
from utils import container


ACCOUNTS = [
    {"id": 7, "platform": "tiktok", "account_id": "tt-1", "label": "Main TT"},
    {"id": 8, "platform": "youtube", "account_id": "yt-1", "label": ""},
]


# ── Blotato: publish_video tags each result with its account ────────────────

def _blotato_with_fakes(monkeypatch, fail_platform=None):
    svc = BlotatoService(base_url="https://backend.blotato.com/v2", api_key="k")

    async def presigned(self, filename):
        return {"presignedUrl": "https://up", "publicUrl": "https://pub/v.mp4"}

    async def upload(self, url, path):
        return None

    async def publish(self, public_url, title, account_id, platform, scheduled_at=""):
        if platform == fail_platform:
            raise Exception("account disconnected")
        return {"postSubmissionId": f"sub-{platform}-{account_id}"}

    monkeypatch.setattr(BlotatoService, "_get_presigned_url", presigned)
    monkeypatch.setattr(BlotatoService, "_upload_file_to_presigned", upload)
    monkeypatch.setattr(BlotatoService, "_publish_post", publish)
    return svc


def test_publish_video_tags_results_with_account(monkeypatch, tmp_path):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    svc = _blotato_with_fakes(monkeypatch)
    results = asyncio.run(svc.publish_video(str(video), "t", "tiktok:tt-1,youtube:yt-1"))
    assert results == [
        {"account": "tiktok:tt-1", "postSubmissionId": "sub-tiktok-tt-1"},
        {"account": "youtube:yt-1", "postSubmissionId": "sub-youtube-yt-1"},
    ]


def test_publish_error_stays_attached_to_its_account(monkeypatch, tmp_path):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    svc = _blotato_with_fakes(monkeypatch, fail_platform="youtube")
    results = asyncio.run(svc.publish_video(str(video), "t", "tiktok:tt-1,youtube:yt-1"))
    assert results[0]["postSubmissionId"] == "sub-tiktok-tt-1"
    assert results[1] == {"account": "youtube:yt-1", "error": "account disconnected"}


def test_get_post_status_reads_documented_endpoint(monkeypatch):
    seen = {}

    class Resp:
        status = 200

        async def text(self):
            return ""

        async def json(self):
            return {"status": "published", "publicUrl": "https://x/post"}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    class Sess:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def get(self, url, headers=None):
            seen["url"] = url
            seen["headers"] = headers
            return Resp()

    monkeypatch.setattr(blotato_mod.aiohttp, "ClientSession", lambda *a, **k: Sess())
    svc = BlotatoService(base_url="https://backend.blotato.com/v2", api_key="k")
    data = asyncio.run(svc.get_post_status("sub-1"))
    assert seen["url"] == "https://backend.blotato.com/v2/posts/sub-1"
    assert seen["headers"] == {"blotato-api-key": "k"}
    assert data["publicUrl"] == "https://x/post"


# ── bookkeeping and polling with fakes ─────────────────────────────────────

class FakeDB:
    def __init__(self):
        self.rows = {}
        self.next_id = 1

    async def add_publication(self, chat_id, user_id, account_row_id, platform, sub_id, status, error_text=None):
        pid = self.next_id
        self.next_id += 1
        self.rows[pid] = {"chat_id": chat_id, "account": account_row_id, "platform": platform,
                          "sub": sub_id, "status": status, "url": None, "error": error_text}
        return pid

    async def update_publication(self, pid, status, public_url=None, error_text=None):
        row = self.rows[pid]
        row["status"] = status
        row["url"] = public_url or row["url"]
        row["error"] = error_text


class FakeMsg:
    def __init__(self):
        self.chat = SimpleNamespace(id=-100)
        self.sent = []

    async def answer(self, text, parse_mode=None):
        self.sent.append(text)


def test_record_publications_one_row_per_selected_account(monkeypatch):
    db = FakeDB()
    monkeypatch.setitem(container.container, DBService, db)
    results = [
        {"account": "tiktok:tt-1", "postSubmissionId": "sub-a"},
        {"account": "youtube:yt-1", "error": "boom"},
    ]
    msg = FakeMsg()

    async def run():
        # scheduled=... so no background poller is started in this test
        return await pub._record_publications(msg, 42, ACCOUNTS, results, scheduled_at="2026-11-01T10:00:00+00:00")

    lines = asyncio.run(run())
    assert len(db.rows) == 2
    assert db.rows[1]["account"] == 7 and db.rows[1]["sub"] == "sub-a"
    assert db.rows[1]["status"] == "scheduled"
    assert db.rows[2]["account"] == 8 and db.rows[2]["status"] == "failed"
    assert any("boom" in line for line in lines)


def test_track_publication_stores_public_url(monkeypatch):
    db = FakeDB()
    db.rows[1] = {"chat_id": -100, "account": 7, "platform": "tiktok", "sub": "sub-a",
                  "status": "submitted", "url": None, "error": None}
    monkeypatch.setitem(container.container, DBService, db)
    monkeypatch.setattr(pub, "_PUB_POLL_INTERVAL", 0)

    states = iter([{"status": "in-progress"}, {"status": "published", "publicUrl": "https://tt/x"}])

    class FakeBlotato:
        async def get_post_status(self, sub_id):
            assert sub_id == "sub-a"
            return next(states)

    monkeypatch.setitem(container.container, BlotatoService, FakeBlotato())
    msg = FakeMsg()
    asyncio.run(pub._track_publication(msg, 1, "sub-a", "tiktok"))
    assert db.rows[1]["status"] == "published"
    assert db.rows[1]["url"] == "https://tt/x"
    assert any("https://tt/x" in text for text in msg.sent)


def test_track_publication_marks_failed_with_message(monkeypatch):
    db = FakeDB()
    db.rows[1] = {"chat_id": -100, "account": 7, "platform": "tiktok", "sub": "s",
                  "status": "submitted", "url": None, "error": None}
    monkeypatch.setitem(container.container, DBService, db)
    monkeypatch.setattr(pub, "_PUB_POLL_INTERVAL", 0)

    class FakeBlotato:
        async def get_post_status(self, sub_id):
            return {"status": "failed", "errorMessage": "video too long"}

    monkeypatch.setitem(container.container, BlotatoService, FakeBlotato())
    msg = FakeMsg()
    asyncio.run(pub._track_publication(msg, 1, "s", "tiktok"))
    assert db.rows[1]["status"] == "failed"
    assert db.rows[1]["error"] == "video too long"


# ── picker keyboard ──────────────────────────────────────────────────────────

def test_picker_marks_selected_and_has_next_cancel():
    kb = get_account_picker_keyboard(ACCOUNTS, {8})
    rows = kb.inline_keyboard
    assert rows[0][0].text.startswith("⬜") and rows[0][0].callback_data == "pubacc:toggle:7"
    assert rows[1][0].text.startswith("✅") and rows[1][0].callback_data == "pubacc:toggle:8"
    assert rows[2][0].callback_data == "pubacc:next"
    assert rows[3][0].callback_data == "pubacc:cancel"


def test_picker_callback_data_fits_telegram_limit():
    kb = get_account_picker_keyboard(
        [{"id": 2_147_483_647, "platform": "youtube", "account_id": "x", "label": ""}], set(),
    )
    for row in kb.inline_keyboard:
        for btn in row:
            assert len(btn.callback_data.encode()) <= 64
