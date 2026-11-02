"""Keyframes mode: 2-4 own photos used in order, one clip between each pair.

Covers the rules and clip lengths, request bodies, the clip-by-pair orchestration,
the upload step (including the warning for unsupported models), the settings and
the script instructions. The classic single-photo flow is checked for regressions.
No network: providers are replaced by fakes.
"""

import asyncio
import json
from types import SimpleNamespace

import pytest

from bot.handlers import common, generation as gen, settings as st
from bot.keyboards import get_advanced_settings_keyboard, get_keyframes_keyboard
from services import higgsfield as hf
from services import keyframes as kf
from services.db import DBService
from services.higgsfield import HiggsfieldService
from services.kling import MODEL_IDS, KlingService
from utils import container


SUPPORTED = ["kling", "kling_v3_std", "higgsfield_kling_std", "higgsfield_minimax_h3"]
UNSUPPORTED = [
    "kling_turbo", "kling_o3_pro", "kling_o3_std", "kling_t2v", "kling_o3_pro_ref",
    "seedance", "seedance_fast", "seedance_25", "seedance_ref", "minimax_h3",
    "higgsfield_seedance_ref",
]


# ── rules and clip lengths ──────────────────────────────────────────────────

@pytest.mark.parametrize("model", SUPPORTED)
def test_supported_models(model):
    assert kf.supports_keyframes(model)


@pytest.mark.parametrize("model", UNSUPPORTED)
def test_other_models_are_not_supported(model):
    assert not kf.supports_keyframes(model)


def test_atlas_kling_uses_only_5_or_10_second_clips():
    assert kf.plan_keyframe_durations("kling", 10, 3) == [5, 5]
    assert kf.plan_keyframe_durations("kling", 20, 3) == [10, 10]
    assert kf.plan_keyframe_durations("kling", 15, 3) == [10, 10]   # 7.5 s -> nearest allowed length
    assert kf.plan_keyframe_durations("kling_v3_std", 5, 3) == [5, 5]  # 5 s minimum per clip
    assert kf.plan_keyframe_durations("kling", 30, 4) == [10, 10, 10]


def test_higgsfield_clip_lengths_are_clamped_to_the_documented_range():
    assert kf.plan_keyframe_durations("higgsfield_kling_std", 9, 3) == [5, 5]
    assert kf.plan_keyframe_durations("higgsfield_kling_std", 4, 3) == [3, 3]       # min 3
    assert kf.plan_keyframe_durations("higgsfield_kling_std", 60, 3) == [15, 15]    # max 15
    assert kf.plan_keyframe_durations("higgsfield_minimax_h3", 4, 3) == [5, 5]      # min 5
    assert kf.plan_keyframe_durations("higgsfield_minimax_h3", 30, 4) == [10, 10, 10]


def test_two_photos_make_one_clip_and_four_make_three():
    assert len(kf.plan_keyframe_durations("kling", 10, 2)) == 1
    assert len(kf.plan_keyframe_durations("kling", 30, 4)) == 3


def test_invalid_input_is_rejected():
    with pytest.raises(ValueError):
        kf.plan_keyframe_durations("seedance", 10, 3)
    with pytest.raises(ValueError):
        kf.plan_keyframe_durations("kling", 10, 1)
    with pytest.raises(ValueError):
        kf.plan_keyframe_durations("kling", 10, 5)


def test_problem_message_is_english_and_lists_supported_models():
    msg = kf.keyframes_problem("seedance", 3)
    assert msg and "cannot use several photos" in msg and "unnecessary costs" in msg
    for label in ("Kling v3 Pro (Atlas)", "Kling 3.0 Std (Higgsfield)", "MiniMax H3 (Higgsfield)"):
        assert label in msg


def test_no_problem_when_the_setting_is_off_or_model_is_fine_or_not_applicable():
    assert kf.keyframes_problem("seedance", 1) is None          # classic single photo
    assert kf.keyframes_problem("kling", 3) is None              # supported
    assert kf.keyframes_problem("seedance_ref", 3) is None       # reference models take several photos
    assert kf.keyframes_problem("seedance_t2v", 3) is None       # text-to-video takes none


# ── request bodies ──────────────────────────────────────────────────────────

def test_higgsfield_kling_sends_last_image_url():
    body = hf.build_request_body(
        "higgsfield_kling_std", prompt="p", duration=5,
        image_urls=["https://a"], end_image_url="https://b",
    )
    assert body["image_url"] == "https://a" and body["last_image_url"] == "https://b"


def test_higgsfield_minimax_sends_end_image_url():
    body = hf.build_request_body(
        "higgsfield_minimax_h3", prompt="p", duration=5,
        image_urls=["https://a"], end_image_url="https://b",
    )
    assert body["image_url"] == "https://a" and body["end_image_url"] == "https://b"


def test_higgsfield_bodies_without_an_end_image_are_unchanged():
    for model in ("higgsfield_kling_std", "higgsfield_minimax_h3"):
        body = hf.build_request_body(model, prompt="p", duration=5, image_urls=["https://a"])
        assert "last_image_url" not in body and "end_image_url" not in body


def test_higgsfield_reference_model_refuses_an_end_frame():
    with pytest.raises(hf.HiggsfieldError):
        hf.build_request_body(
            "higgsfield_seedance_ref", prompt="p", duration=5,
            image_urls=["https://a"], end_image_url="https://b",
        )


class FakeAtlas:
    def __init__(self, tmp_path):
        self.calls = []
        self.static_dir = str(tmp_path)

    async def generate_video(self, model, params):
        self.calls.append((model, params))
        return "https://out/v.mp4"

    async def download(self, url, ext="mp4"):
        return f"downloaded.{ext}"


def _kling(tmp_path):
    svc = KlingService(api_key="k", static_dir=str(tmp_path))
    svc._atlas = FakeAtlas(tmp_path)
    return svc


def test_atlas_kling_sends_end_image(tmp_path):
    svc = _kling(tmp_path)
    asyncio.run(svc.generate_clip(
        prompt="p", image_url="https://a", end_image_url="https://b",
        duration=5, model_id=MODEL_IDS["kling"],
    ))
    model, params = svc._atlas.calls[0]
    assert model == MODEL_IDS["kling"]
    assert params["image"] == "https://a" and params["end_image"] == "https://b"


def test_atlas_kling_without_end_image_is_unchanged(tmp_path):
    svc = _kling(tmp_path)
    asyncio.run(svc.generate_clip(prompt="p", image_url="https://a", duration=5, model_id=MODEL_IDS["kling"]))
    assert "end_image" not in svc._atlas.calls[0][1]


def test_atlas_kling_refuses_an_end_frame_it_cannot_use(tmp_path):
    svc = _kling(tmp_path)
    with pytest.raises(ValueError):   # reference model
        asyncio.run(svc.generate_clip(
            prompt="p", image_urls=["https://a"], end_image_url="https://b",
            duration=5, model_id=MODEL_IDS["kling_o3_pro_ref"],
        ))
    with pytest.raises(ValueError):   # no start image
        asyncio.run(svc.generate_clip(prompt="p", end_image_url="https://b", duration=5))
    assert svc._atlas.calls == []     # nothing was submitted


# ── clip-by-pair orchestration ──────────────────────────────────────────────

class FakeKling:
    def __init__(self):
        self.uploads = []
        self.clips = []

    async def upload_photo(self, path):
        self.uploads.append(path)
        return "https://pub/" + path.replace("\\", "/").rsplit("/", 1)[-1]

    async def generate_clip(self, **kwargs):
        self.clips.append(kwargs)
        return f"clip{len(self.clips)}.mp4"


class FakeHiggsfield:
    def __init__(self):
        self.clips = []

    async def generate_clip(self, model, **kwargs):
        self.clips.append({"model": model, **kwargs})
        return f"hf{len(self.clips)}.mp4"


class FakeGemini:
    def __init__(self):
        self.concat_args = None

    async def concat_videos(self, paths, crossfade=0.5, transitions=None):
        self.concat_args = (list(paths), crossfade, transitions)
        return "joined.mp4"


def _photos(tmp_path, n):
    paths = []
    for i in range(n):
        p = tmp_path / f"photo{i + 1}.jpg"
        p.write_bytes(b"x")
        paths.append(str(p))
    return paths


async def _notify(_text):
    return None


def _run_keyframes(monkeypatch, tmp_path, model, n_photos, target, shots=None, scene="a scene"):
    kling, hfs, gemini = FakeKling(), FakeHiggsfield(), FakeGemini()
    monkeypatch.setitem(container.container, KlingService, kling)
    monkeypatch.setitem(container.container, HiggsfieldService, hfs)
    out = asyncio.run(common._generate_keyframes(
        gemini=gemini,
        settings={"video_model": model, "target_duration": target},
        video_prompt=scene, voiceover_text="", image_paths=_photos(tmp_path, n_photos),
        notify=_notify, shots=shots,
    ))
    return out, kling, hfs, gemini


def test_three_photos_on_atlas_kling_make_two_clips_in_order(monkeypatch, tmp_path):
    out, kling, _, gemini = _run_keyframes(monkeypatch, tmp_path, "kling", 3, 10)
    assert len(kling.clips) == 2
    first, second = kling.clips
    assert (first["image_url"], first["end_image_url"]) == ("https://pub/photo1.jpg", "https://pub/photo2.jpg")
    assert (second["image_url"], second["end_image_url"]) == ("https://pub/photo2.jpg", "https://pub/photo3.jpg")
    assert first["duration"] == second["duration"] == 5
    assert first["model_id"] == MODEL_IDS["kling"]
    assert gemini.concat_args[0] == ["clip1.mp4", "clip2.mp4"]
    assert gemini.concat_args[1] == 0            # hard cut: both sides of the join are the same photo
    assert out["raw_video_path"] == "joined.mp4" and out["has_native_audio"] is False
    assert len(kling.uploads) == 3               # each photo uploaded once, not once per clip


def test_four_photos_on_higgsfield_minimax_make_three_clips(monkeypatch, tmp_path):
    out, _, hfs, gemini = _run_keyframes(monkeypatch, tmp_path, "higgsfield_minimax_h3", 4, 30)
    assert [c["duration"] for c in hfs.clips] == [10, 10, 10]
    assert [(c["image_urls"], c["end_image_url"]) for c in hfs.clips] == [
        (["https://pub/photo1.jpg"], "https://pub/photo2.jpg"),
        (["https://pub/photo2.jpg"], "https://pub/photo3.jpg"),
        (["https://pub/photo3.jpg"], "https://pub/photo4.jpg"),
    ]
    assert all(c["model"] == "higgsfield_minimax_h3" for c in hfs.clips)
    assert gemini.concat_args[0] == ["hf1.mp4", "hf2.mp4", "hf3.mp4"]


def test_two_photos_make_a_single_clip_without_concat(monkeypatch, tmp_path):
    out, kling, _, gemini = _run_keyframes(monkeypatch, tmp_path, "kling_v3_std", 2, 10)
    assert len(kling.clips) == 1 and gemini.concat_args is None
    assert out["raw_video_path"] == "clip1.mp4"


def test_script_shots_become_the_clip_prompts_when_the_count_matches(monkeypatch, tmp_path):
    shots = [{"scene_prompt": "walks to the door", "camera_motion": "push in"},
             {"scene_prompt": "opens it", "camera_motion": ""}]
    _, kling, _, _ = _run_keyframes(monkeypatch, tmp_path, "kling", 3, 10, shots=shots)
    assert kling.clips[0]["prompt"] == "walks to the door. Camera: push in."
    assert kling.clips[1]["prompt"] == "opens it"


def test_every_pair_gets_a_prompt_even_when_the_script_has_a_different_shot_count(monkeypatch, tmp_path):
    shots = [{"scene_prompt": "only one shot"}] * 4
    _, kling, _, _ = _run_keyframes(monkeypatch, tmp_path, "kling", 3, 10, shots=shots, scene="a b")
    assert len(kling.clips) == 2 and all(c["prompt"].strip() for c in kling.clips)


def test_missing_photo_stops_before_any_paid_call(monkeypatch, tmp_path):
    kling, hfs = FakeKling(), FakeHiggsfield()
    monkeypatch.setitem(container.container, KlingService, kling)
    monkeypatch.setitem(container.container, HiggsfieldService, hfs)
    paths = _photos(tmp_path, 3)
    paths[1] = str(tmp_path / "gone.jpg")
    with pytest.raises(RuntimeError, match="upload the photos again"):
        asyncio.run(common._generate_keyframes(
            gemini=FakeGemini(), settings={"video_model": "kling", "target_duration": 10},
            video_prompt="s", voiceover_text="", image_paths=paths, notify=_notify,
        ))
    assert kling.clips == [] and kling.uploads == []


# ── routing: classic flow untouched, unsupported model refused ──────────────

def _route(monkeypatch, model, photos, keyframes):
    seen = {}

    async def fake_keyframes(**kwargs):
        seen["keyframes"] = True
        return {"mode": "keyframes"}

    async def fake_kling(**kwargs):
        seen["kling"] = True
        return {"mode": "classic-kling"}

    async def fake_seedance(**kwargs):
        seen["seedance"] = True
        return {"mode": "classic-seedance"}

    monkeypatch.setattr(common, "_generate_keyframes", fake_keyframes)
    monkeypatch.setattr(common, "_generate_kling", fake_kling)
    monkeypatch.setattr(common, "_generate_seedance", fake_seedance)
    result = asyncio.run(common.generate_video_for_model(
        gemini=None, settings={"video_model": model, "target_duration": 10},
        video_prompt="s", voiceover_text="", image_paths=photos, notify=_notify,
        keyframes=keyframes,
    ))
    return result, seen


def test_keyframes_flag_routes_supported_models_to_keyframes(monkeypatch):
    result, seen = _route(monkeypatch, "kling", ["a", "b", "c"], keyframes=True)
    assert result == {"mode": "keyframes"} and seen == {"keyframes": True}


def test_without_the_flag_the_classic_path_runs_even_with_several_photos(monkeypatch):
    result, seen = _route(monkeypatch, "kling", ["a", "b", "c"], keyframes=False)
    assert result == {"mode": "classic-kling"} and "keyframes" not in seen


def test_a_single_photo_always_uses_the_classic_path(monkeypatch):
    result, seen = _route(monkeypatch, "kling", ["a"], keyframes=True)
    assert result == {"mode": "classic-kling"}


def test_unsupported_model_is_refused_before_any_generation(monkeypatch):
    with pytest.raises(RuntimeError, match="cannot use several photos"):
        _route(monkeypatch, "seedance", ["a", "b", "c"], keyframes=True)


# ── upload step ─────────────────────────────────────────────────────────────

class FakeState:
    def __init__(self, data=None):
        self.data = dict(data or {})
        self.state = None

    async def update_data(self, **kwargs):
        self.data.update(kwargs)

    async def get_data(self):
        return dict(self.data)

    async def set_state(self, state):
        self.state = state


class FakeMessage:
    def __init__(self):
        self.chat = SimpleNamespace(id=-100)
        self.from_user = SimpleNamespace(id=42)
        self.sent = []

    async def answer(self, text, **kwargs):
        self.sent.append(text)


def _start_upload(settings):
    message, state = FakeMessage(), FakeState()
    asyncio.run(gen._start_own_photo_upload(message, state, settings))
    return message, state


def test_keyframes_on_with_a_supported_model_asks_for_all_photos_in_order():
    message, state = _start_upload({"video_model": "kling", "keyframe_photos": 3})
    assert state.data["keyframes_mode"] is True and state.data["own_photo_target_count"] == 3
    assert "Keyframes mode" in message.sent[-1] and "Send photo 1 of 3" in message.sent[-1]
    assert len(message.sent) == 1   # no warning


def test_unsupported_model_gets_the_warning_and_a_single_photo():
    message, state = _start_upload({"video_model": "seedance", "keyframe_photos": 3})
    assert "cannot use several photos" in message.sent[0]
    assert state.data["keyframes_mode"] is False and state.data["own_photo_target_count"] == 1
    assert "Send photo 1 of 1" in message.sent[-1]


def test_classic_single_photo_flow_is_unchanged():
    for settings in ({"video_model": "kling"}, {"video_model": "kling", "keyframe_photos": 1},
                     {"video_model": "seedance", "keyframe_photos": 1}):
        message, state = _start_upload(settings)
        assert state.data["keyframes_mode"] is False and state.data["own_photo_target_count"] == 1
        assert len(message.sent) == 1 and "Send photo 1 of 1" in message.sent[0]
        assert "Keyframes" not in message.sent[0] and "cannot use" not in message.sent[0]


def test_reference_models_keep_collecting_image_count_photos_with_an_info_note():
    message, state = _start_upload({"video_model": "seedance_ref", "keyframe_photos": 3, "image_count": 2})
    assert state.data["keyframes_mode"] is False and state.data["own_photo_target_count"] == 2
    assert "not used with reference models" in message.sent[0]
    assert "cannot use several photos" not in " ".join(message.sent)


def test_reference_models_without_keyframes_are_unchanged():
    message, state = _start_upload({"video_model": "seedance_ref", "image_count": 3})
    assert state.data["own_photo_target_count"] == 3 and len(message.sent) == 1


def test_last_photo_shows_the_plan(monkeypatch):
    class DB:
        async def get_settings(self, user_id, chat_id):
            return {"video_model": "kling", "target_duration": 10}

    previews = []

    async def fake_preview(message, paths, captions=None):
        previews.append(captions)

    monkeypatch.setitem(container.container, DBService, DB())
    monkeypatch.setattr(gen, "_send_image_preview", fake_preview)
    message = FakeMessage()
    state = FakeState({"own_photo_paths": ["a", "b", "c"], "own_photo_target_count": 3, "keyframes_mode": True})
    asyncio.run(gen._finish_or_continue_own_photos(message, state))
    assert previews == [["Keyframe 1/3", "Keyframe 2/3", "Keyframe 3/3"]]
    assert "3 photos → 2 clip(s) of 5s (10s in total)" in message.sent[-1]
    assert state.data["image_paths"] == ["a", "b", "c"]


def test_classic_last_photo_message_has_no_plan(monkeypatch):
    async def fake_preview(message, paths, captions=None):
        return None

    monkeypatch.setattr(gen, "_send_image_preview", fake_preview)
    message = FakeMessage()
    state = FakeState({"own_photo_paths": ["a"], "own_photo_target_count": 1})
    asyncio.run(gen._finish_or_continue_own_photos(message, state))
    assert "clip(s)" not in message.sent[-1] and "Photo(s) received." in message.sent[-1]


# ── price shown to the operator ─────────────────────────────────────────────

def test_cost_uses_the_real_length_in_keyframes_mode():
    data = {"keyframes_mode": True, "photo_source": "own", "image_paths": ["a", "b", "c"]}
    settings = {"video_model": "kling", "target_duration": 5}
    eff = common._cost_settings(settings, data)
    assert eff["target_duration"] == 10          # 2 clips x 5 s: a 5 s setting costs 10 s
    assert "10s" in common._video_cost_line(eff)


def test_cost_is_unchanged_in_classic_mode():
    settings = {"video_model": "kling", "target_duration": 5}
    assert common._cost_settings(settings, {}) is settings
    assert common._cost_settings(settings, {"keyframes_mode": False, "image_paths": ["a"]}) is settings
    # the flag from an earlier run does not apply to a classic (non-own-photo) run
    assert common._cost_settings(settings, {"keyframes_mode": True, "photo_source": "auto"}) is settings


# ── settings ────────────────────────────────────────────────────────────────

class FakeSettingsMessage:
    def __init__(self):
        self.chat = SimpleNamespace(id=-100)
        self.sent = []
        self.edits = []
        self.markups = []

    async def answer(self, text, **kwargs):
        self.sent.append(text)

    async def edit_text(self, text, parse_mode=None, reply_markup=None):
        self.edits.append(text)

    async def edit_reply_markup(self, reply_markup=None):
        self.markups.append(reply_markup)


class FakeCallback:
    def __init__(self, data):
        self.data = data
        self.from_user = SimpleNamespace(id=42)
        self.message = FakeSettingsMessage()

    async def answer(self, text=None, show_alert=False):
        return None


class FakeSettingsDB:
    def __init__(self, data):
        self.data = dict(data)

    async def get_settings(self, user_id, chat_id):
        return dict(self.data)

    async def update_settings(self, user_id, chat_id, values):
        self.data.update(values)
        return dict(self.data)


def test_choosing_keyframes_with_an_unsupported_model_warns_immediately(monkeypatch):
    db = FakeSettingsDB({"video_model": "seedance"})
    monkeypatch.setitem(container.container, DBService, db)
    cb = FakeCallback("settings:keyframes:3")
    asyncio.run(st.settings_keyframes_selected(cb))
    assert db.data["keyframe_photos"] == 3
    assert any("cannot use several photos" in t for t in cb.message.sent)


def test_choosing_keyframes_with_a_supported_model_does_not_warn(monkeypatch):
    db = FakeSettingsDB({"video_model": "kling"})
    monkeypatch.setitem(container.container, DBService, db)
    cb = FakeCallback("settings:keyframes:3")
    asyncio.run(st.settings_keyframes_selected(cb))
    assert db.data["keyframe_photos"] == 3
    assert not any("cannot use" in t for t in cb.message.sent)


def test_turning_keyframes_off_returns_to_the_classic_flow(monkeypatch):
    db = FakeSettingsDB({"video_model": "seedance", "keyframe_photos": 3})
    monkeypatch.setitem(container.container, DBService, db)
    cb = FakeCallback("settings:keyframes:1")
    asyncio.run(st.settings_keyframes_selected(cb))
    assert db.data["keyframe_photos"] == 1
    assert not any("cannot use" in t for t in cb.message.sent)


def test_invalid_keyframe_value_is_rejected(monkeypatch):
    db = FakeSettingsDB({})
    monkeypatch.setitem(container.container, DBService, db)
    for data in ("settings:keyframes:9", "settings:keyframes:x"):
        asyncio.run(st.settings_keyframes_selected(FakeCallback(data)))
    assert "keyframe_photos" not in db.data


def test_switching_video_model_warns_when_keyframes_are_on(monkeypatch):
    from services.gemini import GeminiService
    monkeypatch.setitem(container.container, GeminiService, GeminiService(api_key=""))
    db = FakeSettingsDB({"video_model": "kling", "keyframe_photos": 3})
    monkeypatch.setitem(container.container, DBService, db)

    cb = FakeCallback("video-model:seedance")
    asyncio.run(st.handle_video_model_selected(cb))
    assert any("cannot use several photos" in t for t in cb.message.sent)
    assert any("Selected" in t for t in cb.message.edits)     # the selection itself still went through
    assert db.data["video_model"] == "seedance"


def test_switching_to_a_supported_model_or_with_keyframes_off_does_not_warn(monkeypatch):
    from services.gemini import GeminiService
    monkeypatch.setitem(container.container, GeminiService, GeminiService(api_key=""))

    for saved, target in (({"keyframe_photos": 3}, "kling_v3_std"), ({"keyframe_photos": 1}, "seedance"), ({}, "seedance")):
        db = FakeSettingsDB({"video_model": "kling", **saved})
        monkeypatch.setitem(container.container, DBService, db)
        cb = FakeCallback(f"video-model:{target}")
        asyncio.run(st.handle_video_model_selected(cb))
        assert not any("cannot use" in t for t in cb.message.sent), (saved, target)


def test_keyframes_submenu_marks_current_and_reports_clip_counts():
    rows = get_keyframes_keyboard(3).inline_keyboard
    labels = [r[0].text for r in rows]
    assert labels[0].startswith("Off") and labels[2].startswith("✅ 3 photos → 2 clips")
    assert [r[0].callback_data for r in rows[:4]] == [f"settings:keyframes:{n}" for n in (1, 2, 3, 4)]
    assert rows[-1][0].callback_data == "settings:advanced"


def test_advanced_menu_shows_the_keyframes_state():
    def labels(**kw):
        return [b.text for row in get_advanced_settings_keyboard(**kw).inline_keyboard for b in row]
    assert "🎞 Keyframes: off" in labels()
    assert "🎞 Keyframes: 3 photos" in labels(keyframe_photos=3)


# ── script instructions ─────────────────────────────────────────────────────

class FakeRouter:
    def __init__(self, reply):
        self.reply = reply
        self.system = None
        self.images = None
        self.providers_map = {}

    async def generate_text(self, prompt, system_prompt="", model="", image_paths=None):
        self.system, self.images = system_prompt, image_paths
        return self.reply


def _script_json(n_shots):
    return json.dumps({
        "spoken_text": "hello", "caption": "#x",
        "shots": [{"index": i, "duration_seconds": 5, "scene_prompt": f"s{i}",
                   "camera_motion": "static", "transition": "cut"} for i in range(n_shots)],
    })


def _build(model, photos, keyframes, target=10):
    router = FakeRouter(_script_json(2))
    asyncio.run(common._build_video_prompt(
        "idea", {"video_model": model, "target_duration": target}, router,
        own_photo_paths=photos, keyframes=keyframes,
    ))
    return router


def test_keyframes_script_is_told_to_write_one_shot_per_pair():
    router = _build("kling", ["a", "b", "c"], keyframes=True)
    assert "KEYFRAMES" in router.system and "EXACTLY 2 elements" in router.system
    assert "photo 1" in router.system and "photo 2" in router.system
    assert router.images == ["a", "b", "c"]            # the model really sees all photos


def test_four_photos_ask_for_three_shots():
    router = _build("higgsfield_kling_std", ["a", "b", "c", "d"], keyframes=True, target=30)
    assert "EXACTLY 3 elements" in router.system


def test_classic_single_photo_script_is_unchanged():
    router = _build("kling", ["a"], keyframes=False)
    assert "KEYFRAMES" not in router.system
    assert "ATTACHED PHOTO:" in router.system and "starting frame" in router.system
    assert "EXACTLY 1 elements" in router.system


def test_keyframe_instructions_are_not_used_for_an_unsupported_model():
    router = _build("seedance", ["a", "b", "c"], keyframes=True)
    assert "KEYFRAMES" not in router.system
