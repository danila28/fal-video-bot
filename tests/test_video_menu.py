"""Two-level video model menu: service list, model list, labels and callback data."""

from bot.keyboards import (
    get_video_models_keyboard,
    get_video_service_keyboard,
    video_model_price,
    video_model_title,
)
from services.gemini import GeminiService
from services.pricing import video_provider


def _rows(markup):
    return markup.inline_keyboard


# ── labels ───────────────────────────────────────────────────────────────────

def test_title_strips_emoji_and_price():
    assert video_model_title("🖼🆕 Seedance 2.5 I2V · ~$0.09/s (price unverified)") == "Seedance 2.5 I2V"
    assert video_model_title("🎭 Seedance 2.0 Ref · Higgsfield · price not verified") == "Seedance 2.0 Ref"


def test_price_part_is_what_follows_the_separator():
    assert video_model_price("🖼 Kling v3 Pro I2V · $0.095/s") == "$0.095/s"
    assert video_model_price("no separator here") == ""


def test_every_catalog_entry_has_a_usable_title_and_service():
    entries = GeminiService(api_key="").get_video_models()
    for entry in entries:
        if entry.get("separator"):
            continue
        assert video_model_title(entry["price"]), entry["name"]
        assert video_provider(entry["name"]) in ("atlas", "higgsfield")


# ── level 1: services ────────────────────────────────────────────────────────

def test_service_menu_marks_current_service_and_counts_models():
    kb = get_video_service_keyboard("higgsfield", {"atlas": 21, "higgsfield": 3}, True)
    atlas, hf = _rows(kb)[0][0], _rows(kb)[1][0]
    assert atlas.callback_data == "vsvc:atlas" and "21 models" in atlas.text
    assert hf.callback_data == "vsvc:higgsfield" and hf.text.startswith("✅ ")


def test_service_menu_without_key_shows_disabled_higgsfield_row():
    kb = get_video_service_keyboard("", {"atlas": 21, "higgsfield": 3}, False)
    hf = _rows(kb)[1][0]
    assert hf.callback_data == "noop"
    assert "not configured" in hf.text


# ── level 2: models of one service ──────────────────────────────────────────

def test_model_menu_marks_current_and_has_back_to_services():
    entries = [
        {"separator": True, "label": "── Kling ──"},
        {"name": "kling", "price": "🖼 Kling v3 Pro I2V · $0.095/s"},
        {"name": "minimax_h3", "price": "🖼🆕 MiniMax H3 I2V · ~$0.09/s (price unverified)"},
    ]
    kb = get_video_models_keyboard(entries, "minimax_h3")
    rows = _rows(kb)
    assert rows[0][0].callback_data == "noop"                       # family header
    assert rows[1][0].callback_data == "video-model:kling"
    assert rows[2][0].text.startswith("✅ ")
    assert rows[2][0].callback_data == "video-model:minimax_h3"
    assert rows[-1][0].callback_data == "vsvc:back"


def test_real_catalog_menus_fit_telegram_limits():
    entries = GeminiService(api_key="").get_video_models()
    atlas = [e for e in entries if not e.get("separator") and video_provider(e["name"]) == "atlas"]
    hf = [e for e in entries if not e.get("separator") and video_provider(e["name"]) == "higgsfield"]
    for group, current in ((atlas, "seedance"), (hf, "higgsfield_kling_std")):
        kb = get_video_models_keyboard(group, current)
        for row in _rows(kb):
            for btn in row:
                assert len(btn.callback_data.encode()) <= 64
                assert btn.text  # no empty labels


def test_higgsfield_models_are_grouped_under_their_service():
    entries = GeminiService(api_key="").get_video_models()
    hf_names = [e["name"] for e in entries if not e.get("separator") and video_provider(e["name"]) == "higgsfield"]
    assert hf_names == ["higgsfield_kling_std", "higgsfield_minimax_h3", "higgsfield_seedance_ref"]
