from aiogram.types import (
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    ReplyKeyboardMarkup,
    KeyboardButton,
)


SETTINGS_BUTTON_TEXT = "⚙️ Settings"
GENERATE_BUTTON_TEXT = "🎬 Generate video"


def get_persistent_keyboard():
    """Reply keyboard always visible at the bottom of the chat."""
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text=GENERATE_BUTTON_TEXT)],
            [KeyboardButton(text=SETTINGS_BUTTON_TEXT)],
        ],
        resize_keyboard=True,
        is_persistent=True,
    )


def get_prompt_keyboard():
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Continue", callback_data="prompt_ok")],
            [InlineKeyboardButton(text="Regenerate", callback_data="prompt_regenerate")],
        ]
    )


def get_image_keyboard():
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Continue", callback_data="image_ok")],
            [InlineKeyboardButton(text="Change prompt", callback_data="image_prompt_change")],
            [InlineKeyboardButton(text="Regenerate", callback_data="image_regenerate")],
        ]
    )


def get_video_keyboard(subtitles_on: bool = True):
    subs_label = "🔤 Subtitles: ON" if subtitles_on else "🔤 Subtitles: OFF"
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Continue", callback_data="video_ok")],
            [InlineKeyboardButton(text="Change prompt", callback_data="video_prompt_change")],
            [InlineKeyboardButton(text="Regenerate", callback_data="video_regenerate")],
            [InlineKeyboardButton(text="✏️ Edit video", callback_data="video_edit")],
            [InlineKeyboardButton(text=subs_label, callback_data="video_subtitles_toggle")],
        ]
    )


def get_accounts_keyboard(accounts: list[dict]):
    rows = []
    for acc in accounts:
        platform_icon = "▶️" if acc["platform"] == "youtube" else "🎵"
        label = acc["label"] or acc["account_id"]
        rows.append([
            InlineKeyboardButton(
                text=f"{platform_icon} {acc['platform'].capitalize()} — {label}",
                callback_data="accounts:noop",
            ),
            InlineKeyboardButton(
                text="🗑 Remove",
                callback_data=f"accounts:remove:{acc['id']}",
            ),
        ])
    rows.append([InlineKeyboardButton(text="➕ Add account", callback_data="accounts:add")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def get_duration_keyboard(current: int = 30):
    rows = []
    for d in [15, 30, 45, 60]:
        mark = "✅ " if d == current else ""
        rows.append([InlineKeyboardButton(
            text=f"{mark}{d}s",
            callback_data=f"settings:duration:{d}",
        )])
    rows.append([InlineKeyboardButton(
        text="✏️ Custom duration",
        callback_data="settings:duration:custom",
    )])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def get_video_prompt_keyboard():
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="▶️ Generate video", callback_data="vp_ok")],
            [
                InlineKeyboardButton(text="🔄 Regen scene", callback_data="vp_scene_regen"),
                InlineKeyboardButton(text="✏️ Edit scene", callback_data="vp_scene_edit"),
            ],
            [
                InlineKeyboardButton(text="🔄 Regen voiceover", callback_data="vp_vo_regen"),
                InlineKeyboardButton(text="✏️ Edit voiceover", callback_data="vp_vo_edit"),
            ],
        ]
    )


def get_publish_time_keyboard():
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="📤 Publish now", callback_data="publish_time:now")],
            [InlineKeyboardButton(text="📅 Schedule", callback_data="publish_time:schedule")],
        ]
    )


def get_publish_keyboard():
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="publish", callback_data="publish")],
            [InlineKeyboardButton(text="cancel", callback_data="cancel")],
        ]
    )


def get_resolution_keyboard(current: str = "720p"):
    options = [
        ("720p",  "720p — default, fastest"),
        ("1080p", "1080p — Seedance only, slower"),
    ]
    rows = []
    for val, desc in options:
        mark = "✅ " if val == current else ""
        rows.append([InlineKeyboardButton(
            text=f"{mark}{desc}", callback_data=f"settings:resolution:{val}"
        )])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def get_speed_keyboard(current: float = 1.0):
    options = [1.0, 1.15, 1.3, 1.5]
    rows = []
    for s in options:
        mark = "✅ " if abs(s - current) < 0.01 else ""
        label = f"{mark}{s:.2f}×" if s != 1.0 else f"{mark}1.0× (normal)"
        rows.append([InlineKeyboardButton(text=label, callback_data=f"settings:speed:{s}")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def get_settings_keyboard(is_t2v: bool = False):
    """Main settings page — model selection and content style."""
    model_row = [InlineKeyboardButton(text="🧠 Text model", callback_data="settings:text_model")]
    # T2V models generate straight from text — the image model is unused, hide it.
    if not is_t2v:
        model_row.append(InlineKeyboardButton(text="🖼 Image model", callback_data="settings:image_model"))
    return InlineKeyboardMarkup(
        inline_keyboard=[
            model_row,
            [
                InlineKeyboardButton(text="🎬 Video model", callback_data="settings:video_model"),
            ],
            [
                InlineKeyboardButton(text="🎨 Стиль контента", callback_data="settings:style"),
            ],
            [
                InlineKeyboardButton(text="⚙️ More settings", callback_data="settings:advanced"),
            ],
        ]
    )


def get_style_keyboard(presets: dict, current_key: str = "", is_t2v: bool = False):
    """Content-style menu: built-in niche presets + one-shot AI custom niche
    + manual editing of the underlying prompts."""
    rows = []
    for key, p in presets.items():
        mark = "✅ " if key == current_key else ""
        rows.append([InlineKeyboardButton(
            text=f"{mark}{p['label']}", callback_data=f"style:set:{key}"
        )])
    custom_mark = "✅ " if current_key == "custom" else ""
    rows.append([InlineKeyboardButton(
        text=f"{custom_mark}✍️ Своя ниша (опишу словами)", callback_data="style:custom"
    )])
    edit_row = [InlineKeyboardButton(text="✏️ Промпт сюжета", callback_data="settings:plot_prompt")]
    # For T2V models the reference image stage is skipped — hide its editor.
    if not is_t2v:
        edit_row.append(InlineKeyboardButton(text="✏️ Промпт фото", callback_data="settings:image_prompt"))
    rows.append(edit_row)
    rows.append([InlineKeyboardButton(text="← Back", callback_data="settings:back")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def get_advanced_settings_keyboard(
    subtitles_default_on: bool = True,
    grade_on: bool = False,
    target_duration: int = 30,
    utc_offset: int = 0,
    sfx_on: bool = False,
    video_speed: float = 1.0,
    video_resolution: str = "720p",
    image_count: int = 1,
):
    """Advanced settings sub-page — toggles, quality, publishing."""
    subs_label  = "🔤 Subtitles: ON"  if subtitles_default_on else "🔤 Subtitles: OFF"
    grade_label = "🎨 Grade: ON"      if grade_on             else "🎨 Grade: OFF"
    sfx_label   = "🔊 SFX: ON"       if sfx_on               else "🔊 SFX: OFF"
    spd_label   = f"⚡ {video_speed:.2f}×" if video_speed != 1.0 else "⚡ Speed: normal"
    tz_sign     = "+" if utc_offset >= 0 else ""
    img_label   = f"🖼 {image_count} photo{'s' if image_count > 1 else ''}"
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=subs_label, callback_data="settings:subtitles_toggle")],
            [
                InlineKeyboardButton(text=grade_label,       callback_data="settings:grade_toggle"),
                InlineKeyboardButton(text="🎚 Grade params", callback_data="settings:grade_params"),
            ],
            [
                InlineKeyboardButton(text=sfx_label, callback_data="settings:sfx_toggle"),
                InlineKeyboardButton(text=spd_label, callback_data="settings:speed"),
            ],
            [
                InlineKeyboardButton(text=f"📐 {video_resolution}", callback_data="settings:resolution"),
                InlineKeyboardButton(text=f"⏱ {target_duration}s", callback_data="settings:duration"),
            ],
            [
                InlineKeyboardButton(text=img_label, callback_data="settings:image_count"),
            ],
            [
                InlineKeyboardButton(text="🚫 Negative prompt", callback_data="settings:negative_prompt"),
                InlineKeyboardButton(text="🎙 Voice",            callback_data="settings:voice_id"),
            ],
            [
                InlineKeyboardButton(text="🎵 Music",               callback_data="settings:music_path"),
                InlineKeyboardButton(text=f"🕐 UTC{tz_sign}{utc_offset}", callback_data="settings:timezone"),
            ],
            [
                InlineKeyboardButton(text="📤 Accounts",      callback_data="settings:accounts"),
                InlineKeyboardButton(text="📋 Show settings", callback_data="settings:show"),
            ],
            [
                InlineKeyboardButton(text="← Back", callback_data="settings:back"),
            ],
        ]
    )


def get_image_count_keyboard(current: int = 1):
    options = [
        (1, "1 photo — standard"),
        (2, "2 photos — better variety"),
        (3, "3 photos — more reference"),
        (4, "4 photos — maximum reference"),
    ]
    rows = []
    for val, desc in options:
        mark = "✅ " if val == current else ""
        rows.append([InlineKeyboardButton(
            text=f"{mark}{desc}", callback_data=f"settings:image_count:{val}"
        )])
    return InlineKeyboardMarkup(inline_keyboard=rows)
