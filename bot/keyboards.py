from aiogram.types import (
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    ReplyKeyboardMarkup,
    KeyboardButton,
)


SETTINGS_BUTTON_TEXT = "⚙️ Settings"
GENERATE_BUTTON_TEXT = "🎬 Generate video"
HISTORY_BUTTON_TEXT  = "📼 History"


def get_persistent_keyboard():
    """Reply keyboard always visible at the bottom of the chat."""
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text=GENERATE_BUTTON_TEXT)],
            [KeyboardButton(text=SETTINGS_BUTTON_TEXT), KeyboardButton(text=HISTORY_BUTTON_TEXT)],
        ],
        resize_keyboard=True,
        is_persistent=True,
    )


def get_prompt_keyboard():
    """Keyboard for confirming prompts"""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Continue", callback_data="prompt_ok")],
            [InlineKeyboardButton(text="Regenerate", callback_data="prompt_regenerate")],
        ]
    )


def get_image_keyboard():
    """Keyboard for image actions"""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Continue", callback_data="image_ok")],
            [InlineKeyboardButton(text="Change prompt", callback_data="image_prompt_change")],
            [InlineKeyboardButton(text="Regenerate", callback_data="image_regenerate")],
        ]
    )


def get_video_keyboard(subtitles_on: bool = True):
    """Keyboard for video actions. Includes per-video subtitle toggle."""
    subs_label = "🔤 Subtitles: ON" if subtitles_on else "🔤 Subtitles: OFF"
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Continue", callback_data="video_ok")],
            [InlineKeyboardButton(text="Change prompt", callback_data="video_prompt_change")],
            [InlineKeyboardButton(text="Regenerate", callback_data="video_regenerate")],
            [InlineKeyboardButton(text=subs_label, callback_data="video_subtitles_toggle")],
            [InlineKeyboardButton(text="🎞 Send raw video", callback_data="video_send_raw")],
        ]
    )


def get_accounts_keyboard(accounts: list[dict]):
    """Keyboard for managing per-chat accounts.

    Each account gets its own row with a Remove button.
    A permanent 'Add account' row sits at the bottom.
    """
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


def get_duration_keyboard(current: int = 15):
    """Keyboard for selecting target video duration in settings."""
    rows = []
    for d in [15, 30, 45, 60]:
        mark = "✅ " if d == current else ""
        rows.append([InlineKeyboardButton(
            text=f"{mark}{d}s",
            callback_data=f"settings:duration:{d}",
        )])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def get_video_prompt_keyboard():
    """Keyboard shown after the video prompt is generated — per-part controls."""
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
    """Keyboard shown after the user enters the video title."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="📤 Publish now", callback_data="publish_time:now")],
            [InlineKeyboardButton(text="📅 Schedule", callback_data="publish_time:schedule")],
        ]
    )


def get_publish_keyboard():
    """Keyboard for publish"""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="publish", callback_data="publish")],
            [InlineKeyboardButton(text="cancel", callback_data="cancel")],
        ]
    )


def get_resolution_keyboard(current: str = "720p"):
    """Keyboard for selecting video resolution (Seedance only — Kling/OmniHuman
    output is fixed by the model)."""
    options = [
        ("720p", "720p — default, fastest"),
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
    """Keyboard for selecting video playback speed."""
    options = [1.0, 1.15, 1.3, 1.5]
    rows = []
    for s in options:
        mark = "✅ " if abs(s - current) < 0.01 else ""
        label = f"{mark}{s:.2f}×" if s != 1.0 else f"{mark}1.0× (normal)"
        rows.append([InlineKeyboardButton(text=label, callback_data=f"settings:speed:{s}")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def get_settings_keyboard(
    subtitles_default_on: bool = True,
    grade_on: bool = False,
    target_duration: int = 15,
    utc_offset: int = 0,
    sfx_on: bool = False,
    video_speed: float = 1.0,
    video_resolution: str = "720p",
):
    """Keyboard for /settings menu. Reflects current toggles."""
    subs_label = (
        "🔤 Subtitles default: ON"
        if subtitles_default_on
        else "🔤 Subtitles default: OFF"
    )
    grade_label = "🎨 Colour grade: ON" if grade_on else "🎨 Colour grade: OFF"
    sfx_label   = "🔊 SFX / ASMR: ON"  if sfx_on  else "🔊 SFX / ASMR: OFF"
    spd_label   = f"⚡ Speed: {video_speed:.2f}×" if video_speed != 1.0 else "⚡ Speed: 1.0× (normal)"
    res_label   = f"📐 Resolution: {video_resolution}"
    dur_label = f"⏱ Duration: {target_duration}s"
    tz_sign = "+" if utc_offset >= 0 else ""
    tz_label = f"🕐 Timezone: UTC{tz_sign}{utc_offset}"
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="🧠 Text model",  callback_data="settings:text_model"),
                InlineKeyboardButton(text="🖼 Image model", callback_data="settings:image_model"),
            ],
            [
                InlineKeyboardButton(text="🎬 Video model",  callback_data="settings:video_model"),
                InlineKeyboardButton(text="📋 Show settings", callback_data="settings:show"),
            ],
            [
                InlineKeyboardButton(text="📝 Plot prompt",  callback_data="settings:plot_prompt"),
                InlineKeyboardButton(text="🖼 Image prompt", callback_data="settings:image_prompt"),
            ],
            [
                InlineKeyboardButton(text="🎬 Video prompt", callback_data="settings:video_prompt"),
            ],
            [
                InlineKeyboardButton(text=subs_label, callback_data="settings:subtitles_toggle"),
            ],
            [
                InlineKeyboardButton(text=grade_label, callback_data="settings:grade_toggle"),
                InlineKeyboardButton(text="🎚 Grade params", callback_data="settings:grade_params"),
            ],
            [
                InlineKeyboardButton(text=sfx_label, callback_data="settings:sfx_toggle"),
                InlineKeyboardButton(text=spd_label, callback_data="settings:speed"),
            ],
            [
                InlineKeyboardButton(text=res_label, callback_data="settings:resolution"),
            ],
            [
                InlineKeyboardButton(text="🚫 Negative prompt", callback_data="settings:negative_prompt"),
                InlineKeyboardButton(text="🎙 Voice", callback_data="settings:voice_id"),
            ],
            [
                InlineKeyboardButton(text="🎵 Background music", callback_data="settings:music_path"),
            ],
            [
                InlineKeyboardButton(text=dur_label, callback_data="settings:duration"),
                InlineKeyboardButton(text=tz_label,  callback_data="settings:timezone"),
            ],
            [
                InlineKeyboardButton(text="📤 Accounts", callback_data="settings:accounts"),
                InlineKeyboardButton(text="🪪 My ID",    callback_data="settings:my_id"),
            ],
        ]
    )
