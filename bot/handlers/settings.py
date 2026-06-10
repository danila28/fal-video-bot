"""Settings callbacks, model selection, and settings state handlers."""

import html
import json
import logging

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.types import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
    CallbackQuery,
)
from bot.guard import IsAllowed
from bot.keyboards import (
    get_accounts_keyboard,
    get_duration_keyboard,
    get_resolution_keyboard,
    get_settings_keyboard,
    get_speed_keyboard,
    SETTINGS_BUTTON_TEXT,
    HISTORY_BUTTON_TEXT,
)
from bot.states import GenerationState
from services.db import DBService
from services.gemini import GeminiService
from utils import container
from utils.consts import allowed_users
from bot.handlers.common import DEFAULT_TARGET_DURATION, DURATION_SEGMENTS

router = Router()
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# BASIC SETTINGS COMMANDS
# ─────────────────────────────────────────────

@router.message(F.text == SETTINGS_BUTTON_TEXT, IsAllowed(allowed_users))
async def handle_settings_button(message: Message, state: FSMContext):
    """Reply-keyboard button — opens inline settings menu in any state."""
    await state.clear()
    db = container.inject(DBService)
    settings = await db.get_settings(message.from_user.id, message.chat.id)
    subs_on = settings.get("subtitles_enabled", True)
    grade_on = settings.get("colour_grade_enabled", False)
    sfx_on = settings.get("sfx_enabled", False)
    speed = float(settings.get("video_speed", 1.0))
    res = settings.get("video_resolution", "720p")
    dur = settings.get("target_duration", DEFAULT_TARGET_DURATION)
    tz = settings.get("utc_offset_hours", 0)
    await message.answer("⚙️ Settings", reply_markup=get_settings_keyboard(subs_on, grade_on, dur, tz, sfx_on, speed, res))


@router.message(Command("id"))
async def handle_id(message: Message):
    await message.answer(f"Your Telegram ID: {message.from_user.id}")


@router.message(F.text == HISTORY_BUTTON_TEXT, IsAllowed(allowed_users))
@router.message(Command("history"), IsAllowed(allowed_users))
async def handle_history(message: Message):
    """Show the last 10 generated videos for this chat (re-sent from Telegram CDN)."""
    db = container.inject(DBService)
    rows = await db.get_video_history(message.from_user.id, message.chat.id, limit=10)
    if not rows:
        await message.answer("📭 No videos in history yet.")
        return

    await message.answer(f"📼 Last {len(rows)} video(s):")
    for row in rows:
        gen_label = {"initial": "🎬", "regen": "🔄", "extend": "➕"}.get(row["gen_type"], "🎬")
        ts = row["created_at"].strftime("%Y-%m-%d %H:%M")
        caption = f"{gen_label} <b>{html.escape(row['title'] or '—')}</b>\n<i>{ts}</i>"
        try:
            await message.answer_video(row["file_id"], caption=caption, parse_mode="HTML")
        except Exception as e:
            logger.warning(f"History re-send failed for file_id={row['file_id']}: {e}")
            await message.answer(
                f"{gen_label} {html.escape(row['title'] or '—')} — <i>video unavailable</i>",
                parse_mode="HTML",
            )


# ─────────────────────────────────────────────
# SETTINGS CALLBACKS
# ─────────────────────────────────────────────

@router.callback_query(lambda c: c.data == "settings:show", IsAllowed(allowed_users))
async def settings_show(callback: CallbackQuery):
    await callback.answer()
    try:
        db = container.inject(DBService)
        settings = await db.get_settings(callback.from_user.id, callback.message.chat.id)
        if not settings:
            await callback.message.answer("Settings are empty.")
            return
        raw = json.dumps(settings, indent=2, ensure_ascii=False)
        chunk_size = 3500
        for i in range(0, len(raw), chunk_size):
            chunk = raw[i: i + chunk_size]
            await callback.message.answer(f"<pre>{html.escape(chunk)}</pre>", parse_mode="HTML")
    except Exception as e:
        await callback.message.answer(f"Error showing settings: {e}")


@router.callback_query(lambda c: c.data == "settings:my_id", IsAllowed(allowed_users))
async def settings_my_id(callback: CallbackQuery):
    await callback.answer()
    await callback.message.answer(f"Your Telegram ID: {callback.from_user.id}")


@router.callback_query(lambda c: c.data == "settings:subtitles_toggle", IsAllowed(allowed_users))
async def settings_subtitles_toggle(callback: CallbackQuery):
    db = container.inject(DBService)
    user_id = callback.from_user.id
    chat_id = callback.message.chat.id
    settings = await db.get_settings(user_id, chat_id)
    new_value = not settings.get("subtitles_enabled", True)
    await db.update_settings(user_id, chat_id, {"subtitles_enabled": new_value})
    grade_on = settings.get("colour_grade_enabled", False)
    sfx_on = settings.get("sfx_enabled", False)
    speed = float(settings.get("video_speed", 1.0))
    res = settings.get("video_resolution", "720p")
    dur = settings.get("target_duration", DEFAULT_TARGET_DURATION)
    tz = settings.get("utc_offset_hours", 0)
    await callback.answer(f"Subtitles default: {'ON' if new_value else 'OFF'}")
    try:
        await callback.message.edit_reply_markup(
            reply_markup=get_settings_keyboard(new_value, grade_on, dur, tz, sfx_on, speed, res)
        )
    except Exception:
        await callback.message.answer(
            "⚙️ Settings", reply_markup=get_settings_keyboard(new_value, grade_on, dur, tz, sfx_on, speed, res)
        )


@router.callback_query(lambda c: c.data == "settings:grade_toggle", IsAllowed(allowed_users))
async def settings_grade_toggle(callback: CallbackQuery):
    db = container.inject(DBService)
    user_id = callback.from_user.id
    chat_id = callback.message.chat.id
    settings = await db.get_settings(user_id, chat_id)
    new_value = not settings.get("colour_grade_enabled", False)
    await db.update_settings(user_id, chat_id, {"colour_grade_enabled": new_value})
    subs_on = settings.get("subtitles_enabled", True)
    sfx_on = settings.get("sfx_enabled", False)
    speed = float(settings.get("video_speed", 1.0))
    res = settings.get("video_resolution", "720p")
    dur = settings.get("target_duration", DEFAULT_TARGET_DURATION)
    tz = settings.get("utc_offset_hours", 0)
    await callback.answer(f"Colour grade: {'ON' if new_value else 'OFF'}")
    try:
        await callback.message.edit_reply_markup(
            reply_markup=get_settings_keyboard(subs_on, new_value, dur, tz, sfx_on, speed, res)
        )
    except Exception:
        await callback.message.answer(
            "⚙️ Settings", reply_markup=get_settings_keyboard(subs_on, new_value, dur, tz, sfx_on, speed, res)
        )


@router.callback_query(lambda c: c.data == "settings:sfx_toggle", IsAllowed(allowed_users))
async def settings_sfx_toggle(callback: CallbackQuery):
    db = container.inject(DBService)
    user_id = callback.from_user.id
    chat_id = callback.message.chat.id
    settings = await db.get_settings(user_id, chat_id)
    new_value = not settings.get("sfx_enabled", False)
    await db.update_settings(user_id, chat_id, {"sfx_enabled": new_value})
    subs_on = settings.get("subtitles_enabled", True)
    grade_on = settings.get("colour_grade_enabled", False)
    speed = float(settings.get("video_speed", 1.0))
    res = settings.get("video_resolution", "720p")
    dur = settings.get("target_duration", DEFAULT_TARGET_DURATION)
    tz = settings.get("utc_offset_hours", 0)
    await callback.answer(f"SFX / ASMR: {'ON' if new_value else 'OFF'}")
    try:
        await callback.message.edit_reply_markup(
            reply_markup=get_settings_keyboard(subs_on, grade_on, dur, tz, new_value, speed, res)
        )
    except Exception:
        await callback.message.answer(
            "⚙️ Settings", reply_markup=get_settings_keyboard(subs_on, grade_on, dur, tz, new_value, speed, res)
        )


@router.callback_query(lambda c: c.data == "settings:resolution", IsAllowed(allowed_users))
async def settings_resolution(callback: CallbackQuery):
    await callback.answer()
    db = container.inject(DBService)
    settings = await db.get_settings(callback.from_user.id, callback.message.chat.id)
    current = settings.get("video_resolution", "720p")
    await callback.message.answer(
        f"📐 Current resolution: <b>{current}</b>\n\n"
        "• <b>720p</b> — default, fastest, supported by all models\n"
        "• <b>1080p</b> — Seedance only (lip-sync models output is fixed)\n",
        parse_mode="HTML",
        reply_markup=get_resolution_keyboard(current),
    )


@router.callback_query(lambda c: c.data.startswith("settings:resolution:"), IsAllowed(allowed_users))
async def settings_resolution_selected(callback: CallbackQuery):
    await callback.answer()
    try:
        res_val = callback.data.split(":")[-1]
        if res_val not in ("720p", "1080p"):
            await callback.message.answer("❌ Invalid resolution.")
            return
        db = container.inject(DBService)
        await db.update_settings(
            callback.from_user.id, callback.message.chat.id, {"video_resolution": res_val}
        )
        try:
            await callback.message.edit_reply_markup(reply_markup=get_resolution_keyboard(res_val))
        except Exception:
            pass
        await callback.message.answer(f"✅ Resolution set to <b>{res_val}</b>.", parse_mode="HTML")
    except Exception as e:
        await callback.message.answer(f"❌ Error: {e}")


@router.callback_query(lambda c: c.data == "settings:speed", IsAllowed(allowed_users))
async def settings_speed(callback: CallbackQuery):
    await callback.answer()
    db = container.inject(DBService)
    settings = await db.get_settings(callback.from_user.id, callback.message.chat.id)
    current = float(settings.get("video_speed", 1.0))
    await callback.message.answer(
        f"⚡ Current speed: <b>{current:.2f}×</b>\n\n"
        "Choose video playback speed.\n"
        "• 1.0× — normal speed\n"
        "• 1.15× — slightly faster, more dynamic\n"
        "• 1.3× — fast, more scenes fit in the same clip\n"
        "• 1.5× — turbo, maximum scenes per second\n\n"
        "Higher speed = more content visible, shorter clip duration.",
        parse_mode="HTML",
        reply_markup=get_speed_keyboard(current),
    )


@router.callback_query(lambda c: c.data.startswith("settings:speed:"), IsAllowed(allowed_users))
async def settings_speed_selected(callback: CallbackQuery):
    await callback.answer()
    try:
        speed = float(callback.data.split(":")[-1])
        if speed not in (1.0, 1.15, 1.3, 1.5):
            await callback.message.answer("❌ Invalid speed value.")
            return
        db = container.inject(DBService)
        await db.update_settings(
            callback.from_user.id, callback.message.chat.id, {"video_speed": speed}
        )
        try:
            await callback.message.edit_reply_markup(reply_markup=get_speed_keyboard(speed))
        except Exception:
            pass
        label = {1.0: "normal", 1.15: "slightly faster", 1.3: "fast", 1.5: "turbo"}.get(speed, "")
        await callback.message.answer(
            f"✅ Speed set to <b>{speed:.2f}×</b> ({label}).", parse_mode="HTML"
        )
    except Exception as e:
        await callback.message.answer(f"❌ Error: {e}")


@router.callback_query(lambda c: c.data == "settings:grade_params", IsAllowed(allowed_users))
async def settings_grade_params(callback: CallbackQuery, state: FSMContext):
    await callback.answer()
    defaults = GeminiService.GRADE_DEFAULTS
    await callback.message.answer(
        "Send grade params as 3 comma-separated floats: contrast, saturation, sharpen.\n"
        f"Defaults: {defaults['contrast']}, {defaults['saturation']}, {defaults['sharpen']}\n"
        "Ranges: contrast 0.5–2.0, saturation 0.0–3.0, sharpen 0.0–2.0\n\n"
        "Send a single dash (-) to reset to defaults."
    )
    await state.set_state(GenerationState.SET_GRADE_PARAMS)


@router.callback_query(lambda c: c.data == "settings:duration", IsAllowed(allowed_users))
async def settings_duration(callback: CallbackQuery):
    await callback.answer()
    db = container.inject(DBService)
    settings = await db.get_settings(callback.from_user.id, callback.message.chat.id)
    current = settings.get("target_duration", DEFAULT_TARGET_DURATION)
    await callback.message.answer(
        f"⏱ Current target duration: <b>{current}s</b>\n\n"
        "Select how long the generated video should be.\n\n"
        "<b>Seedance</b> — splits into 10-second clips:\n"
        "• 15s → 2 clips\n"
        "• 30s → 3 clips\n"
        "• 45s → 5 clips\n"
        "• 60s → 6 clips\n\n"
        "<b>Kling / OmniHuman</b> — duration is driven by voiceover length "
        "(the model adapts to your TTS audio).",
        parse_mode="HTML",
        reply_markup=get_duration_keyboard(current),
    )


@router.callback_query(lambda c: c.data.startswith("settings:duration:"), IsAllowed(allowed_users))
async def settings_duration_selected(callback: CallbackQuery):
    await callback.answer()
    try:
        duration = int(callback.data.split(":")[-1])
        if duration not in DURATION_SEGMENTS:
            await callback.message.answer("❌ Invalid duration.")
            return
        db = container.inject(DBService)
        await db.update_settings(
            callback.from_user.id, callback.message.chat.id, {"target_duration": duration}
        )
        try:
            await callback.message.edit_reply_markup(reply_markup=get_duration_keyboard(duration))
        except Exception:
            pass
        await callback.message.answer(
            f"✅ Target duration set to <b>{duration}s</b>.", parse_mode="HTML"
        )
    except Exception as e:
        await callback.message.answer(f"❌ Error: {e}")


@router.callback_query(lambda c: c.data == "settings:timezone", IsAllowed(allowed_users))
async def settings_timezone(callback: CallbackQuery, state: FSMContext):
    await callback.answer()
    db = container.inject(DBService)
    settings = await db.get_settings(callback.from_user.id, callback.message.chat.id)
    current = settings.get("utc_offset_hours", 0)
    sign = "+" if current >= 0 else ""
    await callback.message.answer(
        f"🕐 Current timezone: <b>UTC{sign}{current}</b>\n\n"
        "Send your UTC offset as a whole number.\n"
        "Examples:\n"
        "  <code>+3</code>  — Moscow / Riyadh\n"
        "  <code>+2</code>  — Kyiv / Helsinki (winter)\n"
        "  <code>0</code>   — UTC / London (winter)\n"
        "  <code>-5</code>  — New York (winter)\n\n"
        "After setting this, you can enter publish time in your LOCAL time "
        "and the bot will convert it to UTC automatically.",
        parse_mode="HTML",
    )
    await state.set_state(GenerationState.SET_UTC_OFFSET)


@router.callback_query(lambda c: c.data == "settings:text_model", IsAllowed(allowed_users))
async def settings_text_model(callback: CallbackQuery):
    await callback.answer()
    vertex = container.inject(GeminiService)
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(
                text=f"{m['name']}  |  {m['price']}",
                callback_data=f"text-model:{m['name']}",
            )]
            for m in vertex.get_text_models()
        ]
    )
    await callback.message.answer("Choose text model:", reply_markup=keyboard)


@router.callback_query(lambda c: c.data == "settings:image_model", IsAllowed(allowed_users))
async def settings_image_model(callback: CallbackQuery):
    await callback.answer()
    vertex = container.inject(GeminiService)
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(
                text=f"{m['name']}  |  {m['price']}",
                callback_data=f"image-model:{m['name']}",
            )]
            for m in vertex.get_image_models()
        ]
    )
    await callback.message.answer("Choose image model:", reply_markup=keyboard)


@router.callback_query(lambda c: c.data == "settings:video_model", IsAllowed(allowed_users))
async def settings_video_model(callback: CallbackQuery):
    await callback.answer()
    vertex = container.inject(GeminiService)
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(
                text=f"{m['name']}  |  {m['price']}",
                callback_data=f"video-model:{m['name']}",
            )]
            for m in vertex.get_video_models()
        ]
    )
    await callback.message.answer("Choose video model:", reply_markup=keyboard)


@router.callback_query(lambda c: c.data == "settings:plot_prompt", IsAllowed(allowed_users))
async def settings_plot_prompt(callback: CallbackQuery, state: FSMContext):
    await callback.answer()
    await callback.message.answer("Send new system prompt for idea → scene:")
    await state.set_state(GenerationState.SET_SYSTEM_PLOT_PROMPT)


@router.callback_query(lambda c: c.data == "settings:image_prompt", IsAllowed(allowed_users))
async def settings_image_prompt(callback: CallbackQuery, state: FSMContext):
    await callback.answer()
    await callback.message.answer("Send new system prompt for scene → image:")
    await state.set_state(GenerationState.SET_SYSTEM_IMAGE_PROMPT)


@router.callback_query(lambda c: c.data == "settings:video_prompt", IsAllowed(allowed_users))
async def settings_video_prompt(callback: CallbackQuery, state: FSMContext):
    await callback.answer()
    await callback.message.answer("Send new system prompt for scene → video:")
    await state.set_state(GenerationState.SET_SYSTEM_VIDEO_GENERATION_PROMPT)


@router.callback_query(lambda c: c.data == "settings:accounts", IsAllowed(allowed_users))
async def settings_accounts(callback: CallbackQuery, state: FSMContext):
    await callback.answer()
    db = container.inject(DBService)
    accounts = await db.get_chat_accounts(callback.message.chat.id)
    text = (
        f"📤 Accounts for this chat ({len(accounts)}):"
        if accounts else
        "📤 No accounts configured for this chat yet."
    )
    await callback.message.answer(text, reply_markup=get_accounts_keyboard(accounts))


@router.callback_query(lambda c: c.data == "accounts:add", IsAllowed(allowed_users))
async def accounts_add(callback: CallbackQuery, state: FSMContext):
    await callback.answer()
    await callback.message.answer(
        "Send account in format: <b>platform:account_id:label</b>\n\n"
        "• <b>platform</b> — <code>youtube</code> or <code>tiktok</code>\n"
        "• <b>account_id</b> — ID from Blotato\n"
        "• <b>label</b> — friendly name (e.g. Gaming channel)\n\n"
        "Example: <code>youtube:98434:Gaming YouTube</code>",
        parse_mode="HTML",
    )
    await state.set_state(GenerationState.ADD_CHAT_ACCOUNT)


@router.callback_query(lambda c: c.data == "accounts:noop", IsAllowed(allowed_users))
async def accounts_noop(callback: CallbackQuery):
    await callback.answer()


@router.callback_query(lambda c: c.data.startswith("accounts:remove:"), IsAllowed(allowed_users))
async def accounts_remove(callback: CallbackQuery):
    await callback.answer()
    try:
        row_id = int(callback.data.split(":")[-1])
        db = container.inject(DBService)
        deleted = await db.remove_chat_account(row_id, callback.message.chat.id)
        if not deleted:
            await callback.message.answer("❌ Account not found (already removed?)")
            return
        accounts = await db.get_chat_accounts(callback.message.chat.id)
        text = (
            f"📤 Accounts for this chat ({len(accounts)}):"
            if accounts else
            "📤 No accounts configured for this chat yet."
        )
        try:
            await callback.message.edit_text(text, reply_markup=get_accounts_keyboard(accounts))
        except Exception:
            await callback.message.answer(text, reply_markup=get_accounts_keyboard(accounts))
    except Exception as e:
        await callback.message.answer(f"❌ Error removing account: {e}")


@router.callback_query(lambda c: c.data == "settings:music_path", IsAllowed(allowed_users))
async def settings_music_path(callback: CallbackQuery, state: FSMContext):
    await callback.answer()
    db = container.inject(DBService)
    settings = await db.get_settings(callback.from_user.id, callback.message.chat.id)
    current = settings.get("background_music_path") or "not set"
    vol = settings.get("background_music_volume", 0.18)
    await callback.message.answer(
        f"🎵 <b>Background music</b>\n\n"
        f"Current path: <code>{current}</code>\n"
        f"Current volume: <b>{vol}</b>\n\n"
        f"Send the full path to an mp3/wav file on the server.\n"
        f"Example: <code>C:\\botClaude3\\video-gen-bot\\music\\bg.mp3</code>\n\n"
        f"To set volume add it after a space: <code>path/to/file.mp3 0.15</code>\n"
        f"Send a single dash (-) to disable music.",
        parse_mode="HTML",
    )
    await state.set_state(GenerationState.SET_MUSIC_PATH)


@router.message(GenerationState.SET_MUSIC_PATH, IsAllowed(allowed_users))
async def handle_set_music_path(message: Message, state: FSMContext):
    import os as _os
    db = container.inject(DBService)
    raw = message.text.strip()
    if raw == "-":
        await db.update_settings(message.from_user.id, message.chat.id, {"background_music_path": ""})
        await message.answer("✅ Background music disabled.")
        await state.clear()
        return
    parts = raw.rsplit(" ", 1)
    path = parts[0].strip()
    volume = 0.18
    if len(parts) == 2:
        try:
            volume = max(0.01, min(1.0, float(parts[1])))
        except ValueError:
            pass
    # Normalise path separators — forward slashes also work on Windows
    path = _os.path.normpath(path)
    exists = _os.path.isfile(path)
    await db.update_settings(message.from_user.id, message.chat.id, {
        "background_music_path": path,
        "background_music_volume": volume,
    })
    status = "✅ File found." if exists else "⚠️ File not found on disk — double-check the path. Music will be skipped during generation if missing."
    await message.answer(
        f"✅ Music path saved: <code>{path}</code>\nVolume: <b>{volume}</b>\n{status}",
        parse_mode="HTML",
    )
    await state.clear()


@router.callback_query(lambda c: c.data == "settings:voice_id", IsAllowed(allowed_users))
async def settings_voice_id(callback: CallbackQuery, state: FSMContext):
    await callback.answer()
    await callback.message.answer(
        "Send ElevenLabs voice_id.\n"
        "Browse voices at https://elevenlabs.io/app/voice-library — "
        "open one, copy its ID and paste here.\n\n"
        "Send a single dash (-) to fall back to the system default."
    )
    await state.set_state(GenerationState.SET_VOICE_ID)


@router.callback_query(lambda c: c.data == "settings:negative_prompt", IsAllowed(allowed_users))
async def settings_negative_prompt(callback: CallbackQuery, state: FSMContext):
    await callback.answer()
    await callback.message.answer(
        "Send negative prompt — comma-separated concepts to avoid in image/video.\n"
        "Example: blurry, distorted face, extra fingers, watermark, text on screen, low quality\n\n"
        "Send a single dash (-) to clear."
    )
    await state.set_state(GenerationState.SET_NEGATIVE_PROMPT)


# ─────────────────────────────────────────────
# MODEL SELECTION CALLBACKS
# ─────────────────────────────────────────────

@router.callback_query(lambda c: c.data.startswith("text-model:"), IsAllowed(allowed_users))
async def handle_text_model_selected(callback: CallbackQuery):
    await callback.answer()
    await callback.message.edit_reply_markup(reply_markup=None)
    try:
        db = container.inject(DBService)
        model = callback.data.split(":", 1)[1]
        await db.update_settings(callback.from_user.id, callback.message.chat.id, {"text_model": model})
        await callback.message.answer(f"✅ Text model: {model}")
    except Exception as e:
        await callback.message.answer(f"Error: {str(e)}")


@router.callback_query(lambda c: c.data.startswith("image-model:"), IsAllowed(allowed_users))
async def handle_image_model_selected(callback: CallbackQuery):
    await callback.answer()
    await callback.message.edit_reply_markup(reply_markup=None)
    try:
        db = container.inject(DBService)
        model = callback.data.split(":", 1)[1]
        await db.update_settings(callback.from_user.id, callback.message.chat.id, {"image_model": model})
        await callback.message.answer(f"✅ Image model: {model}")
    except Exception as e:
        await callback.message.answer(f"Error: {str(e)}")


@router.callback_query(lambda c: c.data.startswith("video-model:"), IsAllowed(allowed_users))
async def handle_video_model_selected(callback: CallbackQuery):
    await callback.answer()
    await callback.message.edit_reply_markup(reply_markup=None)
    try:
        db = container.inject(DBService)
        model = callback.data.split(":", 1)[1]
        await db.update_settings(callback.from_user.id, callback.message.chat.id, {"video_model": model})
        await callback.message.answer(f"✅ Video model: {model}")
    except Exception as e:
        await callback.message.answer(f"Error: {str(e)}")


# ─────────────────────────────────────────────
# SETTINGS STATE HANDLERS
# ─────────────────────────────────────────────

@router.message(GenerationState.SET_SYSTEM_IMAGE_PROMPT, IsAllowed(allowed_users))
async def handle_set_image_prompt(message: Message, state: FSMContext):
    db = container.inject(DBService)
    await db.update_settings(message.from_user.id, message.chat.id, {"system_image_prompt": message.text})
    await message.answer("✅ System prompt for image saved.")
    await state.clear()


@router.message(GenerationState.SET_SYSTEM_PLOT_PROMPT, IsAllowed(allowed_users))
async def handle_set_plot_prompt(message: Message, state: FSMContext):
    db = container.inject(DBService)
    await db.update_settings(message.from_user.id, message.chat.id, {"system_plot_prompt": message.text})
    await message.answer("✅ System prompt for plot/scene saved.")
    await state.clear()


@router.message(GenerationState.SET_SYSTEM_VIDEO_GENERATION_PROMPT, IsAllowed(allowed_users))
async def handle_set_video_gen_prompt(message: Message, state: FSMContext):
    db = container.inject(DBService)
    await db.update_settings(message.from_user.id, message.chat.id, {"system_video_prompt": message.text})
    await message.answer("✅ System prompt for video generation saved.")
    await state.clear()


@router.message(GenerationState.SET_GRADE_PARAMS, IsAllowed(allowed_users))
async def handle_set_grade_params(message: Message, state: FSMContext):
    db = container.inject(DBService)
    raw = message.text.strip()
    if raw == "-":
        await db.update_settings(message.from_user.id, message.chat.id, {"colour_grade_params": None})
        await message.answer("✅ Grade params reset to defaults.")
        await state.clear()
        return

    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 3:
        await message.answer(
            f"❌ Expected 3 comma-separated numbers, got {len(parts)}. Example: 1.08, 1.15, 0.6"
        )
        return
    try:
        float(parts[0])
        float(parts[1])
        float(parts[2])
    except ValueError:
        await message.answer("❌ Could not parse numbers. Use format: 1.08, 1.15, 0.6")
        return

    parsed = GeminiService.parse_grade_params(raw)
    canonical = f"{parsed['contrast']},{parsed['saturation']},{parsed['sharpen']}"
    await db.update_settings(message.from_user.id, message.chat.id, {"colour_grade_params": canonical})
    await message.answer(
        f"✅ Grade params saved: contrast={parsed['contrast']}, "
        f"saturation={parsed['saturation']}, sharpen={parsed['sharpen']}"
    )
    await state.clear()


@router.message(GenerationState.SET_VOICE_ID, IsAllowed(allowed_users))
async def handle_set_voice_id(message: Message, state: FSMContext):
    db = container.inject(DBService)
    raw = message.text.strip()
    value = "" if raw == "-" else raw
    await db.update_settings(message.from_user.id, message.chat.id, {"voice_id": value})
    await message.answer(f"✅ Voice saved: {value}" if value else "✅ Voice reset to system default.")
    await state.clear()


@router.message(GenerationState.SET_NEGATIVE_PROMPT, IsAllowed(allowed_users))
async def handle_set_negative_prompt(message: Message, state: FSMContext):
    db = container.inject(DBService)
    value = "" if message.text.strip() == "-" else message.text.strip()
    await db.update_settings(message.from_user.id, message.chat.id, {"negative_prompt": value})
    if value:
        await message.answer(f"✅ Negative prompt saved:\n{value}")
    else:
        await message.answer("✅ Negative prompt cleared.")
    await state.clear()


@router.message(GenerationState.SET_UTC_OFFSET, IsAllowed(allowed_users))
async def handle_set_utc_offset(message: Message, state: FSMContext):
    raw = (message.text or "").strip().lstrip("+")
    try:
        offset = int(raw)
        if not (-14 <= offset <= 14):
            await message.answer("❌ Offset must be between -14 and +14.")
            return
    except ValueError:
        await message.answer(
            "❌ Could not parse offset. Send a number like <code>+3</code> or <code>-5</code>.",
            parse_mode="HTML",
        )
        return
    db = container.inject(DBService)
    await db.update_settings(message.from_user.id, message.chat.id, {"utc_offset_hours": offset})
    sign = "+" if offset >= 0 else ""
    await message.answer(f"✅ Timezone set to <b>UTC{sign}{offset}</b>.", parse_mode="HTML")
    await state.clear()


@router.message(GenerationState.ADD_CHAT_ACCOUNT, IsAllowed(allowed_users))
async def handle_add_chat_account(message: Message, state: FSMContext):
    raw = message.text.strip() if message.text else ""
    parts = raw.split(":", 2)
    if len(parts) < 2:
        await message.answer(
            "❌ Wrong format. Use: <code>platform:account_id:label</code>\n"
            "Example: <code>youtube:98434:Gaming YouTube</code>",
            parse_mode="HTML",
        )
        return

    platform = parts[0].strip().lower()
    account_id = parts[1].strip()
    label = parts[2].strip() if len(parts) == 3 else ""

    if platform not in ("youtube", "tiktok"):
        await message.answer(
            f"❌ Unknown platform: <b>{platform}</b>. Use <code>youtube</code> or <code>tiktok</code>.",
            parse_mode="HTML",
        )
        return
    if not account_id:
        await message.answer("❌ account_id cannot be empty.")
        return

    db = container.inject(DBService)
    await db.add_chat_account(message.chat.id, platform, account_id, label)
    await state.clear()

    accounts = await db.get_chat_accounts(message.chat.id)
    await message.answer(
        f"✅ Account added!\n\n📤 Accounts for this chat ({len(accounts)}):",
        reply_markup=get_accounts_keyboard(accounts),
    )
