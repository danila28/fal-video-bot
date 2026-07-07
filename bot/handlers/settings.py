"""Settings callbacks, model selection, and settings state handlers."""

import html
import json
import logging

from aiogram import F, Router
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
    get_advanced_settings_keyboard,
    get_duration_keyboard,
    get_image_count_keyboard,
    get_resolution_keyboard,
    get_settings_keyboard,
    get_speed_keyboard,
    get_style_keyboard,
    SETTINGS_BUTTON_TEXT,
)
from bot.states import GenerationState
from services.db import DBService
from services.gemini import GeminiService
from utils import container
from utils.consts import allowed_users
from utils.presets import (
    CUSTOM_NICHE_META_PROMPT,
    DEFAULT_PRESET_KEY,
    STYLE_PRESETS,
    parse_custom_niche_json,
)
from bot.handlers.common import (
    DEFAULT_TARGET_DURATION,
    DURATION_SEGMENTS,
    _T2V_VIDEO_MODELS,
)

router = Router()
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# SETTINGS ENTRY
# ─────────────────────────────────────────────

async def _is_t2v_selected(user_id: int, chat_id: int) -> bool:
    db = container.inject(DBService)
    s = await db.get_settings(user_id, chat_id)
    return (s.get("video_model") or "") in _T2V_VIDEO_MODELS


@router.message(F.text == SETTINGS_BUTTON_TEXT, IsAllowed(allowed_users))
async def handle_settings_button(message: Message, state: FSMContext):
    await state.clear()
    is_t2v = await _is_t2v_selected(message.from_user.id, message.chat.id)
    await message.answer("⚙️ Settings", reply_markup=get_settings_keyboard(is_t2v))


@router.callback_query(lambda c: c.data == "settings:back", IsAllowed(allowed_users))
async def settings_back(callback: CallbackQuery):
    await callback.answer()
    is_t2v = await _is_t2v_selected(callback.from_user.id, callback.message.chat.id)
    try:
        await callback.message.edit_text("⚙️ Settings", reply_markup=get_settings_keyboard(is_t2v))
    except Exception:
        await callback.message.answer("⚙️ Settings", reply_markup=get_settings_keyboard(is_t2v))


@router.callback_query(lambda c: c.data == "settings:advanced", IsAllowed(allowed_users))
async def settings_advanced(callback: CallbackQuery):
    await callback.answer()
    db = container.inject(DBService)
    s = await db.get_settings(callback.from_user.id, callback.message.chat.id)
    kb = get_advanced_settings_keyboard(
        subtitles_default_on=s.get("subtitles_enabled", True),
        grade_on=s.get("colour_grade_enabled", False),
        target_duration=s.get("target_duration", DEFAULT_TARGET_DURATION),
        utc_offset=s.get("utc_offset_hours", 0),
        sfx_on=s.get("sfx_enabled", False),
        video_speed=float(s.get("video_speed", 1.0)),
        video_resolution=s.get("video_resolution", "720p"),
        image_count=int(s.get("image_count", 1)),
    )
    try:
        await callback.message.edit_text("⚙️ More settings", reply_markup=kb)
    except Exception:
        await callback.message.answer("⚙️ More settings", reply_markup=kb)


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


@router.callback_query(lambda c: c.data == "settings:subtitles_toggle", IsAllowed(allowed_users))
async def settings_subtitles_toggle(callback: CallbackQuery):
    db = container.inject(DBService)
    user_id = callback.from_user.id
    chat_id = callback.message.chat.id
    s = await db.get_settings(user_id, chat_id)
    new_value = not s.get("subtitles_enabled", True)
    await db.update_settings(user_id, chat_id, {"subtitles_enabled": new_value})
    kb = get_advanced_settings_keyboard(
        subtitles_default_on=new_value,
        grade_on=s.get("colour_grade_enabled", False),
        target_duration=s.get("target_duration", DEFAULT_TARGET_DURATION),
        utc_offset=s.get("utc_offset_hours", 0),
        sfx_on=s.get("sfx_enabled", False),
        video_speed=float(s.get("video_speed", 1.0)),
        video_resolution=s.get("video_resolution", "720p"),
    )
    await callback.answer(f"Subtitles: {'ON' if new_value else 'OFF'}")
    try:
        await callback.message.edit_reply_markup(reply_markup=kb)
    except Exception:
        await callback.message.answer("⚙️ More settings", reply_markup=kb)


@router.callback_query(lambda c: c.data == "settings:grade_toggle", IsAllowed(allowed_users))
async def settings_grade_toggle(callback: CallbackQuery):
    db = container.inject(DBService)
    user_id = callback.from_user.id
    chat_id = callback.message.chat.id
    s = await db.get_settings(user_id, chat_id)
    new_value = not s.get("colour_grade_enabled", False)
    await db.update_settings(user_id, chat_id, {"colour_grade_enabled": new_value})
    kb = get_advanced_settings_keyboard(
        subtitles_default_on=s.get("subtitles_enabled", True),
        grade_on=new_value,
        target_duration=s.get("target_duration", DEFAULT_TARGET_DURATION),
        utc_offset=s.get("utc_offset_hours", 0),
        sfx_on=s.get("sfx_enabled", False),
        video_speed=float(s.get("video_speed", 1.0)),
        video_resolution=s.get("video_resolution", "720p"),
    )
    await callback.answer(f"Colour grade: {'ON' if new_value else 'OFF'}")
    try:
        await callback.message.edit_reply_markup(reply_markup=kb)
    except Exception:
        await callback.message.answer("⚙️ More settings", reply_markup=kb)


@router.callback_query(lambda c: c.data == "settings:sfx_toggle", IsAllowed(allowed_users))
async def settings_sfx_toggle(callback: CallbackQuery):
    db = container.inject(DBService)
    user_id = callback.from_user.id
    chat_id = callback.message.chat.id
    s = await db.get_settings(user_id, chat_id)
    new_value = not s.get("sfx_enabled", False)
    await db.update_settings(user_id, chat_id, {"sfx_enabled": new_value})
    kb = get_advanced_settings_keyboard(
        subtitles_default_on=s.get("subtitles_enabled", True),
        grade_on=s.get("colour_grade_enabled", False),
        target_duration=s.get("target_duration", DEFAULT_TARGET_DURATION),
        utc_offset=s.get("utc_offset_hours", 0),
        sfx_on=new_value,
        video_speed=float(s.get("video_speed", 1.0)),
        video_resolution=s.get("video_resolution", "720p"),
    )
    await callback.answer(f"SFX / ASMR: {'ON' if new_value else 'OFF'}")
    try:
        await callback.message.edit_reply_markup(reply_markup=kb)
    except Exception:
        await callback.message.answer("⚙️ More settings", reply_markup=kb)


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
        "• 1.0× — normal speed\n"
        "• 1.15× — slightly faster\n"
        "• 1.3× — fast\n"
        "• 1.5× — turbo",
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
        "The video is split into the fewest clips possible (max ~10-15s each, "
        "depending on the model) and generated for your exact requested length "
        "— e.g. 20s → two 10s clips, 5s → one 5s clip.",
        parse_mode="HTML",
        reply_markup=get_duration_keyboard(current),
    )


@router.callback_query(lambda c: c.data.startswith("settings:duration:"), IsAllowed(allowed_users))
async def settings_duration_selected(callback: CallbackQuery, state: FSMContext):
    await callback.answer()
    duration_str = callback.data.split(":")[-1]

    if duration_str == "custom":
        await callback.message.answer(
            "Enter video duration in seconds (5-300):\n"
            "Examples: 9, 10, 12, 15, 20, 30"
        )
        await state.set_state(GenerationState.SET_CUSTOM_DURATION)
        return

    try:
        duration = int(duration_str)
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


@router.message(GenerationState.SET_CUSTOM_DURATION, IsAllowed(allowed_users))
async def handle_custom_duration(message: Message, state: FSMContext):
    try:
        duration = int(message.text.strip())
        if duration < 5 or duration > 300:
            await message.answer("❌ Duration must be between 5-300 seconds. Try again.")
            return

        db = container.inject(DBService)
        await db.update_settings(
            message.from_user.id, message.chat.id, {"target_duration": duration}
        )
        await message.answer(
            f"✅ Target duration set to <b>{duration}s</b>.", parse_mode="HTML"
        )
    except ValueError:
        await message.answer("❌ Please enter a valid number (5-300).")
    except Exception as e:
        await message.answer(f"❌ Error: {e}")
    finally:
        await state.clear()


@router.callback_query(lambda c: c.data == "settings:image_count", IsAllowed(allowed_users))
async def settings_image_count(callback: CallbackQuery):
    await callback.answer()
    db = container.inject(DBService)
    settings = await db.get_settings(callback.from_user.id, callback.message.chat.id)
    current = int(settings.get("image_count", 1))
    await callback.message.answer(
        f"🖼 Current: <b>{current} photo{'s' if current > 1 else ''}</b>\n\n"
        "Choose how many reference images to generate per video.\n\n"
        "<b>How it works:</b>\n"
        "• <b>I2V models</b> (Kling, Seedance): images are cycled across clips "
        "— clip 1 uses photo 1, clip 2 uses photo 2, etc.\n\n"
        "More photos → more visual variety and better character anchoring.",
        parse_mode="HTML",
        reply_markup=get_image_count_keyboard(current),
    )


@router.callback_query(lambda c: c.data.startswith("settings:image_count:"), IsAllowed(allowed_users))
async def settings_image_count_selected(callback: CallbackQuery):
    await callback.answer()
    try:
        count = int(callback.data.split(":")[-1])
        if count not in (1, 2, 3, 4):
            await callback.message.answer("❌ Invalid value.")
            return
        db = container.inject(DBService)
        await db.update_settings(
            callback.from_user.id, callback.message.chat.id, {"image_count": count}
        )
        try:
            await callback.message.edit_reply_markup(reply_markup=get_image_count_keyboard(count))
        except Exception:
            pass
        await callback.message.answer(
            f"✅ Image count set to <b>{count} photo{'s' if count > 1 else ''}</b>.", parse_mode="HTML"
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
        "Examples: <code>+3</code>, <code>0</code>, <code>-5</code>",
        parse_mode="HTML",
    )
    await state.set_state(GenerationState.SET_UTC_OFFSET)


@router.callback_query(lambda c: c.data == "settings:text_model", IsAllowed(allowed_users))
async def settings_text_model(callback: CallbackQuery):
    await callback.answer()
    gemini = container.inject(GeminiService)
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(
                text=f"{m['name']}  |  {m['price']}",
                callback_data=f"text-model:{m['name']}",
            )]
            for m in gemini.get_text_models()
        ]
    )
    await callback.message.answer("Choose text model:", reply_markup=keyboard)


@router.callback_query(lambda c: c.data == "settings:image_model", IsAllowed(allowed_users))
async def settings_image_model(callback: CallbackQuery):
    await callback.answer()
    gemini = container.inject(GeminiService)
    rows = []
    for m in gemini.get_image_models():
        if m.get("separator"):
            rows.append([InlineKeyboardButton(text=m["label"], callback_data="noop")])
        else:
            rows.append([InlineKeyboardButton(
                text=f"{m['name']}  |  {m['price']}",
                callback_data=f"image-model:{m['name']}",
            )])
    keyboard = InlineKeyboardMarkup(inline_keyboard=rows)
    await callback.message.answer("Choose image model:", reply_markup=keyboard)


@router.callback_query(lambda c: c.data == "settings:video_model", IsAllowed(allowed_users))
async def settings_video_model(callback: CallbackQuery):
    await callback.answer()
    gemini = container.inject(GeminiService)
    rows = []
    for m in gemini.get_video_models():
        if m.get("separator"):
            rows.append([InlineKeyboardButton(text=m["label"], callback_data="noop")])
        else:
            rows.append([InlineKeyboardButton(
                text=m["price"],
                callback_data=f"video-model:{m['name']}",
            )])
    keyboard = InlineKeyboardMarkup(inline_keyboard=rows)
    await callback.message.answer("Choose video model:", reply_markup=keyboard)


@router.callback_query(lambda c: c.data == "settings:plot_prompt", IsAllowed(allowed_users))
async def settings_plot_prompt(callback: CallbackQuery, state: FSMContext):
    await callback.answer()
    db = container.inject(DBService)
    s = await db.get_settings(callback.from_user.id, callback.message.chat.id)
    current = s.get("system_plot_prompt") or ""
    text = "✏️ Send the new plot prompt text (idea → concept):"
    if current:
        text += f"\n\nCurrent:\n<code>{html.escape(current[:800])}</code>"
    await callback.message.answer(text, parse_mode="HTML")
    await state.set_state(GenerationState.SET_SYSTEM_PLOT_PROMPT)


@router.callback_query(lambda c: c.data == "settings:image_prompt", IsAllowed(allowed_users))
async def settings_image_prompt(callback: CallbackQuery, state: FSMContext):
    await callback.answer()
    db = container.inject(DBService)
    s = await db.get_settings(callback.from_user.id, callback.message.chat.id)
    current = s.get("system_image_prompt") or ""
    text = "✏️ Send the new image prompt text (concept → frame):"
    if current:
        text += f"\n\nCurrent:\n<code>{html.escape(current[:800])}</code>"
    await callback.message.answer(text, parse_mode="HTML")
    await state.set_state(GenerationState.SET_SYSTEM_IMAGE_PROMPT)


# ─────────────────────────────────────────────
# CONTENT STYLE (niche presets + one-shot AI custom niche)
# ─────────────────────────────────────────────

def _video_model_label(model_key: str) -> str:
    """Human-readable label for a video_model settings key."""
    from services.kling import MODEL_LABELS as _KLING_LABELS
    from services.seedance import MODEL_LABELS as _SEEDANCE_LABELS
    return _KLING_LABELS.get(model_key) or _SEEDANCE_LABELS.get(model_key) or model_key


async def _show_style_menu(callback: CallbackQuery):
    db = container.inject(DBService)
    s = await db.get_settings(callback.from_user.id, callback.message.chat.id)
    current = s.get("content_preset") or ""
    is_t2v = (s.get("video_model") or "") in _T2V_VIDEO_MODELS

    lines = ["🎨 <b>Content style</b>\n"]
    lines.append(
        "Pick a niche — it sets the plot and image style together, consistently. "
        "What you get with each:\n"
    )
    for key, p in STYLE_PRESETS.items():
        mark = " ✅" if key == current else ""
        lines.append(f"{p['label']}{mark}\n<i>{p['description']}</i>\n")
    lines.append(
        "✍️ <b>Custom niche</b> — describe your topic in a sentence or two, "
        "the AI will build a style for it once and save it.\n"
    )
    if not current:
        if s.get("system_plot_prompt") or s.get("system_image_prompt"):
            # Legacy user: prompts saved before presets existed
            lines.append("<i>Currently: ✏️ your previously saved prompts (manual setup).</i>")
        else:
            lines.append("<i>Nothing selected yet — 🎯 Universal is used by default.</i>")
    elif current == "manual":
        lines.append("<i>Currently: ✏️ manual prompt setup.</i>")
    elif current == "custom":
        lines.append("<i>Currently: ✍️ custom niche (AI-generated and saved).</i>")
    if is_t2v:
        lines.append(
            "\n<i>ℹ️ You have a text-to-video model selected — the photo stage is "
            "skipped, the image prompt is unused.</i>"
        )

    kb = get_style_keyboard(STYLE_PRESETS, current_key=current, is_t2v=is_t2v)
    try:
        await callback.message.edit_text("\n".join(lines), parse_mode="HTML", reply_markup=kb)
    except Exception:
        await callback.message.answer("\n".join(lines), parse_mode="HTML", reply_markup=kb)


@router.callback_query(lambda c: c.data == "settings:style", IsAllowed(allowed_users))
async def settings_style(callback: CallbackQuery):
    await callback.answer()
    await _show_style_menu(callback)


@router.callback_query(lambda c: c.data.startswith("style:set:"), IsAllowed(allowed_users))
async def handle_style_selected(callback: CallbackQuery):
    key = callback.data.split(":")[-1]
    preset = STYLE_PRESETS.get(key)
    if preset is None:
        await callback.answer("Unknown preset", show_alert=True)
        return
    await callback.answer()
    db = container.inject(DBService)
    new_settings = {
        "content_preset": key,
        "system_plot_prompt": preset["plot"],
        "system_image_prompt": preset["image"],
        # Optional per-niche voiceover/style override (e.g. ASMR whisper).
        # Always written — switching presets must clear the previous one.
        "system_video_prompt": preset.get("video", ""),
    }
    # Each niche ships with a recommended video model (e.g. humor → cheap T2V,
    # ASMR → Seedance Fast). Applied with the preset; announced to the user below.
    model_note = ""
    preset_model = preset.get("video_model")
    if preset_model:
        new_settings["video_model"] = preset_model
        model_note = (
            f"\n🎬 Video model switched to: <b>{_video_model_label(preset_model)}</b> "
            "(recommended for this niche — you can change it in 🎬 Video model)."
        )
    await db.update_settings(callback.from_user.id, callback.message.chat.id, new_settings)
    await callback.message.answer(
        f"✅ Style applied: <b>{preset['label']}</b>\n<i>{preset['description']}</i>\n"
        + model_note +
        "\n\nThe plot and image styles are now tuned for this niche. "
        "You can still hand-edit the texts with the «✏️» buttons in the style menu.",
        parse_mode="HTML",
    )
    await _show_style_menu(callback)


@router.callback_query(lambda c: c.data == "style:custom", IsAllowed(allowed_users))
async def handle_style_custom(callback: CallbackQuery, state: FSMContext):
    await callback.answer()
    await callback.message.answer(
        "✍️ Describe your niche/style in 1-3 sentences.\n\n"
        "Example: <i>vintage watch restoration, calm expert tone, "
        "workshop aesthetic</i>\n\n"
        "The AI will build a plot + image style for it once and save it — "
        "after that it's used as-is, no regeneration.",
        parse_mode="HTML",
    )
    await state.set_state(GenerationState.SET_CUSTOM_NICHE)


@router.message(GenerationState.SET_CUSTOM_NICHE, IsAllowed(allowed_users))
async def handle_custom_niche_input(message: Message, state: FSMContext):
    niche = (message.text or "").strip()
    if len(niche) < 5:
        await message.answer("❌ Too short — describe the niche in at least a few words.")
        return
    await message.answer("⏳ Building a style for your niche…")
    try:
        gemini = container.inject(GeminiService)
        db = container.inject(DBService)
        settings = await db.get_settings(message.from_user.id, message.chat.id)
        raw = await gemini.generate_text(
            niche, CUSTOM_NICHE_META_PROMPT, settings.get("text_model") or ""
        )
        parsed = parse_custom_niche_json(raw)
        if parsed is None:
            await message.answer(
                "❌ Couldn't build a style (the AI returned an unexpected format). "
                "Try again or rephrase the description."
            )
            return
        plot, image = parsed
        await db.update_settings(
            message.from_user.id, message.chat.id,
            {
                "content_preset": "custom",
                "system_plot_prompt": plot,
                "system_image_prompt": image,
                # Clear any per-niche voiceover override left by a previous preset
                "system_video_prompt": "",
            },
        )
        await message.answer(
            "✅ Style for your niche saved!\n\n"
            f"<b>Plot:</b>\n<code>{html.escape(plot[:600])}</code>\n\n"
            f"<b>Image:</b>\n<code>{html.escape(image[:600])}</code>\n\n"
            "It's saved as plain text and will be used for all future videos. "
            "You can adjust it with the «✏️» buttons in the style menu.",
            parse_mode="HTML",
        )
        await state.clear()
    except Exception as e:
        logger.error(f"Custom niche generation failed: {e}")
        await message.answer(f"❌ Error: {e}\nTry again.")


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
        f"To set volume add it after a space: <code>path/to/file.mp3 0.15</code>\n"
        f"Send a single dash (-) to disable music.",
        parse_mode="HTML",
    )
    await state.set_state(GenerationState.SET_MUSIC_PATH)


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
        "Example: blurry, distorted face, extra fingers, watermark, text on screen\n\n"
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


@router.callback_query(lambda c: c.data == "noop", IsAllowed(allowed_users))
async def handle_noop(callback: CallbackQuery):
    await callback.answer()


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
    # Manual edit detaches the saved text from any named preset
    await db.update_settings(
        message.from_user.id, message.chat.id,
        {"system_image_prompt": message.text, "content_preset": "manual"},
    )
    await message.answer("✅ Image prompt saved (style: manual setup).")
    await state.clear()


@router.message(GenerationState.SET_SYSTEM_PLOT_PROMPT, IsAllowed(allowed_users))
async def handle_set_plot_prompt(message: Message, state: FSMContext):
    db = container.inject(DBService)
    # Manual edit detaches the saved text from any named preset
    await db.update_settings(
        message.from_user.id, message.chat.id,
        {"system_plot_prompt": message.text, "content_preset": "manual"},
    )
    await message.answer("✅ Plot prompt saved (style: manual setup).")
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
        float(parts[0]); float(parts[1]); float(parts[2])
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
    path = _os.path.normpath(path)
    exists = _os.path.isfile(path)
    await db.update_settings(message.from_user.id, message.chat.id, {
        "background_music_path": path,
        "background_music_volume": volume,
    })
    status = "✅ File found." if exists else "⚠️ File not found on disk — double-check the path."
    await message.answer(
        f"✅ Music path saved: <code>{path}</code>\nVolume: <b>{volume}</b>\n{status}",
        parse_mode="HTML",
    )
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
