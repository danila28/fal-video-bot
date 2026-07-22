"""Remix from link flow: analyze reference video → extract formula → generate new video."""

import html
import logging
import time

from aiogram import F, Router
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, Message

from bot.guard import IsAllowed
from bot.keyboards import get_remix_formula_keyboard, get_prompt_keyboard
from bot.states import RemixState, GenerationState
from services.db import DBService
from services.downloader import DownloaderService
from services.gemini import GeminiService
from utils import container
from utils.consts import allowed_users
from utils.tg import send_long_message
from bot.handlers.common import (
    DEFAULT_TARGET_DURATION,
    _is_generating,
)

router = Router()
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# REMIX ENTRY POINT
# ─────────────────────────────────────────────

@router.callback_query(F.data == "remix:start", IsAllowed(allowed_users))
async def handle_remix_start(query: CallbackQuery, state: FSMContext):
    """Start remix-from-link flow."""
    await query.answer()

    if await _is_generating(state):
        await query.message.answer("⚠️ Generation in progress — please wait until it finishes.")
        return

    db = container.inject(DBService)
    chat_accounts = await db.get_chat_accounts(query.message.chat.id)
    if not chat_accounts:
        await query.message.answer("No accounts configured for this chat\nTap ⚙️ Settings → 📤 Accounts")
        return

    await state.clear()
    await query.message.answer(
        "Send a link to a TikTok, Instagram Reels, YouTube Shorts, or other video\n\n"
        "Examples:\n"
        "• https://www.tiktok.com/@username/video/123456\n"
        "• https://www.instagram.com/reel/ABC123/\n"
        "• https://youtube.com/shorts/ABC123"
    )
    await state.set_state(RemixState.WAITING_LINK)


@router.message(RemixState.WAITING_LINK, IsAllowed(allowed_users))
async def handle_remix_link(message: Message, state: FSMContext):
    """Receive video link, download, and analyze."""
    url = message.text.strip()

    if not url.startswith(("http://", "https://")):
        await message.answer("❌ Please send a valid URL starting with http:// or https://")
        return

    # Supported platforms
    supported = ("tiktok.com", "instagram.com", "youtube.com", "youtu.be", "x.com", "twitter.com", "reels")
    if not any(platform in url for platform in supported):
        await message.answer(
            "⚠️ Unsupported platform. Supported:\n"
            "• TikTok (tiktok.com)\n"
            "• Instagram Reels (instagram.com/reel)\n"
            "• YouTube Shorts (youtube.com/shorts)\n"
            "• Twitter/X (x.com, twitter.com)\n\n"
            "Send a valid link or click 🎬 Generate video to try something else"
        )
        return

    await state.update_data(remix_url=url)
    await state.set_state(RemixState.ANALYZING)

    status_msg = await message.answer("⏳ Downloading video…")

    try:
        downloader = container.inject(DownloaderService)
        video_path = await downloader.download(url)
        await status_msg.edit_text("⏳ Analyzing video with AI (this may take a moment)…")

        gemini = container.inject(GeminiService)
        db = container.inject(DBService)
        settings = await db.get_settings(message.from_user.id, message.chat.id)
        target_duration = settings.get("target_duration", DEFAULT_TARGET_DURATION)

        analysis_result = await gemini.analyze_reference_video(
            video_path,
            target_duration=target_duration
        )

        await state.update_data(
            remix_analysis=analysis_result,
            remix_video_path=video_path,
            generation_in_progress=True,
            generation_started_at=time.time(),
        )

        # Show extracted formula
        await _show_remix_formula(message, state, analysis_result)
        await state.set_state(RemixState.CONFIRM_FORMULA)

    except Exception as e:
        logger.error(f"Remix analysis failed for {url}: {e}")
        error_msg = str(e)
        if "402" in error_msg or "payment" in error_msg.lower():
            error_display = "❌ Payment required for ElevenLabs voice. Check ⚙️ Settings → 🎙 Voice"
        elif "yt-dlp" in error_msg.lower() or "download" in error_msg.lower():
            error_display = f"❌ Failed to download video: {error_msg}\n\nTry a different link or platform"
        else:
            error_display = f"❌ Failed to analyze video: {error_msg}"

        await status_msg.edit_text(f"{error_display}\n\nClick 🎬 Generate video to try again")
        await state.clear()


# ─────────────────────────────────────────────
# FORMULA PREVIEW & CONFIRMATION
# ─────────────────────────────────────────────

async def _show_remix_formula(message: Message, state: FSMContext, analysis: dict) -> None:
    """Display extracted video formula for review."""
    shots = analysis.get("shots", [])
    metadata = analysis.get("metadata", {})
    title = analysis.get("title", "")
    voiceover = analysis.get("voiceover", "")

    tone = metadata.get("detected_tone", "unknown")
    tempo = metadata.get("detected_tempo", "unknown")
    lang = metadata.get("detected_language", "unknown")

    formula_text = "📊 Extracted Video Formula\n\n"

    if title:
        formula_text += f"🎯 <b>Hook:</b> {html.escape(title)}\n\n"

    if voiceover:
        vo_preview = voiceover[:200] + ("…" if len(voiceover) > 200 else "")
        formula_text += f"🗣 <b>Voiceover:</b> {html.escape(vo_preview)}\n\n"

    formula_text += f"🎭 <b>Tone:</b> {tone}\n"
    formula_text += f"⚡ <b>Tempo:</b> {tempo}\n"
    formula_text += f"🌐 <b>Language:</b> {lang}\n\n"

    formula_text += f"<b>Structure ({len(shots)} scenes):</b>\n"
    for i, shot in enumerate(shots, 1):
        prompt = shot.get("scene_prompt", "")
        duration = shot.get("duration_seconds", 0)
        transition = shot.get("transition", "cut")
        prompt_preview = prompt[:100] + ("…" if len(prompt) > 100 else "")
        formula_text += f"{i}. {html.escape(prompt_preview)} ({duration}s, {transition})\n"

    formula_text += "\n✅ Look good? Generate new video based on this formula"

    await send_long_message(
        message,
        formula_text,
        keyboard=get_remix_formula_keyboard(),
    )


@router.callback_query(F.data == "remix:confirm_formula", IsAllowed(allowed_users))
async def handle_confirm_formula(query: CallbackQuery, state: FSMContext):
    """Confirm the extracted formula and proceed to full generation pipeline."""
    await query.answer()

    data = await state.get_data()
    analysis = data.get("remix_analysis", {})

    if not analysis:
        await query.message.answer("❌ Analysis data lost. Please start over.")
        await state.clear()
        return

    voiceover = analysis.get("voiceover", "")
    title = analysis.get("title", "")
    shots = analysis.get("shots", [])

    # Build a scenario text that will skip the enhancement step
    # and go directly to video generation
    scenario_text = title or "Video based on reference"
    if voiceover:
        scenario_text += f"\n\nVoiceover: {voiceover}"

    # Convert shots to video_prompt JSON format that matches the existing format
    video_prompt_json = {
        "title": title,
        "voiceover": voiceover,
        "shots": shots,
    }

    import json
    video_prompt = json.dumps(video_prompt_json, ensure_ascii=False, indent=2)

    # Split into scene and voiceover for display
    from bot.handlers.common import _split_video_prompt
    gemini = container.inject(GeminiService)
    scene, vo = _split_video_prompt(video_prompt, gemini)

    # Update state to pretend this came from normal flow
    # Skip image generation step and go straight to video
    await state.update_data(
        raw_prompt=scenario_text,
        enhance_prompt=scenario_text,
        own_script=True,  # Treat as ready-made script
        image_paths=[],
        image_path=None,
        image_prompt="",
        video_scene=scene,
        remix_mode=True,
    )

    await query.message.answer("✅ Using extracted formula. Showing video structure…")

    # Show video prompt for final review before generation
    from bot.handlers.common import _show_video_prompt
    await _show_video_prompt(query.message, scene, vo)
    await state.set_state(GenerationState.CONFIRM_VIDEO_PROMPT)


@router.callback_query(F.data == "remix:edit_formula", IsAllowed(allowed_users))
async def handle_edit_formula(query: CallbackQuery, state: FSMContext):
    """Allow editing of extracted formula (for now, just re-analyze)."""
    await query.answer()
    await query.message.answer(
        "Feature coming soon! For now, you can:\n"
        "1. Click 'Generate' to proceed with the current formula\n"
        "2. Or start over with a different video link"
    )


@router.callback_query(F.data == "remix:cancel", IsAllowed(allowed_users))
async def handle_remix_cancel(query: CallbackQuery, state: FSMContext):
    """Cancel remix flow and return to main menu."""
    await query.answer()
    await query.message.answer("Cancelled. Click 🎬 Generate video to start over.")
    await state.clear()
