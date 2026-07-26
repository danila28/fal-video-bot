"""bot.handlers package — re-exports a single combined router.

Split into sub-modules:
  common.py     — shared helpers and constants (no router)
  settings.py   — settings callbacks + model selection + state handlers
  generation.py — idea → image → video prompt → video generation
  editor.py     — AI Editor: edit user's own video via Gemini Omni Flash
  publish.py    — publish flow + cancel
"""

from aiogram import Router

from bot.handlers.settings import router as _settings_router
from bot.handlers.generation import router as _generation_router
from bot.handlers.editor import router as _editor_router
from bot.handlers.publish import router as _publish_router

# Order matters: the editor router goes BEFORE generation so the 🎨 AI Editor
# menu button reaches its entry handler from any generation state (generation's
# state handlers otherwise swallow the button text as user input, e.g. as an
# "idea" in RAW_PROMPT right after /start). The editor's own text handlers
# exclude menu-button texts so 🎬/⚙️ keep working from editor states too.
router = Router()
router.include_router(_settings_router)
router.include_router(_editor_router)
router.include_router(_generation_router)
router.include_router(_publish_router)

__all__ = ["router"]
