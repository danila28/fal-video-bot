"""bot.handlers package — re-exports a single combined router.

Split into sub-modules:
  common.py     — shared helpers and constants (no router)
  settings.py   — settings callbacks + model selection + state handlers
  generation.py — idea → image → video prompt → video generation
  publish.py    — publish flow + cancel
"""

from aiogram import Router

from bot.handlers.settings import router as _settings_router
from bot.handlers.generation import router as _generation_router
from bot.handlers.publish import router as _publish_router

router = Router()
router.include_router(_settings_router)
router.include_router(_generation_router)
router.include_router(_publish_router)

__all__ = ["router"]
