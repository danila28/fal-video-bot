"""Operator-facing error hints and the start-image count for Higgsfield models."""

from bot.handlers.common import _explain_provider_error
from bot.handlers.generation import _image_count_for
from services.llm import AnthropicTextProvider


GOOGLE_FREE_TIER = (
    "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your "
    "current quota ... Quota exceeded for metric: "
    "generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 0, "
    "model: gemini-3-pro-image'}}"
)
ANTHROPIC_EMPTY = (
    "Anthropic request failed: Error code: 400 - {'type': 'error', 'error': {'message': "
    "'Your credit balance is too low to access the Anthropic API.'}}"
)
GEMINI_404 = (
    "Failed to generate text via Gemini: 404 NOT_FOUND. {'error': {'code': 404, "
    "'message': 'models/gemini-3.1-pro is not found for API version v1beta'}}"
)


def test_google_free_tier_error_gets_an_instruction():
    hint = _explain_provider_error(Exception(GOOGLE_FREE_TIER))
    assert "billing" in hint and "Atlas" in hint
    assert "RESOURCE_EXHAUSTED" not in hint


def test_empty_anthropic_balance_gets_an_instruction():
    hint = _explain_provider_error(Exception(ANTHROPIC_EMPTY))
    assert "console.anthropic.com" in hint


def test_unavailable_model_id_gets_an_instruction():
    assert "Settings" in _explain_provider_error(Exception(GEMINI_404))


def test_unknown_errors_pass_through_unchanged():
    assert _explain_provider_error(Exception("boom")) == "boom"
    # a real 429 with a quota above zero is an ordinary rate limit, not the free-tier case
    plain = "429 RESOURCE_EXHAUSTED. Too many requests per minute"
    assert _explain_provider_error(Exception(plain)) == plain


def test_higgsfield_image_to_video_uses_one_start_frame():
    for model in ("higgsfield_kling_std", "higgsfield_minimax_h3"):
        assert _image_count_for({"video_model": model, "image_count": 4}) == 1


def test_other_models_keep_the_configured_image_count():
    assert _image_count_for({"video_model": "higgsfield_seedance_ref", "image_count": 4}) == 4
    assert _image_count_for({"video_model": "kling", "image_count": 3}) == 3
    assert _image_count_for({"image_count": 2}) == 2
    assert _image_count_for({}) == 1


def test_claude_provider_is_review_only():
    assert AnthropicTextProvider.in_menu is False
