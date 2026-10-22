"""Local price table for video cost estimates.

Prices are per second of output, in USD. Atlas prices are copied from the menu
labels in services/gemini.py (those labels say "verify before relying on exact
numbers"), so every Atlas price is marked unverified. Higgsfield documents no
per-second price on any page we checked, so its models have no price here and
the bot says so instead of inventing one.

Cost ledger rows store est_usd together with price_verified=False until a
price is checked against the provider's own price list.
"""

# model name (as in settings.video_model) -> USD per output second
VIDEO_PRICE_PER_SECOND: dict[str, float] = {
    "seedance": 0.09,
    "seedance_fast": 0.072,
    "seedance_mini": 0.045,
    "seedance_t2v": 0.09,
    "seedance_mini_t2v": 0.045,
    "seedance_ref": 0.09,
    "seedance_fast_ref": 0.072,
    "seedance_25": 0.09,
    "seedance_25_t2v": 0.09,
    "seedance_25_ref": 0.09,
    "kling": 0.095,
    "kling_v3_std": 0.071,
    "kling_turbo": 0.095,
    "kling_o3_pro": 0.126,
    "kling_o3_std": 0.10,
    "kling_t2v": 0.095,
    "kling_turbo_t2v": 0.095,
    "kling_o3_pro_ref": 0.126,
    "kling_o3_std_ref": 0.10,
    "minimax_h3": 0.09,
    "minimax_h3_t2v": 0.09,
    "minimax_h3_ref": 0.09,
}


def video_provider(model: str) -> str:
    """Which paid API serves this video model (used as the ledger 'provider')."""
    return "higgsfield" if (model or "").lower().startswith("higgsfield_") else "atlas"


def estimate_video_usd(model: str, seconds: float) -> float | None:
    """Estimated cost of `seconds` of video, or None when the model has no known price."""
    rate = VIDEO_PRICE_PER_SECOND.get((model or "").lower())
    if rate is None or seconds <= 0:
        return None
    return round(rate * seconds, 4)


def format_video_cost_line(model: str, seconds: float) -> str:
    """One HTML-safe line for the video prompt screen."""
    usd = estimate_video_usd(model, seconds)
    if usd is None:
        return (
            "💰 Cost estimate: not available for this model "
            "(check the provider's console for the price)."
        )
    rate = VIDEO_PRICE_PER_SECOND[(model or "").lower()]
    return (
        f"💰 Cost estimate: ~${usd:.2f} for {seconds:g}s "
        f"(${rate:g}/s, unverified price)."
    )
