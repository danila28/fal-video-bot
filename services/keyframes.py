"""Keyframes mode: 2-4 own photos used in order, one clip between each pair.

With photos P1..Pn the video is made of n-1 clips: clip i starts exactly on Pi
and ends exactly on Pi+1 (first frame -> last frame). The clips are joined with
hard cuts, which is seamless because the last frame of one clip and the first
frame of the next are the same photo.

Only models whose API has a documented end-frame parameter are supported:
  Atlas       kling, kling_v3_std      -> "end_image"          (Atlas model pages)
  Higgsfield  higgsfield_kling_std     -> "last_image_url"     (Higgsfield docs)
              higgsfield_minimax_h3    -> "end_image_url"      (Higgsfield docs)
Not supported (no confirmed end-frame field): Kling v3 Turbo, Kling O3, Seedance,
MiniMax on Atlas, reference models and text-to-video models.

The setting is `keyframe_photos` (1 = off, the classic single start photo).
"""

from dataclasses import dataclass

MIN_PHOTOS = 2
MAX_PHOTOS = 4


@dataclass(frozen=True)
class KeyframeModel:
    provider: str                       # "atlas" | "higgsfield"
    label: str
    min_dur: int                        # seconds per clip
    max_dur: int
    allowed: tuple[int, ...] | None = None   # exact clip lengths when the API accepts only some


KEYFRAME_MODELS: dict[str, KeyframeModel] = {
    # Atlas Kling v3 single-clip calls are used with 5 s or 10 s only: the project
    # found that other lengths are silently rendered at a different length.
    "kling": KeyframeModel("atlas", "Kling v3 Pro (Atlas)", 5, 10, allowed=(5, 10)),
    "kling_v3_std": KeyframeModel("atlas", "Kling v3 Std (Atlas)", 5, 10, allowed=(5, 10)),
    "higgsfield_kling_std": KeyframeModel("higgsfield", "Kling 3.0 Std (Higgsfield)", 3, 15),
    "higgsfield_minimax_h3": KeyframeModel("higgsfield", "MiniMax H3 (Higgsfield)", 5, 15),
}


def supports_keyframes(video_model: str) -> bool:
    return (video_model or "").lower() in KEYFRAME_MODELS


def plan_keyframe_durations(video_model: str, target_seconds: float, photo_count: int) -> list[int]:
    """Seconds for each of the photo_count-1 clips, as close to an even split of
    target_seconds as the model allows. The sum can differ from the target when
    the model's clip lengths force it (e.g. 5 s minimum per clip)."""
    spec = KEYFRAME_MODELS.get((video_model or "").lower())
    if spec is None:
        raise ValueError(f"{video_model!r} does not support keyframes")
    if not MIN_PHOTOS <= photo_count <= MAX_PHOTOS:
        raise ValueError(f"keyframes need {MIN_PHOTOS}-{MAX_PHOTOS} photos, got {photo_count}")

    per_clip = int(target_seconds / (photo_count - 1) + 0.5)
    if spec.allowed:
        # nearest allowed length; a tie goes to the shorter clip (cheaper)
        per_clip = min(spec.allowed, key=lambda a: (abs(a - per_clip), a))
    else:
        per_clip = max(spec.min_dur, min(spec.max_dur, per_clip))
    return [per_clip] * (photo_count - 1)


def supported_labels() -> str:
    return ", ".join(spec.label for spec in KEYFRAME_MODELS.values())


def unsupported_message() -> str:
    """Shown instead of letting the user pay for photos that would never be used."""
    return (
        "⚠️ The selected video model cannot use several photos in order (keyframes). "
        "To avoid unnecessary costs only 1 photo will be used.\n"
        f"To use several photos choose one of: {supported_labels()} "
        "in Settings → Video model."
    )


def keyframes_problem(video_model: str, wanted_photos: int) -> str | None:
    """Warning text when several photos are wanted but the model cannot use them
    in order; None when everything is fine or the setting does not apply.

    Reference models take several photos on their own (as references, not as
    ordered keyframes) and text-to-video models take none, so neither is a problem."""
    model = (video_model or "").lower()
    if wanted_photos <= 1 or supports_keyframes(model):
        return None
    if model.endswith("_ref") or "t2v" in model:
        return None
    return unsupported_message()
