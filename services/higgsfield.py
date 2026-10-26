"""Higgsfield video models: Seedance, Kling and MiniMax served through Higgsfield.

Endpoints and body fields come from the Higgsfield API docs:
  POST https://api.higgsfield.ai/<endpoint>
       -> {"status": "queued", "request_id", "status_url", "cancel_url"}
  GET  <status_url>  (documented as "poll status_url until status is completed")
       status: queued | in_progress | completed | failed | nsfw | canceled
       completed -> {"video": {"url": "<mp4>"}}
  Auth header: "Authorization: Key <HIGGSFIELD_API_KEY>".

Media inputs must be public URLs. Local photos are uploaded through the same
Atlas upload endpoint the Kling service already uses (KlingService.upload_photo).

Supported models (settings video_model):
  higgsfield_kling_std      Kling 3.0 Standard, image-to-video (one photo)
  higgsfield_minimax_h3     MiniMax H3, image-to-video (one photo)
  higgsfield_seedance_ref   Seedance 2.5, reference-to-video (one or more photos)
"""

import asyncio
import logging
import os
import uuid
from dataclasses import dataclass

import aiohttp

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.higgsfield.ai"
_POLL_INTERVAL = 5           # seconds between status checks
_POLL_TIMEOUT = 600          # 10 minutes max wait
_POLL_MAX_TRANSIENT_ERRORS = 6
_KLING_MAX_PROMPT_CHARS = 2500

_STATUS_FAILED = {"failed", "nsfw", "canceled"}
_SEEDANCE_RESOLUTIONS = {"480p", "720p", "1080p"}   # Seedance 2.5 has no 4k


class HiggsfieldError(Exception):
    """Any failure of a Higgsfield request. Message is safe to show to the user."""


@dataclass(frozen=True)
class HiggsfieldModel:
    endpoint: str       # path after the base URL, e.g. "minimax/h3/image-to-video"
    label: str          # human-readable name for notify messages
    min_dur: int        # seconds per clip
    max_dur: int
    reference: bool     # True = takes several images in "image_urls"
    has_audio: bool     # True = accepts a sound flag


HF_MODELS: dict[str, HiggsfieldModel] = {
    "higgsfield_kling_std": HiggsfieldModel(
        endpoint="kling-video/v3.0/std/image-to-video",
        label="Kling 3.0 Std (Higgsfield)",
        min_dur=3, max_dur=15, reference=False, has_audio=True,
    ),
    "higgsfield_minimax_h3": HiggsfieldModel(
        endpoint="minimax/h3/image-to-video",
        label="MiniMax H3 (Higgsfield)",
        min_dur=5, max_dur=15, reference=False, has_audio=False,
    ),
    "higgsfield_seedance_ref": HiggsfieldModel(
        endpoint="bytedance/seedance-2.5/reference-to-video",
        label="Seedance 2.5 Ref (Higgsfield)",
        min_dur=4, max_dur=30, reference=True, has_audio=True,
    ),
}


def build_request_body(
    model_key: str,
    *,
    prompt: str,
    duration: int,
    image_urls: list[str],
    resolution: str = "720p",
    sound: bool = False,
) -> dict:
    """Body for one clip. Only fields documented for each endpoint are sent."""
    spec = HF_MODELS.get(model_key)
    if spec is None:
        raise HiggsfieldError(f"Unknown Higgsfield model: {model_key}")
    if not image_urls:
        raise HiggsfieldError(f"{spec.label} needs at least one public image URL")

    duration = max(spec.min_dur, min(spec.max_dur, int(duration)))
    prompt = (prompt or "").strip() or "Smooth cinematic motion."

    if model_key == "higgsfield_kling_std":
        return {
            "prompt": prompt[:_KLING_MAX_PROMPT_CHARS],
            "image_url": image_urls[0],
            "duration": duration,
            "sound": "on" if sound else "off",
        }
    if model_key == "higgsfield_minimax_h3":
        return {
            "prompt": prompt,
            "image_url": image_urls[0],
            "duration": duration,
            "resolution": "2K",
            "aspect_ratio": "9:16",
            "aigc_watermark": False,
        }
    # higgsfield_seedance_ref
    if resolution not in _SEEDANCE_RESOLUTIONS:
        resolution = "720p"
    return {
        "prompt": prompt,
        "image_urls": image_urls[:30],   # documented maximum for Seedance 2.5
        "duration": duration,
        "resolution": resolution,
        "aspect_ratio": "9:16",
        "generate_audio": bool(sound),
    }


def extract_video_url(status_payload: dict) -> str | None:
    video = status_payload.get("video")
    if isinstance(video, dict) and video.get("url"):
        return video["url"]
    return None


class HiggsfieldService:
    """Submit, poll and download one Higgsfield clip at a time."""

    def __init__(self, api_key: str, static_dir: str = ""):
        self._api_key = api_key
        if not static_dir:
            static_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static"
            )
        self.static_dir = static_dir
        os.makedirs(static_dir, exist_ok=True)

    def _headers(self, json_body: bool = False) -> dict:
        headers = {"Authorization": f"Key {self._api_key}"}
        if json_body:
            headers["Content-Type"] = "application/json"
            headers["Idempotency-Key"] = str(uuid.uuid4())
        return headers

    async def generate_clip(
        self,
        model_key: str,
        *,
        prompt: str,
        duration: int,
        image_urls: list[str],
        resolution: str = "720p",
        sound: bool = False,
    ) -> str:
        """Submit one clip, wait for it, download it. Returns the local mp4 path."""
        spec = HF_MODELS[model_key]
        body = build_request_body(
            model_key, prompt=prompt, duration=duration,
            image_urls=image_urls, resolution=resolution, sound=sound,
        )
        async with aiohttp.ClientSession() as session:
            submitted = await self._submit(session, spec.endpoint, body)
            video_url = await self._wait(session, submitted, spec.label)
        return await self.download(video_url)

    async def _submit(self, session: aiohttp.ClientSession, endpoint: str, body: dict) -> dict:
        url = f"{_BASE_URL}/{endpoint}"
        logger.info(f"Higgsfield submit {endpoint}")
        async with session.post(url, headers=self._headers(json_body=True), json=body) as resp:
            text = await resp.text()
            if resp.status not in (200, 201, 202):
                raise HiggsfieldError(f"Higgsfield submit failed HTTP {resp.status}: {text[:300]}")
            data = await resp.json()
        if not data.get("request_id"):
            raise HiggsfieldError(f"No request_id in Higgsfield response: {str(data)[:300]}")
        return data

    async def _wait(self, session: aiohttp.ClientSession, submitted: dict, label: str) -> str:
        status_url = submitted.get("status_url") or (
            f"{_BASE_URL}/requests/{submitted['request_id']}/status"
        )
        elapsed = 0
        transient = 0
        while elapsed < _POLL_TIMEOUT:
            await asyncio.sleep(_POLL_INTERVAL)
            elapsed += _POLL_INTERVAL
            try:
                async with session.get(status_url, headers=self._headers()) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        if resp.status >= 500 and transient < _POLL_MAX_TRANSIENT_ERRORS:
                            transient += 1
                            continue
                        raise HiggsfieldError(f"Higgsfield status failed HTTP {resp.status}: {text[:200]}")
                    payload = await resp.json()
            except aiohttp.ClientError as e:
                if transient < _POLL_MAX_TRANSIENT_ERRORS:
                    transient += 1
                    continue
                raise HiggsfieldError(f"Higgsfield status network error: {e}")
            transient = 0

            status = payload.get("status")
            if status == "completed":
                url = extract_video_url(payload)
                if not url:
                    raise HiggsfieldError(f"{label}: completed but no video url in response")
                return url
            if status == "nsfw":
                raise HiggsfieldError(
                    f"{label}: rejected by the content filter (not charged). Try another photo or script."
                )
            if status in _STATUS_FAILED:
                err = payload.get("error") or status
                raise HiggsfieldError(f"{label}: generation {status}: {err}")
        raise HiggsfieldError(f"{label}: timeout after {_POLL_TIMEOUT}s")

    async def download(self, url: str, ext: str = "mp4") -> str:
        dest = os.path.join(self.static_dir, f"{uuid.uuid4()}.{ext}")
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    raise HiggsfieldError(f"Higgsfield download failed HTTP {resp.status}")
                with open(dest, "wb") as fh:
                    async for chunk in resp.content.iter_chunked(65536):
                        fh.write(chunk)
        logger.info(f"Higgsfield downloaded → {dest}")
        return dest
