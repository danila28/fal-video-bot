"""Shared Atlas Cloud HTTP client used by all generation services."""

import asyncio
import logging
import os
import uuid

import aiohttp

logger = logging.getLogger(__name__)

_BASE_URL      = "https://api.atlascloud.ai/api/v1"
_POLL_INTERVAL = 5    # seconds between status checks
_POLL_TIMEOUT  = 600  # 10 minutes max wait
_POLL_MAX_TRANSIENT_ERRORS = 6  # consecutive 5xx/network errors tolerated before giving up


class AtlasClient:
    """Thin async wrapper around the Atlas Cloud REST API.

    Shared by all generation services so the upload / generate / poll /
    download pattern is implemented exactly once.
    """

    def __init__(self, api_key: str, static_dir: str = ""):
        self._api_key = api_key
        if not static_dir:
            static_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static"
            )
        self.static_dir = static_dir
        os.makedirs(static_dir, exist_ok=True)

    @property
    def _auth_headers(self) -> dict:
        return {"Authorization": f"Bearer {self._api_key}"}

    @property
    def _json_headers(self) -> dict:
        return {**self._auth_headers, "Content-Type": "application/json"}

    # ── Upload ─────────────────────────────────────────────────────────────

    async def upload_file(self, file_path: str) -> str:
        """Upload a local file to Atlas Cloud storage. Returns the public URL."""
        async with aiohttp.ClientSession() as session:
            form = aiohttp.FormData()
            with open(file_path, "rb") as fh:
                form.add_field(
                    "file", fh,
                    filename=os.path.basename(file_path),
                )
                async with session.post(
                    f"{_BASE_URL}/model/uploadMedia",
                    headers=self._auth_headers,
                    data=form,
                ) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        raise Exception(
                            f"Atlas upload failed HTTP {resp.status}: {text[:200]}"
                        )
                    result = await resp.json()

        data = result.get("data") or {}
        url = result.get("url") or data.get("url") or data.get("download_url")
        if not url:
            raise Exception(f"No URL in Atlas upload response: {result}")
        logger.info(f"Uploaded {file_path} → {url}")
        return url

    # ── Generation ─────────────────────────────────────────────────────────

    async def generate_video(self, model: str, params: dict) -> str:
        """Submit a video generation job and poll until complete. Returns output URL."""
        return await self._submit_and_poll("generateVideo", model, params)

    async def generate_image(self, model: str, params: dict) -> str:
        """Submit an image generation job and poll until complete. Returns output URL."""
        return await self._submit_and_poll("generateImage", model, params)

    async def _submit_and_poll(self, endpoint: str, model: str, params: dict) -> str:
        async with aiohttp.ClientSession() as session:
            # ── Submit ────────────────────────────────────────────────
            payload = {"model": model, **params}
            logger.info(f"Atlas {endpoint} payload: {payload}")
            async with session.post(
                f"{_BASE_URL}/model/{endpoint}",
                headers=self._json_headers,
                json=payload,
            ) as resp:
                if resp.status not in (200, 201):
                    text = await resp.text()
                    raise Exception(
                        f"Atlas {endpoint} submit failed HTTP {resp.status}: {text[:300]}"
                    )
                data = await resp.json()

            # Normalise: both {"data": {"id": ...}} and {"id": ...} are seen.
            inner = data.get("data") or data
            pred_id = inner.get("id")
            if not pred_id:
                raise Exception(f"No prediction id in Atlas response: {data}")
            logger.info(f"Atlas {endpoint}: model={model} id={pred_id}")

            # ── Poll ─────────────────────────────────────────────────
            poll_url = f"{_BASE_URL}/model/prediction/{pred_id}"
            elapsed = 0
            consecutive_errors = 0
            while elapsed < _POLL_TIMEOUT:
                await asyncio.sleep(_POLL_INTERVAL)
                elapsed += _POLL_INTERVAL

                try:
                    async with session.get(
                        poll_url, headers=self._auth_headers
                    ) as resp:
                        if resp.status != 200:
                            text = await resp.text()
                            # Atlas's gateway occasionally returns a transient
                            # 5xx (HTML error page) mid-generation — retry a
                            # few times instead of failing the whole job.
                            if resp.status >= 500 and consecutive_errors < _POLL_MAX_TRANSIENT_ERRORS:
                                consecutive_errors += 1
                                logger.warning(
                                    f"Atlas poll transient HTTP {resp.status} "
                                    f"({consecutive_errors}/{_POLL_MAX_TRANSIENT_ERRORS}), "
                                    f"retrying: {pred_id}"
                                )
                                continue
                            raise Exception(
                                f"Atlas poll failed HTTP {resp.status}: {text[:200]}"
                            )
                        poll = await resp.json()
                except aiohttp.ClientError as e:
                    if consecutive_errors < _POLL_MAX_TRANSIENT_ERRORS:
                        consecutive_errors += 1
                        logger.warning(
                            f"Atlas poll network error "
                            f"({consecutive_errors}/{_POLL_MAX_TRANSIENT_ERRORS}), "
                            f"retrying: {pred_id}: {e}"
                        )
                        continue
                    raise Exception(f"Atlas poll network error: {e}")

                consecutive_errors = 0

                # Normalise: {"data": {...}} or flat {"status": ...}
                status_data = poll.get("data") or poll
                status = status_data.get("status")

                if status in ("completed", "succeeded"):
                    # Video uses "outputs", image uses "output"
                    outputs = (
                        status_data.get("outputs")
                        or status_data.get("output")
                        or []
                    )
                    if not outputs:
                        raise Exception(
                            f"Atlas completed but no outputs in response: {poll}"
                        )
                    return outputs[0]

                if status == "failed":
                    error = status_data.get("error") or "unknown error"
                    raise Exception(f"Atlas generation failed: {error}")

                logger.debug(
                    f"Atlas poll: {pred_id} → {status} ({elapsed}s elapsed)"
                )

        raise Exception(
            f"Atlas generation timeout after {_POLL_TIMEOUT}s: model={model}"
        )

    # ── Download ───────────────────────────────────────────────────────────

    async def download(self, url: str, ext: str = "mp4") -> str:
        """Download a file from URL into static dir. Returns local path."""
        filename = f"{uuid.uuid4()}.{ext}"
        dest = os.path.join(self.static_dir, filename)
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    raise Exception(
                        f"Atlas download failed HTTP {resp.status}: {url}"
                    )
                with open(dest, "wb") as fh:
                    async for chunk in resp.content.iter_chunked(65536):
                        fh.write(chunk)
        logger.info(f"Downloaded → {dest}")
        return dest
