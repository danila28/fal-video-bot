import aiohttp
import json
import logging
import os
import mimetypes

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

logger = logging.getLogger(__name__)

# Network calls to Blotato are idempotent enough at the granularity of each
# request (presign / upload / publish) — 429 and 5xx are common with social
# APIs, retry is the right default. NOTE: full publish_video flow is *not*
# wrapped because partial state (file uploaded, post not created) shouldn't
# trigger a re-upload.
_retry_net = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING),
)


class BlotatoService:
    """Blotato REST API integration (presigned upload flow)"""

    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    @_retry_net
    async def _get_presigned_url(self, filename: str) -> dict:
        """Step 1: get presigned URL"""

        url = f"{self.base_url}/media/uploads"

        headers = {
            "Content-Type": "application/json",
            "blotato-api-key": self.api_key,
        }

        payload = {
            "filename": filename,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as resp:

                text = await resp.text()

                if resp.status != 201:
                    logger.error(f"Presigned URL error {resp.status}: {text}")
                    raise Exception(f"Blotato presigned error: {text}")

                return await resp.json()

    @_retry_net
    async def _upload_file_to_presigned(self, presigned_url: str, file_path: str):
        """Step 2: upload file via PUT"""

        content_type, _ = mimetypes.guess_type(file_path)
        content_type = content_type or "application/octet-stream"

        with open(file_path, "rb") as f:
            async with aiohttp.ClientSession() as session:
                async with session.put(
                    presigned_url,
                    data=f,
                    headers={"Content-Type": content_type},
                ) as resp:

                    if resp.status not in (200, 201):
                        text = await resp.text()
                        logger.error(f"Upload error {resp.status}: {text}")
                        raise Exception(f"Upload failed: {text}")

    @_retry_net
    async def _publish_post(
        self,
        public_url: str,
        title: str,
        account_id: str,
        platform: str,
        scheduled_at: str = "",
    ):
        """Step 3: create post.

        `scheduled_at` — ISO 8601 UTC datetime string, e.g. "2025-06-15T14:30:00Z".
        When empty the post is published immediately.
        """

        url = f"{self.base_url}/posts"

        headers = {
            "Content-Type": "application/json",
            "blotato-api-key": self.api_key,
        }

        content = {
            "text": title,
            "mediaUrls": [public_url],
            "platform": platform,
        }

        # --- TARGET BLOCK ---
        target = {
            "targetType": platform,
        }

        # --- PLATFORM-SPECIFIC REQUIRED FIELDS ---

        if platform == "tiktok":
            target.update(
                {
                    "privacyLevel": "PUBLIC_TO_EVERYONE",
                    "disabledComments": False,
                    "disabledDuet": False,
                    "disabledStitch": False,
                    "isBrandedContent": False,
                    "isYourBrand": False,
                    "isAiGenerated": True,
                }
            )

        elif platform == "youtube":
            target.update(
                {
                    "title": title,
                    "privacyStatus": "public",
                    "shouldNotifySubscribers": True,
                }
            )

        post_block: dict = {
            "accountId": account_id,
            "content": content,
            "target": target,
        }

        # scheduledTime must be a root-level sibling of "post", not nested inside it.
        payload: dict = {"post": post_block}
        if scheduled_at:
            payload["scheduledTime"] = scheduled_at

        logger.info(
            f"Blotato publish → platform={platform} account={account_id} "
            f"scheduled_at={scheduled_at!r}\n"
            f"payload: {json.dumps(payload, ensure_ascii=False, indent=2)}"
        )

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as resp:

                text = await resp.text()

                if resp.status >= 400:
                    raise Exception(f"{platform} publish error {resp.status}: {text}")

                return await resp.json()

    async def publish_video(
        self,
        video_path: str,
        title: str,
        accounts: str,
        scheduled_at: str = "",
    ):
        """
        Full flow:
        1. presigned URL
        2. upload file
        3. publish post(s)

        `scheduled_at` — ISO 8601 UTC string (e.g. "2025-06-15T14:30:00Z").
        Pass empty string to publish immediately.
        """

        if not os.path.exists(video_path):
            raise Exception("Video file does not exist")

        filename = os.path.basename(video_path)

        # --- STEP 1 ---
        presigned_data = await self._get_presigned_url(filename)
        presigned_url = presigned_data["presignedUrl"]
        public_url = presigned_data["publicUrl"]

        # --- STEP 2 ---
        await self._upload_file_to_presigned(presigned_url, video_path)

        # --- STEP 3 ---
        results = []

        # accounts: "youtube:123,tiktok:456"
        pairs = [acc.strip() for acc in accounts.split(",") if acc.strip()]

        for pair in pairs:
            try:
                platform, account_id = pair.split(":", 1)

                result = await self._publish_post(
                    public_url=public_url,
                    title=title,
                    account_id=account_id,
                    platform=platform,
                    scheduled_at=scheduled_at,
                )

                results.append(result)

            except Exception as e:
                logger.error(f"Failed for {pair}: {e}")
                results.append({"account": pair, "error": str(e)})

        return results
