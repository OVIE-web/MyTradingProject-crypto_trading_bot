# src/routers/webhook.py
from __future__ import annotations

import hashlib
import hmac
import logging
import os

from dotenv import load_dotenv
from fastapi import APIRouter, Header, HTTPException, Request, status

load_dotenv()

logger = logging.getLogger(__name__)
router = APIRouter()

WEBHOOK_SECRET = os.getenv("GITHUB_WEBHOOK_SECRET", "")


def verify_signature(payload: bytes, signature: str) -> bool:
    """Verify GitHub X-Hub-Signature-256 header."""
    if not WEBHOOK_SECRET:
        logger.warning("GITHUB_WEBHOOK_SECRET not set")
        return False

    expected = "sha256=" + hmac.new(WEBHOOK_SECRET.encode(), payload, hashlib.sha256).hexdigest()

    return hmac.compare_digest(expected, signature)


@router.post("/", status_code=status.HTTP_200_OK)
async def github_webhook(
    request: Request,
    x_hub_signature_256: str = Header(default=""),
    x_github_event: str = Header(default=""),
) -> dict[str, str]:
    """
    Receive and verify GitHub webhook payloads.
    """
    payload = await request.body()

    # Verify signature
    if not verify_signature(payload, x_hub_signature_256):
        logger.warning("Invalid webhook signature received")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid signature")

    # Handle push event
    if x_github_event == "push":
        logger.info("Push event received from GitHub")
        # Add your bot trigger logic here
        return {"status": "Push event received"}

    return {"status": f"Event '{x_github_event}' received but not handled"}
