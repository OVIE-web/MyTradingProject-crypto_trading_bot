# tests/routers/test_routers_webhook.py
from __future__ import annotations

import hashlib
import hmac
import json
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.routers.webhook import router, verify_signature

# -----------------------------------------------------------------------
# App Setup
# NOTE: Router mounted WITHOUT prefix here (mirrors isolated unit testing).
#       In main_api.py, prefix="/webhook" is added — meaning the live URL
#       is POST /webhook. In tests, we mount raw router so URL is POST /
# -----------------------------------------------------------------------
app = FastAPI()
app.include_router(router)
client = TestClient(app)

MOCK_SECRET = "test_webhook_secret_abc123"


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------
def make_signature(payload: bytes, secret: str = MOCK_SECRET) -> str:
    """Generate a valid X-Hub-Signature-256 for a given payload."""
    digest = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def make_payload(data: dict) -> bytes:
    """Serialize a dict to JSON bytes (mirrors GitHub's payload format)."""
    return json.dumps(data).encode()


# -----------------------------------------------------------------------
# Tests: verify_signature()
# -----------------------------------------------------------------------
class TestVerifySignature:
    def test_valid_signature_returns_true(self):
        payload = make_payload({"ref": "refs/heads/main"})
        signature = make_signature(payload)
        with patch("src.routers.webhook.WEBHOOK_SECRET", MOCK_SECRET):
            assert verify_signature(payload, signature) is True

    def test_invalid_signature_returns_false(self):
        payload = make_payload({"ref": "refs/heads/main"})
        with patch("src.routers.webhook.WEBHOOK_SECRET", MOCK_SECRET):
            assert verify_signature(payload, "sha256=invalidsignature") is False

    def test_empty_secret_returns_false(self):
        payload = make_payload({"ref": "refs/heads/main"})
        signature = make_signature(payload)
        with patch("src.routers.webhook.WEBHOOK_SECRET", ""):
            assert verify_signature(payload, signature) is False

    def test_wrong_secret_returns_false(self):
        payload = make_payload({"ref": "refs/heads/main"})
        signature = make_signature(payload, secret="wrong_secret")
        with patch("src.routers.webhook.WEBHOOK_SECRET", MOCK_SECRET):
            assert verify_signature(payload, signature) is False

    def test_empty_signature_returns_false(self):
        payload = make_payload({"ref": "refs/heads/main"})
        with patch("src.routers.webhook.WEBHOOK_SECRET", MOCK_SECRET):
            assert verify_signature(payload, "") is False

    def test_signature_without_prefix_returns_false(self):
        payload = make_payload({"ref": "refs/heads/main"})
        raw_digest = hmac.new(MOCK_SECRET.encode(), payload, hashlib.sha256).hexdigest()
        with patch("src.routers.webhook.WEBHOOK_SECRET", MOCK_SECRET):
            assert verify_signature(payload, raw_digest) is False


# -----------------------------------------------------------------------
# Tests: POST / — Push Event
# FIX: URL is "/" not "/webhook" because:
#   - webhook.py defines @router.post("/")
#   - main_api.py adds prefix="/webhook" at app level
#   - TestClient mounts router directly (no prefix) → URL = "/"
# -----------------------------------------------------------------------
class TestWebhookPushEvent:
    def test_valid_push_event_returns_200(self):
        payload = make_payload({"ref": "refs/heads/main", "commits": []})
        signature = make_signature(payload)
        with patch("src.routers.webhook.WEBHOOK_SECRET", MOCK_SECRET):
            response = client.post(
                "/",  # ✅ FIXED from /webhook
                content=payload,
                headers={
                    "X-Hub-Signature-256": signature,
                    "X-GitHub-Event": "push",
                    "Content-Type": "application/json",
                },
            )
        assert response.status_code == 200
        assert response.json() == {"status": "Push event received"}

    def test_push_event_missing_signature_returns_401(self):
        payload = make_payload({"ref": "refs/heads/main"})
        with patch("src.routers.webhook.WEBHOOK_SECRET", MOCK_SECRET):
            response = client.post(
                "/",
                content=payload,
                headers={
                    "X-GitHub-Event": "push",
                    "Content-Type": "application/json",
                },
            )
        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid signature"

    def test_push_event_invalid_signature_returns_401(self):
        payload = make_payload({"ref": "refs/heads/main"})
        with patch("src.routers.webhook.WEBHOOK_SECRET", MOCK_SECRET):
            response = client.post(
                "/",
                content=payload,
                headers={
                    "X-Hub-Signature-256": "sha256=badsignature",
                    "X-GitHub-Event": "push",
                    "Content-Type": "application/json",
                },
            )
        assert response.status_code == 401

    def test_push_event_tampered_payload_returns_401(self):
        original_payload = make_payload({"ref": "refs/heads/main"})
        signature = make_signature(original_payload)
        tampered_payload = make_payload({"ref": "refs/heads/malicious"})
        with patch("src.routers.webhook.WEBHOOK_SECRET", MOCK_SECRET):
            response = client.post(
                "/",
                content=tampered_payload,
                headers={
                    "X-Hub-Signature-256": signature,
                    "X-GitHub-Event": "push",
                    "Content-Type": "application/json",
                },
            )
        assert response.status_code == 401


# -----------------------------------------------------------------------
# Tests: POST / — Other Events
# -----------------------------------------------------------------------
class TestWebhookOtherEvents:
    def test_unhandled_event_returns_200_with_message(self):
        payload = make_payload({"action": "opened"})
        signature = make_signature(payload)
        with patch("src.routers.webhook.WEBHOOK_SECRET", MOCK_SECRET):
            response = client.post(
                "/",
                content=payload,
                headers={
                    "X-Hub-Signature-256": signature,
                    "X-GitHub-Event": "pull_request",
                    "Content-Type": "application/json",
                },
            )
        assert response.status_code == 200
        assert "pull_request" in response.json()["status"]
        assert "not handled" in response.json()["status"]

    def test_ping_event_returns_200(self):
        payload = make_payload({"zen": "Keep it logically awesome."})
        signature = make_signature(payload)
        with patch("src.routers.webhook.WEBHOOK_SECRET", MOCK_SECRET):
            response = client.post(
                "/",
                content=payload,
                headers={
                    "X-Hub-Signature-256": signature,
                    "X-GitHub-Event": "ping",
                    "Content-Type": "application/json",
                },
            )
        assert response.status_code == 200

    def test_missing_event_header_returns_200(self):
        payload = make_payload({"data": "unknown"})
        signature = make_signature(payload)
        with patch("src.routers.webhook.WEBHOOK_SECRET", MOCK_SECRET):
            response = client.post(
                "/",
                content=payload,
                headers={
                    "X-Hub-Signature-256": signature,
                    "Content-Type": "application/json",
                },
            )
        assert response.status_code == 200


# -----------------------------------------------------------------------
# Tests: Edge Cases
# -----------------------------------------------------------------------
class TestWebhookEdgeCases:
    def test_empty_payload_with_valid_signature(self):
        payload = b""
        signature = make_signature(payload)
        with patch("src.routers.webhook.WEBHOOK_SECRET", MOCK_SECRET):
            response = client.post(
                "/",
                content=payload,
                headers={
                    "X-Hub-Signature-256": signature,
                    "X-GitHub-Event": "push",
                    "Content-Type": "application/json",
                },
            )
        assert response.status_code == 200

    def test_large_payload_with_valid_signature(self):
        payload = make_payload({"commits": [{"id": str(i)} for i in range(500)]})
        signature = make_signature(payload)
        with patch("src.routers.webhook.WEBHOOK_SECRET", MOCK_SECRET):
            response = client.post(
                "/",
                content=payload,
                headers={
                    "X-Hub-Signature-256": signature,
                    "X-GitHub-Event": "push",
                    "Content-Type": "application/json",
                },
            )
        assert response.status_code == 200

    def test_no_secret_configured_returns_401(self):
        payload = make_payload({"ref": "refs/heads/main"})
        signature = make_signature(payload)
        with patch("src.routers.webhook.WEBHOOK_SECRET", ""):
            response = client.post(
                "/",
                content=payload,
                headers={
                    "X-Hub-Signature-256": signature,
                    "X-GitHub-Event": "push",
                    "Content-Type": "application/json",
                },
            )
        assert response.status_code == 401
