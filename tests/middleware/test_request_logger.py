"""Tests for the request logging middleware."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.middleware import request_logger
from app.middleware.request_logger import (
    PROCESS_TIME_HEADER,
    REQUEST_ID_HEADER,
    RequestLoggerMiddleware,
    register_request_logger,
)


def create_app() -> FastAPI:
    """Create a tiny app with request logging middleware installed."""
    app = FastAPI()
    register_request_logger(app)

    @app.get("/ok")
    def ok() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/boom")
    def boom() -> None:
        raise RuntimeError("boom")

    return app


def test_register_request_logger_adds_middleware() -> None:
    """The registration helper should install RequestLoggerMiddleware."""
    app = FastAPI()

    register_request_logger(app)

    assert app.user_middleware[0].cls is RequestLoggerMiddleware


def test_request_logger_preserves_incoming_request_id(mocker) -> None:
    """Existing request IDs should be propagated to the response headers."""
    mock_info = mocker.patch.object(request_logger.LOG, "info")
    client = TestClient(create_app())

    response = client.get("/ok", headers={REQUEST_ID_HEADER: "request-123"})

    assert response.status_code == 200
    assert response.headers[REQUEST_ID_HEADER] == "request-123"
    assert float(response.headers[PROCESS_TIME_HEADER]) >= 0
    assert mock_info.call_count == 2
    assert "request_started" in mock_info.call_args_list[0].args[0]
    assert "request_completed" in mock_info.call_args_list[1].args[0]


def test_request_logger_generates_request_id_when_missing(monkeypatch) -> None:
    """Requests without IDs should receive a generated request ID."""
    monkeypatch.setattr(request_logger, "uuid4", lambda: "generated-id")
    client = TestClient(create_app())

    response = client.get("/ok")

    assert response.status_code == 200
    assert response.headers[REQUEST_ID_HEADER] == "generated-id"
    assert float(response.headers[PROCESS_TIME_HEADER]) >= 0


def test_request_logger_logs_failed_requests(mocker) -> None:
    """Unhandled route exceptions should be logged before being re-raised."""
    mock_exception = mocker.patch.object(request_logger.LOG, "exception")
    client = TestClient(create_app(), raise_server_exceptions=False)

    response = client.get("/boom", headers={REQUEST_ID_HEADER: "request-err"})

    assert response.status_code == 500
    mock_exception.assert_called_once()
    assert "request_failed" in mock_exception.call_args.args[0]
    assert "request-err" in mock_exception.call_args.args
