"""Tests for the error handling middleware functions."""

from __future__ import annotations

import json

from fastapi import FastAPI, status
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.requests import Request

from app.middleware import error_handler


def make_request(path: str = "/example") -> Request:
    """Build a lightweight Starlette request for direct handler tests."""
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": path,
            "raw_path": path.encode(),
            "headers": [],
            "query_string": b"",
            "scheme": "http",
            "server": ("testserver", 80),
            "client": ("127.0.0.1", 12345),
        }
    )


def response_json(response) -> dict:
    """Decode JSONResponse content for assertions."""
    return json.loads(response.body.decode())


def test_error_payload_contains_standard_shape() -> None:
    """Middleware error payloads should share one response contract."""
    payload = error_handler._error_payload(
        status_code=404,
        error="not_found",
        message="Missing",
        path="/missing",
        details=[{"field": "name"}],
    )

    assert payload["success"] is False
    assert payload["error"]["code"] == 404
    assert payload["error"]["type"] == "not_found"
    assert payload["error"]["message"] == "Missing"
    assert payload["error"]["path"] == "/missing"
    assert payload["error"]["details"] == [{"field": "name"}]
    assert "timestamp" in payload["error"]


def test_http_exception_handler_returns_consistent_json(mocker) -> None:
    """Expected HTTP errors should be converted to the shared JSON shape."""
    mock_log_info = mocker.patch.object(error_handler.LOG, "info")
    request = make_request("/missing")
    exc = StarletteHTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Not found",
        headers={"X-Test": "yes"},
    )

    response = error_handler.http_exception_handler(request, exc)
    payload = response_json(response)

    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert response.headers["x-test"] == "yes"
    assert payload["error"]["type"] == "http_error"
    assert payload["error"]["message"] == "Not found"
    assert payload["error"]["path"] == "/missing"
    mock_log_info.assert_called_once()


def test_http_exception_handler_logs_server_errors_as_errors(mocker) -> None:
    """5xx HTTP exceptions should use error-level logging."""
    mock_log_error = mocker.patch.object(error_handler.LOG, "error")
    request = make_request("/broken")
    exc = StarletteHTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Service unavailable",
    )

    response = error_handler.http_exception_handler(request, exc)

    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert response_json(response)["error"]["message"] == "Service unavailable"
    mock_log_error.assert_called_once()


def test_http_exception_handler_delegates_non_http_exceptions(mocker) -> None:
    """If wired incorrectly, non-HTTP exceptions should still get a safe 500."""
    mock_unhandled = mocker.patch.object(
        error_handler,
        "unhandled_exception_handler",
        return_value="fallback-response",
    )
    request = make_request("/fallback")
    exc = RuntimeError("boom")

    response = error_handler.http_exception_handler(request, exc)

    assert response == "fallback-response"
    mock_unhandled.assert_called_once_with(request, exc)


def test_validation_exception_handler_returns_details(mocker) -> None:
    """Validation errors should include FastAPI validation details."""
    mock_log_info = mocker.patch.object(error_handler.LOG, "info")
    request = make_request("/items")
    exc = RequestValidationError(
        [
            {
                "type": "missing",
                "loc": ("body", "symbol"),
                "msg": "Field required",
                "input": {},
            }
        ]
    )

    response = error_handler.validation_exception_handler(request, exc)
    payload = response_json(response)

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    assert payload["error"]["type"] == "validation_error"
    assert payload["error"]["message"] == "Request validation failed"
    assert payload["error"]["details"][0]["loc"] == ["body", "symbol"]
    mock_log_info.assert_called_once()


def test_validation_exception_handler_handles_unexpected_exception_type() -> None:
    """The validation handler should still produce a 422 payload if called directly."""
    request = make_request("/items")

    response = error_handler.validation_exception_handler(request, RuntimeError("bad"))
    payload = response_json(response)

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    assert payload["error"]["details"] == []


def test_unhandled_exception_handler_returns_safe_500(mocker) -> None:
    """Unexpected exceptions should hide internals but log the full exception."""
    mock_log_exception = mocker.patch.object(error_handler.LOG, "exception")
    request = make_request("/explode")

    response = error_handler.unhandled_exception_handler(request, RuntimeError("secret"))
    payload = response_json(response)

    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert payload["error"]["type"] == "internal_server_error"
    assert payload["error"]["message"] == "An unexpected error occurred"
    assert payload["error"]["path"] == "/explode"
    mock_log_exception.assert_called_once()


def test_register_error_handlers_registers_expected_handlers() -> None:
    """FastAPI should receive all middleware exception handlers."""
    app = FastAPI()

    error_handler.register_error_handlers(app)

    assert app.exception_handlers[StarletteHTTPException] is error_handler.http_exception_handler
    assert app.exception_handlers[RequestValidationError] is (
        error_handler.validation_exception_handler
    )
    assert app.exception_handlers[Exception] is error_handler.unhandled_exception_handler
