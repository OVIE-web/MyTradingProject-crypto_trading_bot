from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.logging import get_logger

type ExceptionHandler = Callable[[Request, Exception], JSONResponse]

LOG = get_logger(__name__)


def _request_path(request: Request) -> str:
    return str(request.url.path)


def _error_payload(
    *,
    status_code: int,
    error: str,
    message: str,
    path: str,
    details: Any | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "success": False,
        "error": {
            "code": status_code,
            "type": error,
            "message": message,
            "path": path,
            "timestamp": datetime.now(UTC).isoformat(),
        },
    }

    if details is not None:
        payload["error"]["details"] = details

    return payload


def http_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Return consistent JSON for expected HTTP errors."""
    if not isinstance(exc, StarletteHTTPException):
        return unhandled_exception_handler(request, exc)

    message = str(exc.detail) if exc.detail else "HTTP error"

    if exc.status_code >= status.HTTP_500_INTERNAL_SERVER_ERROR:
        LOG.error(
            "HTTP exception. status=%s path=%s detail=%s",
            exc.status_code,
            _request_path(request),
            message,
        )
    else:
        LOG.info(
            "HTTP exception. status=%s path=%s detail=%s",
            exc.status_code,
            _request_path(request),
            message,
        )

    return JSONResponse(
        status_code=exc.status_code,
        content=_error_payload(
            status_code=exc.status_code,
            error="http_error",
            message=message,
            path=_request_path(request),
        ),
        headers=getattr(exc, "headers", None),
    )


def validation_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Return consistent JSON for request validation errors."""
    details = exc.errors() if isinstance(exc, RequestValidationError) else []
    LOG.info("Validation error. path=%s errors=%s", _request_path(request), details)

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=_error_payload(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            error="validation_error",
            message="Request validation failed",
            path=_request_path(request),
            details=details,
        ),
    )


def unhandled_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Return safe JSON for unexpected exceptions and log full traceback."""
    LOG.exception("Unhandled exception. path=%s error=%s", _request_path(request), exc)

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=_error_payload(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error="internal_server_error",
            message="An unexpected error occurred",
            path=_request_path(request),
        ),
    )


def register_error_handlers(app: FastAPI) -> None:
    """Register API exception handlers on the FastAPI app."""
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
