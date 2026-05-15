from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from uuid import uuid4

from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.logging import get_logger

LOG = get_logger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"
PROCESS_TIME_HEADER = "X-Process-Time-ms"


class RequestLoggerMiddleware(BaseHTTPMiddleware):
    """Log request/response metadata and propagate a request ID."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER) or str(uuid4())
        start_time = time.perf_counter()

        client_host = request.client.host if request.client else "unknown"
        method = request.method
        path = request.url.path

        LOG.info(
            "request_started request_id=%s method=%s path=%s client=%s",
            request_id,
            method,
            path,
            client_host,
        )

        try:
            response = await call_next(request)
        except Exception:
            duration_ms = (time.perf_counter() - start_time) * 1000
            LOG.exception(
                "request_failed request_id=%s method=%s path=%s client=%s duration_ms=%.2f",
                request_id,
                method,
                path,
                client_host,
                duration_ms,
            )
            raise

        duration_ms = (time.perf_counter() - start_time) * 1000
        response.headers[REQUEST_ID_HEADER] = request_id
        response.headers[PROCESS_TIME_HEADER] = f"{duration_ms:.2f}"

        LOG.info(
            "request_completed request_id=%s method=%s path=%s status_code=%s duration_ms=%.2f",
            request_id,
            method,
            path,
            response.status_code,
            duration_ms,
        )

        return response


def register_request_logger(app: FastAPI) -> None:
    """Register request logging middleware on the FastAPI app."""
    app.add_middleware(RequestLoggerMiddleware)
