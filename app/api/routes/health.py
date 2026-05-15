from __future__ import annotations

from fastapi import APIRouter, status
from pydantic import BaseModel

router = APIRouter()


class HealthResponse(BaseModel):
    """Response returned by the health check endpoint."""

    status: str
    service: str


@router.get(
    "/",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
)
def health_check() -> HealthResponse:
    """Return a lightweight API health status."""
    return HealthResponse(status="ok", service="OvieX-Quant-Engine-API")
