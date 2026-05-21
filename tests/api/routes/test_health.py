from __future__ import annotations

from fastapi import status
from fastapi.testclient import TestClient

from app.api.routes.health import HealthResponse, health_check, router
from app.main import app

client = TestClient(app)


def test_health_check_returns_expected_response_model() -> None:
    """The route function should return the typed health response."""
    response = health_check()

    assert isinstance(response, HealthResponse)
    assert response.status == "ok"
    assert response.service == "OvieX-Quant-Engine-API"


def test_health_endpoint_returns_ok_payload() -> None:
    """GET /health/ should expose the public health-check payload."""
    response = client.get("/health/")

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {
        "status": "ok",
        "service": "OvieX-Quant-Engine-API",
    }


def test_health_router_defines_root_get_route() -> None:
    """The health router should register a GET route at its local root."""
    routes = [route for route in router.routes if getattr(route, "path", None) == "/"]

    assert routes
    assert "GET" in routes[0].methods  # type: ignore
