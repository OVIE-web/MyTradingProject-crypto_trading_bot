"""Tests for FastAPI main application."""

from fastapi.testclient import TestClient

from src.main_api import app

# Initialize test client
client = TestClient(app)


class TestHealthEndpoint:
    """Test GET / health check endpoint."""

    def test_health_check_success(self) -> None:
        """Test GET / returns message."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "Trading Bot API" in data["message"]

    def test_health_check_has_content(self) -> None:
        """Test health endpoint returns non-empty message."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)
        assert len(data.get("message", "")) > 0


class TestTokenEndpoint:
    """Test POST /token endpoint."""

    def test_token_with_invalid_credentials(self) -> None:
        """Test token endpoint with invalid credentials."""
        response = client.post(
            "/token",
            data={"username": "invalid", "password": "invalid"},
        )
        # Should fail with 401
        assert response.status_code == 401

    def test_token_endpoint_exists(self) -> None:
        """Test token endpoint is accessible."""
        response = client.post(
            "/token",
            data={"username": "test", "password": "test"},
        )
        # May succeed or fail, but should not crash
        assert response.status_code in [200, 401, 422]


class TestUserMeEndpoint:
    """Test GET /users/me endpoint."""

    def test_users_me_requires_auth(self) -> None:
        """Test /users/me requires authentication."""
        response = client.get("/users/me")
        # Should fail without auth token
        assert response.status_code in [401, 403]

    def test_users_me_with_invalid_token(self) -> None:
        """Test /users/me with invalid token."""
        response = client.get(
            "/users/me",
            headers={"Authorization": "Bearer invalid_token"},
        )
        # Should fail with invalid token
        assert response.status_code in [401, 403]


class TestErrorHandling:
    """Test error handling."""

    def test_invalid_endpoint(self) -> None:
        """Test accessing non-existent endpoint."""
        response = client.get("/invalid-endpoint")
        assert response.status_code == 404

    def test_invalid_method(self) -> None:
        """Test invalid HTTP method."""
        response = client.post("/")
        # GET endpoint accessed with POST should fail
        assert response.status_code in [405, 422]


class TestApiStructure:
    """Test API structure and basic functionality."""

    def test_root_returns_json(self) -> None:
        """Test root endpoint returns valid JSON."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)

    def test_api_has_trades_router(self) -> None:
        """Test API includes trades router."""
        # Trades endpoints should be available
        # Will fail with auth, but should exist
        response = client.get("/trades/")
        assert response.status_code in [200, 401, 403]
