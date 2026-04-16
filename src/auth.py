"""
Authentication and security module.

Handles JWT token creation, validation, and user authentication logic.
Follows project security standards and Python 3.12+ best practices.
"""

import logging
import secrets
from datetime import UTC, datetime, timedelta
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from pydantic import BaseModel

from src.settings import settings

# Configure logging
logger = logging.getLogger(__name__)

# Constants
ALGORITHM = "HS256"

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


class Token(BaseModel):
    """Token response schema."""

    access_token: str
    token_type: str


class TokenData(BaseModel):
    """Token payload data."""

    username: str | None = None


def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    """Create a new JWT access token.

    Args:
        data: Payload data to encode in the token.
        expires_delta: Optional expiration time delta.

    Returns:
        Encoded JWT token string.

    Raises:
        HTTPException: If token creation fails.
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(UTC) + expires_delta
    else:
        expire = datetime.now(UTC) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire})

    try:
        encoded_jwt = jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    except Exception as e:
        logger.error(f"Failed to create access token: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not generate authentication token",
        )


def verify_password(plain_password: str, password: str) -> bool:
    """Verify password using constant-time comparison."""
    return secrets.compare_digest(plain_password, password)


def authenticate_user(username: str, password: str) -> dict[str, str] | None:
    """Authenticate a user against environment credentials.

    Args:
        username: The username to check.
        password: The password to check.

    Returns:
        A user dictionary if valid, None otherwise.
    """
    if secrets.compare_digest(username, settings.ADMIN_USERNAME):
        if verify_password(password, settings.ADMIN_PASSWORD):
            return {"username": username}
    return None


async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]) -> str:
    """Validate the current user from JWT token.

    Args:
        token: The JWT token from Authorization header.

    Returns:
        The username extracted from the token.

    Raises:
        HTTPException: If token is invalid or expired.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[ALGORITHM])
        username: str | None = payload.get("sub")

        if username is None or not secrets.compare_digest(username, settings.ADMIN_USERNAME):
            raise credentials_exception
        return username
    except JWTError as e:
        logger.warning(f"JWT validation error: {e}")
        raise credentials_exception
