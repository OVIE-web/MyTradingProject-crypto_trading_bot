# src/auth.py
from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext

from src.settings import settings

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Security constants
# ------------------------------------------------------------------
ISSUER = "crypto-trading-bot"
AUDIENCE = "crypto-trading-api"

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ------------------------------------------------------------------
# Fake admin user (intentional for now)
# ------------------------------------------------------------------
_fake_users_db: dict[str, dict[str, str]] = {
    settings.ADMIN_USERNAME: {
        "username": settings.ADMIN_USERNAME,
        "hashed_password": pwd_context.hash(settings.ADMIN_PASSWORD),
    }
}

# ------------------------------------------------------------------
# Password helpers
# ------------------------------------------------------------------
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def authenticate_user(username: str, password: str) -> dict[str, str] | None:
    user = _fake_users_db.get(username)
    if not user:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    return user


# ------------------------------------------------------------------
# JWT creation
# ------------------------------------------------------------------
def create_access_token(
    *,
    subject: str,
    expires_delta: timedelta | None = None,
) -> str:
    now = datetime.now(tz=UTC)
    expire = now + (
        expires_delta
        if expires_delta is not None
        else timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )

    payload: dict[str, Any] = {
        "sub": subject,
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
        "iss": ISSUER,
        "aud": AUDIENCE,
    }

    return jwt.encode(
        payload,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )


# ------------------------------------------------------------------
# JWT decoding (USED BY /users/me)
# ------------------------------------------------------------------
def get_current_user(token: str = Depends(oauth2_scheme)) -> dict[str, str]:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired authentication token",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
            audience=AUDIENCE,
            issuer=ISSUER,
            options={"require": ["exp", "sub", "iss", "aud"]},
        )
    except JWTError as exc:
        logger.warning("JWT decode failed: %s", exc)
        raise credentials_exception

    username = payload.get("sub")
    if not isinstance(username, str):
        raise credentials_exception

    user = _fake_users_db.get(username)
    if user is None:
        raise credentials_exception

    return {"username": username}


# -------------------- End of auth.py --------------------
