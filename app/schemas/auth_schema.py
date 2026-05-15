from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator


class TokenResponse(BaseModel):
    """OAuth2 bearer token response."""

    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    """Decoded JWT token data."""

    username: str | None = None


class LoginRequest(BaseModel):
    """JSON login request schema for future non-form auth endpoints."""

    username: str = Field(..., min_length=1, max_length=80)
    password: str = Field(..., min_length=1)


class CurrentUserResponse(BaseModel):
    """Authenticated user response."""

    username: str


class UserBase(BaseModel):
    """Shared user fields exposed through API schemas."""

    username: str = Field(..., min_length=1, max_length=80)
    email: EmailStr | None = None
    role: str = Field(default="trader", max_length=40)
    is_active: bool = True
    is_superuser: bool = False

    @field_validator("username")
    @classmethod
    def normalize_username(cls, value: str) -> str:
        username = value.strip()
        if not username:
            raise ValueError("username must not be empty")
        return username


class UserCreate(UserBase):
    """Schema for creating a database-backed user."""

    password: str = Field(..., min_length=8)


class UserRead(UserBase):
    """Schema for returning stored user records."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    last_login_at: datetime | None = None
    created_at: datetime
    updated_at: datetime
