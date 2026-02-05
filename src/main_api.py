from __future__ import annotations  # OPTIONAL but nice now

import logging
from datetime import timedelta
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel

from src.auth import authenticate_user, create_access_token, get_current_user
from src.routers import trades
from src.settings import settings

logger = logging.getLogger(__name__)

app = FastAPI(title="Crypto Trading Bot API", version="1.0")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


class Token(BaseModel):
    access_token: str
    token_type: str


app.include_router(trades.router, prefix="/trades", tags=["Trades"])


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "🚀 Trading Bot API is running!"}


@app.post("/token", response_model=Token)
def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
) -> Token:
    user = authenticate_user(form_data.username, form_data.password)
    if not (isinstance(user, dict) and "username" in user):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)

    access_token = create_access_token(
        subject=user["username"],
        expires_delta=access_token_expires,
    )

    return Token(
        access_token=access_token,
        token_type="bearer",
    )


@app.get("/users/me")
def read_users_me(
    current_user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    return current_user
