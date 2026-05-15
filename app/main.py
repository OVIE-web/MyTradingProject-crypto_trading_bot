from __future__ import annotations  # OPTIONAL but nice now

import sys
from datetime import timedelta
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

from app.api.routes import health, predictions, trades, webhook
from app.core.env_loader import load_environment
from app.core.logging import get_logger, setup_logging
from app.core.security import authenticate_user, create_access_token, get_current_user
from app.core.settings import settings
from app.middleware.error_handler import register_error_handlers
from app.middleware.request_logger import register_request_logger
from app.schemas.auth_schema import CurrentUserResponse, TokenResponse

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

load_environment()
setup_logging()

logger = get_logger(__name__)

app = FastAPI(
    title="OvieX Quant Engine API",
    description="OvieX Quant Trading Engine Bot API",
    version="0.0.1-beta",
)
register_request_logger(app)
register_error_handlers(app)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


app.include_router(predictions.router, prefix="/predictions", tags=["Predictions"])
app.include_router(trades.router, prefix="/trades", tags=["Trades"])
app.include_router(webhook.router, prefix="/webhook", tags=["Webhook"])
app.include_router(health.router, prefix="/health", tags=["Health"])


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "🚀 OvieX Quant Engine API is running!",
        "version": "0.0.1-beta",
        "status": "healthy",
    }


@app.post("/token", response_model=TokenResponse)
def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
) -> TokenResponse:
    user = authenticate_user(form_data.username, form_data.password)
    if not (isinstance(user, dict) and "username" in user):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)

    access_token = create_access_token(
        data={"sub": user["username"]},
        expires_delta=access_token_expires,
    )

    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
    )


@app.get("/users/me", response_model=CurrentUserResponse)
def read_users_me(
    current_user: str = Depends(get_current_user),
) -> CurrentUserResponse:
    return CurrentUserResponse(username=current_user)
