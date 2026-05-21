from __future__ import annotations

import app.schemas as schemas
from app.schemas.auth_schema import (
    CurrentUserResponse,
    LoginRequest,
    TokenData,
    TokenResponse,
    UserBase,
    UserCreate,
    UserRead,
)
from app.schemas.prediction_schema import (
    FeaturesInput,
    PredictionCreate,
    PredictionRead,
    PredictionResponse,
    ReloadModelResponse,
)
from app.schemas.trade_schema import (
    TradeBase,
    TradeCreate,
    TradeDeleteResponse,
    TradeRead,
    TradeUpdate,
)


def test_schema_package_exports_public_api() -> None:
    assert schemas.__all__ == [
        "CurrentUserResponse",
        "FeaturesInput",
        "LoginRequest",
        "PredictionCreate",
        "PredictionRead",
        "PredictionResponse",
        "ReloadModelResponse",
        "TokenData",
        "TokenResponse",
        "TradeBase",
        "TradeCreate",
        "TradeDeleteResponse",
        "TradeRead",
        "TradeUpdate",
        "UserBase",
        "UserCreate",
        "UserRead",
    ]


def test_schema_package_exports_match_source_modules() -> None:
    assert schemas.CurrentUserResponse is CurrentUserResponse
    assert schemas.FeaturesInput is FeaturesInput
    assert schemas.LoginRequest is LoginRequest
    assert schemas.PredictionCreate is PredictionCreate
    assert schemas.PredictionRead is PredictionRead
    assert schemas.PredictionResponse is PredictionResponse
    assert schemas.ReloadModelResponse is ReloadModelResponse
    assert schemas.TokenData is TokenData
    assert schemas.TokenResponse is TokenResponse
    assert schemas.TradeBase is TradeBase
    assert schemas.TradeCreate is TradeCreate
    assert schemas.TradeDeleteResponse is TradeDeleteResponse
    assert schemas.TradeRead is TradeRead
    assert schemas.TradeUpdate is TradeUpdate
    assert schemas.UserBase is UserBase
    assert schemas.UserCreate is UserCreate
    assert schemas.UserRead is UserRead
