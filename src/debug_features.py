"""
Debug script to identify feature mismatch between model and prediction data.
Run this to see what's happening.
"""

import os

import pandas as pd
import xgboost as xgb

from src.binance_manager import BinanceManager
from src.feature_engineer import calculate_technical_indicators

# Load the model
MODEL_SAVE_PATH = os.getenv("MODEL_SAVE_PATH", "src/models/xgboost_model.json")
model = xgb.XGBClassifier()
model.load_model(MODEL_SAVE_PATH)

# Get expected features from model
expected_features = model.get_booster().feature_names
num_features = model.n_features_in_

print(f"✅ Model expects {num_features} features")
print(f"✅ Expected feature names: {expected_features}")
print()

# Get actual features from your data pipeline
binance = BinanceManager()
klines = binance.get_latest_ohlcv("BTCUSDT", interval="1h", limit=50)
df = pd.DataFrame(
    klines,
    columns=[
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "num_trades",
        "taker_buy_base",
        "taker_buy_quote",
        "ignore",
    ],
)

# Apply feature engineering
features = calculate_technical_indicators(df)

print(f"❌ Current features shape: {features.shape}")
print(f"❌ Current feature names: {list(features.columns)}")
print()

# Check numeric features
numeric_features = features.select_dtypes(include=["int64", "int32", "float64", "float32", "bool"])
print(f"📊 After filtering to numeric: {numeric_features.shape}")
print(f"📊 Numeric feature names: {list(numeric_features.columns)}")
print()

# Show what should be selected
if expected_features:
    missing = [f for f in expected_features if f not in numeric_features.columns]
    extra = [f for f in numeric_features.columns if f not in expected_features]
    if missing:
        print(f"⚠️  Missing from data: {missing}")
    if extra:
        print(f"⚠️  Extra in data: {extra}")
