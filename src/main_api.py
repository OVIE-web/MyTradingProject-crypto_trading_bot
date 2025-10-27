# src/main_api.py
from fastapi import FastAPI
from src.routers import trades

app = FastAPI(title="Crypto Trading Bot API", version="1.0")

# Register Routers
app.include_router(trades.router, prefix="/trades", tags=["Trades"])

@app.get("/")
def root():
    return {"message": "🚀 Trading Bot API is running!"}
