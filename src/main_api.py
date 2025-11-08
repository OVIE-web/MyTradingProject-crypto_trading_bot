import os
import logging
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from datetime import datetime, timedelta
from jose import jwt
from src.auth import authenticate_user, create_access_token
from src.routers import trades
from src.config import settings
from pydantic import BaseModel
from typing import Dict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

app = FastAPI(title="Crypto Trading Bot API", version="1.0")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

SECRET_KEY = os.getenv("JWT_SECRET_KEY")
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")

def create_access_token(data: Dict[str, str], expires_delta: timedelta = timedelta(hours=1)) -> str:
    if SECRET_KEY is None:
        raise ValueError("SECRET_KEY is not set in environment variables")
    
    to_encode = data.copy()
    now = datetime.now()  # Define the 'now' variable
    expire = now + expires_delta
    to_encode["exp"] = expire.strftime("%s")
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

class Token(BaseModel):
    access_token: str
    token_type: str


# Register Routers
app.include_router(trades.router, prefix="/trades", tags=["Trades"])

@app.get("/")
def root():
    return {"message": "🚀 Trading Bot API is running!"}


@app.post("/token", response_model=Token)
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=400, detail="Invalid credentials")
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me")
def read_users_token(token: str = Depends(oauth2_scheme)):
    return {"message": f"Hello, user with token {token}"}