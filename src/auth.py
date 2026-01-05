# src/auth.py
import logging
from datetime import datetime, timedelta
from typing import Dict

import pytz
from dotenv import load_dotenv
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext

from src.config import settings

# Load environment variables
load_dotenv()

# Configure logging once (you can also do this globally in main_api.py)
logging.basicConfig(
    level=logging.INFO,  # or DEBUG for more detail
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


# --- JWT and Password Configuration ---
SECRET_KEY = settings.JWT_SECRET_KEY
ALGORITHM = settings.JWT_ALGORITHM
ACCESS_TOKEN_EXPIRE_MINUTES = settings.ACCESS_TOKEN_EXPIRE_MINUTES

ADMIN_USERNAME = settings.ADMIN_USERNAME
ADMIN_PASSWORD = settings.ADMIN_PASSWORD


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


fake_users_db = {
    ADMIN_USERNAME: {
        "username": ADMIN_USERNAME,
        "hashed_password": pwd_context.hash(ADMIN_PASSWORD),
    }
}


# --- Helper Functions ---
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def authenticate_user(username: str, password: str):
    user = fake_users_db.get(username)
    if not user or not verify_password(password, user["hashed_password"]):
        return False
    return user


def create_access_token(
    data: Dict[str, str], expires_delta: timedelta = timedelta(minutes=15)
) -> str:
    """
    Create a JWT access token with the given data and expiration time.

    Args:
        data (Dict[str, str]): A dictionary containing the data to be encoded in the JWT.
        expires_delta (timedelta, optional): The expiration time of the JWT. Defaults to timedelta(minutes=15).

    Returns:
        str: The encoded JWT access token.

    Raises:
        ValueError: If the expires_delta value is negative.
    """
    if expires_delta < timedelta(0):  # handle negative values
        raise ValueError("Invalid expires_delta value")
    to_encode = data.copy()
    now = datetime.now(pytz.utc)
    expire = now + expires_delta
    to_encode["exp"] = expire.strftime("%s")
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(token: str = Depends(oauth2_scheme)):
    """Decode and verify JWT token"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str | None = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = fake_users_db.get(username)
    if user is None:
        raise credentials_exception
    return user
