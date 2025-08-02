import secrets
from datetime import datetime, timedelta
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer  # Fixed import
from jose import jwt, JWTError
from sqlalchemy.orm import Session
from api.db import get_session
from api.models import User
import logging
import os
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# JWT Configuration
SECRET_KEY = os.environ["JWT_SECRET_KEY"]
ALGORITHM = os.environ["JWT_ALGORITHM"]
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.environ["JWT_EXPIRATION_HOURS"])

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")  # Create the OAuth2 scheme

# Store for token blacklist (use Redis in production)
blacklisted_tokens = set()


def set_token_expiration():
    """Set the expiration time for access tokens."""
    return timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)


def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(token: str):
    try:
        if token in blacklisted_tokens:
            raise HTTPException(status_code=401, detail="Token has been revoked")

        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        user_id: int = payload.get("user_id")
        if email is None or user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return {"email": email, "user_id": user_id}
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


async def get_current_user(
    token: str = Depends(oauth2_scheme), session: Session = Depends(get_session)
):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        logger.info(f"Getting current user from token: {token[:20]}...")

        # The token is already extracted by OAuth2PasswordBearer
        token_data = verify_token(token)

        logger.info(f"Token verified, looking up user_id: {token_data['user_id']}")

        user = session.get(User, token_data["user_id"])
        if user is None:
            logger.warning(f"User not found for user_id: {token_data['user_id']}")
            raise credentials_exception

        logger.info(f"User found: {user.email}")
        return user
    except HTTPException:
        logger.error("HTTPException in get_current_user")
        raise credentials_exception
    except Exception as e:
        logger.error(f"Unexpected error in get_current_user: {str(e)}")
        raise credentials_exception
