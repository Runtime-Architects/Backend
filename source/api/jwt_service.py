"""
jwt.py

This module provides utility functions for creating, verifying, and managing
JWT (JSON Web Tokens) for user authentication and authorization.
"""

import logging
import os
import secrets
import sys
from datetime import datetime, timedelta

from dotenv import load_dotenv
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlalchemy.orm import Session

from api.db import get_session
from api.models import User


load_dotenv()

# Logging Config
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

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
    """Sets the expiration time for the access token.

    Returns:
        timedelta: The duration for which the access token is valid,
        set to a specified number of minutes defined by
        ACCESS_TOKEN_EXPIRE_MINUTES.
    """
    return timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)


def create_access_token(data: dict, expires_delta: timedelta = None):
    """Creates an access token by encoding the provided data with an expiration time.

    Args:
        data (dict): The data to encode in the access token.
        expires_delta (timedelta, optional): The time duration for which the token is valid.
            If not provided, a default expiration time will be used.

    Returns:
        str: The encoded JWT access token.

    Raises:
        Exception: If there is an error during the encoding process.
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(token: str):
    """Verify the provided JWT token.

    This function checks if the token is blacklisted and decodes it to extract the user's email and user ID. If the token is invalid or has been revoked, an HTTPException is raised.

    Args:
        token (str): The JWT token to verify.

    Returns:
        dict: A dictionary containing the user's email and user ID if the token is valid.

    Raises:
        HTTPException: If the token is blacklisted, invalid, or cannot be decoded.
    """
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
    """Retrieve the current user based on the provided authentication token.

    This asynchronous function extracts the user information from the given token. It verifies the token and retrieves the corresponding user from the database session. If the token is invalid or the user cannot be found, an HTTPException is raised.

    Args:
        token (str, optional): The authentication token used to identify the user. Defaults to the value provided by the `oauth2_scheme` dependency.
        session (Session, optional): The database session used to query user information. Defaults to the value provided by the `get_session` dependency.

    Raises:
        HTTPException: If the token is invalid or the user cannot be found.

    Returns:
        User: The user object corresponding to the authenticated token.
    """
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
