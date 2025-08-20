"""
auth_routes.py

This module consists ofFast API Routes and Pydantic Data Models for Authentication
"""

import base64
import json
import logging
import sys
import traceback
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer
from pydantic import BaseModel
from sqlmodel import Session, select
from webauthn import (
    generate_authentication_options,
    generate_registration_options,
    verify_authentication_response,
    verify_registration_response,
)
from webauthn.helpers.cose import COSEAlgorithmIdentifier
from webauthn.helpers.structs import AuthenticationCredential, RegistrationCredential

from api.db import get_session
from api.jwt_service import (
    blacklisted_tokens,
    create_access_token,
    get_current_user,
    set_token_expiration,
)
from api.models import Credential, User


# Add this helper function at the top of your file
def normalize_base64(data):
    """Normalize a Base64 encoded string by removing trailing equal signs and adding necessary padding.

    Args:
        data (str): The Base64 encoded string to normalize.

    Returns:
        str: The normalized Base64 encoded string with proper padding.

    Raises:
        TypeError: If the input data is not a string.
    """
    if isinstance(data, str):
        # Remove any existing padding
        data = data.rstrip("=")
        # Add proper padding
        missing_padding = len(data) % 4
        if missing_padding:
            data += "=" * (4 - missing_padding)
    return data


# Add logging configuration
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


class RegisterRequest(BaseModel):
    email: str


class LoginRequest(BaseModel):
    email: str


router = APIRouter(prefix="/auth", tags=["authentication"])
security = HTTPBearer()

# Store challenges temporarily (use Redis in production)
challenges = {}

# Configuration
# Configuration
import os
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(dotenv_path)


RP_ID = os.getenv("WEBAUTHN_RP_ID", "localhost")
RP_NAME = os.getenv("WEBAUTHN_RP_NAME", "Sustainable Development")
ORIGIN = os.getenv("WEBAUTHN_ORIGIN", "http://localhost:3000")


@router.post("/register/begin")
async def begin_registration(
    request: RegisterRequest, session: Session = Depends(get_session)
):
    """Begin the registration process for a new user.

    This asynchronous function checks if a user with the provided email already exists in the database. If the user does exist, it raises an HTTP 400 error. If not, it generates registration options, including a challenge for the user to respond to, and returns these options in a structured format.

    Args:
        request (RegisterRequest): The registration request containing user details.
        session (Session, optional): The database session dependency. Defaults to Depends(get_session).

    Returns:
        dict: A dictionary containing the registration options, including the challenge and user information.

    Raises:
        HTTPException: If the user already exists (400) or if an unexpected error occurs (500).
    """
    try:
        # Check if user already exists
        existing_user = session.exec(
            select(User).where(User.email == request.email)
        ).first()

        if existing_user:
            raise HTTPException(status_code=400, detail="User already exists")

        # Generate registration options
        options = generate_registration_options(
            rp_id=RP_ID,
            rp_name=RP_NAME,
            user_id=request.email.encode("utf-8"),
            user_name=request.email,
            user_display_name=request.email,
            supported_pub_key_algs=[
                COSEAlgorithmIdentifier.ECDSA_SHA_256,
                COSEAlgorithmIdentifier.RSASSA_PKCS1_v1_5_SHA_256,
            ],
        )

        # Store challenge
        challenge_b64 = base64.urlsafe_b64encode(options.challenge).decode("utf-8")
        challenges[challenge_b64] = {
            "email": request.email,
            "timestamp": datetime.now().isoformat(),
            "challenge_bytes": options.challenge,
        }

        # Build options dictionary
        options_dict = {
            "challenge": challenge_b64,
            "rp": {"id": options.rp.id, "name": options.rp.name},
            "user": {
                "id": base64.urlsafe_b64encode(options.user.id).decode("utf-8"),
                "name": options.user.name,
                "displayName": options.user.display_name,
            },
            "pubKeyCredParams": [
                {"type": "public-key", "alg": alg.alg}
                for alg in options.pub_key_cred_params
            ],
            "timeout": options.timeout,
            "attestation": options.attestation,
        }

        # Add authenticatorSelection if exists
        if options.authenticator_selection:
            options_dict["authenticatorSelection"] = {
                "userVerification": options.authenticator_selection.user_verification
            }

        return options_dict

    except HTTPException as http_exc:
        logger.error(f"HTTP Exception: {http_exc.detail}")
        raise http_exc
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/register/complete")
async def complete_registration(
    credential: dict, session: Session = Depends(get_session)
):
    """Completes the registration process for a user by verifying the provided credentials.

    This asynchronous function takes a credential dictionary and a database session, verifies the registration response, and stores the user and credential information in the database. It also handles challenges to ensure the integrity of the registration process.

    Args:
        credential (dict): A dictionary containing the registration credential data, including the response with clientDataJSON.
        session (Session, optional): A database session object. Defaults to a session obtained from the `get_session` dependency.

    Returns:
        dict: A dictionary containing a success message and the user ID of the newly registered user.

    Raises:
        HTTPException: If there is a challenge mismatch, the registration verification fails, or any other error occurs during the registration process.
    """
    try:
        logger.info(f"Starting registration completion")
        logger.info(f"Received credential keys: {list(credential.keys())}")
        logger.info(f"Available challenges: {list(challenges.keys())}")

        # Decode clientDataJSON to extract challenge
        def add_padding(data):
            missing_padding = len(data) % 4
            if missing_padding:
                data += "=" * (4 - missing_padding)
            return data

        client_data_json_raw = credential["response"]["clientDataJSON"]
        client_data_json_padded = normalize_base64(client_data_json_raw)
        client_data = json.loads(
            base64.urlsafe_b64decode(client_data_json_padded).decode("utf-8")
        )
        client_challenge_b64 = client_data["challenge"]

        # Find matching challenge
        challenge_data = challenges.get(client_challenge_b64)
        challenge_key_to_delete = client_challenge_b64  # Track which key to delete
        logger.info(f"Found challenge data: {challenge_data is not None}")

        if not challenge_data:
            # Try finding by normalizing both client and stored challenges
            client_challenge_normalized = normalize_base64(client_challenge_b64).rstrip(
                "="
            )

            for stored_challenge, stored_data in challenges.items():
                stored_challenge_normalized = normalize_base64(stored_challenge).rstrip(
                    "="
                )
                if stored_challenge_normalized == client_challenge_normalized:
                    challenge_data = stored_data
                    challenge_key_to_delete = (
                        stored_challenge  # Use the actual stored key
                    )
                    logger.info(f"Found challenge match after normalization")
                    break

        if not challenge_data:
            logger.error(f"Challenge mismatch - Client: {client_challenge_b64}")
            logger.error(f"Available challenges: {list(challenges.keys())}")
            logger.error(
                f"Challenge lengths - Client: {len(client_challenge_b64)}, Available: {[len(k) for k in challenges.keys()]}"
            )
            raise HTTPException(status_code=400, detail="Challenge mismatch or expired")

        logger.info(f"Challenge match found for email: {challenge_data['email']}")

        # Perform verification - this will raise an exception if verification fails
        verification = verify_registration_response(
            credential=credential,
            expected_challenge=challenge_data["challenge_bytes"],
            expected_origin=ORIGIN,
            expected_rp_id=RP_ID,
        )

        logger.info("Registration verification successful")

        # Save user and credential
        user = User(email=challenge_data["email"])
        session.add(user)
        session.commit()
        session.refresh(user)

        credential_record = Credential(
            user_id=user.id,
            credential_id=credential["id"],
            public_key=base64.urlsafe_b64encode(
                verification.credential_public_key
            ).decode("utf-8"),
            sign_count=verification.sign_count,
            created_at=datetime.now().isoformat(),
        )
        session.add(credential_record)
        session.commit()

        # Clean up challenge using the correct key
        del challenges[challenge_key_to_delete]
        logger.info("Registration completed successfully")

        return {"message": "Registration successful", "user_id": user.id}

    except HTTPException as http_exc:
        logger.error(f"HTTP Exception: {http_exc.detail}")
        raise http_exc
    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Registration failed")


@router.post("/login/begin")
async def begin_authentication(
    request: LoginRequest, session: Session = Depends(get_session)
):
    """Begin the authentication process for a user.

    This asynchronous function handles the initial steps of user authentication by verifying the user's email, retrieving their credentials, and generating authentication options.

    Args:
        request (LoginRequest): The login request containing the user's email.
        session (Session, optional): The database session dependency. Defaults to Depends(get_session).

    Returns:
        dict: A dictionary containing the authentication options, including the challenge, timeout, rpId, user verification method, and allowed credentials.

    Raises:
        HTTPException: If the user is not found, no credentials are associated with the user, or if an unexpected error occurs during the authentication process.
    """
    try:
        # First, check if user exists and get their credentials
        user = session.exec(select(User).where(User.email == request.email)).first()

        if not user:
            raise HTTPException(status_code=400, detail="User not found")

        # Get user's credentials
        user_credentials = session.exec(
            select(Credential).where(Credential.user_id == user.id)
        ).all()

        if not user_credentials:
            raise HTTPException(status_code=400, detail="No credentials found for user")

        # Helper function to add padding
        def add_padding(data):
            missing_padding = len(data) % 4
            if missing_padding:
                data += "=" * (4 - missing_padding)
            return data

        # Build allowCredentials list for the client
        allow_credentials = []
        for cred in user_credentials:
            allow_credentials.append(
                {
                    "type": "public-key",
                    "id": cred.credential_id,  # Use the stored credential ID as-is
                    "transports": [
                        "usb",
                        "nfc",
                        "ble",
                        "internal",
                    ],  # Add common transports
                }
            )

        # Build allowCredentials list for the webauthn library (needs bytes)
        webauthn_allow_credentials = []
        for cred in user_credentials:
            try:
                # Add padding if needed and decode to bytes
                padded_id = add_padding(cred.credential_id)
                credential_bytes = base64.urlsafe_b64decode(padded_id)
                webauthn_allow_credentials.append(
                    {"type": "public-key", "id": credential_bytes}
                )
            except Exception as e:
                logger.error(f"Error processing credential {cred.credential_id}: {e}")
                continue

        options = generate_authentication_options(
            rp_id=RP_ID, allow_credentials=webauthn_allow_credentials
        )

        challenge_b64 = base64.urlsafe_b64encode(options.challenge).decode("utf-8")
        challenges[challenge_b64] = {
            "email": request.email,
            "user_id": user.id,  # Store user ID for easier lookup
            "timestamp": datetime.now().isoformat(),
            "challenge_bytes": options.challenge,
        }

        options_dict = {
            "challenge": challenge_b64,
            "timeout": options.timeout,
            "rpId": options.rp_id,
            "userVerification": options.user_verification,
            "allowCredentials": allow_credentials,
        }

        return options_dict

    except HTTPException as http_exc:
        logger.error(f"HTTP Exception: {http_exc.detail}")
        raise http_exc
    except Exception as e:
        logger.error(f"Authentication initiation error: {str(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Authentication initiation failed")


# Update your complete_authentication function:
@router.post("/login/complete")
async def complete_authentication(
    credential: dict, session: Session = Depends(get_session)
):
    """Completes the authentication process using the provided credentials.

    This asynchronous function verifies the provided authentication credentials against stored data,
    validates the client challenge, and updates the user's sign count in the database. If successful,
    it generates and returns an access token for the authenticated user.

    Args:
        credential (dict): A dictionary containing the authentication credentials, including
            the response with clientDataJSON, authenticatorData, signature, and userHandle.
        session (Session, optional): A database session dependency. Defaults to Depends(get_session).

    Returns:
        dict: A dictionary containing the authentication success message, access token, token type,
            expiration time, and user information (ID and email).

    Raises:
        HTTPException: If any errors occur during the authentication process, including:
            - Challenge mismatch or expired
            - Invalid credential ID format
            - Credential not found
            - User ID mismatch
            - Authentication signature verification failure
            - General authentication failure
    """
    try:
        logger.info(f"Starting authentication completion")
        logger.info(f"Received credential keys: {list(credential.keys())}")

        # Get challenge from clientDataJSON - DO NOT MODIFY THE ORIGINAL
        client_data_json_raw = credential["response"]["clientDataJSON"]
        client_data_json_padded = normalize_base64(client_data_json_raw)
        client_data = json.loads(
            base64.urlsafe_b64decode(client_data_json_padded).decode("utf-8")
        )
        client_challenge_b64 = client_data["challenge"]

        logger.info(f"Client challenge: {client_challenge_b64}")
        logger.info(f"Client challenge length: {len(client_challenge_b64)}")

        # Find matching challenge - try both with and without padding
        challenge_data = challenges.get(client_challenge_b64)
        challenge_key_to_delete = client_challenge_b64  # Track which key to delete

        if not challenge_data:
            # Try with normalized padding
            client_challenge_normalized = normalize_base64(client_challenge_b64)
            challenge_data = challenges.get(client_challenge_normalized)
            if challenge_data:
                challenge_key_to_delete = client_challenge_normalized

            if not challenge_data:
                # Try to find by comparing normalized versions
                for stored_challenge, stored_data in challenges.items():
                    if normalize_base64(stored_challenge).rstrip(
                        "="
                    ) == client_challenge_b64.rstrip("="):
                        challenge_data = stored_data
                        challenge_key_to_delete = (
                            stored_challenge  # Use the actual stored key
                        )
                        logger.info(f"Found challenge match with normalization")
                        break

        if not challenge_data:
            logger.error(f"Challenge not found")
            logger.error(f"Client challenge: {client_challenge_b64}")
            logger.error(f"Available challenges: {list(challenges.keys())}")
            raise HTTPException(status_code=400, detail="Challenge mismatch or expired")

        logger.info(f"Challenge match found for email: {challenge_data['email']}")

        # Find the credential - IMPORTANT: Store credential IDs as raw bytes, not base64
        credential_id = credential["id"]
        logger.info(f"Looking for credential ID: {credential_id}")

        # Convert the incoming credential ID to bytes for comparison
        try:
            credential_id_bytes = base64.urlsafe_b64decode(
                normalize_base64(credential_id)
            )
        except Exception as e:
            logger.error(f"Error decoding credential ID: {e}")
            raise HTTPException(status_code=400, detail="Invalid credential ID format")

        # Find credential by matching the actual credential ID
        stored_credential = None
        all_credentials = session.exec(
            select(Credential).where(Credential.user_id == challenge_data["user_id"])
        ).all()

        for cred in all_credentials:
            try:
                # Try to decode the stored credential ID
                stored_id_bytes = base64.urlsafe_b64decode(
                    normalize_base64(cred.credential_id)
                )
                if stored_id_bytes == credential_id_bytes:
                    stored_credential = cred
                    logger.info(f"Found matching credential")
                    break
            except Exception as e:
                logger.warning(
                    f"Error decoding stored credential ID {cred.credential_id}: {e}"
                )
                continue

        if not stored_credential:
            logger.error(f"Credential not found for ID: {credential_id}")
            raise HTTPException(status_code=400, detail="Credential not found")

        logger.info(f"Found stored credential for user_id: {stored_credential.user_id}")

        # Verify the user matches
        if stored_credential.user_id != challenge_data["user_id"]:
            logger.error(f"User ID mismatch")
            raise HTTPException(
                status_code=400, detail="Credential and challenge user mismatch"
            )

        # CRITICAL: Use the original clientDataJSON from the credential
        # DO NOT use any normalized or modified version
        formatted_credential = {
            "id": credential_id,  # Keep as base64 string
            "rawId": credential_id,  # Sometimes needed
            "response": {
                "clientDataJSON": credential["response"][
                    "clientDataJSON"
                ],  # Use ORIGINAL
                "authenticatorData": credential["response"]["authenticatorData"],
                "signature": credential["response"]["signature"],
                "userHandle": credential["response"].get("userHandle"),  # May be None
            },
            "type": "public-key",  # Ensure this is set
        }

        logger.info(
            f"Formatted credential structure: {list(formatted_credential.keys())}"
        )
        logger.info(f"Response keys: {list(formatted_credential['response'].keys())}")

        # Perform verification with the formatted credential
        try:
            verification = verify_authentication_response(
                credential=formatted_credential,  # Use formatted credential
                expected_challenge=challenge_data["challenge_bytes"],
                expected_origin=ORIGIN,
                expected_rp_id=RP_ID,
                credential_public_key=base64.urlsafe_b64decode(
                    normalize_base64(stored_credential.public_key)
                ),
                credential_current_sign_count=stored_credential.sign_count,
            )

            logger.info(f"Authentication verification successful: {verification}")
            logger.info(f"New sign count: {verification.new_sign_count}")

        except Exception as verification_error:
            logger.error(f"Verification failed with error: {str(verification_error)}")
            logger.error(f"Error type: {type(verification_error).__name__}")

            # Additional debugging
            logger.error(
                f"Challenge bytes (hex): {challenge_data['challenge_bytes'].hex()}"
            )
            logger.error(f"Expected origin: {ORIGIN}")
            logger.error(f"Expected RP ID: {RP_ID}")
            logger.error(f"Client data origin: {client_data.get('origin')}")
            logger.error(f"Client data type: {client_data.get('type')}")

            # Check the signature format
            try:
                sig_bytes = base64.urlsafe_b64decode(
                    normalize_base64(credential["response"]["signature"])
                )
                logger.error(f"Signature length: {len(sig_bytes)}")
                logger.error(f"Signature first 10 bytes: {sig_bytes[:10].hex()}")
            except Exception as sig_error:
                logger.error(f"Signature decode error: {sig_error}")

            raise HTTPException(
                status_code=400,
                detail=f"Authentication signature verification failed: {str(verification_error)}",
            )

        # Update sign count
        stored_credential.sign_count = verification.new_sign_count
        session.commit()
        logger.info("Updated sign count in database")

        # Clean up challenge
        del challenges[challenge_key_to_delete]

        user = session.get(User, stored_credential.user_id)
        logger.info(f"Authentication completed successfully for user: {user.email}")

        # Create access token
        access_token_expires = set_token_expiration()
        access_token = create_access_token(
            data={"sub": user.email, "user_id": user.id},
            expires_delta=access_token_expires,
        )

        return {
            "message": "Authentication successful",
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": access_token_expires.total_seconds(),
            "user": {"id": user.id, "email": user.email},
        }

    except HTTPException as http_exc:
        logger.error(f"HTTP Exception: {http_exc.detail}")
        raise http_exc
    except Exception as e:
        logger.error(f"Authentication error: {str(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Authentication failed")


@router.post("/logout")
async def logout(token: str = Depends(security)):
    """Logs out a user by blacklisting their token.

    This asynchronous function takes a token as a dependency and adds it to a blacklist. If the operation is successful, it returns a success message. In case of an error, it logs the error and raises an HTTPException with a 500 status code.

    Args:
        token (str, optional): The token to be blacklisted. Defaults to the value provided by the security dependency.

    Returns:
        dict: A message indicating the logout was successful.

    Raises:
        HTTPException: If an error occurs during the logout process.
    """
    try:
        # Add token to blacklist
        blacklisted_tokens.add(token.credentials)
        logger.info(f"Token {token.credentials} has been blacklisted")
        return {"message": "Logged out successfully"}
    except Exception as e:
        logger.error(f"Logout error: {str(e)}")
        raise HTTPException(status_code=500, detail="Logout failed")


@router.get("/me")
async def get_me(token: str = Depends(security)):
    """Retrieve the current user's information.

    This asynchronous function fetches the current user's email and ID based on the provided token. If the token is valid and the user is found, their information is returned. In case of an error during the retrieval process, an HTTPException is raised with a 500 status code.

    Args:
        token (str, optional): The authentication token used to identify the user.
            Defaults to the value provided by the Depends function from the security module.

    Returns:
        dict: A dictionary containing the user's email and ID.

    Raises:
        HTTPException: If there is an error retrieving user information.
    """
    try:
        user = get_current_user(token=token)
        return {"email": user.email, "id": user.id}
    except Exception as e:
        logger.error(f"Get me error: {str(e)}")
        raise HTTPException(
            status_code=500, detail="Could not retrieve user information"
        )


@router.get("/refresh-token")
async def refresh_token(token: str = Depends(security)):
    """Refreshes the access token for the current user.

    This asynchronous function takes a token as a dependency, validates it, and generates a new access token if the user is authenticated. In case of an error during the process, it logs the error and raises an HTTPException.

    Args:
        token (str): The current user's token, provided as a dependency.

    Returns:
        dict: A dictionary containing the new access token and its type.

    Raises:
        HTTPException: If there is an error while refreshing the token.
    """
    try:
        user = get_current_user(token=token)
        new_access_token = create_access_token(
            data={"sub": user.email, "user_id": user.id},
            expires_delta=set_token_expiration(),
        )
        return {"access_token": new_access_token, "token_type": "bearer"}
    except Exception as e:
        logger.error(f"Refresh token error: {str(e)}")
        raise HTTPException(status_code=500, detail="Could not refresh token")
