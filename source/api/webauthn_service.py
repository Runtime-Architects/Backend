from webauthn import generate_registration_options, verify_registration_response
from webauthn import generate_authentication_options, verify_authentication_response
from webauthn.helpers.structs import (
    AuthenticatorSelectionCriteria,
    UserVerificationRequirement,
    RegistrationCredential,
    AuthenticationCredential,
)
from webauthn.helpers.cose import COSEAlgorithmIdentifier
import base64
import json


class WebAuthnService:
    def __init__(self, rp_id: str = "localhost", rp_name: str = "Your App"):
        self.rp_id = rp_id
        self.rp_name = rp_name
        self.expected_origin = (
            f"https://{rp_id}" if rp_id != "localhost" else "http://localhost:8000"
        )

    def generate_registration_options(
        self, user_id: str, username: str, display_name: str
    ):
        """Generate options for passkey registration"""
        options = generate_registration_options(
            rp_id=self.rp_id,
            rp_name=self.rp_name,
            user_id=user_id.encode("utf-8"),
            user_name=username,
            user_display_name=display_name,
            supported_pub_key_algs=[
                COSEAlgorithmIdentifier.ECDSA_SHA_256,
                COSEAlgorithmIdentifier.RSASSA_PKCS1_v1_5_SHA_256,
            ],
            authenticator_selection=AuthenticatorSelectionCriteria(
                user_verification=UserVerificationRequirement.PREFERRED,
            ),
        )

        # Convert to dict manually
        options_dict = {
            "challenge": base64.urlsafe_b64encode(options.challenge).decode("utf-8"),
            "rp": {
                "id": options.rp.id,
                "name": options.rp.name,
            },
            "user": {
                "id": base64.urlsafe_b64encode(options.user.id).decode("utf-8"),
                "name": options.user.name,
                "displayName": options.user.display_name,
            },
            "pubKeyCredParams": [
                {"type": param.type, "alg": param.alg}
                for param in options.pub_key_cred_params
            ],
            "timeout": options.timeout,
            "attestation": options.attestation,
            "authenticatorSelection": (
                {
                    "userVerification": options.authenticator_selection.user_verification,
                }
                if options.authenticator_selection
                else None
            ),
        }

        return options_dict

    def verify_registration(
        self, credential: RegistrationCredential, expected_challenge: str
    ):
        """Verify passkey registration"""
        return verify_registration_response(
            credential=credential,
            expected_challenge=base64.urlsafe_b64decode(expected_challenge),
            expected_origin=self.expected_origin,
            expected_rp_id=self.rp_id,
        )

    def generate_authentication_options(self, user_credentials: list = None):
        """Generate options for passkey authentication"""
        options = generate_authentication_options(
            rp_id=self.rp_id,
            allow_credentials=user_credentials,
            user_verification=UserVerificationRequirement.PREFERRED,
        )

        # Convert to dict manually
        options_dict = {
            "challenge": base64.urlsafe_b64encode(options.challenge).decode("utf-8"),
            "timeout": options.timeout,
            "rpId": options.rp_id,
            "userVerification": options.user_verification,
            "allowCredentials": (
                [
                    {
                        "type": cred.type,
                        "id": base64.urlsafe_b64encode(cred.id).decode("utf-8"),
                        "transports": cred.transports,
                    }
                    for cred in options.allow_credentials
                ]
                if options.allow_credentials
                else []
            ),
        }

        return options_dict

    def verify_authentication(
        self,
        credential: AuthenticationCredential,
        expected_challenge: str,
        credential_public_key: bytes,
        credential_sign_count: int,
    ):
        """Verify passkey authentication"""
        return verify_authentication_response(
            credential=credential,
            expected_challenge=base64.urlsafe_b64decode(expected_challenge),
            expected_origin=self.expected_origin,
            expected_rp_id=self.rp_id,
            credential_public_key=credential_public_key,
            credential_current_sign_count=credential_sign_count,
        )
