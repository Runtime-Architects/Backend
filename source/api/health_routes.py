"""
health_routes.py

This module contains FastAPI Routes to check the health of the Agents and Azure CLient
"""

import logging
import os
import sys
from datetime import datetime

from fastapi import APIRouter

# Logging Config
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check():
    """Perform a health check on the system components and return their statuses.

    This asynchronous function checks the initialization status of agents, the connection status of the OpenAI client, the health of the OpenAI API, the existence of the data directory, and the count of JSON files within it. It also verifies if the Azure AI API key is configured. Based on these checks, it determines the overall health status of the system and returns a structured response.

    Returns:
        dict: A dictionary containing the overall health status, a message describing the status, a timestamp of the check, details about the components' statuses, any API error messages, the version of the service, and the uptime check timestamp.

    Raises:
        Exception: Logs an error and returns an unhealthy status if any unexpected error occurs during the health check process.
    """
    try:
        from agents.agent_workflow import initialize_agents, client

        team_flow = initialize_agents()

        # Check if agents are initialized
        agents_status = "initialized" if team_flow is not None else "not_initialized"

        # Get current timestamp
        current_time = datetime.now().isoformat()

        # Check OpenAI client status
        openai_client_status = "connected" if client is not None else "not_connected"

        # Test actual OpenAI API connectivity
        openai_api_status = "unknown"
        api_error_message = None

        if client is not None:
            try:
                openai_api_status = "healthy"
            except Exception as api_error:
                error_str = str(api_error)
                if "429" in error_str or "quota" in error_str.lower():
                    openai_api_status = "quota_exceeded"
                elif "401" in error_str or "invalid" in error_str.lower():
                    openai_api_status = "invalid_key"
                elif "403" in error_str:
                    openai_api_status = "forbidden"
                else:
                    openai_api_status = "error"
                api_error_message = error_str

        # Check data directory and files
        data_dir_exists = os.path.exists("src/data")
        data_files_count = 0
        if data_dir_exists:
            try:
                data_files_count = len(
                    [f for f in os.listdir("src/data") if f.endswith(".json")]
                )
            except:
                data_files_count = 0

        # Check if required environment variables are set
        api_key = os.getenv("AZURE_AI_API_KEY")
        api_key_configured = bool(api_key)

        # Determine overall status
        overall_status = "healthy"

        if openai_api_status in ["quota_exceeded", "invalid_key", "forbidden"]:
            overall_status = "warning"
        elif openai_api_status == "error":
            overall_status = "error"
        elif agents_status != "initialized" or openai_client_status != "connected":
            overall_status = "warning"

        # Create appropriate message
        if overall_status == "error":
            if openai_api_status == "quota_exceeded":
                message = (
                    "OpenAI API quota exceeded. Please check your billing details."
                )
            elif openai_api_status == "invalid_key":
                message = "OpenAI API key is invalid. Please check your API key."
            elif openai_api_status == "forbidden":
                message = "OpenAI API access forbidden. Please check your API key permissions."
            else:
                message = "System error detected."
        elif overall_status == "warning":
            message = "Some components have issues but system is partially operational."
        else:
            message = "AutoGen Agents and Services are operational."

        return {
            "status": overall_status,
            "message": message,
            "timestamp": current_time,
            "components": {
                "agents_status": agents_status,
                "openai_client_status": openai_client_status,
                "openai_api_status": openai_api_status,
                "api_key_configured": api_key_configured,
                "data_directory_exists": data_dir_exists,
                "data_files_count": data_files_count,
            },
            "api_error": api_error_message,
            "version": "1.0.0",
            "uptime_check": current_time,
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "message": f"AutoGen Business Insights API encountered an error: {str(e)}",
        }
