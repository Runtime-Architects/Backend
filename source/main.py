"""
main.py

Main FastAPI application entry point.
Sets up middleware, routes, and initializes database and agents.
"""

import logging
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from agents.agent_workflow import initialize_agents
from api.auth_routes import router as auth_router
from api.chat_routes import router as chat_router
from api.conversation_routes import router as conversation_router
from api.db import create_db_and_tables
from api.health_routes import router as health_router


# Logging Config
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)

from dotenv import load_dotenv

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan for the FastAPI application.

    This asynchronous function manages the startup and shutdown processes of the FastAPI application. It initializes the database and tables, sets up the agent factory, and logs the status of the application during these phases.

    Args:
        app (FastAPI): The FastAPI application instance.

    Yields:
        None: This function yields control back to the FastAPI application.

    Raises:
        Exception: If there is an error during the startup process, it logs the error and raises the exception.
    """
    # Startup
    logger.info("Starting AutoGen Business Insights API...")
    create_db_and_tables()
    try:
        app.state.agent_factory = initialize_agents
        logger.info("API startup completed successfully")
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down Sustainable CIty API...")


# Initialize FastAPI app
app = FastAPI(
    title="Sustainable City",
    description="API for generating business insights reports using AutoGen agents",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ["CORS_ORIGINS"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth_router)
app.include_router(conversation_router)
app.include_router(chat_router)
app.include_router(health_router)


@app.get("/")
async def root():
    """Returns a dictionary containing information about the Sustainable City AI.

    This asynchronous function provides a message and links to documentation and client resources.

    Returns:
        dict: A dictionary with the following keys:
            - message (str): A message indicating the purpose of the service.
            - docs (str): The URL path to the documentation.
            - client (str): The URL path to the client interface.
    """
    return {
        "message": "Sustainable City AI",
        "docs": "/docs",
        "client": "/client",
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=os.environ["APP_HOST"],
        port=int(os.environ["APP_PORT"]),
        reload=True,
        timeout_keep_alive=300,
        timeout_graceful_shutdown=30,
    )
