import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from api.db import create_db_and_tables
from api.auth_routes import router as auth_router
from api.conversation_routes import router as conversation_router
from api.chat_routes import router as chat_router
from api.health_routes import router as health_router
from agents.agent_workflow import initialize_agents


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from dotenv import load_dotenv

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
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
    """Root endpoint"""
    return {
        "message": "AutoGen Business Insights API",
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
