"""
db.py

This module consists of functions to create Database and Tables and use Database Session
"""

import os
from pathlib import Path
from typing import Generator

from dotenv import load_dotenv
from sqlmodel import Session, SQLModel, create_engine

from .models import Credential, User


load_dotenv()

# Create data directory if it doesn't exist
data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

# Database URL - stores in data/ directory
DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
print("Using DB URL:", DATABASE_URL)

engine = create_engine(
    DATABASE_URL,
    echo=False,  # Set to True for SQL query debugging
    pool_pre_ping=True,  # Verify connections before use
    pool_recycle=300,  # Recycle connections every 5 minutes
)


def create_db_and_tables():
    """Creates the database and tables using SQLModel.

    This function initializes the database by creating all tables defined in the SQLModel metadata.
    It also prints a confirmation message indicating the location of the created database.

    Returns:
        None
    """
    # Create all tables in the database
    SQLModel.metadata.create_all(engine)
    print(f"Database created at: COSMOSDB")


def get_session() -> Generator[Session, None, None]:
    """Yields a database session.

    This function creates a new database session using the provided engine and yields it for use.
    The session is automatically closed when the generator is exhausted.

    Yields:
        Session: A database session object.

    Usage:
        with get_session() as session:
            # Perform database operations using the session
    """
    with Session(engine) as session:
        yield session
