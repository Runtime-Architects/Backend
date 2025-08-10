from dotenv import load_dotenv
from sqlmodel import create_engine, Session, SQLModel
from typing import Generator
import os
from pathlib import Path
from .models import User, Credential

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
    """Create database and tables"""
    # Create all tables in the database
    SQLModel.metadata.create_all(engine)
    print(f"Database created at: COSMOSDB")


def get_session() -> Generator[Session, None, None]:
    with Session(engine) as session:
        yield session
