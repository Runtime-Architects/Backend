from sqlmodel import create_engine, Session, SQLModel
from typing import Generator
from dotenv import load_dotenv
import os
from pathlib import Path

# Load environment variables from .env file
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

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
    pool_recycle=300,    # Recycle connections every 5 minutes
)

def create_db_and_tables():
    """Create database and tables"""
    # Import your models here to ensure they're registered
    from models import User, Credential
    # Create all tables in the database
    SQLModel.metadata.create_all(engine)
    print(f"Database created at: COSMOSDB")

def get_session() -> Generator[Session, None, None]:
    with Session(engine) as session:
        yield session