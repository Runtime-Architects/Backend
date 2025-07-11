from sqlmodel import create_engine, Session, SQLModel
from typing import Generator
import os
from pathlib import Path

# Create data directory if it doesn't exist
data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

# Database URL - stores in data/ directory
DATABASE_URL = f"sqlite:///{data_dir}/app.db"

engine = create_engine(
    DATABASE_URL, 
    connect_args={"check_same_thread": False},
    echo=False  # Set to True for SQL query debugging
)

def create_db_and_tables():
    """Create database and tables"""
    # Import your models here to ensure they're registered
    from models import User, Credential
    # Create all tables in the database
    SQLModel.metadata.create_all(engine)
    print(f"Database created at: {DATABASE_URL}")

def get_session() -> Generator[Session, None, None]:
    with Session(engine) as session:
        yield session