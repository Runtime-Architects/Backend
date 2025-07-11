from sqlmodel import SQLModel, Field, Relationship
from typing import Optional, List
import json

class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    email: str = Field(index=True, unique=True)
    # Remove plain passkey field and add WebAuthn support
    credentials: List["Credential"] = Relationship(back_populates="user")

class Credential(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id")
    credential_id: str = Field(index=True, unique=True)
    public_key: str  # Store as base64 encoded string
    sign_count: int = Field(default=0)
    created_at: str
    
    user: Optional[User] = Relationship(back_populates="credentials")