from sqlmodel import SQLModel, Field, Relationship
from typing import Optional, List
import json

class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    email: str = Field(index=True, unique=True)
    # Remove plain passkey field and add WebAuthn support
    credentials: List["Credential"] = Relationship(back_populates="user")
    conversations: List["Conversation"] = Relationship(back_populates="user")

class Credential(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id")
    credential_id: str = Field(index=True, unique=True)
    public_key: str  # Store as base64 encoded string
    sign_count: int = Field(default=0)
    created_at: str
    
    user: Optional[User] = Relationship(back_populates="credentials")

class Conversation(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id")
    title: Optional[str] = Field(default=None)  # Auto-generated or user-defined title
    created_at: str
    updated_at: str
    
    user: Optional[User] = Relationship(back_populates="conversations")
    messages: List["Message"] = Relationship(back_populates="conversation")

class Message(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    conversation_id: int = Field(foreign_key="conversation.id")
    role: str  # "user" or "assistant"
    content: str
    timestamp: str
    
    conversation: Optional[Conversation] = Relationship(back_populates="messages")