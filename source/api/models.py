"""
models.py

This module contains Data Models created for the Database
"""

import json
from typing import List, Optional

from sqlmodel import Field, Relationship, SQLModel


class User(SQLModel, table=True):
    """User model representing a user in the system.

    Attributes:
        id (Optional[int]): The unique identifier for the user. This is the primary key and defaults to None.
        email (str): The email address of the user. This field is indexed and must be unique.
        credentials (List[Credential]): A list of credentials associated with the user. This establishes a relationship with the Credential model.
        conversations (List[Conversation]): A list of conversations associated with the user. This establishes a relationship with the Conversation model.
    """

    id: Optional[int] = Field(default=None, primary_key=True)
    email: str = Field(index=True, unique=True)
    # Remove plain passkey field and add WebAuthn support
    credentials: List["Credential"] = Relationship(back_populates="user")
    conversations: List["Conversation"] = Relationship(back_populates="user")


class Credential(SQLModel, table=True):
    """Credential model representing user credentials in the database.

    Attributes:
        id (Optional[int]): The unique identifier for the credential.
            Defaults to None and is the primary key.
        user_id (int): The identifier for the associated user,
            linked to the User model via a foreign key.
        credential_id (str): A unique identifier for the credential,
            indexed for efficient querying.
        public_key (str): The public key associated with the credential,
            stored as a base64 encoded string.
        sign_count (int): The number of times the credential has been used
            to sign, defaults to 0.
        created_at (str): The timestamp indicating when the credential was created.
        user (Optional[User]): The user associated with this credential,
            establishing a relationship with the User model.
    """

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id")
    credential_id: str = Field(index=True, unique=True)
    public_key: str  # Store as base64 encoded string
    sign_count: int = Field(default=0)
    created_at: str

    user: Optional[User] = Relationship(back_populates="credentials")


class Conversation(SQLModel, table=True):
    """A class representing a conversation in a database.

    Attributes:
        id (Optional[int]): The unique identifier for the conversation.
            Automatically generated if not provided.
        user_id (int): The identifier for the user associated with the conversation.
        title (Optional[str]): The title of the conversation, which can be auto-generated
            or user-defined.
        created_at (str): The timestamp indicating when the conversation was created.
        updated_at (str): The timestamp indicating when the conversation was last updated.
        user (Optional[User]): The user associated with the conversation, establishing a
            relationship with the User model.
        messages (List[Message]): A list of messages associated with the conversation,
            establishing a relationship with the Message model.
    """

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id")
    title: Optional[str] = Field(default=None)  # Auto-generated or user-defined title
    created_at: str
    updated_at: str

    user: Optional[User] = Relationship(back_populates="conversations")
    messages: List["Message"] = Relationship(back_populates="conversation")


class Message(SQLModel, table=True):
    """Message class represents a message in a conversation.

    Attributes:
        id (Optional[int]): The unique identifier for the message. Defaults to None and is the primary key.
        conversation_id (int): The identifier for the associated conversation, linked to the Conversation model.
        role (str): The role of the message sender, either "user" or "assistant".
        content (str): The content of the message.
        timestamp (str): The timestamp when the message was created.
        conversation (Optional[Conversation]): The associated Conversation object, with a back reference to messages.
    """

    id: Optional[int] = Field(default=None, primary_key=True)
    conversation_id: int = Field(foreign_key="conversation.id")
    role: str  # "user" or "assistant"
    content: str
    timestamp: str

    conversation: Optional[Conversation] = Relationship(back_populates="messages")
