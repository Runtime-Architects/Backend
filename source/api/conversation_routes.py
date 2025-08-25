"""
conversation_routes.py

This module contains FastAPI Routes for Conversations
"""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session

from api.auth_routes import get_current_user
from api.db import get_session
from api.models import Conversation, Message, User


router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.get("/")
async def get_conversations(
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Retrieve conversations for the current user.

    This asynchronous function fetches all conversations associated with the
    current user from the database, ordered by the most recently updated.

    Args:
        current_user (User, optional): The user for whom to retrieve conversations.
            Defaults to the result of the `get_current_user` dependency.
        session (Session, optional): The database session to use for the query.
            Defaults to the result of the `get_session` dependency.

    Returns:
        dict: A dictionary containing a list of conversations under the key
        "conversations".
    """
    """Await docstring generation..."""
    conversations = (
        session.query(Conversation)
        .filter(Conversation.user_id == current_user.id)
        .order_by(Conversation.updated_at.desc())
        .all()
    )

    return {"conversations": conversations}


@router.get("/{conversation_id}")
async def get_conversation(
    conversation_id: int,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Retrieve a conversation and its associated messages.

    This asynchronous function fetches a conversation based on the provided conversation ID and the current user. It ensures that the conversation belongs to the current user. If the conversation is not found, a 404 HTTP exception is raised. The function also retrieves all messages associated with the conversation, ordered by their timestamp.

    Args:
        conversation_id (int): The ID of the conversation to retrieve.
        current_user (User, optional): The user making the request. Defaults to the result of `get_current_user`.
        session (Session, optional): The database session to use for the query. Defaults to the result of `get_session`.

    Returns:
        dict: A dictionary containing the conversation and a list of associated messages.

    Raises:
        HTTPException: If the conversation is not found, a 404 error is raised.
    """
    conversation = (
        session.query(Conversation)
        .filter(
            Conversation.id == conversation_id, Conversation.user_id == current_user.id
        )
        .first()
    )

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    messages = (
        session.query(Message)
        .filter(Message.conversation_id == conversation_id)
        .order_by(Message.timestamp.asc())
        .all()
    )

    return {"conversation": conversation, "messages": messages}


@router.delete("/{conversation_id}")
async def delete_conversation(
    conversation_id: int,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Deletes a conversation and its associated messages from the database.

    This asynchronous function takes a conversation ID and the current user, verifies that the conversation belongs to the user, and deletes it along with all associated messages. If the conversation is not found, a 404 HTTP exception is raised.

    Args:
        conversation_id (int): The ID of the conversation to be deleted.
        current_user (User, optional): The user requesting the deletion. Defaults to the result of `get_current_user`.
        session (Session, optional): The database session to use for the operation. Defaults to the result of `get_session`.

    Raises:
        HTTPException: If the conversation is not found, a 404 error is raised.

    Returns:
        dict: A message indicating the successful deletion of the conversation.
    """
    conversation = (
        session.query(Conversation)
        .filter(
            Conversation.id == conversation_id, Conversation.user_id == current_user.id
        )
        .first()
    )

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Delete all messages first
    session.query(Message).filter(Message.conversation_id == conversation_id).delete()

    # Delete the conversation
    session.delete(conversation)
    session.commit()

    return {"message": "Conversation deleted successfully"}
