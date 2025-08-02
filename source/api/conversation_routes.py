from fastapi import APIRouter, HTTPException, Depends
from sqlmodel import Session
from datetime import datetime

from api.db import get_session
from api.models import Conversation, Message, User
from api.auth_routes import get_current_user

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.get("/")
async def get_conversations(
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Get all conversations for the authenticated user"""
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
    """Get a specific conversation with all messages"""
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
    """Delete a conversation and all its messages"""
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
