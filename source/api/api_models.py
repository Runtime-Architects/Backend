"""
api_models.py

This module consists of Pydantic Data Models used by Chat Routes
"""

from pydantic import BaseModel


class QuestionRequest(BaseModel):
    """A class representing a request for a question in a conversation.

    Attributes:
        question (str): The question being asked.
        conversation_id (int, optional): The ID of the conversation. Defaults to None.
    """

    question: str
    conversation_id: int = None


class MessageResponse(BaseModel):
    """MessageResponse is a model that represents a response message.

    Attributes:
        role (str): The role of the entity sending the message (e.g., "user", "assistant").
        content (str): The content of the message being sent.
    """

    role: str
    content: str


class APIResponse(BaseModel):
    """APIResponse represents the structure of a response from the API.

    Attributes:
        status (str): The status of the API response.
        message (MessageResponse): An object containing the message details of the API response.
    """

    status: str
    message: MessageResponse
