from pydantic import BaseModel


class QuestionRequest(BaseModel):
    question: str
    conversation_id: int = None


class MessageResponse(BaseModel):
    role: str
    content: str


class APIResponse(BaseModel):
    status: str
    message: MessageResponse
