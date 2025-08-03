import os
from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse, FileResponse

from api.api_models import QuestionRequest, APIResponse, MessageResponse
from api.models import User
from api.auth_routes import get_current_user
from api.autogen_service import run_autogen_task, run_autogen_task_streaming
from api.streamer import StreamEventManager
import logging

logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])


@router.post("/ask", response_model=APIResponse)
async def ask_endpoint(request: QuestionRequest):
    """
    Main endpoint to process questions and return business insights
    """
    try:
        logger.info(f"Received question: {request.question[:100]}...")

        # Run the AutoGen task
        response_content = await run_autogen_task(request.question)

        # Return the response in the expected format
        return APIResponse(
            status="Success",
            message=MessageResponse(role="assistant", content=response_content),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in ask endpoint: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {str(e)}"
        )


@router.post("/ask-stream")
async def ask_stream_endpoint(
    request_body: QuestionRequest,
    request: Request,
    current_user: User = Depends(get_current_user)
):
    """
    Streaming endpoint that provides real-time updates during agent processing
    Requires authentication
    """
    team_flow_factory = request.app.state.agent_factory
    team_flow = await team_flow_factory()

    async def generate_stream():
        event_manager = StreamEventManager()
        async for chunk in run_autogen_task_streaming(
            event_manager,
            request_body.question,
            current_user.id,
            request_body.conversation_id,
            team_flow
        ):
            yield chunk

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )

@router.get("/images/{image_name}")
async def get_image(image_name: str):
    """Get an image by name"""
    base_dir = os.path.abspath("plots")
    image_path = os.path.normpath(os.path.join(base_dir, image_name))
    if not image_path.startswith(base_dir):
        raise HTTPException(status_code=400, detail="Invalid image path")
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(image_path, media_type="image/png", filename=image_name)


@router.post("/conversations/{conversation_id}/messages")
async def add_message_to_conversation(
    conversation_id: int,
    request_body: QuestionRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
):
    """
    Add a new message to an existing conversation
    """
    from sqlmodel import Session
    from api.db import get_session
    from api.models import Conversation

    # Verify conversation ownership
    session = next(get_session())

    conversation = (
        session.query(Conversation)
        .filter(
            Conversation.id == conversation_id,
            Conversation.user_id == current_user.id
        )
        .first()
    )
    session.close()

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    team_flow_factory = request.app.state.agent_factory
    team_flow = await team_flow_factory()

    async def generate_stream():
        event_manager = StreamEventManager()
        async for chunk in run_autogen_task_streaming(
            event_manager,
            request_body.question,
            current_user.id,
            conversation_id,
            team_flow
        ):
            yield chunk

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
