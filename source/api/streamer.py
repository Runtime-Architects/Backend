"""
streamer.py

This module contains the implementation of SSE and the Pydantic Models used by it
"""

import asyncio
import logging
import sys
from datetime import datetime
from enum import Enum
from typing import Any, AsyncGenerator, Callable, Dict

from fastapi.responses import StreamingResponse
from pydantic import BaseModel


# Logging Config
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


class StreamEventType(str, Enum):
    """Enum class representing the different types of stream events.

    Attributes:
        STARTED (str): Indicates that the stream has started.
        AGENT_THINKING (str): Indicates that the agent is currently thinking.
        AGENT_RESPONSE (str): Indicates that the agent has provided a response.
        TOOL_EXECUTION (str): Indicates that a tool is being executed.
        ERROR (str): Indicates that an error has occurred.
        COMPLETED (str): Indicates that the stream has completed.
    """

    STARTED = "started"
    AGENT_THINKING = "agent_thinking"
    AGENT_RESPONSE = "agent_response"
    TOOL_EXECUTION = "tool_execution"
    ERROR = "error"
    COMPLETED = "completed"


class StreamEvent(BaseModel):
    """A class representing a stream event.

    Attributes:
        event_type (StreamEventType): The type of the stream event.
        timestamp (str): The timestamp of the event.
        agent_name (str, optional): The name of the agent associated with the event. Defaults to an empty string.
        message (str, optional): A message related to the event. Defaults to an empty string.
        data (Dict[str, Any], optional): Additional data associated with the event. Defaults to an empty dictionary.
    """

    event_type: StreamEventType
    timestamp: str
    agent_name: str = ""
    message: str = ""
    data: Dict[str, Any] = {}


class StreamEventManager:
    """StreamEventManager is a class that manages streaming events and tracks progress.

    Attributes:
        events (list): A list to store emitted events.
        current_step (int): The current step in the progress tracking.
        total_steps (int): The total number of steps for progress tracking.

    Methods:
        emit_event(event_type, agent_name="", message="", data=None):
            Create and emit a new streaming event.

        get_progress_percentage():
            Calculate the current progress percentage.

        increment_step():
            Increment the current step counter for progress tracking.
    """

    def __init__(self):
        """Initialize the event manager with empty events list and progress tracking."""
        self.events = []
        self.current_step = 0
        self.total_steps = 10

    async def emit_event(
        self,
        event_type: StreamEventType,
        agent_name: str = "",
        message: str = "",
        data: Dict[str, Any] = None,
    ) -> StreamEvent:
        """
        Create and emit a new streaming event.

        Args:
            event_type: Type of the event being emitted
            agent_name: Name of the agent generating the event
            message: Event message content
            data: Additional event data dictionary

        Returns:
            StreamEvent: The created and stored event
        """
        event = StreamEvent(
            event_type=event_type,
            timestamp=datetime.now().isoformat(),
            agent_name=agent_name,
            message=message,
            data=data or {},
        )
        self.events.append(event)
        logger.info(f"Event: {event_type} | Agent: {agent_name} | Message: {message}")
        return event

    def get_progress_percentage(self) -> int:
        """
        Calculate the current progress percentage.

        Returns:
            int: Progress percentage (0-100)
        """
        return min(int((self.current_step / self.total_steps) * 100), 100)

    def increment_step(self):
        """Increment the current step counter for progress tracking."""
        self.current_step += 1


class StreamingLogHandler(logging.Handler):
    """Custom logging handler that converts log records to streaming events."""

    def __init__(self, event_manager: StreamEventManager):
        """
        Initialize the streaming log handler.

        Args:
            event_manager: The event manager to emit events through
        """
        super().__init__()
        self.event_manager = event_manager
        self.event_queue = asyncio.Queue()

    def emit(self, record):
        """
        Capture log records and queue them for processing.

        Args:
            record: The log record to process
        """
        try:
            # Queue the log record for async processing
            asyncio.create_task(self.event_queue.put(record))
        except Exception:
            pass  # Avoid logging errors in the log handler

    async def process_logs(self) -> AsyncGenerator[str, None]:
        """
        Process queued log records and emit streaming events.

        Yields:
            str: Server-sent event formatted strings
        """
        while True:
            try:
                record = await asyncio.wait_for(self.event_queue.get(), timeout=0.1)

                # Determine if this is an AutoGen-related log
                if (
                    "autogen" in record.name.lower()
                    or "agent" in record.getMessage().lower()
                ):
                    # Extract agent information from log message
                    message = record.getMessage()

                    # Emit appropriate event
                    if record.levelno >= logging.ERROR:
                        event_type = StreamEventType.ERROR
                    elif "thinking" in message.lower():
                        event_type = StreamEventType.AGENT_THINKING
                    else:
                        event_type = StreamEventType.AGENT_RESPONSE

                    event = await self.event_manager.emit_event(
                        event_type,
                        agent_name=record.name,
                        message=(
                            message[:200] + "..." if len(message) > 200 else message
                        ),
                        data={"log_level": record.levelname},
                    )
                    yield f'data: {{"event": {event.model_dump_json()}}}\n\n'

            except asyncio.TimeoutError:
                # No logs to process, yield control
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Error processing logs: {e}")
                break


class SSEStreamer:
    """Server-Sent Events streamer for real-time agent communication."""

    def __init__(self, agent_function: Callable[..., AsyncGenerator[str, None]]):
        """
        Initialize the SSE streamer.

        Args:
            agent_function: The async generator function that produces streaming data
        """
        self.agent_function = agent_function
        self.event_manager = StreamEventManager()

    async def stream(self, *args, **kwargs) -> StreamingResponse:
        """
        Create a streaming response for real-time communication.

        Args:
            *args: Positional arguments to pass to the agent function
            **kwargs: Keyword arguments to pass to the agent function

        Returns:
            StreamingResponse: FastAPI streaming response for SSE
        """

        async def event_generator():
            """
            Internal generator that handles the streaming logic and error handling.

            Yields:
                str: Server-sent event formatted data
            """
            try:
                async for event_data in self.agent_function(
                    self.event_manager, *args, **kwargs
                ):
                    yield event_data
                    await asyncio.sleep(0.01)
            except Exception as e:
                logger.exception("Error while streaming response")
                error_event = await self.event_manager.emit_event(
                    StreamEventType.ERROR,
                    message=f"Stream failed: {str(e)}",
                    data={"error": str(e)},
                )
                yield f'data: {{"event": {error_event.model_dump_json()}}}\n\n'

        return StreamingResponse(event_generator(), media_type="text/event-stream")
