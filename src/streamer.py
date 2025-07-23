from fastapi.responses import StreamingResponse
from typing import AsyncGenerator, Callable, Any, Dict
import asyncio
import logging
from datetime import datetime
from enum import Enum
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class StreamEventType(str, Enum):
    """Enumeration of different stream event types for agent communication."""
    STARTED = "started"
    AGENT_THINKING = "agent_thinking"
    AGENT_RESPONSE = "agent_response"
    TOOL_EXECUTION = "tool_execution"
    ERROR = "error"
    COMPLETED = "completed"

class StreamEvent(BaseModel):
    """Pydantic model representing a streaming event with metadata."""
    event_type: StreamEventType
    timestamp: str
    agent_name: str = ""
    message: str = ""
    data: Dict[str, Any] = {}

class StreamEventManager:
    """Manages streaming events and progress tracking for agent communication."""
    
    def __init__(self):
        """Initialize the event manager with empty events list and progress tracking."""
        self.events = []
        self.current_step = 0
        self.total_steps = 15
        
    async def emit_event(self, event_type: StreamEventType, agent_name: str = "", 
                        message: str = "", data: Dict[str, Any] = None) -> StreamEvent:
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
            data=data or {}
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
                if 'autogen' in record.name.lower() or 'agent' in record.getMessage().lower():
                    # Extract agent information from log message
                    message = record.getMessage()
                    
                    # Emit appropriate event
                    if record.levelno >= logging.ERROR:
                        event_type = StreamEventType.ERROR
                    elif 'thinking' in message.lower():
                        event_type = StreamEventType.AGENT_THINKING
                    else:
                        event_type = StreamEventType.AGENT_RESPONSE
                    
                    event = await self.event_manager.emit_event(
                        event_type,
                        agent_name=record.name,
                        message=message[:200] + "..." if len(message) > 200 else message,
                        data={"log_level": record.levelname}
                    )
                    yield f"data: {{\"event\": {event.model_dump_json()}}}\n\n"
                    
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
                async for event_data in self.agent_function(self.event_manager, *args, **kwargs):
                    yield event_data
                    await asyncio.sleep(0.01)
            except Exception as e:
                logger.exception("Error while streaming response")
                error_event = await self.event_manager.emit_event(
                    StreamEventType.ERROR,
                    message=f"Stream failed: {str(e)}",
                    data={"error": str(e)}
                )
                yield f"data: {{\"event\": {error_event.model_dump_json()}}}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")