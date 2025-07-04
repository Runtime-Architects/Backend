from fastapi.responses import StreamingResponse
from typing import AsyncGenerator, Callable, Any, Dict
import asyncio
import logging
from datetime import datetime
from enum import Enum
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class StreamEventType(str, Enum):
    STARTED = "started"
    AGENT_THINKING = "agent_thinking"
    AGENT_RESPONSE = "agent_response"
    TOOL_EXECUTION = "tool_execution"
    ERROR = "error"
    COMPLETED = "completed"

class StreamEvent(BaseModel):
    event_type: StreamEventType
    timestamp: str
    agent_name: str = ""
    message: str = ""
    data: Dict[str, Any] = {}

class StreamEventManager:
    def __init__(self):
        self.events = []
        self.current_step = 0
        self.total_steps = 25
        
    async def emit_event(self, event_type: StreamEventType, agent_name: str = "", 
                        message: str = "", data: Dict[str, Any] = None) -> StreamEvent:
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
        return min(int((self.current_step / self.total_steps) * 100), 100)
    
    def increment_step(self):
        self.current_step += 1


class StreamingLogHandler(logging.Handler):
    def __init__(self, event_manager: StreamEventManager):
        super().__init__()
        self.event_manager = event_manager
        self.event_queue = asyncio.Queue()
    
    def emit(self, record):
        """Capture log records and queue them for processing"""
        try:
            # Queue the log record for async processing
            asyncio.create_task(self.event_queue.put(record))
        except Exception:
            pass  # Avoid logging errors in the log handler
    
    async def process_logs(self) -> AsyncGenerator[str, None]:
        """Process queued log records and emit streaming events"""
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
    def __init__(self, agent_function: Callable[..., AsyncGenerator[str, None]]):
        self.agent_function = agent_function
        self.event_manager = StreamEventManager()

    async def stream(self, *args, **kwargs) -> StreamingResponse:
        async def event_generator():
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