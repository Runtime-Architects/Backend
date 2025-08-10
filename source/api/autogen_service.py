import logging
from datetime import datetime
from typing import AsyncGenerator
from fastapi import HTTPException

from autogen_agentchat.messages import StopMessage
from api.db import get_session
from api.models import Conversation, Message
from api.streamer import StreamEventManager, StreamEventType

# from agents.agent_workflow import team_flow

logger = logging.getLogger(__name__)


async def run_autogen_task(question: str) -> str:
    """Simple non-streaming AutoGen task execution"""
    # Implementation for non-streaming version
    # You can keep your existing logic here
    pass


async def run_autogen_task_streaming(
    event_manager: StreamEventManager,
    question: str,
    user_id: int,
    conversation_id: int,
    team_flow,
) -> AsyncGenerator[str, None]:
    """Run the AutoGen task with real-time message monitoring using AutoGen's streaming"""
    if team_flow is None:
        error_event = await event_manager.emit_event(
            StreamEventType.ERROR,
            message="AutoGen agents not initialized",
            data={"error": "AutoGen agents not initialized"},
        )
        yield f'data: {{"event": {error_event.model_dump_json()}}}\n\n'
        return

    try:
        logger.info(f"Starting analysis for: {question[:100]}...")

        # Handle conversation creation or retrieval
        from api.db import get_session

        session = next(get_session())
        try:
            if conversation_id:
                # Use existing conversation
                conversation = (
                    session.query(Conversation)
                    .filter(
                        Conversation.id == conversation_id,
                        Conversation.user_id == user_id,
                    )
                    .first()
                )

                if not conversation:
                    raise HTTPException(
                        status_code=404, detail="Conversation not found"
                    )

                # Update the conversation timestamp
                conversation.updated_at = datetime.now().isoformat()
                session.commit()
            else:
                # Create new conversation
                conversation = Conversation(
                    user_id=user_id,
                    title=question[:50] + "..." if len(question) > 50 else question,
                    created_at=datetime.now().isoformat(),
                    updated_at=datetime.now().isoformat(),
                )
                session.add(conversation)
                session.commit()
                session.refresh(conversation)

            conversation_id = conversation.id

            # Save user message
            user_message = Message(
                conversation_id=conversation_id,
                role="user",
                content=question,
                timestamp=datetime.now().isoformat(),
            )
            session.add(user_message)
            session.commit()
        finally:
            session.close()

        # Emit starting event
        event = await event_manager.emit_event(
            StreamEventType.STARTED,
            agent_name="user",
            message=f"Starting analysis for: {question[:100]}...",
            data={
                "question": question,
                "progress": 0,
                "conversation_id": conversation_id,
            },
        )
        yield f'data: {{"event": {event.model_dump_json()}}}\n\n'

        # Create the task
        # In the streaming function, after retrieving the conversation
        if conversation_id and conversation:
            # Get recent messages for context
            recent_messages = (
                session.query(Message)
                .filter(Message.conversation_id == conversation_id)
                .order_by(Message.timestamp.desc())
                .limit(10)
                .all()
            )

            # Build context string
            context_messages = []
            for msg in reversed(recent_messages):  # Reverse to get chronological order
                context_messages.append(f"{msg.role}: {msg.content}")

            context = "\n".join(context_messages) if context_messages else ""

            # Create context-aware task
            task = f"""Previous conversation context:
            
            {context}

            Current user question: {question}

            Generate a comprehensive business insights report that considers the conversation history and responds to the current question."""

        else:
            # Create the task as before for new conversations
            task = f"Generate a comprehensive business insights report based on this request: {question}"

        # Track the final response - collect all meaningful messages
        all_agent_responses = []
        final_report = ""

        # Use AutoGen's streaming capability
        async for message in team_flow.run_stream(task=task):
            # Check for StopMessage (indicates completion)
            if isinstance(message, StopMessage):
                # Compile final response - prioritize ReportAgent, then fallback to all responses
                report_agent_messages = [
                    resp
                    for resp in all_agent_responses
                    if resp["source"] == "ReportAgent"
                ]

                if report_agent_messages:
                    # Use ReportAgent's responses
                    final_report = "\n\n".join(
                        [resp["content"] for resp in report_agent_messages]
                    )
                elif all_agent_responses:
                    # Fallback: use all substantial responses
                    final_report = "\n\n".join(
                        [
                            f"**{resp['source']}**: {resp['content']}"
                            for resp in all_agent_responses
                        ]
                    )
                else:
                    final_report = "Task completed but no response was generated."

                # Clean up the final report
                final_report = final_report.replace("TERMINATE", "").strip()

                # Save final response to database
                if final_report:
                    session = next(get_session())
                    try:
                        assistant_message = Message(
                            conversation_id=conversation_id,
                            role="assistant",
                            content=final_report,
                            timestamp=datetime.now().isoformat(),
                        )
                        session.add(assistant_message)

                        # Update conversation timestamp
                        conversation = session.get(Conversation, conversation_id)
                        if conversation:
                            conversation.updated_at = datetime.now().isoformat()

                        session.commit()
                        logger.info(
                            f"Successfully saved assistant response to database for conversation {conversation_id}"
                        )
                        logger.info(
                            f"Final report length: {len(final_report)} characters"
                        )
                        logger.info(
                            f"Final report content: {final_report[:100]}..."
                        )  # Log first 100 chars
                    except Exception as db_error:
                        logger.error(
                            f"Failed to save assistant response: {str(db_error)}"
                        )
                        session.rollback()
                    finally:
                        session.close()
                else:
                    logger.warning("Final report is empty, not saving to database")

                event = await event_manager.emit_event(
                    StreamEventType.COMPLETED,
                    agent_name="GraphManager",
                    message="Workflow completed successfully",
                    data={
                        "progress": 100,
                        "conversation_id": conversation_id,
                        "final_response": final_report,
                    },
                )
                yield f'data: {{"event": {event.model_dump_json()}}}\n\n'
                break

            # Process regular messages
            if hasattr(message, "source") and hasattr(message, "content"):
                event_manager.increment_step()

                # Safely extract and process content
                content_str = ""
                content_type = "text"

                if isinstance(message.content, str):
                    content_str = message.content
                elif isinstance(message.content, list):
                    # Handle function calls or structured content
                    if message.content and hasattr(message.content[0], "name"):
                        # This is likely a function call
                        content_type = "function_call"
                        func_names = [
                            item.name
                            for item in message.content
                            if hasattr(item, "name")
                        ]
                        content_str = f"Calling functions: {', '.join(func_names)}"
                    else:
                        content_str = str(message.content)
                else:
                    content_str = str(message.content)

                # Store ALL substantial agent responses (not just ReportAgent)
                if (
                    len(content_str.strip()) > 50
                    and "TERMINATE" not in content_str
                    and content_type != "function_call"
                ):
                    clean_content = content_str.strip()
                    # Avoid duplicates
                    if not any(
                        resp["content"] == clean_content for resp in all_agent_responses
                    ):
                        all_agent_responses.append(
                            {"source": message.source, "content": clean_content}
                        )
                        logger.info(
                            f"Collected response from {message.source}: {len(clean_content)} chars"
                        )

                # Determine event type based on content
                if "TERMINATE" in content_str:
                    event_type = StreamEventType.COMPLETED
                    try:
                        # If TERMINATE is found, we assume the task is complete
                        final_report = content_str.replace("TERMINATE", "").strip()
                        assistant_message = Message(
                            conversation_id=conversation_id,
                            role="assistant",
                            content=final_report,
                            timestamp=datetime.now().isoformat(),
                        )
                        session.add(assistant_message)
                        await session.commit()
                    except Exception as db_error:
                        logger.error(f"Failed to save final report: {str(db_error)}")
                        session.rollback()

                elif content_type == "function_call":
                    event_type = StreamEventType.TOOL_EXECUTION
                elif any(
                    keyword in content_str.lower()
                    for keyword in ["thinking", "planning", "analyzing", "processing"]
                ):
                    event_type = StreamEventType.AGENT_THINKING
                else:
                    event_type = StreamEventType.AGENT_RESPONSE

                # Create display message - truncate if too long
                display_message = (
                    content_str[:100] + "..." if len(content_str) > 100 else content_str
                )

                agent_responses = {
                    "CarbonAgent": "",
                    "PolicyAgent": "",
                    "DataAnalysisAgent": "",
                }

                if (
                    message.source in agent_responses
                    and len(content_str.strip()) > 50
                    and content_type != "function_call"
                ):
                    agent_responses[message.source] = content_str

                # Generate supportive content based on agent and context
                supportive_content = ""

                if message.source == "CarbonAgent":
                    # Use the stored CarbonAgent response
                    if agent_responses["CarbonAgent"]:
                        max_length = 5000
                        carbon_response = agent_responses["CarbonAgent"]
                        if len(carbon_response) > max_length:
                            supportive_content = carbon_response[:max_length] + "..."
                        else:
                            supportive_content = carbon_response
                    else:
                        supportive_content = "Carbon analysis in progress"

                    # Add plot information if available
                    import re

                    plot_pattern = r"(co2plot_[a-zA-Z0-9_\-]+\.png|carbon_[a-zA-Z0-9_\-]+\.png|plot_[a-zA-Z0-9_\-]+\.png)"
                    plots_found = re.findall(plot_pattern, content_str)

                    if plots_found:
                        if (
                            supportive_content
                            and supportive_content != "Carbon analysis in progress"
                        ):
                            supportive_content += f"\n\nPlots: {', '.join(plots_found)}"
                        else:
                            supportive_content = (
                                f"Generated plots: {', '.join(plots_found)}"
                            )

                elif message.source == "PolicyAgent":
                    policy_response = agent_responses["PolicyAgent"]
                    if policy_response:
                        supportive_content = policy_response

                content_str = content_str.replace("TERMINATE", "").strip()

                event = await event_manager.emit_event(
                    event_type,
                    agent_name=message.source,
                    message=display_message,
                    data={
                        "progress": min(event_manager.get_progress_percentage(), 95),
                        "content_type": content_type,
                        "conversation_id": conversation_id,
                        "full_content": (
                            content_str if len(content_str) < 5000 else None
                        ),
                        "context": (
                            supportive_content
                            if len(supportive_content) < 5000
                            else None
                        ),
                    },
                )

                yield f'data: {{"event": {event.model_dump_json()}}}\n\n'

        # Final check - if no responses were collected during streaming, try to get them from the completed flow
        if not all_agent_responses:
            logger.warning(
                "No responses collected during streaming, this might indicate an issue with the workflow"
            )

    except Exception as e:
        logger.error(f"Error in streaming AutoGen task: {str(e)}")
        error_event = await event_manager.emit_event(
            StreamEventType.ERROR,
            message=f"Task failed: {str(e)}",
            data={"error": str(e)},
        )
        yield f'data: {{"event": {error_event.model_dump_json()}}}\n\n'
