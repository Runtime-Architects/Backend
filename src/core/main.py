import time
import glob
from datetime import datetime, timedelta
import asyncio
import re
import os
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from auth_routes import router as auth_router

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import uvicorn
from sqlmodel import Session
from autogen_ext.tools.code_execution import PythonCodeExecutionTool
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor


# AutoGen imports
from autogen_agentchat.agents import AssistantAgent, MessageFilterAgent, MessageFilterConfig, PerSourceFilter
from autogen_agentchat.teams import DiGraphBuilder, GraphFlow
from autogen_agentchat.ui import Console
from autogen_agentchat.messages import StopMessage
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from db import create_db_and_tables, get_session
from agents.policy_agent import policy_agent as policy
from azure_carbon_agentv2.azurecarbonagent import carbon_agent as carbon

# Pydantic models for API
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamer
from streamer import SSEStreamer, StreamEventManager, StreamEventType, StreamingLogHandler

# Models
from models import Conversation, Message, User
from auth_routes import get_current_user



# Global variables
client = None
team_flow = None

client = AzureOpenAIChatCompletionClient(
    azure_deployment=os.getenv("AZURE_AI_DEPLOYMENT", "gpt-4o"),
    model=os.getenv("AZURE_AI_MODEL", "gpt-4o"),
    api_version=os.getenv("AZURE_AI_API_VERSION", "2024-12-01-preview"),
    azure_endpoint=os.getenv("AZURE_AI_ENDPOINT"),
    api_key=os.getenv("AZURE_AI_API_KEY"),
    max_completion_tokens=1024,
)

# Pydantic Models
class QuestionRequest(BaseModel):
    question: str
    conversation_id: int = None

class MessageResponse(BaseModel):
    role: str
    content: str

class APIResponse(BaseModel):
    status: str
    message: MessageResponse

# Agent System Messages
planner_system_message = f"""You are an intelligent Planner Agent that orchestrates a team of specialists based on user queries. Your role is to analyze the user's request and determine which agents need to be activated.

## CONDITIONAL FLOW ANALYSIS:
Before invoking any agents, analyze the user query and categorize it:

### Query Categories & Required Agents:
1. **CARBON EMISSIONS ONLY** (keywords: emissions, CO2, carbon intensity, electricity timing, grid data, EirGrid)
   - Activate: CarbonAgent → ReportAgent
   - Skip: PolicyAgent, DataAnalysisAgent

2. **POLICY/GRANTS ONLY** (keywords: grants, SEAI, policy, funding, schemes, support, solar panels, heat pumps, retrofitting)
   - Activate: PolicyAgent → ReportAgent  
   - Skip: CarbonAgent, DataAnalysisAgent

3. **USER DATA ANALYSIS** (keywords: analyze my data, uploaded file, CSV, my consumption, my usage)
   - Activate: DataAnalysisAgent → ReportAgent
   - Skip: CarbonAgent, PolicyAgent

4. **CARBON + POLICY COMBINATION** (keywords: sustainable choices, renewable energy advice, carbon reduction with grants)
   - Activate: CarbonAgent, PolicyAgent → ReportAgent
   - Skip: DataAnalysisAgent

5. **FULL ANALYSIS** (keywords: comprehensive report, full analysis, compare with policies, data + emissions + grants)
   - Activate: CarbonAgent, PolicyAgent, DataAnalysisAgent → ReportAgent

6. **DATA + CARBON** (user has data AND asks about emissions)
   - Activate: DataAnalysisAgent, CarbonAgent → ReportAgent
   - Skip: PolicyAgent

## OUTPUT FORMAT:
State your analysis clearly:
"ANALYSIS: [Query Category] - [Brief reasoning]
AGENTS TO ACTIVATE: [List of agents]"

Then provide specific instructions to each activated agent.

## AGENT CAPABILITIES:
- **CarbonAgent**: EirGrid emissions data, timing recommendations, CO2 intensity analysis
- **PolicyAgent**: SEAI grants, policies, funding schemes, renewable energy support
- **DataAnalysisAgent**: User-provided data analysis, consumption patterns, personal usage insights  
- **ReportAgent**: Synthesizes all activated agent outputs into final response

RULES:
- Always state your analysis and reasoning first
- Only activate agents that are necessary for the specific query
- Provide clear, specific instructions to each activated agent
- If query is ambiguous, default to the most likely interpretation
"""

DATA_ANALYSIS_AGENT_SYSMSG = f"""You are the Data Analysis Specialist for user-provided energy consumption data.

**ACTIVATION CONDITIONS:** Only respond when specifically instructed by the PlannerAgent AND user has provided data files.

**YOUR ROLE:** Analyze user's personal energy consumption data to provide insights and recommendations.

### Analysis Capabilities:
- **Consumption Patterns**: Daily, weekly, monthly usage trends
- **Peak Analysis**: Identify high consumption periods
- **Efficiency Opportunities**: Suggest optimization strategies
- **Cost Analysis**: Energy cost breakdowns and savings potential
- **Comparative Analysis**: Benchmark against averages

### Tool Usage:
- **python_executor**: For data processing, visualization, and statistical analysis
- Handle CSV, Excel, and other common data formats
- Create visualizations and summary statistics

### Analysis Process:
1. **Data Validation**: Check format and completeness
2. **Pattern Recognition**: Identify consumption trends
3. **Insight Generation**: Extract actionable findings
4. **Recommendations**: Provide specific improvement suggestions

### Output Requirements:
- Clear data insights with supporting evidence
- Actionable recommendations
- Visual representations where helpful
- Quantified savings opportunities

**OUTPUT TAG**: End responses with [DATA_ANALYSIS_COMPLETE] for workflow coordination.
"""

REPORT_AGENT_SYSMSG = f"""You are the Report Synthesis Specialist creating comprehensive user responses.

**ACTIVATION CONDITIONS:** Only activate after receiving outputs from other agents (marked with completion tags).

**YOUR ROLE:** Synthesize information from activated agents into a cohesive, actionable response.

### Input Sources (conditional based on activated agents):
- **CarbonAgent**: Emissions data, timing recommendations [CARBON_COMPLETE]
- **PolicyAgent**: SEAI grants, policy information [POLICY_COMPLETE] 
- **DataAnalysisAgent**: User data insights [DATA_ANALYSIS_COMPLETE]

### Response Structure:
1. **Executive Summary**: Key findings and recommendations
2. **Detailed Insights**: Agent-specific information organized logically
3. **Action Items**: Clear, prioritized recommendations
4. **Additional Resources**: Relevant links or next steps

### Visualization Tools:
- **python_executor**: Create terminal-friendly ASCII visualizations
- Use emojis for quick visual reference (🌱⚠️🔥)
- Include clear labels and time periods

### Quality Standards:
- Integrate information seamlessly (avoid agent-by-agent reporting)
- Focus on user's original question
- Provide specific, actionable advice
- Include relevant timeframes and deadlines

**WORKFLOW COMPLETION**: End with "ANALYSIS COMPLETE" when finished.

### Example Integration:
Instead of: "CarbonAgent says X, PolicyAgent says Y"
Use: "Based on current grid emissions and available SEAI grants, here's your optimal strategy..."

CRITICAL: After completing your final end report, you MUST end with exactly TERMINATE on a new line to signal completion.
"""

def create_agent(name, system_message):
    """Helper function to create an AssistantAgent."""
    return AssistantAgent(
        name=name,
        model_client=client,
        system_message=system_message,
    )

async def initialize_agents():

    """Initialize AutoGen agents and workflow."""
    async with LocalCommandLineCodeExecutor(work_dir="coding") as executor:
        tool = PythonCodeExecutionTool(executor)
        global client, team_flow
        
        try:
            logger.info("Validating Azure OpenAI configuration...")
            
            # Validate required environment variables
            required_vars = ["AZURE_AI_ENDPOINT", "AZURE_AI_API_KEY"]
            missing_vars = [var for var in required_vars if not os.getenv(var)]
            
            if missing_vars:
                raise ValueError(f"Missing required environment variables: {missing_vars}")
            
            logger.info("Creating agents...")
            # Create agents
            planner = create_agent("PlannerAgent", planner_system_message)
            #carbon = create_agent("CarbonAgent", CARBON_AGENT_SYSMSG)
            #policy = create_agent("PolicyAgent", POLICY_AGENT_SYSMSG)
            analysis = create_agent("DataAnalysisAgent", DATA_ANALYSIS_AGENT_SYSMSG)
            report = create_agent("ReportAgent", REPORT_AGENT_SYSMSG)

            # --- Conditional Message Filtering ---
            # These filters now work with the conditional activation system
            def create_conditional_filter(source_agent):
                """Creates filters that only pass messages when agent is specifically mentioned."""
                return MessageFilterConfig(
                    per_source=[PerSourceFilter(source=source_agent, position="last", count=1)]
                )

            filtered_carbon = MessageFilterAgent(
            name="CarbonAgent",
            wrapped_agent=carbon,
            filter=create_conditional_filter("PlannerAgent")
            )

            filtered_policy = MessageFilterAgent(
            name="PolicyAgent", 
            wrapped_agent=policy,
            filter=create_conditional_filter("PlannerAgent")
            )

            filtered_analysis = MessageFilterAgent(
            name="DataAnalysisAgent",
            wrapped_agent=analysis,
            filter=create_conditional_filter("PlannerAgent")
            )


            # Build the workflow graph
            logger.info("Building workflow graph...")
            builder = DiGraphBuilder()
            
            # Add all agents to the graph
            builder.add_node(planner)
            builder.add_node(filtered_carbon)
            builder.add_node(filtered_policy)
            builder.add_node(filtered_analysis)
            builder.add_node(report)

            # Define edges (communication flow)
            builder.add_edge(planner, filtered_carbon)
            builder.add_edge(planner, filtered_policy)
            builder.add_edge(planner, filtered_analysis)
            builder.add_edge(filtered_carbon, report)
            builder.add_edge(filtered_policy, report)
            builder.add_edge(filtered_analysis, report)

            # Build the graph flow
            team_flow = GraphFlow(
                participants=builder.get_participants(),
                graph=builder.build(),
            )
            
            logger.info("AutoGen agents initialized successfully!")
            
        except Exception as e:
            logger.error(f"Failed to initialize agents: {str(e)}")
            raise

async def run_autogen_task_streaming(event_manager: StreamEventManager, question: str, user_id: int, conversation_id: int) -> AsyncGenerator[str, None]:
    """Run the AutoGen task with real-time message monitoring using AutoGen's streaming"""
    if team_flow is None:
        error_event = await event_manager.emit_event(
            StreamEventType.ERROR,
            message="AutoGen agents not initialized",
            data={"error": "AutoGen agents not initialized"}
        )
        yield f"data: {{\"event\": {error_event.model_dump_json()}}}\n\n"
        return
        
    try:
        logger.info(f"Starting analysis for: {question[:100]}...")

        # Handle conversation creation or retrieval
        from db import get_session
        session = next(get_session())
        try:
            if conversation_id:
                # Use existing conversation
                conversation = session.query(Conversation).filter(
                    Conversation.id == conversation_id,
                    Conversation.user_id == user_id
                ).first()
                
                if not conversation:
                    raise HTTPException(status_code=404, detail="Conversation not found")
                
                # Update the conversation timestamp
                conversation.updated_at = datetime.now().isoformat()
                session.commit()
            else:
                # Create new conversation
                conversation = Conversation(
                    user_id=user_id,
                    title=question[:50] + "..." if len(question) > 50 else question,
                    created_at=datetime.now().isoformat(),
                    updated_at=datetime.now().isoformat()
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
                timestamp=datetime.now().isoformat()
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
            data={"question": question, "progress": 0, "conversation_id": conversation_id}
        )
        yield f"data: {{\"event\": {event.model_dump_json()}}}\n\n"
        
        # Create the task
        # In the streaming function, after retrieving the conversation
        if conversation_id and conversation:
            # Get recent messages for context
            recent_messages = session.query(Message).filter(
                Message.conversation_id == conversation_id
            ).order_by(Message.timestamp.desc()).limit(10).all()

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
                report_agent_messages = [resp for resp in all_agent_responses if resp['source'] == 'ReportAgent']
                
                if report_agent_messages:
                    # Use ReportAgent's responses
                    final_report = "\n\n".join([resp['content'] for resp in report_agent_messages])
                elif all_agent_responses:
                    # Fallback: use all substantial responses
                    final_report = "\n\n".join([f"**{resp['source']}**: {resp['content']}" for resp in all_agent_responses])
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
                            timestamp=datetime.now().isoformat()
                        )
                        session.add(assistant_message)
                        
                        # Update conversation timestamp
                        conversation = session.get(Conversation, conversation_id)
                        if conversation:
                            conversation.updated_at = datetime.now().isoformat()
                        
                        session.commit()
                        logger.info(f"Successfully saved assistant response to database for conversation {conversation_id}")
                        logger.info(f"Final report length: {len(final_report)} characters")
                        logger.info(f"Final report content: {final_report[:100]}...")  # Log first 100 chars
                    except Exception as db_error:
                        logger.error(f"Failed to save assistant response: {str(db_error)}")
                        session.rollback()
                    finally:
                        session.close()
                else:
                    logger.warning("Final report is empty, not saving to database")
                
                event = await event_manager.emit_event(
                    StreamEventType.COMPLETED,
                    agent_name="GraphManager",
                    message="Workflow completed successfully",
                    data={"progress": 100, "conversation_id": conversation_id, "final_response": final_report}
                )
                yield f"data: {{\"event\": {event.model_dump_json()}}}\n\n"
                break
                
            # Process regular messages
            if hasattr(message, 'source') and hasattr(message, 'content'):
                event_manager.increment_step()
                
                # Safely extract and process content
                content_str = ""
                content_type = "text"
                
                if isinstance(message.content, str):
                    content_str = message.content
                elif isinstance(message.content, list):
                    # Handle function calls or structured content
                    if message.content and hasattr(message.content[0], 'name'):
                        # This is likely a function call
                        content_type = "function_call"
                        func_names = [item.name for item in message.content if hasattr(item, 'name')]
                        content_str = f"Calling functions: {', '.join(func_names)}"
                    else:
                        content_str = str(message.content)
                else:
                    content_str = str(message.content)
                
                # Store ALL substantial agent responses (not just ReportAgent)
                if len(content_str.strip()) > 50 and "TERMINATE" not in content_str and content_type != "function_call":
                    clean_content = content_str.strip()
                    # Avoid duplicates
                    if not any(resp['content'] == clean_content for resp in all_agent_responses):
                        all_agent_responses.append({
                            'source': message.source,
                            'content': clean_content
                        })
                        logger.info(f"Collected response from {message.source}: {len(clean_content)} chars")
                
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
                            timestamp=datetime.now().isoformat()
                        )
                        session.add(assistant_message)
                        await session.commit()
                    except Exception as db_error:
                        logger.error(f"Failed to save final report: {str(db_error)}")
                        session.rollback()

                elif content_type == "function_call":
                    event_type = StreamEventType.TOOL_EXECUTION
                elif any(keyword in content_str.lower() for keyword in ["thinking", "planning", "analyzing", "processing"]):
                    event_type = StreamEventType.AGENT_THINKING
                else:
                    event_type = StreamEventType.AGENT_RESPONSE
                
                # Create display message - truncate if too long
                display_message = content_str[:100] + "..." if len(content_str) > 100 else content_str
                
                agent_responses = {
                    "CarbonAgent": "",
                    "PolicyAgent": "", 
                    "DataAnalysisAgent": ""
                }

                if message.source in agent_responses and len(content_str.strip()) > 50 and content_type != "function_call":
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
                    plot_pattern = r'(co2plot_[a-zA-Z0-9_\-]+\.png|carbon_[a-zA-Z0-9_\-]+\.png|plot_[a-zA-Z0-9_\-]+\.png)'
                    plots_found = re.findall(plot_pattern, content_str)
                    
                    if plots_found:
                        if supportive_content and supportive_content != "Carbon analysis in progress":
                            supportive_content += f"\n\nPlots: {', '.join(plots_found)}"
                        else:
                            supportive_content = f"Generated plots: {', '.join(plots_found)}"
                
                elif message.source == "PolicyAgent":
                    policy_response = agent_responses["PolicyAgent"]
                    if policy_response:
                        supportive_content = policy_response

                event = await event_manager.emit_event(
                    event_type,
                    agent_name=message.source,
                    message=display_message,
                    data={
                        "progress": min(event_manager.get_progress_percentage(), 95),
                        "content_type": content_type,
                        "conversation_id": conversation_id,
                        "full_content": content_str if len(content_str) < 5000 else None,
                        "context": supportive_content if len(supportive_content) < 5000 else None,
                    }
                )

                yield f"data: {{\"event\": {event.model_dump_json()}}}\n\n"
        
        # Final check - if no responses were collected during streaming, try to get them from the completed flow
        if not all_agent_responses:
            logger.warning("No responses collected during streaming, this might indicate an issue with the workflow")
        
    except Exception as e:
        logger.error(f"Error in streaming AutoGen task: {str(e)}")
        error_event = await event_manager.emit_event(
            StreamEventType.ERROR,
            message=f"Task failed: {str(e)}",
            data={"error": str(e)}
        )
        yield f"data: {{\"event\": {error_event.model_dump_json()}}}\n\n"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup
    logger.info("Starting AutoGen Business Insights API...")
    create_db_and_tables()
    try:
        await initialize_agents()
        logger.info("API startup completed successfully")
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down AutoGen Business Insights API...")

# Initialize FastAPI app
app = FastAPI(
    title="AutoGen Business Insights API",
    description="API for generating business insights reports using AutoGen agents",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)

@app.post("/ask-stream")
async def ask_stream_endpoint(request: QuestionRequest, current_user: User = Depends(get_current_user)):
    """
    Streaming endpoint that provides real-time updates during agent processing
    Requires authentication
    """
    from fastapi.responses import StreamingResponse
    
    async def generate_stream():
        event_manager = StreamEventManager()
        async for chunk in run_autogen_task_streaming(
            event_manager, 
            request.question, 
            current_user.id, 
            request.conversation_id 
        ):
            yield chunk
    
    return StreamingResponse(generate_stream(), media_type="text/event-stream", 
                           headers={"Cache-Control": "no-cache", "Connection": "keep-alive"})

@app.get("/conversations")
async def get_conversations(current_user: User = Depends(get_current_user), session: Session = Depends(get_session)):
    """Get all conversations for the authenticated user"""
    conversations = session.query(Conversation).filter(
        Conversation.user_id == current_user.id
    ).order_by(Conversation.updated_at.desc()).all()
    
    return {"conversations": conversations}

@app.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: int, current_user: User = Depends(get_current_user), session: Session = Depends(get_session)):
    """Get a specific conversation with all messages"""
    conversation = session.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == current_user.id
    ).first()
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    messages = session.query(Message).filter(
        Message.conversation_id == conversation_id
    ).order_by(Message.timestamp.asc()).all()
    
    return {
        "conversation": conversation,
        "messages": messages
    }

@app.post("/conversations/{conversation_id}/messages")
async def add_message_to_conversation(
    conversation_id: int, 
    request: QuestionRequest, 
    current_user: User = Depends(get_current_user)
):
    """Add a new message to an existing conversation"""
    from fastapi.responses import StreamingResponse
    
    # Verify conversation ownership
    session = next(get_session())
    conversation = session.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == current_user.id
    ).first()
    session.close()
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    async def generate_stream():
        event_manager = StreamEventManager()
        async for chunk in run_autogen_task_streaming(
            event_manager, 
            request.question, 
            current_user.id, 
            conversation_id
        ):
            yield chunk
    
    return StreamingResponse(generate_stream(), media_type="text/event-stream", 
                           headers={"Cache-Control": "no-cache", "Connection": "keep-alive"})

@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: int, current_user: User = Depends(get_current_user), session: Session = Depends(get_session)):
    """Delete a conversation and all its messages"""
    conversation = session.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == current_user.id
    ).first()
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    # Delete all messages first
    session.query(Message).filter(Message.conversation_id == conversation_id).delete()
    
    # Delete the conversation
    session.delete(conversation)
    session.commit()
    
    return {"message": "Conversation deleted successfully"}


@app.get("/health")
async def health_check():
    """Enhanced health check endpoint with system status."""
    try:
        # Check if agents are initialized
        agents_status = "initialized" if team_flow is not None else "not_initialized"
        
        # Get current timestamp
        current_time = datetime.now().isoformat()
        
        # Check OpenAI client status
        openai_client_status = "connected" if client is not None else "not_connected"
        
        # Test actual OpenAI API connectivity
        openai_api_status = "unknown"
        api_error_message = None
        
        if client is not None:
            try:
                # Make a minimal API call to test connectivity
                test_response = await client.create([{"role": "user", "content": "test"}])
                openai_api_status = "healthy"
            except Exception as api_error:
                error_str = str(api_error)
                if "429" in error_str or "quota" in error_str.lower():
                    openai_api_status = "quota_exceeded"
                elif "401" in error_str or "invalid" in error_str.lower():
                    openai_api_status = "invalid_key"
                elif "403" in error_str:
                    openai_api_status = "forbidden"
                else:
                    openai_api_status = "error"
                api_error_message = error_str
        
        # Check data directory and files
        data_dir_exists = os.path.exists("src/data")
        data_files_count = 0
        if data_dir_exists:
            try:
                data_files_count = len([f for f in os.listdir("src/data") if f.endswith('.json')])
            except:
                data_files_count = 0
        
        # Check if required environment variables are set
        api_key = os.getenv("AZURE_AI_API_KEY")
        api_key_configured = bool(api_key)
        
        # Determine overall status
        overall_status = "healthy"
        
        if openai_api_status in ["quota_exceeded", "invalid_key", "forbidden"]:
            overall_status = "warning"
        elif openai_api_status == "error":
            overall_status = "error"
        elif agents_status != "initialized" or openai_client_status != "connected":
            overall_status = "warning"
        
        # Create appropriate message
        if overall_status == "error":
            if openai_api_status == "quota_exceeded":
                message = "OpenAI API quota exceeded. Please check your billing details."
            elif openai_api_status == "invalid_key":
                message = "OpenAI API key is invalid. Please check your API key."
            elif openai_api_status == "forbidden":
                message = "OpenAI API access forbidden. Please check your API key permissions."
            else:
                message = "System error detected."
        elif overall_status == "warning":
            message = "Some components have issues but system is partially operational."
        else:
            message = "AutoGen Agents and Services are operational."
        
        return {
            "status": overall_status,
            "message": message,
            "timestamp": current_time,
            "components": {
                "agents_status": agents_status,
                "openai_client_status": openai_client_status,
                "openai_api_status": openai_api_status,
                "api_key_configured": api_key_configured,
                "data_directory_exists": data_dir_exists,
                "data_files_count": data_files_count
            },
            "api_error": api_error_message,
            "version": "1.0.0",
            "uptime_check": current_time
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "message": f"AutoGen Business Insights API encountered an error: {str(e)}"
        }
    
@app.get("/images/{image_name}")
async def get_image(image_name: str):
    """Get an image by name"""
    image_path = f"plots/{image_name}"
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(image_path)

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "AutoGen Business Insights API", "docs": "/docs", "client": "/client"}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",  
        host="0.0.0.0",
        port=8000,
        reload=True,
        timeout_keep_alive=300,  
        timeout_graceful_shutdown=30,
    )