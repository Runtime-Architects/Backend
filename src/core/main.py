import time
from datetime import datetime
import asyncio
import os
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from auth_routes import router as auth_router

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import uvicorn

# AutoGen imports
from autogen_agentchat.agents import AssistantAgent, MessageFilterAgent, MessageFilterConfig, PerSourceFilter
from autogen_agentchat.teams import DiGraphBuilder, GraphFlow
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from db import create_db_and_tables, get_session

# Pydantic models for API
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Azure OpenAI Configuration
AZURE_ENDPOINT = "https://runtime-architects-ai-hub-dev.cognitiveservices.azure.com/"
MODEL = "gpt-4o"
AZURE_DEPLOYMENT = "gpt-4o"
API_KEY = "KEY"
API_VERSION = "2024-12-01-preview"

# Global variables
client = None
team_flow = None

# Pydantic Models
class QuestionRequest(BaseModel):
    question: str

class MessageResponse(BaseModel):
    role: str
    content: str

class APIResponse(BaseModel):
    status: str
    message: MessageResponse

# Agent System Messages
planner_system_message = f"""You are the Planner Agent orchestrating a team of specialists. Your role is to decompose complex tasks into structured workflows with clear dependencies.

Your responsibilities:
1. Task Decomposition: Break objectives into atomic sub-tasks for:
   - Carbon Agent (emissions data retrieval): 
        - Has the access to emission tool which can retrieve carbon emissions and analyse the data to classify the data into low:[], medium:[], high:[]
   -Policy Agent (policy data retrieval):
        - Has the access to search tool which can retrieve policies and analyse them to decide and report them based on the query
   -Data Analysis Agent ():
        - Has the access to python executor tool, which can execute python scripts, which it uses to analyse data given by the user. Only use it if necesssary.
   - Report Agent (visualization and summarization): 
        - Has access to the python executor tool, which can execute python scripts. It summarises the data 

RULES:
- State the plan you are following clearly
- ONLY output what agents are to be invoked

The goal is to help energy-conscious consumers make sustainable choices by clear, actionable advice about electricity usage, renewable energy, and carbon reduction using markdown. 
"""

CARBON_AGENT_SYSMSG = f"""You are an intelligent assistant with access to specialized tools. Today's date and time is: {datetime.now()}.

### Available Tools:
- **PythonCodeExecutionTool**: For general programming tasks
- **Carbon Data Retriever**: Fetches raw CO2 intensity data (use when you need unprocessed data)
- **Daily Analyzer**: For analysis day/days (15 minute granularity)
- **Weekly Analyzer**: For analysis week/weeks (hourly granularity)
- **Monthly Analyzer**: For analysis of month/months (day granularity)

TOOL USAGE RULES:
- For CO2 intensity queries, ALWAYS use the emission_tool with these exact parameters:
  - Date format MUST be YYYY-MM-DD (e.g., '2025-06-24')
  - Region MUST be one of:
    * 'roi' for Republic of Ireland (Ireland)
    * 'ni' for Northern Ireland
    * 'all' for both Republic of Ireland (Ireland) & Northern Ireland

- **Time Period** determines which analyzer to use:
    - 1 day to 6 days → Daily Analyzer
    - 7 days to 21 days → Weekly Analyzer
    - greater than 21 days → Monthly Analyzer

- When using the tool:
1. Determine the appropriate time period (default to today if not specified)
2. Identify the region (default to 'all' if not specified)
3. Use the emission_tool to get current data
4. Analyze the results to provide a data-driven answer

### Response Format Guidelines:
1. Start with analysis type and time period covered
2. Show key findings with emojis (🌱 for low, ⚠️ for medium, 🔥 for high emissions)
3. Provide actionable recommendations
4. Include any notable trends or comparisons

When providing recommendations for the current day, always consider the current time. 
Only suggest activities or actions for future time slots—never for times that have already passed. 
For example, if it is currently 16:00, recommendations should only apply to 16:00 onward today.
"""

POLICY_AGENT_SYSMSG = '''
You are a SEAI Policy Agent that answers questions using only the provided SEAI documents.

INSTRUCTIONS:
- When a user asks a question, first identify key terms and synonyms that may appear in SEAI policy documents
- Use the run_curl_search tool to query those terms against SEAI search
- Reformulate queries if no relevant documents are found
- Answer questions using ONLY information from the returned documents
- If information is not in the documents, say: "I don't have that information in the available SEAI documents"
- Be conversational and helpful
- Don't mention JSON, technical details, or internal tools
- Cite relevant document titles when answering
'''

REPORT_AGENT_SYSMSG = f"""You are the Report Agent creating terminal-friendly visualizations. Today's date and time is: {datetime.now()}. You turn analysis into human-readable dashboards.

Your responsibilities:
1. Only use data processed from from CarbonAgent, and PolicyAgent
2. Include only the timings recieved from CarbonAgent while generating report or summaries. 
2. Create clear visualizations of the data
3. Generate summary insights and recommendations

TOOLS:
- python_executor: ONLY for creating visualizations from provided data

RULES:
- NEVER try to fetch raw data yourself 
- always use the processed data from CarbonAgent
- For visualization:
  - Use ASCII art for terminal display
  - Include clear labels and time periods
  - Add emoji indicators (🌱 for low, ⚠️ for medium, 🔥 for high)
  
EXAMPLE OUTPUT:
```ascii
CO2 Intensity Trend (ROI) - {datetime.now().date()}
┌─────────────────────────────────────┐
│  High (🔥) ███▄                      │
│ Medium (⚠️) █  █▄▄                   │
│  Low (🌱) █    ███▄▄▄               │
└─────────────────────────────────────┘
Best Time: 02:00-05:00 (🌱 Lowest Intensity)
Always include:

Date/period covered

Clear intensity classification

Specific usage recommendations 

When done, SAY "TERMINATE"
"""

data_analysis_agent_system_message = (
    "You are a data analyst responsible for synthesizing inputs from the Carbon Agent and Policy Agent. "
    "Analyze the relationship between emissions data and sustainability policies to identify trends, anomalies, and actionable insights. "
    "Present your findings clearly, using bullet points or tables if helpful, for use in a final report. "
)

def create_agent(name, system_message):
    """Helper function to create an AssistantAgent."""
    return AssistantAgent(
        name=name,
        model_client=client,
        system_message=system_message,
        model_client_stream=True,
    )

async def initialize_agents():
    """Initialize AutoGen agents and workflow."""
    global client, team_flow
    
    try:
        logger.info("Initializing Azure OpenAI client...")
        client = AzureOpenAIChatCompletionClient(
            azure_deployment=AZURE_DEPLOYMENT,
            model=MODEL,
            api_version=API_VERSION,
            azure_endpoint=AZURE_ENDPOINT,
            api_key=API_KEY,
            max_completion_tokens=1024,
        )
        
        logger.info("Creating agents...")
        # Create agents
        planner = create_agent("PlannerAgent", planner_system_message)
        carbon = create_agent("CarbonAgent", CARBON_AGENT_SYSMSG)
        policy = create_agent("PolicyAgent", POLICY_AGENT_SYSMSG)
        analysis = create_agent("DataAnalysisAgent", data_analysis_agent_system_message)
        report = create_agent("ReportAgent", REPORT_AGENT_SYSMSG)

        # Create filtered agents
        filtered_carbon = MessageFilterAgent(
            name="CarbonAgent",
            wrapped_agent=carbon,
            filter=MessageFilterConfig(per_source=[PerSourceFilter(source="PlannerAgent", position="last", count=1)]),
        )

        filtered_policy = MessageFilterAgent(
            name="PolicyAgent",
            wrapped_agent=policy,
            filter=MessageFilterConfig(per_source=[PerSourceFilter(source="PlannerAgent", position="last", count=1)]),
        )

        filtered_analysis = MessageFilterAgent(
            name="DataAnalysisAgent",
            wrapped_agent=analysis,
            filter=MessageFilterConfig(per_source=[PerSourceFilter(source="PlannerAgent", position="last", count=1)]),
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

async def run_autogen_task(question: str) -> str:
    """Run the AutoGen task and return the aggregated response"""
    if team_flow is None:
        raise HTTPException(status_code=500, detail="AutoGen agents not initialized")
    
    try:
        logger.info(f"Starting AutoGen task with question: {question[:100]}...")

        # Create the task directly for the planner agent
        task = f"Generate a comprehensive business insights report based on this request: {question}"

        # Run the team task - the planner agent will orchestrate everything
        result = await team_flow.run(task=task)

        # Extract the final aggregated response from the planner agent
        final_response = ""

        # Look for the planner's final aggregated response in the conversation
        for message in result.messages:
            if message.source == "PlannerAgent":
                # Check if this is the final comprehensive response (before TERMINATE)
                if any(
                    keyword in message.content
                    for keyword in [
                        "Executive Summary",
                        "Final Aggregated Response",
                        "COMPREHENSIVE FINAL RESPONSE",
                    ]
                ):
                    final_response = message.content
                    break

        # If no specific final response found, use the last substantial planner message before TERMINATE
        if not final_response:
            planner_messages = [
                msg for msg in result.messages if msg.source == "PlannerAgent"
            ]
            if planner_messages:
                # Find the last substantial message before TERMINATE
                for msg in reversed(planner_messages):
                    if "TERMINATE" not in msg.content and len(msg.content) > 100:
                        final_response = msg.content
                        break

        # Fallback: if still no response, aggregate all meaningful responses
        if not final_response:
            logger.warning(
                "No final aggregated response found from planner. Attempting to aggregate all responses."
            )
            all_responses = []
            for message in result.messages:
                if len(message.content) > 50 and "TERMINATE" not in message.content:
                    all_responses.append(f"**{message.source}**: {message.content}")
            final_response = (
                "\n\n".join(all_responses)
                if all_responses
                else "No meaningful response generated."
            )

        logger.info(
            f"AutoGen task completed successfully. Response length: {len(final_response)}"
        )
        return final_response

    except Exception as e:
        logger.error(f"Error in AutoGen task: {str(e)}")
        raise HTTPException(status_code=500, detail=f"AutoGen task failed: {str(e)}")

# Streaming event types and manager classes
from enum import Enum
from typing import Optional, Dict, Any

class StreamEventType(str, Enum):
    STARTED = "started"
    AGENT_THINKING = "agent_thinking"
    AGENT_RESPONSE = "agent_response"
    TOOL_EXECUTION = "tool_execution"
    COMPLETED = "completed"
    ERROR = "error"

class StreamEvent(BaseModel):
    event_type: StreamEventType
    agent_name: Optional[str] = None
    message: str
    timestamp: datetime
    data: Optional[Dict[str, Any]] = None

class StreamEventManager:
    def __init__(self):
        self.step_count = 0
        self.total_steps = 20  # Estimated total steps
        
    def increment_step(self):
        self.step_count += 1
        
    def get_progress_percentage(self) -> int:
        return min(int((self.step_count / self.total_steps) * 100), 95)
    
    async def emit_event(
        self,
        event_type: StreamEventType,
        agent_name: Optional[str] = None,
        message: str = "",
        data: Optional[Dict[str, Any]] = None
    ) -> StreamEvent:
        return StreamEvent(
            event_type=event_type,
            agent_name=agent_name,
            message=message,
            timestamp=datetime.now(),
            data=data or {}
        )

async def run_autogen_task_streaming(event_manager: StreamEventManager, question: str) -> AsyncGenerator[str, None]:
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
        logger.info(f"Starting streaming AutoGen task with question: {question[:100]}...")
        
        # Emit starting event
        event = await event_manager.emit_event(
            StreamEventType.STARTED,
            message=f"Starting analysis for: {question[:100]}...",
            data={"question": question, "progress": 0}
        )
        yield f"data: {{\"event\": {event.model_dump_json()}}}\n\n"
        
        # Create the task
        task = f"Generate a comprehensive business insights report based on this request: {question}"
        
        # Use AutoGen's streaming capability
        async for message in team_flow.run_stream(task=task):
            # Process real AutoGen messages
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
                
                # Determine event type based on content
                if "TERMINATE" in content_str:
                    event_type = StreamEventType.COMPLETED
                elif content_type == "function_call":
                    event_type = StreamEventType.TOOL_EXECUTION
                elif any(keyword in content_str.lower() for keyword in ["thinking", "planning", "analyzing", "processing"]):
                    event_type = StreamEventType.AGENT_THINKING
                else:
                    event_type = StreamEventType.AGENT_RESPONSE
                
                # Create display message truncate if too long
                display_message = content_str[:100] + "..." if len(content_str) > 100 else content_str
                
                event = await event_manager.emit_event(
                    event_type,
                    agent_name=message.source,
                    message=display_message,
                    data={
                        "progress": min(event_manager.get_progress_percentage(), 95),
                        "content_type": content_type
                    }
                )
                yield f"data: {{\"event\": {event.model_dump_json()}}}\n\n"
        
        # Final completion event
        event = await event_manager.emit_event(
            StreamEventType.COMPLETED,
            agent_name="PlannerAgent",
            message="Task completed successfully",
            data={"progress": 100}
        )
        yield f"data: {{\"event\": {event.model_dump_json()}}}\n\n"
        
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
@app.post("/ask", response_model=APIResponse)
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

@app.post("/ask-stream")
async def ask_stream_endpoint(request: QuestionRequest):
    """
    Streaming endpoint that provides real-time updates during agent processing
    """
    from fastapi.responses import StreamingResponse
    
    async def generate_stream():
        event_manager = StreamEventManager()
        async for chunk in run_autogen_task_streaming(event_manager, request.question):
            yield chunk
    
    return StreamingResponse(generate_stream(), media_type="text/event-stream", 
                           headers={"Cache-Control": "no-cache", "Connection": "keep-alive"})

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
        api_key_configured = bool(API_KEY and API_KEY.startswith(("sk-", "Bearer")))
        
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

@app.get("/test-stream")
async def test_stream():
    """Test streaming endpoint with mock events"""
    from fastapi.responses import StreamingResponse
    
    async def mock_stream():
        test_events = [
            "System: Test stream started",
            "TestAgent: Processing test request...",
            "TestAgent: Generated test response",
            "System: Test completed successfully"
        ]
        
        for event in test_events:
            yield f"data: {event}\n\n"
            await asyncio.sleep(1)
    
    return StreamingResponse(mock_stream(), media_type="text/plain")

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "AutoGen Business Insights API", "docs": "/docs", "client": "/client"}

@app.get("/client")
async def serve_client():
    """Serve the streaming client HTML"""
    try:
        return FileResponse("streaming_client.html", media_type="text/html")
    except FileNotFoundError:
        return {"message": "Client HTML file not found. Please ensure streaming_client.html exists."}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",  
        host="0.0.0.0",
        port=8000,
        reload=True,
        timeout_keep_alive=300,  
        timeout_graceful_shutdown=30,
    )