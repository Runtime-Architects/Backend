import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import logging
from contextlib import asynccontextmanager
import uvicorn
import queue
import threading
import time
from datetime import datetime
from typing import AsyncGenerator, Dict, Any, Optional, List
from enum import Enum
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.agents.openai import OpenAIAssistantAgent
from openai import AsyncOpenAI
from autogen_agentchat.conditions import (
    TextMentionTermination,
    MaxMessageTermination,
)
from autogen_agentchat.teams import SelectorGroupChat
from autogen_core.tools import FunctionTool
import sys
from co2_analysis import CO2IntensityAnalyzer
from run_eirgrid_downloader import main as eirgrid_main
import json
import os

# MCP imports
try:
    from mcp import types
    from mcp.server import Server
    from mcp.server.models import InitializationOptions
    from mcp.types import (
        Resource, 
        Tool, 
        TextContent, 
        ImageContent, 
        EmbeddedResource
    )
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("MCP not available. Install with: pip install mcp")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create a separate logger for streaming events
stream_logger = logging.getLogger("stream_events")
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter('%(asctime)s - STREAM - %(levelname)s - %(message)s'))
stream_logger.addHandler(stream_handler)
stream_logger.setLevel(logging.INFO)

# --- Configuration Variables ---
openai_api_key = "key"
openai_model = "gpt-4o-mini"
openai_carbon_assistant_id = ""
openai_analysis_assistant_id = ""
openai_report_assistant_id = ""

# Pydantic models
class QuestionRequest(BaseModel):
    question: str

class MessageResponse(BaseModel):
    role: str
    content: str

class APIResponse(BaseModel):
    status: str
    message: MessageResponse

class StreamEventType(str, Enum):
    STARTED = "started"
    AGENT_THINKING = "agent_thinking"
    AGENT_RESPONSE = "agent_response"
    TOOL_EXECUTION = "tool_execution"
    MCP_TOOL_CALL = "mcp_tool_call"
    ERROR = "error"
    COMPLETED = "completed"

class StreamEvent(BaseModel):
    event_type: StreamEventType
    timestamp: str
    agent_name: str = ""
    message: str = ""
    data: Dict[str, Any] = {}

class StreamResponse(BaseModel):
    event: StreamEvent
    message: MessageResponse

# MCP Tool Integration
class MCPToolIntegration:
    """Integration class to use MCP tools within AutoGen agents"""
    
    def __init__(self):
        self.available_tools: List[Dict[str, Any]] = []
        self.enabled = MCP_AVAILABLE
        if self.enabled:
            self._setup_tools()
    
    def _setup_tools(self):
        """Setup available MCP tools"""
        self.available_tools = [
            {
                "name": "get_emission_analysis",
                "description": "Get CO2 intensity analysis for specified date range and region",
                "schema": {
                    "type": "object",
                    "properties": {
                        "startdate": {"type": "string", "description": "Start date (YYYY-MM-DD)"},
                        "enddate": {"type": "string", "description": "End date (YYYY-MM-DD)"},
                        "region": {"type": "string", "enum": ["all", "roi", "ni"], "description": "Region to analyze"}
                    },
                    "required": ["startdate", "enddate", "region"]
                }
            },
            {
                "name": "generate_synthetic_data",
                "description": "Generate synthetic business data for analysis",
                "schema": {
                    "type": "object",
                    "properties": {
                        "data_type": {"type": "string", "enum": ["sales", "products", "emissions"], "description": "Type of data to generate"},
                        "period": {"type": "string", "description": "Time period for the data"},
                        "categories": {"type": "array", "items": {"type": "string"}, "description": "Data categories to include"}
                    },
                    "required": ["data_type", "period"]
                }
            },
            {
                "name": "analyze_business_data",
                "description": "Analyze business data and generate insights",
                "schema": {
                    "type": "object",
                    "properties": {
                        "data": {"type": "object", "description": "Data to analyze"},
                        "analysis_type": {"type": "string", "enum": ["trend", "comparison", "summary"], "description": "Type of analysis to perform"}
                    },
                    "required": ["data", "analysis_type"]
                }
            },
            {
                "name": "create_ascii_dashboard",
                "description": "Create ASCII dashboard from analyzed data",
                "schema": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "Dashboard title"},
                        "data": {"type": "object", "description": "Analyzed data for visualization"},
                        "chart_types": {"type": "array", "items": {"type": "string"}, "description": "Types of charts to include"}
                    },
                    "required": ["title", "data"]
                }
            }
        ]
    
    async def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get available MCP tools"""
        return self.available_tools
    
    async def call_mcp_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Call an MCP tool and return the result"""
        try:
            if tool_name == "get_emission_analysis":
                result = await self._get_emission_analysis(
                    arguments["startdate"],
                    arguments["enddate"], 
                    arguments["region"]
                )
            elif tool_name == "generate_synthetic_data":
                result = await self._generate_synthetic_data(
                    arguments["data_type"],
                    arguments["period"],
                    arguments.get("categories", [])
                )
            elif tool_name == "analyze_business_data":
                result = await self._analyze_business_data(
                    arguments["data"],
                    arguments["analysis_type"]
                )
            elif tool_name == "create_ascii_dashboard":
                result = await self._create_ascii_dashboard(
                    arguments["title"],
                    arguments["data"],
                    arguments.get("chart_types", ["bar"])
                )
            else:
                result = {"error": f"Unknown MCP tool: {tool_name}"}
            
            return json.dumps(result, indent=2)
                
        except Exception as e:
            error_result = {"error": f"MCP tool execution failed: {str(e)}"}
            return json.dumps(error_result, indent=2)
    
    async def _get_emission_analysis(self, startdate: str, enddate: str, region: str) -> Dict[str, Any]:
        """Get emission analysis using existing function"""
        try:
            result = await get_emission_analysis(startdate, enddate, region)
            return {"status": "success", "data": result, "tool": "get_emission_analysis"}
        except Exception as e:
            return {"status": "error", "message": str(e), "tool": "get_emission_analysis"}
    
    async def _generate_synthetic_data(self, data_type: str, period: str, categories: List[str]) -> Dict[str, Any]:
        """Generate synthetic data"""
        import random
        from datetime import datetime, timedelta
        
        # Generate synthetic data based on type
        if data_type == "sales":
            data = []
            start_date = datetime.now() - timedelta(days=30)
            for i in range(30):
                date = start_date + timedelta(days=i)
                data.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "sales": random.randint(1000, 5000),
                    "category": random.choice(categories or ["Electronics", "Clothing", "Food"])
                })
        elif data_type == "emissions":
            data = []
            for category in (categories or ["Transport", "Energy", "Manufacturing"]):
                data.append({
                    "category": category,
                    "co2_kg": random.randint(100, 1000),
                    "period": period
                })
        elif data_type == "products":
            data = []
            for category in (categories or ["Product A", "Product B", "Product C"]):
                data.append({
                    "product": category,
                    "revenue": random.randint(10000, 50000),
                    "units_sold": random.randint(100, 500),
                    "period": period
                })
        else:
            data = {"type": data_type, "period": period, "placeholder": True}
        
        return {"status": "success", "data": data, "type": data_type, "tool": "generate_synthetic_data"}
    
    async def _analyze_business_data(self, data: Dict[str, Any], analysis_type: str) -> Dict[str, Any]:
        """Analyze business data"""
        if analysis_type == "trend":
            # Simple trend analysis
            if isinstance(data.get("data"), list) and len(data["data"]) > 0:
                values = []
                for item in data["data"]:
                    if "sales" in item:
                        values.append(item["sales"])
                    elif "revenue" in item:
                        values.append(item["revenue"])
                    elif "co2_kg" in item:
                        values.append(item["co2_kg"])
                
                if values:
                    trend = "increasing" if values[-1] > values[0] else "decreasing"
                    return {
                        "analysis_type": "trend",
                        "trend": trend,
                        "avg_value": sum(values) / len(values),
                        "total_points": len(values),
                        "min_value": min(values),
                        "max_value": max(values),
                        "tool": "analyze_business_data"
                    }
        elif analysis_type == "summary":
            data_points = data.get("data", [])
            return {
                "analysis_type": "summary",
                "total_records": len(data_points),
                "data_structure": str(type(data_points)),
                "sample_keys": list(data_points[0].keys()) if data_points else [],
                "tool": "analyze_business_data"
            }
        
        return {
            "analysis_type": analysis_type, 
            "status": "analyzed", 
            "data_summary": str(data)[:200],
            "tool": "analyze_business_data"
        }
    
    async def _create_ascii_dashboard(self, title: str, data: Dict[str, Any], chart_types: List[str]) -> Dict[str, Any]:
        """Create ASCII dashboard"""
        
        # Extract key metrics from data
        metrics = {}
        if "trend" in data:
            metrics["Trend"] = data["trend"]
        if "avg_value" in data:
            metrics["Average"] = f"{data['avg_value']:.2f}"
        if "total_records" in data:
            metrics["Records"] = str(data["total_records"])
        
        dashboard = f"""
╔══════════════════════════════════════════════════════════════╗
║                        {title:^36}                        ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  📊 Dashboard Summary:                                       ║
║  • Analysis Type: {data.get('analysis_type', 'N/A'):<42} ║
║  • Status: {data.get('status', 'Generated'):<49} ║
"""
        
        # Add metrics
        for key, value in metrics.items():
            dashboard += f"║  • {key}: {value:<54} ║\n"
        
        dashboard += f"""║                                                              ║
║  📈 Chart Types: {', '.join(chart_types):<43} ║
║                                                              ║
║  Generated via MCP Tool Integration                          ║
║  Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S"):<45} ║
╚══════════════════════════════════════════════════════════════╝
"""
        
        return {
            "status": "success", 
            "dashboard": dashboard,
            "metrics": metrics,
            "tool": "create_ascii_dashboard"
        }

# Read the content of the Report template
def read_file() -> str:
    """Reads the content of the report_template.txt file."""
    try:
        with open("report_template.txt", "r") as file:
            return file.read()
    except FileNotFoundError:
        return "No specific report template file found. Proceed with a general business insights report structure."

# Enhanced system prompts with MCP integration
planner_sys_prompt = """
# **Planner Agent System Prompt: Enhanced with MCP (Model Context Protocol) Integration**

### **Role Definition**
You are the **Planning Agent**, orchestrating the creation of a Monthly Business Insights Report using both traditional agents and **MCP (Model Context Protocol) tools**. You coordinate between:

1.  **CarbonAgent**: Generates synthetic raw data tables (enhanced with MCP tools)
2.  **DataAnalysisAgent**: Analyzes data and generates insights (can use MCP tools)
3.  **ReportAgent**: Creates ASCII dashboard reports (enhanced with MCP tools)
4.  **MCP Tools**: Direct access to specialized functions via Model Context Protocol

### **MCP Integration**
You have access to MCP tools that can be called directly:
- `get_emission_analysis`: Real CO2 intensity data analysis
- `generate_synthetic_data`: Advanced synthetic data generation
- `analyze_business_data`: Business data analysis capabilities
- `create_ascii_dashboard`: Professional ASCII dashboard creation

### **Enhanced Workflow with MCP**

1.  **Task Planning & MCP Tool Assessment**:
    - Analyze the request to determine if MCP tools can enhance the workflow
    - Decide whether to use traditional agents, MCP tools, or a hybrid approach
    - For data generation: Consider MCP's `generate_synthetic_data` tool
    - For analysis: Consider MCP's `analyze_business_data` tool
    - For dashboards: Consider MCP's `create_ascii_dashboard` tool

2.  **Hybrid Execution Strategy**:
    - **Option A**: Pure MCP workflow for simple requests
    - **Option B**: Agent-MCP hybrid for complex analysis
    - **Option C**: Traditional agent workflow with MCP enhancement

3.  **MCP Tool Coordination**:
    - When using MCP tools, call them with appropriate parameters
    - Integrate MCP tool results with agent outputs
    - Ensure seamless data flow between MCP tools and agents

4.  **Final Response Aggregation**:
    - Combine outputs from all sources (agents + MCP tools)
    - Provide comprehensive final response including:
        * Executive Summary
        * Data Sources (agent-generated vs MCP-generated)
        * Analysis Results (hybrid insights)
        * Complete Dashboard
        * Recommendations
    - After final response, state "**TERMINATE**"

### **MCP Tool Usage Guidelines**
- Use MCP tools when they can provide higher quality or more efficient results
- Always specify tool parameters clearly
- Integrate MCP results with agent workflows seamlessly
- Log MCP tool usage for transparency

### **Enhanced Instructions**
- Assess each request for optimal MCP tool utilization
- Coordinate between traditional agents and MCP tools
- Provide clear attribution of data sources and analysis methods
- Ensure all outputs meet the same quality standards regardless of source

---

### **Context**
*Here is the report template for your action, enhanced with MCP capabilities*

"""

carbon_system_prompt = """
### **Role Definition**
You are the **CarbonAgent**, responsible for **generating synthetic raw data tables** with **MCP (Model Context Protocol) enhancement capabilities**.

### **Enhanced Capabilities with MCP**
- You can leverage MCP tools for advanced data generation
- Access to `generate_synthetic_data` MCP tool for sophisticated synthetic data
- Integration with `get_emission_analysis` for real emission data when needed
- Ability to combine real and synthetic data sources

### **Instructions**

1. **Enhanced Data Generation**:
    - Use MCP tools when they can provide better quality synthetic data
    - Generate structured, tabular data in Markdown format
    - Combine real emission data (via MCP) with synthetic business data
    - Clearly label data sources (synthetic vs real via MCP)

2. **MCP Tool Integration**:
    - Call MCP `generate_synthetic_data` for complex data requirements
    - Use `get_emission_analysis` when real emission data enhances the dataset
    - Specify data types, periods, and categories appropriately

3. **Quality Standards**:
    - Ensure all data (MCP-generated or traditional) meets analysis requirements
    - Provide clear data structure and labeling
    - Include metadata about data generation method

### **Tool Priority**:
1. MCP tools for advanced requirements
2. Traditional Carbon Footprint Estimator for simple emission data
3. Code interpreter for date/time context
"""

data_analysis_system_prompt = """
### **Role Definition**
You are the **DataAnalysisAgent** with **MCP (Model Context Protocol) enhanced analysis capabilities**.

### **MCP Integration**
- Access to `analyze_business_data` MCP tool for advanced analysis
- Can process both traditional agent data and MCP-generated data
- Enhanced analytical capabilities through MCP tool integration

### **Instructions**

1.  **Start Condition**:
    -   Process when explicitly instructed by PlanningAgent
    -   Can work with data from any source (agents or MCP tools)

2.  **Enhanced Analysis**:
    -   Use MCP `analyze_business_data` tool for complex analysis requirements
    -   Perform traditional analysis for standard requirements
    -   Combine insights from multiple analysis methods
    -   Generate comprehensive summaries incorporating all data sources

3.  **Output Standards**:
    -   Clear attribution of analysis methods used
    -   Integration of MCP tool results with traditional analysis
    -   Structured output ready for ReportAgent consumption
"""

report_system_prompt = """
### **Role Definition**
You are the **ReportAgent** with **MCP (Model Context Protocol) enhanced dashboard creation capabilities**.

### **MCP Integration**
- Access to `create_ascii_dashboard` MCP tool for professional dashboard generation
- Enhanced ASCII visualization capabilities
- Integration with analysis from multiple sources

### **Instructions**

1.  **Start Condition**:
    -   Process when explicitly instructed by PlanningAgent
    -   Work with analysis from any source (agents or MCP tools)

2.  **Enhanced Dashboard Creation**:
    -   Use MCP `create_ascii_dashboard` tool for sophisticated dashboards
    -   Traditional code_interpreter for custom ASCII graphics
    -   Combine multiple visualization approaches
    -   Ensure professional, readable ASCII output

3.  **Quality Standards**:
    -   Professional layout using ASCII characters
    -   Clear data visualization and summaries
    -   Attribution of all data sources and analysis methods
    -   Comprehensive "Prepared By" section including MCP tool usage
"""

async def get_emission_analysis(startdate: str, enddate: str, region: str) -> any:
    file_path = f'data/co2_intensity_all_{startdate.replace("-", "")}.json'

    def call_as_cli():
        # Simulate command line arguments
        sys.argv = [
            "run_eirgrid_downloader.py",
            "--areas",
            "co2_intensity",
            "--start",
            startdate,
            "--end",
            enddate,
            "--region",
            region,
            "--forecast",
            "--output-dir",
            "./data",
        ]
        return eirgrid_main()

    # Step 1: Try to load data from existing file
    try:
        if os.path.exists(file_path):
            with open(file_path, "r") as file:
                scraper_data = json.load(file)
        else:
            raise FileNotFoundError
    except (FileNotFoundError, json.JSONDecodeError):
        # Step 2: If failed, call the CLI-based scraper
        call_as_cli()
        try:
            with open(file_path, "r") as file:
                scraper_data = json.load(file)
        except Exception as e:
            raise Exception(
                "Failed to retrieve emission data even after scraping."
            ) from e

    # Step 3: Analyze data
    analyzer = CO2IntensityAnalyzer(scraper_data)
    intensity_periods = analyzer.get_combined_periods()

    return intensity_periods

# Create a function tool
emission_tool = FunctionTool(
    func=get_emission_analysis,
    description="Gets the CO2 intensity levels, for the specified start date and end date, the tool also takes regions all (Republic of Ireland & Northern Ireland), roi (Republic of Ireland), ni (Northern Ireland)",
)

# MCP Tool Function for agents
async def call_mcp_tool_function(tool_name: str, **kwargs) -> str:
    """Function that agents can call to use MCP tools"""
    if mcp_integration and mcp_integration.enabled:
        return await mcp_integration.call_mcp_tool(tool_name, kwargs)
    return json.dumps({"error": "MCP not available", "fallback": "using traditional methods"})

# Create properly typed functions for MCP tools
async def mcp_generate_synthetic_data(data_type: str, period: str, categories: Optional[List[str]] = None) -> str:
    """Generate synthetic business data using MCP tools"""
    return await call_mcp_tool_function(
        "generate_synthetic_data", 
        data_type=data_type, 
        period=period, 
        categories=categories or []
    )

async def mcp_analyze_business_data(data: Dict[str, Any], analysis_type: str) -> str:
    """Analyze business data using MCP tools"""
    return await call_mcp_tool_function(
        "analyze_business_data",
        data=data,
        analysis_type=analysis_type
    )

async def mcp_create_ascii_dashboard(title: str, data: Dict[str, Any], chart_types: Optional[List[str]] = None) -> str:
    """Create ASCII dashboard using MCP tools"""
    return await call_mcp_tool_function(
        "create_ascii_dashboard",
        title=title,
        data=data,
        chart_types=chart_types or ["bar"]
    )

# Create MCP function tools for agents with proper type annotations
mcp_generate_data_tool = FunctionTool(
    func=mcp_generate_synthetic_data,
    description="Generate synthetic business data using MCP tools (data_type: sales/products/emissions, period: time period, categories: list of categories)",
)

mcp_analyze_data_tool = FunctionTool(
    func=mcp_analyze_business_data,
    description="Analyze business data using MCP tools (data: data object, analysis_type: trend/comparison/summary)",
)

mcp_create_dashboard_tool = FunctionTool(
    func=mcp_create_ascii_dashboard,
    description="Create ASCII dashboard using MCP tools (title: dashboard title, data: analyzed data, chart_types: list of chart types)",
)

# Global variables for agents and MCP
openai_model_client = None
openai_client = None
team = None
mcp_integration: Optional[MCPToolIntegration] = None

class StreamEventManager:
    """Enhanced event manager with MCP tool tracking"""
    
    def __init__(self):
        self.events = []
        self.current_step = 0
        self.total_steps = 4
        self.mcp_calls = 0
        
    async def emit_event(self, event_type: StreamEventType, agent_name: str = "", 
                        message: str = "", data: Dict[str, Any] = None) -> StreamEvent:
        """Emit a streaming event and log it"""
        event = StreamEvent(
            event_type=event_type,
            timestamp=datetime.now().isoformat(),
            agent_name=agent_name,
            message=message,
            data=data or {}
        )
        
        self.events.append(event)
        
        # Track MCP tool calls
        if event_type == StreamEventType.MCP_TOOL_CALL:
            self.mcp_calls += 1
        
        # Log the event
        stream_logger.info(f"Event: {event_type} | Agent: {agent_name} | Message: {message}")
        
        return event
    
    def get_progress_percentage(self) -> int:
        """Calculate progress percentage"""
        return min(int((self.current_step / self.total_steps) * 100), 100)
    
    def increment_step(self):
        """Increment current step"""
        self.current_step += 1

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize agents and MCP on startup"""
    global openai_model_client, openai_client, team, mcp_integration

    logger.info("Initializing AutoGen agents and MCP integration...")

    # Initialize MCP integration
    mcp_integration = MCPToolIntegration()
    if mcp_integration.enabled:
        available_tools = await mcp_integration.get_available_tools()
        logger.info(f"MCP enabled with tools: {[tool['name'] for tool in available_tools]}")
    else:
        logger.warning("MCP not available - falling back to traditional tools only")

    # Create the OpenAI chat completion client
    openai_model_client = OpenAIChatCompletionClient(
        model=openai_model,
        api_key=openai_api_key,
        temperature=0,
    )

    # Create an OpenAI client (Async version for Agents API)
    openai_client = AsyncOpenAI(api_key=openai_api_key)

    # Create the planner agent with MCP-enhanced prompt
    planner_agent = AssistantAgent(
        "PlanningAgent",
        description="An agent for planning tasks with MCP tool integration, responsible for coordinating agents and MCP tools for optimal results.",
        model_client=openai_model_client,
        system_message=planner_sys_prompt + read_file(),
    )

    # Create enhanced agents with MCP tool access
    carbon_agent = OpenAIAssistantAgent(
        name="CarbonAgent",
        description="An agent responsible for data generation with MCP enhancement capabilities.",
        client=openai_client,
        model=openai_model,
        temperature=0,
        instructions=carbon_system_prompt,
        assistant_id=openai_carbon_assistant_id,
        tools=["code_interpreter", emission_tool, mcp_generate_data_tool],
        tool_resources={},
    )

    data_analysis_agent = OpenAIAssistantAgent(
        name="DataAnalysisAgent",
        description="An agent for data analysis with MCP tool integration.",
        client=openai_client,
        temperature=0,
        model=openai_model,
        instructions=data_analysis_system_prompt,
        assistant_id=openai_analysis_assistant_id,
        tools=["code_interpreter", mcp_analyze_data_tool],
    )

    report_agent = OpenAIAssistantAgent(
        name="ReportAgent",
        description="An agent for creating ASCII dashboards with MCP enhancement.",
        client=openai_client,
        temperature=0,
        model=openai_model,
        instructions=report_system_prompt,
        assistant_id=openai_report_assistant_id,
        tools=["code_interpreter", mcp_create_dashboard_tool],
    )

    # Define termination condition
    termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(
        max_messages=40
    )

    # Create team
    team = SelectorGroupChat(
        [planner_agent, carbon_agent, data_analysis_agent, report_agent],
        model_client=openai_model_client,
        termination_condition=termination,
    )

    logger.info("AutoGen agents and MCP integration initialized successfully")
    yield
    logger.info("Shutting down...")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="AutoGen API with MCP",
    description="API for generating business insights reports using AutoGen agents enhanced with Model Context Protocol (MCP)",
    version="2.0.0",
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

async def run_autogen_task(question: str) -> str:
    """Run the AutoGen task with MCP integration and return the aggregated response"""
    try:
        logger.info(f"Starting AutoGen task with MCP integration: {question[:100]}...")

        # Create the task with MCP awareness
        task = f"""Generate a comprehensive business insights report based on this request: {question}

Available MCP Tools:
{json.dumps([tool['name'] for tool in await mcp_integration.get_available_tools()], indent=2) if mcp_integration and mcp_integration.enabled else "None - using traditional tools only"}

Use MCP tools when they can enhance the analysis quality and efficiency."""

        # Run the team task
        result = await team.run(task=task)

        # Extract the final aggregated response
        final_response = ""

        # Look for the planner's final aggregated response
        for message in result.messages:
            if message.source == "PlanningAgent":
                if any(
                    keyword in message.content
                    for keyword in [
                        "Executive Summary",
                        "Final Aggregated Response",
                        "COMPREHENSIVE FINAL RESPONSE",
                        "MCP Integration Summary",
                    ]
                ):
                    final_response = message.content
                    break

        # Fallback logic (same as before)
        if not final_response:
            planner_messages = [
                msg for msg in result.messages if msg.source == "PlanningAgent"
            ]
            if planner_messages:
                for msg in reversed(planner_messages):
                    if "TERMINATE" not in msg.content and len(msg.content) > 100:
                        final_response = msg.content
                        break

        if not final_response:
            all_responses = []
            for message in result.messages:
                if len(message.content) > 50 and "TERMINATE" not in message.content:
                    all_responses.append(f"**{message.source}**: {message.content}")
            final_response = (
                "\n\n".join(all_responses)
                if all_responses
                else "No meaningful response generated."
            )

        # Add MCP usage summary if available
        if mcp_integration and mcp_integration.enabled:
            mcp_summary = f"\n\n---\n**MCP Integration Summary:**\n- MCP Tools Available: {len(await mcp_integration.get_available_tools())}\n- Enhanced with Model Context Protocol capabilities\n- Hybrid agent-MCP workflow execution"
            final_response += mcp_summary

        logger.info(f"AutoGen task with MCP completed successfully. Response length: {len(final_response)}")
        return final_response

    except Exception as e:
        logger.error(f"Error in AutoGen task with MCP: {str(e)}")
        raise HTTPException(status_code=500, detail=f"AutoGen task failed: {str(e)}")

async def run_autogen_task_streaming(question: str, event_manager: StreamEventManager) -> AsyncGenerator[str, None]:
    """Run the AutoGen task with MCP integration and streaming events"""
    try:
        logger.info(f"Starting streaming AutoGen task with MCP: {question[:100]}...")
        
        # Emit starting event
        event = await event_manager.emit_event(
            StreamEventType.STARTED,
            message=f"Starting MCP-enhanced analysis for: {question[:100]}...",
            data={
                "question": question, 
                "progress": 0, 
                "mcp_enabled": mcp_integration.enabled if mcp_integration else False
            }
        )
        yield f"data: {{\"event\": {event.model_dump_json()}}}\n\n"
        await asyncio.sleep(0.5)
        
        # Emit MCP tool availability event
        if mcp_integration and mcp_integration.enabled:
            tools = await mcp_integration.get_available_tools()
            event = await event_manager.emit_event(
                StreamEventType.MCP_TOOL_CALL,
                agent_name="MCPIntegration",
                message=f"MCP tools available: {', '.join([tool['name'] for tool in tools])}",
                data={"available_tools": [tool['name'] for tool in tools], "progress": 5}
            )
            yield f"data: {{\"event\": {event.model_dump_json()}}}\n\n"
            await asyncio.sleep(0.5)
        
        # Emit planner thinking event
        event_manager.increment_step()
        event = await event_manager.emit_event(
            StreamEventType.AGENT_THINKING,
            agent_name="PlanningAgent",
            message="Planning MCP-enhanced task coordination...",
            data={"progress": event_manager.get_progress_percentage()}
        )
        yield f"data: {{\"event\": {event.model_dump_json()}}}\n\n"
        await asyncio.sleep(1)

        # Create the enhanced task
        task = f"""Generate a comprehensive business insights report based on this request: {question}

Available MCP Tools:
{json.dumps([tool['name'] for tool in await mcp_integration.get_available_tools()], indent=2) if mcp_integration and mcp_integration.enabled else "None - using traditional tools only"}

Use MCP tools when they can enhance the analysis quality and efficiency."""
        
        # Enhanced simulation steps
        simulation_steps = [
            ("CarbonAgent", "Generating data with MCP enhancement..."),
            ("MCPIntegration", "Executing synthetic data generation tool"),
            ("CarbonAgent", "Data tables generated successfully"),
            ("DataAnalysisAgent", "Analyzing data with MCP tools..."),
            ("MCPIntegration", "Running business data analysis"),
            ("DataAnalysisAgent", "Analysis completed with enhanced insights"),
            ("ReportAgent", "Creating ASCII dashboard with MCP tools..."),
            ("MCPIntegration", "Generating professional dashboard"),
            ("ReportAgent", "Enhanced ASCII dashboard completed"),
            ("PlanningAgent", "Aggregating MCP-enhanced results...")
        ]
        
        # Start the AutoGen team task in background
        result_task = asyncio.create_task(team.run(task=task))
        
        step_delay = 1.5  # Faster with MCP efficiency
        for i, (agent, message) in enumerate(simulation_steps):
            if not result_task.done():
                if agent == "MCPIntegration":
                    event_type = StreamEventType.MCP_TOOL_CALL
                else:
                    event_type = StreamEventType.AGENT_RESPONSE
                
                event_manager.increment_step()
                event = await event_manager.emit_event(
                    event_type,
                    agent_name=agent,
                    message=message,
                    data={"progress": min(event_manager.get_progress_percentage(), 95)}
                )
                yield f"data: {{\"event\": {event.model_dump_json()}}}\n\n"
                await asyncio.sleep(step_delay)
            else:
                break

        # Wait for the actual result
        result = await result_task
        
        # Process the final response (same logic as non-streaming version)
        final_response = ""
        agent_responses = {}
        
        for message in result.messages:
            agent_name = message.source
            if agent_name not in agent_responses:
                agent_responses[agent_name] = []
            agent_responses[agent_name].append(message.content)
            stream_logger.info(f"Agent {agent_name} contributed {len(message.content)} characters")

        # Extract final response
        for message in result.messages:
            if message.source == "PlanningAgent":
                if any(
                    keyword in message.content
                    for keyword in [
                        "Executive Summary",
                        "Final Aggregated Response",
                        "COMPREHENSIVE FINAL RESPONSE",
                        "MCP Integration Summary",
                    ]
                ):
                    final_response = message.content
                    break

        if not final_response:
            planner_messages = [
                msg for msg in result.messages if msg.source == "PlanningAgent"
            ]
            if planner_messages:
                for msg in reversed(planner_messages):
                    if "TERMINATE" not in msg.content and len(msg.content) > 100:
                        final_response = msg.content
                        break

        if not final_response:
            all_responses = []
            for message in result.messages:
                if len(message.content) > 50 and "TERMINATE" not in message.content:
                    all_responses.append(f"**{message.source}**: {message.content}")
            final_response = (
                "\n\n".join(all_responses)
                if all_responses
                else "No meaningful response generated."
            )

        # Add MCP summary
        if mcp_integration and mcp_integration.enabled:
            mcp_summary = f"\n\n---\n**MCP Integration Summary:**\n- MCP Tools Available: {len(await mcp_integration.get_available_tools())}\n- MCP Tool Calls: {event_manager.mcp_calls}\n- Enhanced with Model Context Protocol capabilities"
            final_response += mcp_summary

        # Emit completion event
        event = await event_manager.emit_event(
            StreamEventType.COMPLETED,
            agent_name="PlanningAgent",
            message="MCP-enhanced task completed successfully",
            data={
                "progress": 100,
                "final_response": final_response,
                "total_messages": len(result.messages),
                "agent_count": len(agent_responses),
                "response_length": len(final_response),
                "mcp_calls": event_manager.mcp_calls,
                "mcp_enabled": mcp_integration.enabled if mcp_integration else False
            }
        )
        yield f"data: {{\"event\": {event.model_dump_json()}}}\n\n"

        logger.info(f"Streaming AutoGen task with MCP completed. Response length: {len(final_response)}")
        
    except Exception as e:
        logger.error(f"Error in streaming AutoGen task with MCP: {str(e)}")
        stream_logger.error(f"MCP streaming task failed: {str(e)}")
        
        error_event = await event_manager.emit_event(
            StreamEventType.ERROR,
            message=f"MCP-enhanced task failed: {str(e)}",
            data={"error": str(e), "progress": event_manager.get_progress_percentage()}
        )
        yield f"data: {{\"event\": {error_event.model_dump_json()}}}\n\n"

@app.post("/ask", response_model=APIResponse)
async def ask_endpoint(request: QuestionRequest):
    """
    Main endpoint to process questions with MCP enhancement
    """
    try:
        logger.info(f"Received question for MCP-enhanced processing: {request.question[:100]}...")

        # Run the AutoGen task with MCP integration
        response_content = await run_autogen_task(request.question)

        return APIResponse(
            status="Success",
            message=MessageResponse(role="assistant", content=response_content),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in MCP-enhanced ask endpoint: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {str(e)}"
        )

@app.post("/ask-stream")
async def ask_stream_endpoint(request: QuestionRequest):
    """
    Streaming endpoint with MCP integration
    """
    async def generate_stream():
        event_manager = StreamEventManager()
        
        try:
            stream_logger.info(f"Starting MCP-enhanced streaming request: {request.question[:100]}...")
            
            # Create the streaming generator with MCP
            async for event_data in run_autogen_task_streaming(request.question, event_manager):
                yield event_data
                
        except Exception as e:
            logger.error(f"Error in MCP streaming endpoint: {str(e)}")
            stream_logger.error(f"MCP streaming failed: {str(e)}")
            
            # Send error event
            error_event = await event_manager.emit_event(
                StreamEventType.ERROR,
                message=f"MCP stream failed: {str(e)}",
                data={"error": str(e)}
            )
            yield f"data: {{\"event\": {error_event.model_dump_json()}}}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
        }
    )

# MCP-specific endpoints
@app.get("/mcp/status")
async def mcp_status():
    """Get MCP integration status"""
    if mcp_integration:
        tools = await mcp_integration.get_available_tools()
        return {
            "mcp_enabled": mcp_integration.enabled,
            "available_tools": [tool['name'] for tool in tools],
            "total_tools": len(tools),
            "status": "operational" if mcp_integration.enabled else "disabled"
        }
    return {"mcp_enabled": False, "status": "not_initialized"}

@app.get("/mcp/tools")
async def list_mcp_tools():
    """List available MCP tools"""
    if mcp_integration and mcp_integration.enabled:
        tools = await mcp_integration.get_available_tools()
        return {"tools": tools, "status": "success"}
    return {"tools": [], "status": "mcp_not_available"}

@app.post("/mcp/execute")
async def execute_mcp_tool_endpoint(request: dict):
    """Execute an MCP tool directly"""
    tool_name = request.get("tool_name")
    arguments = request.get("arguments", {})
    
    if not tool_name:
        raise HTTPException(status_code=400, detail="tool_name is required")
    
    if mcp_integration and mcp_integration.enabled:
        result = await mcp_integration.call_mcp_tool(tool_name, arguments)
        return {"result": result, "status": "success", "tool_used": tool_name}
    
    raise HTTPException(status_code=503, detail="MCP not available")

@app.get("/health")
async def health_check():
    """Enhanced health check with MCP status"""
    try:
        # Existing health checks
        agents_status = "initialized" if team is not None else "not_initialized"
        current_time = datetime.now().isoformat()
        openai_client_status = "connected" if openai_client is not None else "not_connected"
        
        # MCP-specific checks
        mcp_status = "not_available"
        mcp_tools_count = 0
        if mcp_integration:
            if mcp_integration.enabled:
                mcp_status = "operational"
                tools = await mcp_integration.get_available_tools()
                mcp_tools_count = len(tools)
            else:
                mcp_status = "disabled"
        
        # Data directory checks
        data_dir_exists = os.path.exists("src/data")
        data_files_count = 0
        if data_dir_exists:
            try:
                data_files_count = len([f for f in os.listdir("src/data") if f.endswith('.json')])
            except:
                data_files_count = 0
        
        api_key_configured = bool(openai_api_key and openai_api_key.startswith("sk-"))
        
        # Determine overall status
        overall_status = "healthy"
        if agents_status != "initialized" or openai_client_status != "connected":
            overall_status = "degraded"
        
        return {
            "status": overall_status, 
            "message": "AutoGen Agents and MCP Services are operational.",
            "timestamp": current_time,
            "components": {
                "agents_status": agents_status,
                "openai_client_status": openai_client_status,
                "mcp_status": mcp_status,
                "mcp_tools_available": mcp_tools_count,
                "api_key_configured": api_key_configured,
                "data_directory_exists": data_dir_exists,
                "data_files_count": data_files_count
            },
            "version": "2.0.0",
            "features": ["AutoGen Agents", "MCP Integration", "Streaming API"],
            "uptime_check": current_time
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy", 
            "message": f"AutoGen Business Insights API with MCP encountered an error: {str(e)}"
        }

@app.get("/test-stream")
async def test_stream():
    """Test streaming endpoint with MCP events"""
    async def generate_test_stream():
        event_manager = StreamEventManager()
        
        test_events = [
            (StreamEventType.STARTED, "System", "MCP test stream started"),
            (StreamEventType.MCP_TOOL_CALL, "MCPIntegration", "Testing MCP tool availability"),
            (StreamEventType.AGENT_THINKING, "TestAgent", "Processing with MCP enhancement..."),
            (StreamEventType.AGENT_RESPONSE, "TestAgent", "Generated MCP-enhanced response"),
            (StreamEventType.COMPLETED, "System", "MCP test completed successfully")
        ]
        
        for event_type, agent, message in test_events:
            event = await event_manager.emit_event(event_type, agent, message)
            yield f"data: {{\"event\": {event.model_dump_json()}}}\n\n"
            await asyncio.sleep(1)
    
    return StreamingResponse(
        generate_test_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

@app.get("/")
async def root():
    """Root endpoint with MCP information"""
    mcp_status = "enabled" if (mcp_integration and mcp_integration.enabled) else "disabled"
    return {
        "message": "AutoGen Business Insights API with MCP Integration", 
        "version": "2.0.0",
        "mcp_status": mcp_status,
        "docs": "/docs", 
        "client": "/client",
        "mcp_tools": "/mcp/tools"
    }

@app.get("/client")
async def serve_client():
    """Serve the streaming client HTML"""
    return FileResponse("streaming_client.html", media_type="text/html")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        timeout_keep_alive=300,
        timeout_graceful_shutdown=30,
    )
