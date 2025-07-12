import asyncio
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_ext.tools.code_execution import PythonCodeExecutionTool
from autogen_agentchat.conditions import (
    TextMentionTermination,
    MaxMessageTermination,
)
from autogen_core.tools import FunctionTool
from co2_analysis_tool import (co2_analysis_daily, co2_analysis_weekly,
                               co2_analysis_monthly)
from scraper_tools.run_eirgrid_downloader import main as eirgrid_main
from datetime import datetime
import sys
import os
from dotenv import load_dotenv
load_dotenv()

#client for agents and selector_chat
model_client = AzureOpenAIChatCompletionClient(
                    azure_deployment=os.getenv("AZURE_DEPLOYMENT"),
                    model=os.getenv("MODEL"),
                    api_version=os.getenv("API_VERSION"),
                    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
                    api_key= os.getenv("API_KEY"), # For key-based authentication.
                )


# =====tools======
executor= LocalCommandLineCodeExecutor(work_dir="coding")
python_tool = PythonCodeExecutionTool(executor)

async def get_emission_analysis(startdate: str, enddate: str, region: str) -> float:
    # Check if the file exists
    file_path = f'data/co2_intensity/co2_intensity_{region}_{startdate}_{enddate}.json'

    if os.path.exists(file_path):
        return {
            "result": "Success, Data scraped.",
            "file_location": file_path
        }
    
    else:
        def call_as_cli():
            # Simulate command line arguments
            sys.argv = [
                'run_eirgrid_downloader.py',
                '--areas', 'co2_intensity',
                '--start', startdate,
                '--end', enddate,
                '--region', region,
                '--forecast',
                '--output-dir', './data'
            ]
            
            # Run the main function
            return eirgrid_main()
        
        call_as_cli()
        
        # Try to load the file again after calling the CLI
        if os.path.exists(file_path):
            return {
            "result": "Success, Data scraped.",
            "file_location": file_path
            }
        else:
            raise Exception(f"Failed to fetch data from scrapper. File not found at {file_path}")
        

# Create a function tool
emission_tool = FunctionTool(
    func=get_emission_analysis,
    description="Gets the CO2 intensity levels. Parameters: startdate (YYYY-MM-DD), enddate (YYYY-MM-DD), region ('all', 'roi', or 'ni')",
    name= "emission_tool"
)


async def analyze_daily_co2(file_path: str) -> dict:
    """Tool 2: Daily Data Analyzer - Analyzes CO2 data with daily granularity"""
    
    analysis = co2_analysis_daily.get_daily_analysis(file_path)
    return {
        "analysis_type": "daily",
        "results": analysis,
    }

daily_analyzer_tool = FunctionTool(
    func=analyze_daily_co2,
    description="Analyzes CO2 data with daily granularity. Best for periods <=6 days. Parameters: startdate, enddate, region",
    name= "analyze_daily_co2"
)


async def analyze_weekly_co2(file_path: str) -> dict:
    """Tool 3: Weekly Data Analyzer - Analyzes CO2 data with hourly granularity"""

    analysis = co2_analysis_weekly.get_weekly_analysis(file_path)
    return {
        "analysis_type": "weekly",
        "results": analysis,
    }

weekly_analyzer_tool = FunctionTool(
    func=analyze_weekly_co2,
    description="Analyzes CO2 data with weekly granularity. Best for periods >=7 days but <= 21 days. Parameters: startdate, enddate, region",
    name= "analyze_weekly_co2"
)



async def analyze_monthly_co2(file_path: str) -> dict:
    """Tool 4: Monthly Data Analyzer - Analyzes CO2 data with daily granularity"""

    analysis = co2_analysis_monthly.get_monthly_analysis(file_path)
    return {
        "analysis_type": "monthly",
        "results": analysis,
    }

monthly_analyzer_tool = FunctionTool(
    func=analyze_monthly_co2,
    description="Analyzes CO2 data with monthly granularity. Best for periods >=31 days. Parameters: startdate, enddate, region",
    name= "analyze_monthly_co2"
)


# =====system prompts======

planner_system_message = f"""You are the Planner Agent orchestrating a team of specialists. Your role is to decompose complex tasks into structured workflows with clear dependencies.

Your responsibilities:
1. Task Decomposition: Break objectives into atomic sub-tasks for:
   - Carbon Agent (emissions data retrieval): 
        - Has the access to emission tool which can retrieve carbon emissions and analyse the data to classify the data into low:[], medium:[], high:[]
   - Report Agent (visualization and summazrization): 
        - Has access to the python executor tool, which can execute python scripts.

RULES:
- State the plan you are following clearly

The goal is to help energy-conscious consumers make sustainable choices by clear, actionable advice about electricity usage, renewable energy, and carbon reduction using markdown. 
"""

carbon_system_message= f"""You are an intelligent assistant with access to specialized tools. Today's date and time is: {datetime.now()}.

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

report_system_message= system_message = report_system_message = f"""You are the Report Agent creating terminal-friendly visualizations. Today's date and time is: {datetime.now()}. You turn analysis into human-readable dashboards.

Your responsibilities:
1. Only use data processed from from CarbonAgent
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
"""""

# ====== Agents =======

planning_agent = AssistantAgent(
    "PlanningAgent",
    description="An agent for planning tasks, this agent should be the first to engage when given a new task.",
    model_client=model_client,
    system_message= planner_system_message
)

user_proxy = UserProxyAgent(
    name="User",
)


carbon_agent = AssistantAgent(
            name="assistant", model_client=model_client, 
            tools=[emission_tool, daily_analyzer_tool, 
                   weekly_analyzer_tool, monthly_analyzer_tool ], 
            reflect_on_tool_use=True,
            max_tool_iterations= 5,
            system_message= carbon_system_message
        )


report_agent = AssistantAgent(
    "ReportAgent",
    description="An agent responsible for generating the final dashboard report with text and graphs.",
    tools=[python_tool],
    model_client=model_client,
    reflect_on_tool_use=True,
    system_message= carbon_system_message
)



# Define a termination condition that stops the task if the planner accepts the final report and says TERMINATE.
termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(max_messages=40) # Increased max messages for more complex flow

selector_prompt = """Select an agent to perform task.

{roles}

Current conversation context:
{history}

Read the above conversation, then select an agent from {participants} to perform the next task.
Make sure the planner agent has assigned tasks before other agents start working.
Only select one agent.
"""


# Create a team to collaborate and accomplish the task
team = SelectorGroupChat(
    [planning_agent, user_proxy, carbon_agent, report_agent],
    model_client=model_client,
    termination_condition=termination,
)

async def main():
    # task = "What are the effects of CO2 on the environment"
    task = "What is the best time to use my appliances today in Ireland?"    
    await Console(team.run_stream(task=task))
    print("********* Dashboard Report Generation Complete ****************")

if __name__ == "__main__":
    asyncio.run(main())

