import os
import sys
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_ext.tools.code_execution import PythonCodeExecutionTool
from autogen_core.tools import FunctionTool
from autogen_agentchat.ui import Console
from datetime import datetime
import asyncio
from co2_analysis_tool import (co2_analysis_daily, co2_analysis_weekly,
                               co2_analysis_monthly)
from scraper_tools.run_eirgrid_downloader import main as eirgrid_main
import json

from dotenv import load_dotenv

load_dotenv()

import time

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


async def analyze_daily_co2(file_path: str) -> dict:
    """Tool 2: Daily Data Analyzer - Analyzes CO2 data with daily granularity"""
    
    analysis = co2_analysis_daily.get_daily_analysis(file_path)
    return {
        "analysis_type": "daily",
        "results": analysis,
    }

async def analyze_weekly_co2(file_path: str) -> dict:
    """Tool 3: Weekly Data Analyzer - Analyzes CO2 data with hourly granularity"""

    analysis = co2_analysis_weekly.get_weekly_analysis(file_path)
    return {
        "analysis_type": "weekly",
        "results": analysis,
    }


async def analyze_monthly_co2(file_path: str) -> dict:
    """Tool 4: Monthly Data Analyzer - Analyzes CO2 data with daily granularity"""

    analysis = co2_analysis_monthly.get_monthly_analysis(file_path)
    return {
        "analysis_type": "monthly",
        "results": analysis,
    }


daily_analyzer_tool = FunctionTool(
    func=analyze_daily_co2,
    description="Analyzes CO2 data with daily granularity. Best for periods <=6 days. Parameters: startdate, enddate, region",
    name= "analyze_daily_co2"
)

weekly_analyzer_tool = FunctionTool(
    func=analyze_weekly_co2,
    description="Analyzes CO2 data with weekly granularity. Best for periods >=7 days but <= 21 days. Parameters: startdate, enddate, region",
    name= "analyze_weekly_co2"
)

monthly_analyzer_tool = FunctionTool(
    func=analyze_monthly_co2,
    description="Analyzes CO2 data with monthly granularity. Best for periods >=31 days. Parameters: startdate, enddate, region",
    name= "analyze_monthly_co2"
)

# Create a function tool
emission_tool = FunctionTool(
    func=get_emission_analysis,
    description="Gets the CO2 intensity levels. Parameters: startdate (YYYY-MM-DD), enddate (YYYY-MM-DD), region ('all', 'roi', or 'ni')",
    name= "emission_tool"
)

#assistant system message
system_message = f"""You are an intelligent assistant with access to specialized tools. Today's date and time is: {datetime.now()}.

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




async def main() -> None:
    async with LocalCommandLineCodeExecutor(work_dir="coding") as executor:
        tool = PythonCodeExecutionTool(executor)

        client = AzureOpenAIChatCompletionClient(
                    azure_deployment=os.getenv("AZURE_DEPLOYMENT"),
                    model=os.getenv("MODEL"),
                    api_version=os.getenv("API_VERSION"),
                    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
                    api_key= os.getenv("API_KEY"), # For key-based authentication.
                )


        agent = AssistantAgent(
            name="assistant", model_client=client, tools=[emission_tool, daily_analyzer_tool, 
                                                          weekly_analyzer_tool, monthly_analyzer_tool ], 
            reflect_on_tool_use=True,
            max_tool_iterations= 5,
            system_message= system_message
        )

        await Console(
            agent.run_stream(
                task=f"Based on last 2 weeks data, What is the best time to use my appliances today in Ireland?"
            )
        )

if __name__=='__main__':
    start= time.time()
    asyncio.run(main())
    print(f'Time taken: {time.time() - start}')