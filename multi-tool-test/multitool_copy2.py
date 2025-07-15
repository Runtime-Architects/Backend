from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_ext.tools.code_execution import PythonCodeExecutionTool
from autogen_core.tools import FunctionTool
from autogen_agentchat.conditions import TextMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from datetime import datetime, timedelta
import asyncio
import os
from dotenv import load_dotenv
import sys
from co2_analysis import CO2IntensityAnalyzer
from run_eirgrid_downloader import main as eirgrid_main
import json

load_dotenv()

# Tool 1: Data Fetcher
async def fetch_emission_data(startdate: str, enddate: str, region: str) -> dict:
    """Fetches raw CO2 intensity data from the source.
    Parameters:
        startdate (YYYY-MM-DD), enddate (YYYY-MM-DD), 
        region ('all', 'roi', or 'ni')
    Returns raw JSON data."""

    file_path = f'data/co2_intensity/co2_intensity_{region}_{startdate}_{enddate}.json'
    
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
             
    except FileNotFoundError:
        # File doesn't exist, call the CLI function
        def call_as_cli():
            sys.argv = [
                'run_eirgrid_downloader.py',
                '--areas', 'co2_intensity',
                '--start', startdate,
                '--end', enddate,
                '--region', region,
                '--forecast',
                '--output-dir', './data'
            ]
            return eirgrid_main()
        
        call_as_cli()
        
        try:
            with open(file_path, 'r') as file:
                return json.load(file)
            # json.load(file)
        except FileNotFoundError:
            raise Exception(f"Failed to fetch data from scrapper. File not found at {file_path}")

# Define the async tools
async def get_current_data() -> str:
    """Get the current weather in a given location"""
    return f"Data is fetched successfully, follow the next step."

async def get_data_recommendations(location: str) -> str:
    """Get restaurant recommendations in a given location"""
    return f"Today's {location}, time to charge is great"

async def main():
    # Initialize the Azure OpenAI client
    client = AzureOpenAIChatCompletionClient(
        azure_deployment=os.getenv("AZURE_DEPLOYMENT"),
        model=os.getenv("MODEL"),
        api_version=os.getenv("API_VERSION"),
        azure_endpoint=os.getenv("AZURE_ENDPOINT"),
        api_key=os.getenv("API_KEY"),
    )

    # Create tool objects
    weather_tool = FunctionTool(
        func=get_current_data,
        description="Get current data for emission, date format: 'YYYY-MM-DD', region:'roi/ni/all'",
        name="get_current_weather"
    )

    restaurant_tool = FunctionTool(
        func=get_data_recommendations,
        description="Get recommendations usage by location",
        name="get_restaurant_recommendations"
    )

    # Create assistant agent with both tools
    assistant = AssistantAgent(
        "assistant",
        model_client=client,
        tools=[weather_tool, restaurant_tool],
        system_message=f"""You are a helpful AI assistant that can:
        - Get current co2 data
        - Recommend usage by location
        Today is {datetime.now().strftime('%Y-%m-%d')}
        Reply "TERMINATE" when the task is done.""",
        reflect_on_tool_use=True,
        max_tool_iterations=10
    )

    # Termination condition
    termination_condition = TextMessageTermination("assistant")

    # Create the team
    team = RoundRobinGroupChat(
        [assistant],
        termination_condition=termination_condition,
    )

    # Start the conversation
    task = "When can I charge today, in ireland"
    stream = team.run_stream(task=task)
    
    # Use Console to display the conversation
    await Console(stream)
    
    # Close the client when done
    await client.close()

if __name__ == "__main__":
    asyncio.run(main())