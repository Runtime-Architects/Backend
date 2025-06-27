from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_ext.tools.code_execution import PythonCodeExecutionTool
from autogen_core.tools import FunctionTool
from autogen_agentchat.ui import Console
from datetime import datetime
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()
import sys
from co2_analysis import CO2IntensityAnalyzer
from run_eirgrid_downloader import main as eirgrid_main
import json

async def get_emission_analysis(startdate: str, enddate: str, region: str) -> float:
    
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

    try:
        with open(f'data/co2_intensity/co2_intensity_{region}_{startdate}_{enddate}.json', 'r') as file:
            scraper_data = json.load(file)
    except:
        raise Exception
    

    analyzer = CO2IntensityAnalyzer(scraper_data)
    intensity_periods = analyzer.get_combined_periods()
    

    return intensity_periods



# Create a function tool
emission_tool = FunctionTool(
    func=get_emission_analysis,
    description="Gets the CO2 intensity levels. Parameters: startdate (YYYY-MM-DD), enddate (YYYY-MM-DD), region ('all', 'roi', or 'ni')"
)

#assistant system message
system_message = f"""You are an intelligent assistant with access to specialized tools. Today's date and time is: {datetime.now()} You have two main capabilities:

1. Python Code Execution: You can write and execute Python code in a sandboxed environment.
2. CO2 Emission Analysis: You have a specialized tool that can fetch real-time CO2 intensity data for electricity generation in all (Ireland & Northern Ireland), roi (Ireland), ni (Northern Ireland).

Your available tools:
- PythonCodeExecutionTool: For general programming tasks
- emission_tool: Specifically for CO2 intensity data

TOOL USAGE RULES:
- For CO2 intensity queries, ALWAYS use the emission_tool with these exact parameters:
  - Date format MUST be YYYY-MM-DD (e.g., '2025-06-24')
  - Region MUST be one of:
    * 'roi' for Republic of Ireland (Ireland)
    * 'ni' for Northern Ireland
    * 'all' for both Republic of Ireland (Ireland) & Northern Ireland

When any question relates to CO2 emissions, electricity usage patterns, or optimal times for energy consumption, you MUST use the emission_tool to get current data before responding. This includes but is not limited to:
- Questions about EV charging times
- Best times to use appliances
- Electricity carbon intensity
- Energy usage recommendations
- Renewable energy availability
- Carbon footprint of electricity use

For these types of questions:
1. Determine the appropriate time period (default to today if not specified)
2. Identify the region (default to 'all' if not specified)
3. Use the emission_tool to get current data
4. Analyze the results to provide a data-driven answer

For general programming or math questions, use the Python execution tool.

Always verify tool parameters before use and explain your reasoning when choosing tools. If a question could have CO2 implications but isn't explicitly about emissions, consider whether emission data would improve your answer.
 
You provide help energy-conscious consumers make sustainable choices by clear, actionable advice about electricity usage, renewable energy, and carbon reduction using markdown. Include relevant emojis to emphasize 
environmental benefits.
"""

async def main() -> None:
    async with DockerCommandLineCodeExecutor(work_dir="coding") as executor:
        tool = PythonCodeExecutionTool(executor)

        custom_model_client = OpenAIChatCompletionClient(
            model="openai/gpt-4.1",
            base_url="https://models.github.ai/inference",
            api_key=os.getenv("GITHUB_TOKEN"),
            model_info={
                "vision": True,
                "function_calling": True,
                "json_output": True,
                "family": "unknown",
                "structured_output": True,
            },
        )

        agent = AssistantAgent(
            name="assistant", model_client=custom_model_client, tools=[tool, emission_tool], reflect_on_tool_use=True,
            system_message= system_message
        )

        await Console(
            agent.run_stream(
                task=f"What is the best time to use my appliances today in Ireland?"
            )
        )

if __name__=='__main__':
    asyncio.run(main())
