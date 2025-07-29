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
from co2_analysis_tool.co2_analysis import CO2IntensityAnalyzer

from dotenv import load_dotenv

load_dotenv()

import time

async def get_emission_analysis(startdate: str, enddate: str, region: str) -> float:
    # Check if the file exists

    try:
        # Your existing tool logic
        analyzer = CO2IntensityAnalyzer(startdate, enddate, region)
        intensity_periods = analyzer.get_analysis_by_view()

        return intensity_periods
    except Exception as e:
            # Return detailed error info
            return {
                "error": str(e),
                "input_params": {"startdate": startdate, "enddate": enddate, "region": region}
            }


# Create a function tool
emission_tool = FunctionTool(
    func=get_emission_analysis,
    description="Gets the CO2 intensity levels. Parameters: startdate (YYYY-MM-DD), enddate (YYYY-MM-DD), region ('all', 'roi', or 'ni')",
    name= "emission_tool"
)

#assistant system message
system_message = f"""You are an intelligent assistant with access to specialized tools. Today's date and time is: {datetime.now()}.

### Available Tools:
- **Emission tool**- returns carbon intensity

TOOL USAGE RULES:
- For CO2 intensity queries, ALWAYS use the emission_tool with these exact parameters:
  - Date format MUST be YYYY-MM-DD (e.g., '2025-06-24')
  - Region MUST be one of (based on the user query):
    * 'roi' for Republic of Ireland or Ireland
    * 'ni' for Northern Ireland
    * 'all' for both Republic of Ireland (Ireland) & Northern Ireland

- When using the tool:
1. Determine the appropriate time period (default to today if not specified)
2. Identify the region, the user wants data from, intelligently.  (default to 'all' if not specified)
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
            name="assistant", model_client=client, tools=[emission_tool], 
            reflect_on_tool_use=True,
            max_tool_iterations= 10,
            system_message= system_message
        )

        await Console(
            agent.run_stream(
                #task=f"Based on last month's data, which day had the highest emission in Ireland?"
                #task=f"What were the cleanest energy times last Tuesday in Northern Ireland?"
                # task= f"What was the carbon emission for the first two weeks of this month versus the last two weeks of the last month?"
                task= "What is the best time to use my appliances today in Ireland"
                # task=f"Compare ROI and NI emissions for the first week of June"
            )
        )

if __name__=='__main__':
    start= time.time()
    asyncio.run(main())
    print(f'Time taken: {time.time() - start}')