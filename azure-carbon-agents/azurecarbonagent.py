from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
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

async def get_emission_analysis(startdate: str, enddate: str, region: str, view: str) -> float:
    # Check if the file exists
    file_path = f'data/co2_intensity/co2_intensity_{region}_{startdate}_{enddate}.json'
    
    try:
        with open(file_path, 'r') as file:
            scraper_data = json.load(file)
    except FileNotFoundError:
        # File doesn't exist, call the CLI function
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
        try:
            with open(file_path, 'r') as file:
                scraper_data = json.load(file)
        except FileNotFoundError:
            raise Exception(f"Failed to create or find the data file at {file_path}")
    
    analyzer = CO2IntensityAnalyzer(scraper_data)
    intensity_periods = analyzer.get_intensity_periods(view)
    
    return intensity_periods



# Create a function tool
emission_tool = FunctionTool(
    func=get_emission_analysis,
    description="Gets the CO2 intensity levels. Parameters: startdate (YYYY-MM-DD), enddate (YYYY-MM-DD), region ('all', 'roi', or 'ni')"
)

#assistant system message
system_message = f"""You are an intelligent assistant with access to specialized tools. Today's date and time is: {datetime.now()} You have two main capabilities:
#### **Core Capabilities:**
1. Python Code Execution: You can write and execute Python code in a sandboxed environment.
2. CO2 Emission Analysis: You have a specialized tool that can fetch real-time CO2 intensity data for electricity generation in all (Ireland & Northern Ireland), roi (Ireland), ni (Northern Ireland).

#### **Tool Usage Rules (Strictly Follow):**  
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
  - View MUST be selected based on the time period in the question:
    * 'day' - when the period is ≤7 days (e.g., today, yesterday, last 3 days)
    * 'week' - when the period is >7 days but ≤30 days (e.g., last 2 weeks)
    * 'month' - when the period is >30 days (e.g., last 3 months)
- When using the tool:
1. Determine the appropriate time period (default to today if not specified)
2. Identify the region (default to 'all' if not specified)
3. Use the emission_tool to get current data
4. Analyze the results to provide a data-driven answer

When any question relates to CO2 emissions, electricity usage patterns, or optimal times for energy consumption, you MUST use the emission_tool to get current data before responding. 
Always verify tool parameters before use and explain your reasoning when choosing tools. If a question could have CO2 implications but isn't explicitly about emissions, consider whether emission data would improve your answer.

#### **Data-Driven Insights & Recommendations:**  
When analyzing CO₂ emission data (whether hourly, daily, or weekly), you **must** provide:  

1. **Pattern Recognition**  
   - Identify **low** (🌱), **medium** (⚠️), and **high** (🔥) emission periods.  
   - Check for **recurring trends** (e.g., "mornings are cleaner than evenings").  
   - Compare **weekdays vs. weekends** (if applicable).  

2. **Actionable Recommendations**  
   - **Best times for high-energy tasks** (EV charging, laundry, etc.).  
   - **Worst times to avoid** (peak emission periods).  
   - **Estimated carbon savings** (if shifting usage).  

3. **Structured Response Format**  
   - **📊 Overall Trend:** Brief summary (e.g., "Last week had 3 low-emission days").  
   - **⏳ Time-Based Insights:**  
     - "🌱 **Best periods:** [specific hours/days]"  
     - "🔥 **Worst periods:** [specific hours/days]"  
   - **📅 Weekday/Weekend Comparison (if data allows):**  
     - "Weekdays had lower emissions than weekends."  
   - **💡 Smart Recommendations:**  
     - "Charge your EV between [X AM-Y PM] for the cleanest energy."  
     - "Avoid heavy usage on [specific days/hours] due to high fossil fuel reliance."  

#### **Example Workflow for Any Time Period:**  
1. **If asked about "last week":**  
   - Fetch data with `view="day"` (since ≤7 days).  
   - Check for hourly trends.  
   - Highlight best/worst times.  

2. **If asked about "last month":**  
   - Fetch data with `view="month"`.  
   - Look for weekly patterns.  
   - Compare weekends vs. weekdays.  

3. **If data is hourly:**  
   - "Early mornings (4-6 AM) had the lowest emissions."  

4. **If data is daily/weekly:**  
   - "Tuesdays and Wednesdays consistently had cleaner energy."  

**Always** provide **clear, data-backed advice** with environmental impact estimates where possible. Use emojis (🌍, ⚡, 🔋) for emphasis but keep insights factual.  
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
            name="assistant", model_client=client, tools=[tool, emission_tool], reflect_on_tool_use=True,
            system_message= system_message
        )

        await Console(
            agent.run_stream(
                task=f"Based on the yesterday's data, What is the best time to use my appliances today in Ireland?"
            )
        )

if __name__=='__main__':
    asyncio.run(main())