# azurecarbonagent.py
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
from co2_analysis_tool import (
    co2_analysis_daily, 
    co2_analysis_weekly, 
    co2_analysis_monthly
)
from scraper_tools.run_eirgrid_downloader import main as eirgrid_main
import json
import os
from dotenv import load_dotenv

load_dotenv()

client = AzureOpenAIChatCompletionClient(
    azure_deployment=os.environ["AZURE_DEPLOYMENT"],
    model=os.environ["MODEL"],
    api_version=os.environ["API_VERSION"],
    azure_endpoint=os.environ["AZURE_ENDPOINT"],
    api_key=os.environ["API_KEY"],
    max_completion_tokens=1024,
)

import time

def load_examples():
    """Load good and bad examples for consistent responses"""
    try:
        with open('examples.json', 'r') as f:
            examples_data = json.load(f)
        return examples_data
    except FileNotFoundError:
        print("⚠️ Warning: examples.json not found. Creating default examples...")
        # Create minimal examples if file doesn't exist
        return {
            "good_examples": [],
            "bad_examples": [],
            "format_requirements": {
                "required_sections": [
                    "Header with region and data source reference",
                    "Optimal period with specific times and CO2 values", 
                    "Specific recommendations with bullet points",
                    "High emission times to avoid",
                    "Data summary with min/max/average values",
                    "Environmental impact statement with percentage"
                ]
            }
        }

async def get_emission_analysis(startdate: str, enddate: str, region: str) -> dict:
    """Enhanced emission analysis with data extraction for prompt"""
    # Check if the file exists
    file_path = f'data/co2_intensity/co2_intensity_{region}_{startdate}_{enddate}.json'

    if os.path.exists(file_path):
        # Load and analyze the data
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Extract key metrics for the agent
        analysis_result = {
            "result": "Success, Data scraped and analyzed.",
            "file_location": file_path,
            "data_available": True
        }
        
        # Extract time series data
        time_series = None
        if 'data' in data:
            if isinstance(data['data'], list):
                time_series = data['data']
            elif isinstance(data['data'], dict) and 'time_series' in data['data']:
                time_series = data['data']['time_series']
        
        if time_series:
            # Extract CO2 values and times
            values = []
            time_value_pairs = []
            
            for entry in time_series:
                if isinstance(entry, dict) and 'value' in entry:
                    value = entry['value']
                    time_str = entry.get('time', '')
                    values.append(value)
                    if time_str:
                        time_value_pairs.append((time_str, value))
            
            if values:
                min_value = min(values)
                max_value = max(values)
                avg_value = sum(values) / len(values)
                
                # Find optimal and peak times
                min_entries = [entry for entry in time_series if entry.get('value') == min_value]
                max_entries = [entry for entry in time_series if entry.get('value') == max_value]
                
                optimal_times = [entry.get('time', '') for entry in min_entries if entry.get('time')]
                peak_times = [entry.get('time', '') for entry in max_entries if entry.get('time')]
                
                # Add analysis to result
                analysis_result.update({
                    "co2_data": {
                        "min_value": min_value,
                        "max_value": max_value,
                        "daily_average": int(avg_value),
                        "optimal_times": optimal_times[:3],  # First 3 optimal times
                        "peak_times": peak_times[:3],       # First 3 peak times
                        "data_points": len(time_series),
                        "region": region,
                        "date_range": f"{startdate} to {enddate}"
                    }
                })
        
        return analysis_result
    
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
            return await get_emission_analysis(startdate, enddate, region)  # Recursive call
        else:
            raise Exception(f"Failed to fetch data from scrapper. File not found at {file_path}")

async def analyze_daily_co2(startdate: str, enddate: str, region: str = "all") -> dict:
    """Tool 2: Daily Data Analyzer - Analyzes CO2 data with daily granularity"""
    
    # Construct file path based on parameters
    file_path = f'data/co2_intensity/co2_intensity_{region}_{startdate}_{enddate}.json'
    
    # Check if file exists, if not try to get the data first
    if not os.path.exists(file_path):
        try:
            # Try to get the data using the emission analysis function
            await get_emission_analysis(startdate, enddate, region)
        except Exception as e:
            return {"error": f"Failed to retrieve data: {str(e)}", "analysis_type": "daily"}
    
    analysis = co2_analysis_daily.get_daily_analysis(file_path)
    return analysis

async def analyze_weekly_co2(startdate: str, enddate: str, region: str = "all") -> dict:
    """Tool 3: Weekly Data Analyzer - Analyzes CO2 data with weekly granularity"""

    # Construct file path based on parameters
    file_path = f'data/co2_intensity/co2_intensity_{region}_{startdate}_{enddate}.json'
    
    # Check if file exists, if not try to get the data first
    if not os.path.exists(file_path):
        try:
            # Try to get the data using the emission analysis function
            await get_emission_analysis(startdate, enddate, region)
        except Exception as e:
            return {"error": f"Failed to retrieve data: {str(e)}", "analysis_type": "weekly"}
    
    analysis = co2_analysis_weekly.get_weekly_analysis(file_path)
    return analysis

async def analyze_monthly_co2(startdate: str, enddate: str, region: str = "all") -> dict:
    """Tool 4: Monthly Data Analyzer - Analyzes CO2 data with monthly granularity"""

    # Construct file path based on parameters
    file_path = f'data/co2_intensity/co2_intensity_{region}_{startdate}_{enddate}.json'
    
    # Check if file exists, if not try to get the data first
    if not os.path.exists(file_path):
        try:
            # Try to get the data using the emission analysis function
            await get_emission_analysis(startdate, enddate, region)
        except Exception as e:
            return {"error": f"Failed to retrieve data: {str(e)}", "analysis_type": "monthly"}
    
    analysis = co2_analysis_monthly.get_monthly_analysis(file_path)
    return analysis

daily_analyzer_tool = FunctionTool(
    func=analyze_daily_co2,
    description="Analyzes CO2 data with daily granularity. Best for periods <=6 days. Parameters: startdate (YYYY-MM-DD), enddate (YYYY-MM-DD), region ('all', 'roi', or 'ni')",
    name= "analyze_daily_co2"
)

weekly_analyzer_tool = FunctionTool(
    func=analyze_weekly_co2,
    description="Analyzes CO2 data with weekly granularity. Best for periods >=7 days but <= 21 days. Parameters: startdate (YYYY-MM-DD), enddate (YYYY-MM-DD), region ('all', 'roi', or 'ni')",
    name= "analyze_weekly_co2"
)

monthly_analyzer_tool = FunctionTool(
    func=analyze_monthly_co2,
    description="Analyzes CO2 data with monthly granularity. Best for periods >=22 days. Parameters: startdate (YYYY-MM-DD), enddate (YYYY-MM-DD), region ('all', 'roi', or 'ni')",
    name= "analyze_monthly_co2"
)

# Create a function tool
emission_tool = FunctionTool(
    func=get_emission_analysis,
    description="Gets the CO2 intensity levels with detailed analysis. Parameters: startdate (YYYY-MM-DD), enddate (YYYY-MM-DD), region ('all', 'roi', or 'ni')",
    name= "get_emission_analysis"
)

def create_robust_system_message():
    """Create robust system message based on examples and strict formatting"""
    
    examples = load_examples()
    
    # Extract good examples for reference
    good_examples_text = ""
    if examples.get("good_examples"):
        good_examples_text = "\n## GOOD RESPONSE EXAMPLES:\n"
        for example in examples["good_examples"][:3]:  # Include first 3 examples
            good_examples_text += f"\n**Query:** {example['query']}\n"
            good_examples_text += f"**Response:** {example['expected_response'][:300]}...\n"
            good_examples_text += f"**Why Good:** {example['why_good']}\n"
    
    # Extract bad examples for avoidance
    bad_examples_text = ""
    if examples.get("bad_examples"):
        bad_examples_text = "\n## AVOID THESE BAD EXAMPLES:\n"
        for example in examples["bad_examples"][:3]:  # Include first 3 examples
            bad_examples_text += f"\n**Query:** {example['query']}\n"
            bad_examples_text += f"**Bad Response:** {example['bad_response'][:200]}...\n"
            bad_examples_text += f"**Why Bad:** {example['why_bad']}\n"
    
    return f"""You are a Carbon Emissions Analysis Expert that provides precise, data-driven recommendations for optimal energy usage timing in Ireland.

## CORE PRINCIPLES:
1. Base ALL responses strictly on the provided CO2 data from get_emission_analysis tool
2. Maintain consistent response structure and length (400-900 characters)
3. Use specific numerical values and time ranges from the actual data
4. Follow the exact output format specified below
5. Always mention "REAL EirGrid data" or "real data" to indicate data source

## ANALYSIS PROCESS:
Always follow this sequence:
1. Call get_emission_analysis tool with appropriate date range (default: today's date)
2. Parse the returned co2_data for min_value, max_value, optimal_times, peak_times
3. Extract specific time ranges and CO2 intensity values
4. Structure response using the mandatory template below

## MANDATORY OUTPUT FORMAT:
Structure EVERY response using this exact template:

🏠 **Best Times to Use Appliances in Ireland Today (Based on REAL EirGrid Data):**

🌱 **Optimal Period (Lowest Real CO2):**
- **[TIME_RANGE]**: [CO2_VALUE]g CO2/kWh (REAL EirGrid data)
- Perfect for washing machine, dishwasher, and EV charging

⚡ **Specific Appliance Recommendations (Real Data-Based):**
• **Washing Machine**: Start cycle at [SPECIFIC_TIME]
• **Dishwasher**: Schedule for [TIME_RANGE]
• **Electric Vehicle Charging**: Begin charging during real low-emission window
• **Tumble Dryer**: Use during overnight period

🔥 **Avoid High Real Emission Times:**
- **[TIME_RANGE]**: [CO2_VALUE]g CO2/kWh (real peak demand data)

📊 **Today's REAL EirGrid CO2 Data:**
- Minimum: [VALUE]g CO2/kWh (real measurement)
- Maximum: [VALUE]g CO2/kWh (real measurement)
- Daily Average: [VALUE]g CO2/kWh (real average)

🌍 **Environmental Impact**: Using appliances during optimal times reduces your carbon footprint by up to [PERCENTAGE]% compared to peak times (calculated from real data)!

## RESPONSE CONSTRAINTS:
- Response length: 400-900 characters total
- Always include specific times in HH:MM-HH:MM format (e.g., "16:00-19:00")
- Always include exact numerical CO2 values from the data (e.g., "203g CO2/kWh")
- Use present tense for current data, future tense for forecasts
- Never speculate beyond provided data
- Must include at least 5 bullet points or recommendations
- Must include percentage calculation for environmental impact

## DATA HANDLING RULES:
- Always call get_emission_analysis first for any CO2-related query
- Use today's date ({datetime.now().strftime('%Y-%m-%d')}) as default
- Use region 'all' for Ireland unless specified otherwise
- Extract min_value, max_value, daily_average from co2_data in tool response
- Find optimal times from optimal_times array in tool response
- Find peak times from peak_times array in tool response

## CONSISTENCY REQUIREMENTS:
- Identical queries with same data must produce identical core recommendations
- Time ranges must match the actual data returned by the tool
- CO2 values must be exact numbers from the tool response
- Use standard 24-hour time format (e.g., "14:00" not "2 PM")
- Percentage calculations must be based on (max_value - min_value) / max_value * 100

## CALCULATION FORMULAS:
- Environmental Impact Percentage: ((max_value - min_value) / max_value) * 100
- Round CO2 values to nearest whole number
- Time ranges should span 2-4 hours for optimal periods

{good_examples_text}

{bad_examples_text}

## ERROR HANDLING:
- If get_emission_analysis returns no data: "No carbon intensity data available for analysis."
- If query unclear: Focus on general appliance timing recommendations
- If tool fails: "Unable to retrieve current CO2 data. Please try again."

## PROHIBITED ACTIONS:
- Never provide recommendations without calling get_emission_analysis first
- Never use vague terms like "usually", "typically", "generally"
- Never exceed 900 characters or go under 400 characters
- Never deviate from the structured format with required emojis
- Never provide CO2 values not present in the tool response
- Never skip the environmental impact percentage calculation

## CURRENT DATE AND TIME: {datetime.now()}

Remember: The evaluation system expects EXACT adherence to this format. Any deviation will result in evaluation failure even if the content is accurate."""

#assistant system message
system_message = create_robust_system_message()

carbon_agent = AssistantAgent(
            name="CarbonAgent", 
            model_client=client, 
            tools=[emission_tool, daily_analyzer_tool, weekly_analyzer_tool, monthly_analyzer_tool], 
            reflect_on_tool_use=True,
            max_tool_iterations=5,
            system_message=system_message
        )
