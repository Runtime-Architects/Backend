import os
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_core.tools import FunctionTool
from datetime import datetime
import asyncio
from co2_analysis_tool.co2_analysis import CO2IntensityAnalyzer
from prompt_util import (initialize_prompt_history,
                         activate_prompt, get_user_input)
import json
import re
from dotenv import load_dotenv

load_dotenv()


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
  - Region MUST be one of:
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

client = AzureOpenAIChatCompletionClient(
    azure_deployment=os.getenv("AZURE_DEPLOYMENT"),
    model=os.getenv("MODEL"),
    api_version=os.getenv("API_VERSION"),
    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    api_key=os.getenv("API_KEY"),
)

async def main() -> None:
    # Create a dictionary to store the logs
    log_data = {
        "system_prompt": system_message,
        "query": None,
        "log": [],
        "final_output": None
    }

    initialize_prompt_history(system_message)

    agent = AssistantAgent(
        name="assistant", 
        model_client=client, 
        tools=[emission_tool], 
        reflect_on_tool_use=True,
        max_tool_iterations=10,
        system_message=system_message
    )

    query = "Based on the last week's data what is the best time to charge my EV?"
    log_data["query"] = query

    # Collect messages from the stream
    final_output = None
    
    
    # Process the stream
    async for msg in agent.run_stream(task=query):

        # Check if this is a TextMessage with final output
        if hasattr(msg, 'messages'):
            log_entry = {
            "content": str(msg),
            "timestamp": datetime.now().isoformat()
            }
            log_data["log"].append(log_entry)   #store last log since it has the entire log

            for m in msg.messages:
                if hasattr(m, 'type') and m.type == 'TextMessage':
                    final_output = m.content

    print(final_output)  
            
    log_data["final_output"] = str(final_output) if final_output else None

    # Save logs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"agent_logs_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(log_data, f, indent=2, default=str)
    
    print(f"Logs saved to {filename}")


    feedback= get_user_input("\n \n Please provide any feedback if necessary")
    await run_prompt_engineer(filename, feedback)


# prompt_agent.py
async def run_prompt_engineer(fp: str = None, feedback: str = None):
    # Load last log file (passed from carbon_agent) 
    try:
        with open(fp) as f:
            log_data = json.load(f)
    except:
        raise FileNotFoundError

    prompt_agent = AssistantAgent(
            name="prompt_agent", 
            model_client=client,    
            description="An agent responsible for prompt engineering",
            reflect_on_tool_use=True,
            max_tool_iterations= 5,
            system_message= "You are an agent that can perform prompt engineering."
        )

    # Analyze logs + feedback
    task = f"""
    Analyze this log data: {log_data}. 
    Human feedback: {feedback if feedback else "None"}.
    Suggest an improved system message.
    The improved prompt should be in between ==improved_prompt== 
    """
    
    # Generate new prompt

    async for msg in prompt_agent.run_stream(task= task):        
        # Check if this is a TextMessage with final output
        if hasattr(msg, 'messages'):
            for m in msg.messages:
                if hasattr(m, 'type') and m.type == 'TextMessage':
                    final_output = m.content

    print(final_output)

    # Extract improved prompt using regex
    improved_prompt_match = re.search(
    r'==improved_prompt==(.*?)==improved_prompt==', 
    final_output, 
    re.DOTALL)

    if improved_prompt_match:
        improved_prompt = improved_prompt_match.group(1).strip()

        # print(f"\n\nExtracted improved prompt:\n{improved_prompt}")
        
        # Save for human review
        # save_prompt_version(improved_prompt, status="pending")  
    
    else:
        print("\n\n No improved prompt found in response")
        improved_prompt = None
    

    human_proxy= get_user_input("approve? (yes or no): ")
    
    if human_proxy == "yes":
        activate_prompt(improved_prompt)  # Update active prompt
    elif human_proxy == "no":
        print("Updated prompt rejected !!!") # Impelement roll back 
    else:
        print("Wrong command") # Revert to last stable





if __name__ == '__main__':
    asyncio.run(main())

