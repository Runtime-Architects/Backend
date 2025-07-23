import os
import asyncio
import json
import subprocess
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from datetime import datetime
import re
import time
from dotenv import load_dotenv
load_dotenv()


scopes = os.getenv("SCOPES")
search_index_name = os.getenv("SEARCH_INDEX_NAME")
search_api_key = os.getenv("SEARCH_API_KEY")
search_api_version= os.getenv("SEARCH_API_VERSION") 
search_endpoint=  os.getenv("SEARCH_ENDPOINT") 

client = AzureOpenAIChatCompletionClient(
            azure_deployment=os.getenv("AZURE_DEPLOYMENT"),
            model=os.getenv("MODEL"),
            api_version=os.getenv("API_VERSION"),
            azure_endpoint=os.getenv("AZURE_ENDPOINT"),
            api_key=os.getenv("API_KEY"),
        )


policy_agent_system_message = '''
You are a SEAI Policy Agent that answers questions using only the provided SEAI documents.

INSTRUCTIONS:
- When a user asks a question, first identify key terms and synonyms that may appear in SEAI policy documents
- Use the run_curl_search tool to query those terms against SEAI search
- Reformulate queries if no relevant documents are found
- Answer questions using ONLY information from the returned documents
- If information is not in the documents, say: "I don't have the requested information, please visit https://www.seai.ie/"
- Be conversational and helpful
- Don't mention JSON, technical details, or internal tools
- Cite relevant document titles when answering

SUMMARY
Summarize the information from the tools, as clear points 
'''


def run_curl_search(query: str) -> str:
    
    curl_command = [
        "curl", "-X", "POST",
        f"{search_endpoint}/indexes/{search_index_name}/docs/search?api-version={search_api_version}",
        "-H", "Content-Type: application/json",
        "-H", f"api-key: {search_api_key}",
        "-d", json.dumps({
            "search": query,
            "top": 5
        })
    ]
    
    result = subprocess.run(curl_command, capture_output=True, text=True)
    return result.stdout

async def main():

    initialize_prompt_history(policy_agent_system_message)

    agent = AssistantAgent(
        name="policy_agent",
        model_client=client,
        system_message=policy_agent_system_message,
        tools=[run_curl_search],
        reflect_on_tool_use= True,
        max_tool_iterations= 3
    )


    query= "What grants are available for home energy upgrades?"


    # Create a dictionary to store the logs
    log_data = {
        "system_prompt": policy_agent_system_message,
        "query": None,
        "log": [],
        "final_output": None
    }

    log_data["query"] = query

    # Collect messages from the stream
    final_output = None
    
    
    # Process the stream
    async for msg in agent.run_stream(task=query):
        # Store all messages in log
        log_entry = {
            "content": str(msg),
            "timestamp": datetime.now().isoformat()
        }
        log_data["log"].append(log_entry)
        
        # Check if this is a TextMessage with final output
        if hasattr(msg, 'messages'):
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
    print('\n\n')


    feedback=input("> Please provide a feedback if necessary")
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
        r'==improved_prompt==(.*?)(?=$|==)', 
        final_output, 
        re.DOTALL
    )

    if improved_prompt_match:
        improved_prompt = improved_prompt_match.group(1).strip()

        print(f"\n\nExtracted improved prompt:\n{improved_prompt}")
        
        # Save for human review
        save_prompt_version(improved_prompt, status="pending")  
    
    else:
        print("\n\nNo improved prompt found in response")
        improved_prompt = None
    

    human_proxy= input(f"\n\nReview this new system message: {improved_prompt}. Reply APPROVE/RB (Roll back).")
    
    if human_proxy == "APPROVE":
        activate_prompt(improved_prompt)  # Update active prompt
    else:
        rollback_prompt()  # Revert to last stable



def initialize_prompt_history(initial_prompt: str):
    """Initialize prompt history file with the first system prompt"""
    if not os.path.exists("prompt_history.json"):
        history = {
            "versions": [{
                "prompt": initial_prompt,
                "timestamp": datetime.now().isoformat(),
                "status": "active"
            }]
        }
        with open("prompt_history.json", "w") as f:
            json.dump(history, f, indent=2)



def save_prompt_version(prompt: str, status: str = "pending"):
    try:
        with open("prompt_history.json", "r") as f:
            history = json.load(f)
    except FileNotFoundError:
        history = {"versions": []}
    
    history["versions"].append({
        "prompt": prompt,
        "timestamp": datetime.now().isoformat(),
        "status": status
    })
    with open("prompt_history.json", "w") as f:
        json.dump(history, f, indent=2)

def activate_prompt(prompt: str):
    try:
        with open("prompt_history.json", "r") as f:
            history = json.load(f)
    except FileNotFoundError:
        history = {"versions": []}
    
    # Deactivate all others
    for v in history["versions"]:
        v["status"] = "inactive"
    # Activate new prompt
    save_prompt_version(prompt, status="active")

def rollback_prompt():
    try:
        with open("prompt_history.json", "r") as f:
            history = json.load(f)
    except FileNotFoundError:
        print("No prompt history found to rollback to")
        return
    
    try:
        last_active = next(v for v in reversed(history["versions"]) if v["status"] == "active")
        activate_prompt(last_active["prompt"])
    except:
        print("No active prompt found in history")


if __name__ == '__main__':
    asyncio.run(main())

