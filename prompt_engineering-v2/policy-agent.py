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
import time
from dotenv import load_dotenv
load_dotenv()


scopes = os.getenv("SCOPES")
search_index_name = os.getenv("SEARCH_INDEX_NAME")
search_api_key = os.getenv("SEARCH_API_KEY")
search_api_version= os.getenv("SEARCH_API_VERSION") 
search_endpoint=  os.getenv("SEARCH_ENDPOINT") 


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
    client = AzureOpenAIChatCompletionClient(
            azure_deployment=os.getenv("AZURE_DEPLOYMENT"),
            model=os.getenv("MODEL"),
            api_version=os.getenv("API_VERSION"),
            azure_endpoint=os.getenv("AZURE_ENDPOINT"),
            api_key=os.getenv("API_KEY"),
        )

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

if __name__ == '__main__':
    start = time.time()
    asyncio.run(main())
    print(f'Time taken: {time.time() - start}')
