import os
import asyncio
import time
import json
import subprocess
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import DiGraphBuilder, GraphFlow
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient, OpenAIChatCompletionClient
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from dotenv import load_dotenv
load_dotenv()


scopes = os.getenv("SCOPES")
search_index_name = os.getenv("SEARCH_INDEX_NAME")
search_api_key = os.getenv("SEARCH_API_KEY")
search_api_version= os.getenv("SEARCH_API_VERSION") 
search_endpoint=  os.getenv("SEARCH_ENDPOINT") 


POLICY_AGENT_SYSMSG = '''
You are a SEAI Policy Agent that answers questions using only the provided SEAI documents.

INSTRUCTIONS:
- When a user asks a question, first identify key terms and synonyms that may appear in SEAI policy documents
- Use the run_curl_search tool to query those terms against SEAI search
- Reformulate queries if no relevant documents are found
- Answer questions using ONLY information from the returned documents
- If information is not in the documents, say: "I don't have that information in the available SEAI documents"
- Be conversational and helpful
- Don't mention JSON, technical details, or internal tools
- Cite relevant document titles when answering
'''




# def create_agent(name, system_message):
#     return AssistantAgent(
#         name=name,
#         model_client=client,
#         system_message=system_message,
#         model_client_stream=True,
# ) 

import openai


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

    assistant = AssistantAgent(
        name="policy_agent",
        model_client=client,
        system_message=POLICY_AGENT_SYSMSG,
        tools=[run_curl_search],
        model_client_stream=True
    )


    task= "What grants are available for home energy upgrades?"
    initial_message = TextMessage(content=task, source="user")

    stream = assistant.on_messages_stream(
            messages=[initial_message],
            cancellation_token=CancellationToken()
        )
    
    await Console(stream)


if __name__ == "__main__":
    asyncio.run(main())
