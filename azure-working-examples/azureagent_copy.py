import asyncio
from pathlib import Path
import shutil
import venv
from autogen_core.code_executor import CodeBlock
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_ext.tools.code_execution import PythonCodeExecutionTool
from autogen_agentchat.ui import Console
import asyncio
import os
from dotenv import load_dotenv
load_dotenv()
import time
import subprocess


async def main():
    
    client = AzureOpenAIChatCompletionClient(
        azure_deployment=os.getenv("AZURE_DEPLOYMENT"),
        model=os.getenv("MODEL"),
        api_version=os.getenv("API_VERSION"),
        azure_endpoint=os.getenv("AZURE_ENDPOINT"),
        api_key=os.getenv("API_KEY"),  # For key-based authentication.
    )
    
    agent = AssistantAgent(
        name="assistant", 
        model_client=client, 
        reflect_on_tool_use=True,
        system_message="""You are an intelligent agent"""
    )

    await Console(
        agent.run_stream(
            task=f"Generate a plot for 5 random integers and plot them using matplotlib and save it to random.png"
        )
    )


if __name__ == '__main__':
    start= time.time()
    asyncio.run(main())
    end= time.time()

    print(f"**** Time taken to execute the code: {end-start}")