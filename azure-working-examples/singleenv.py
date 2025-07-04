import asyncio
from pathlib import Path
import shutil
import venv
from autogen_core import CancellationToken
from autogen_core.code_executor import CodeBlock
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_ext.tools.code_execution import PythonCodeExecutionTool
from autogen_agentchat.ui import Console
import sys
import asyncio
import os
from dotenv import load_dotenv
load_dotenv()

import time


async def main():
    # Setup working directory
    work_dir = Path("coding")
    work_dir.mkdir(exist_ok=True)
    
    # Create base virtual environment with common packages
    base_venv_dir = work_dir / ".venv_base"
    if not base_venv_dir.exists():
        print("Creating base virtual environment...")
        
        venv_builder = venv.EnvBuilder(with_pip=True)
        venv_builder.create(base_venv_dir)
        venv_context = venv_builder.ensure_directories(base_venv_dir)


        temp_executor = LocalCommandLineCodeExecutor(
        work_dir=base_venv_dir.parent,
        virtual_env_context=venv_context)
    
    # Install common packages
    await temp_executor.execute_code_blocks(
        code_blocks=[
            CodeBlock(language="powershell", code="pip install matplotlib numpy pandas"),
        ],
        cancellation_token=CancellationToken(),
    )
            
    
    # Create executor with working virtual environment context
    executor = LocalCommandLineCodeExecutor(
        work_dir=work_dir,
        virtual_env_context=venv_context
    )
    
    client = AzureOpenAIChatCompletionClient(
        azure_deployment=os.getenv("AZURE_DEPLOYMENT"),
        model=os.getenv("MODEL"),
        api_version=os.getenv("API_VERSION"),
        azure_endpoint=os.getenv("AZURE_ENDPOINT"),
        api_key=os.getenv("API_KEY"),  # For key-based authentication.
    )

    tool = PythonCodeExecutionTool(executor)
    
    agent = AssistantAgent(
        name="assistant", 
        model_client=client, 
        tools=[tool], 
        reflect_on_tool_use=True,
        system_message="""You are a Python coding assistant. 
                       You have access to a Python environment that already has common packages installed.
                       You can install additional packages if needed."""
    )

    await Console(
        agent.run_stream(
            task=f"Generate a plot for 5 random integers and plot them using matplotlib and save it to random.png"
        )
    )

    # if base_venv_dir.exists():
    #     shutil.rmtree(base_venv_dir)


if __name__ == '__main__':
    start= time.time()
    asyncio.run(main())
    end= time.time()

    print(f"**** Time taken to execute the code: {end-start}")