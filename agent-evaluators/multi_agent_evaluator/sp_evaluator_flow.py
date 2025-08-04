"""
Autogen MultiAgent - Prompt Engineering Tool

This script provides an interactive console interface for:
1. Running custom prompt tests
2. Evaluating existing conversation logs
3. Performing prompt engineering analysis
"""

import workflow
import asyncio
import logging
from datetime import datetime
import time
import json
import os
from autogen_agentchat.base import TaskResult
from prompt_util import (prompt_helper, get_user_input, create_log_folder,
                         generate_suggestion)
import glob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    """
    Main function that provides the interactive console interface.
    Handles user input and directs to appropriate functions.
    """
    while True:
        print("\n===== Prompt Engineer =====")
        print("1. Run a custom test")
        print("2. Evaluate existing log")
        print("3. Prompt Engineer (analyze recent logs)")
        print("4. Exit")

        option = get_user_input("Enter your option: ").strip()

        if option == '1':
            await run_test()
        elif option == '2':
            await eval_existing()
        elif option == '3':
            await prompt_engineer()
            break
        elif option == '4':
            break
        else:
            logger.info("Invalid option")
    return

async def run_test():
    """
    Runs a custom prompt test and saves the conversation log.
    
    Steps:
    1. Gets user input for the test prompt
    2. Executes the workflow with the prompt
    3. Captures and logs all steps of the conversation
    4. Saves results to a timestamped JSON file
    5. Provides analysis of the conversation
    """
    task = get_user_input("\n\nEnter the prompt to test: ").strip()

    # Default prompt if none provided
    if not task:
        task = "what is the best time to use my appliance in Ireland?"

    log = []
    start_time = time.time()

    try:
        # Execute the workflow and capture each step
        async for step_result in workflow.flow.run_stream(task=task):
            if not isinstance(step_result, TaskResult):
                # Extract source and content from the step result
                source = getattr(step_result, 'source', 'unknown')
                content = getattr(step_result, 'content', '')
                
                log.append({source: content})
                logger.info(f"---------- TextMessage ({source}) ----------")
                logger.info(f"{content}")

    except KeyboardInterrupt:
        log.append({"User": "Interrupted (KeyboardInterrupt)"})
        logger.info("User interrupted the process")
    except Exception as e:
        log.append({"Error": str(e)})
        logger.error(f"An error occurred: {e}")

    # Calculate execution time
    time_taken = time.time() - start_time

    # Prepare log data structure
    log_data = {
        "planner_system_prompt": workflow.planner_system_message,
        "carbon_system_prompt": workflow.carbon_system_message,
        "policy_system_prompt": workflow.policy_system_message,
        "report_system_prompt": workflow.report_system_message,
        "time_taken": time_taken,
        "query": task,
        "log": log,
        "critic": None
    }

    # Create timestamp and save log
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = create_log_folder("agent_logs")
    filename = f"{work_dir}/agent_logs_{timestamp}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, default=str)
    
    # Provide analysis of the conversation
    await prompt_helper(filename)
    
    print(f"\nLogs saved to {filename}")

async def eval_existing():
    """
    Evaluates an existing conversation log file.
    
    Continuously prompts user for a file path until a valid one is provided,
    then analyzes the conversation.
    """
    while True:
        filepath = get_user_input("Enter the log file to evaluate (or type 'exit' to quit): ").strip()

        if filepath== "exit":
            logger.info("\n\n Exiting option 2...")
            break
        if os.path.isfile(filepath):
            await prompt_helper(filepath)
            break
        else:
            logger.info("\n\nLog file does not exist!")

async def prompt_engineer():
    """
    Performs prompt engineering analysis on recent conversation logs.
    
    Identifies the 10 most recent log files in the agent_logs directory,
    displays them, and generates suggestions for prompt improvement.
    """
    logger.info("===== Prompt Engineer =====")
    folder_path = 'agent_logs'

    if not os.path.exists(folder_path):
        logger.info("\n\n No logs exist! Returning to menu....")
        return

    # Get all log files with their modification times
    files = glob.glob(os.path.join(folder_path, '*'))
    files_with_time = [(f, os.path.getmtime(f)) for f in files]

    # Sort by modification time (newest first)
    files_with_time.sort(key=lambda x: x[1], reverse=True)

    print(f"\nThere are {len(files_with_time)} logs.")
    print("\nThe most recent are:\n")

    # Display the 10 most recent files
    recent_files = [f[0] for f in files_with_time[:10]]
    for file in recent_files:
        print(file)

    get_user_input("\n\nPress Enter to generate suggestions.")

    # Generate and display suggestions
    response = await generate_suggestion(recent_files)
    print("\nSuggestions on agent to focus upon:\n")
    print(response)

if __name__ == '__main__':
    asyncio.run(main())