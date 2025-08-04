import os
import json
import sys
from pathlib import Path
from openai import AzureOpenAI
import logging

# Configure Azure OpenAI client
client = AzureOpenAI(
    api_version=os.getenv("API_VERSION"),
    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    api_key=os.getenv("API_KEY")
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def get_user_input(display: str) -> str:
    """
    Safely gets user input with proper error handling.
    
    Args:
        display (str): The prompt message to display to the user
        
    Returns:
        str: The user's input after stripping whitespace
    """
    while True:
        try:
            response = input(display)
            return response.strip()
        except KeyboardInterrupt:
            logger.error("\nOperation cancelled by user.")
            sys.exit(1)
        except EOFError:
            logger.error("\nEnd of input reached.")
            sys.exit(1)

def create_log_folder(path: str) -> Path:
    """
    Creates a directory at the specified path if it doesn't exist.
    
    Args:
        path (str): The directory path to create
        
    Returns:
        Path: Path object representing the created directory
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj

async def prompt_helper(filename: str) -> None:
    """
    Analyzes agent interaction logs and generates performance critiques.
    
    Args:
        filename (str): Path to the JSON log file to analyze
        
    Raises:
        FileNotFoundError: If the specified log file doesn't exist
        
    Writes:
        Updates the log file with critique information
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            log_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise FileNotFoundError(f"Error loading log file: {str(e)}")
    
    # Initialize critique if not present
    log_data.setdefault("critic", None)
    
    analysis_task = f"""
    Analyze this log data: {log_data}.
    """
    
    try:
        response = client.chat.completions.create(
            model=os.getenv("AZURE_DEPLOYMENT"),
            messages=[
                {
                    "role": "system",
                    "content": """You are a precise evaluator that analyzes the logs of interaction between agents
                    and identifies which agent is not performing well individually.
                    Always generate a report with this structure:
                    
                    AGENT PERFORMANCE
                        - Agent (Overall summary and correctness)
                        - Tool (if used, evaluate the correctness)
                        - Weakness/Improvements (Highlight weaknesses for failures or improvements otherwise)
                    """
                },
                {
                    "role": "user",
                    "content": analysis_task
                }
            ],
            max_tokens=4096
        )
        
        critique = response.choices[0].message.content
        logger.info("\n\n\n===== Critique ======")
        logger.info(critique)
        
        # Update and save log data with critique
        log_data["critic"] = critique
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
            
    except Exception as e:
        logger.error(f"Error generating critique: {str(e)}")
        raise

async def generate_suggestion(recent_files: list) -> str:
    """
    Analyzes multiple log critiques to suggest which agent's prompt needs improvement.
    
    Args:
        recent_files (list): List of paths to recent log files
        
    Returns:
        str: AI-generated suggestion on which agent to focus on
    """
    critiques = []
    
    try:
        for file in recent_files:
            with open(file, 'r', encoding='utf-8') as f:
                log_data = json.load(f)
                if "critic" in log_data and log_data["critic"]:
                    critiques.append(log_data["critic"])
    except Exception as e:
        raise Exception(f"Error processing log files: {str(e)}")
    
    if not critiques:
        return "No valid critiques found in the provided logs."
    
    analysis_task = f"""
    Analyze these performance critiques: {critiques}.
    Based on consistent weaknesses identified, suggest which agent's system prompt 
    should be prioritized for improvement.
    """
    
    try:
        response = client.chat.completions.create(
            model=os.getenv("AZURE_DEPLOYMENT"),
            messages=[
                {
                    "role": "system",
                    "content": """You are a precise evaluator that analyzes multiple performance critiques
                    to identify the agent with the most consistent weaknesses. Provide:
                    1. The agent that needs most improvement
                    2. The specific weaknesses identified
                    3. Suggested focus areas for prompt improvement"""
                },
                {
                    "role": "user",
                    "content": analysis_task
                }
            ],
            max_tokens=4096
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"Error generating suggestions: {str(e)}")
        raise