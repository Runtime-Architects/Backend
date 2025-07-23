import os
from datetime import datetime
import json
import sys


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
    
    # First, set all existing prompts to inactive
    for version in history["versions"]:
        version["status"] = "inactive"
    
    # Then add the new active prompt
    new_version = {
        "prompt": prompt,
        "timestamp": datetime.now().isoformat(),
        "status": "active"
    }
    history["versions"].append(new_version)
    
    # Save the updated history
    with open("prompt_history.json", "w") as f:
        json.dump(history, f, indent=2)


def get_user_input(display: str):
    try:
        response = input(display)
        return response.strip()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except EOFError:
        print("\nEnd of input reached.")
        sys.exit(1)

