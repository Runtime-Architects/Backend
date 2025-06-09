# Custom Agent - AutoGen Custom Agent Package

## Overview

`custom-agent` is a customizable conversational agent framework built on top of Microsoft's AutoGen library. This agent can use LLM's other than the default (OpenAI) supported by Autogen.

## Features

- **Custom Model Integration** - Easily connect your own LLM backend via the CustomModelClient


## Installation

1. Clone the repository
```bash
git clone <repository-url>
cd custom-agent
```

2. Create a virtual environment
```bash
python3.10 -m venv .venv
.venv/bin/activate  
```

## Quick Start

```python
from src.agent import ChatAgent

# Initialize the agent
agent = ChatAgent()

# Get a response
response = agent.get_final_response("What is the current policy on X?")
print(response)
```

## Configuration

1. Create an `OAI_CONFIG_LIST.json` file:
```json
[
    {  
        "model": "custom-model",
        "model_client_cls":"client-class",
        "api_key":"<your-api-key>"
    }
]
```

2. Place it in your working directory or specify the path during initialization:
```python
agent = ChatAgent(config_path="path/to/your/config.json")
```


## Reference

### `ChatAgent` Class

#### Methods:
- `__init__(config_path: str = "OAI_CONFIG_LIST")` - Initialize with optional config path
- `chat(message: str, summary_method: str = "last_msg")` - Returns full ChatResult object
- `get_final_response(message: str)` - Returns just the summary response string

## Examples

See the `examples/` directory for:
- Agent using custom client
- Inference Using CEREBRAS