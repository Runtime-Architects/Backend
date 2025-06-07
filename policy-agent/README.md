# Policy Agent

This project implements a simple agent using the autogen framework in Python 3.10. The agent is designed to perform specific tasks and can be easily extended or modified for various use cases.

## Project Structure

```
simple-agent
├── src
│   ├── agent.py        # Contains the PolicyAgent class and its methods
│   ├── utils.py        # Utility functions for the agent
│   └── __init__.py     # Marks the directory as a Python package
├── requirements.txt     # Lists project dependencies
├── .gitignore           # Specifies files to ignore in Git
├── README.md            # Project documentation
└── setup.py             # Packaging configuration
```

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd simple-agent
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python3.10 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

To use the PolicyAgent, you can create an instance of the `PolicyAgent` class and call its methods to manage its lifecycle:

```python
from src.agent import PolicyAgent

agent = PolicyAgent()
agent.initialize()
agent.run()
agent.shutdown()
```
