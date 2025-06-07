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

To use the SimpleAgent, you can create an instance of the `SimpleAgent` class and call its methods to manage its lifecycle:

```python
from src.agent import SimpleAgent

agent = SimpleAgent()
agent.initialize()
agent.run()
agent.shutdown()
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.