from src.agent import ChatAgent

# Initialize the agent
agent = ChatAgent()

# Get a response
response = agent.get_final_response("What is the current policy on X?")
print(response)