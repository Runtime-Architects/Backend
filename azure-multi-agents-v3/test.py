from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import DiGraphBuilder, GraphFlow
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_core.tools import FunctionTool
import json
import os
import asyncio
import subprocess
from dotenv import load_dotenv
load_dotenv()

# Model client
client = AzureOpenAIChatCompletionClient(
                    azure_deployment=os.getenv("AZURE_DEPLOYMENT"),
                    model=os.getenv("MODEL"),
                    api_version=os.getenv("API_VERSION"),
                    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
                    api_key= os.getenv("API_KEY"), # For key-based authentication.
                )

# Create agents for A→B→(C1,C2)→B scenario
# Create agents
agent_a = AssistantAgent("A", model_client=client, system_message="Initiate parallel tasks. task1, task2")
agent_b1 = AssistantAgent("B1", model_client=client, system_message="Handle task type 1. Say 'B1_COMPLETE' when done.")
agent_b2 = AssistantAgent("B2", model_client=client, system_message="Handle task type 2. Try to calculate what is the circumference of earth Say 'B2_COMPLETE' when done.")
agent_b3 = AssistantAgent("B3", model_client=client, system_message="Handle task type 3. Say 'B3_COMPLETE' when done.")
agent_c = AssistantAgent("C", model_client=client, system_message="Finalize after all B agents complete.")

# Build the graph
builder = DiGraphBuilder()
builder.add_node(agent_a).add_node(agent_b1).add_node(agent_b2).add_node(agent_b3).add_node(agent_c)

# A → B1, B2, B3 (parallel fan-out)
builder.add_edge(agent_a, agent_b1, condition=lambda msg: "task1" in msg.to_model_text().lower())
builder.add_edge(agent_a, agent_b2, condition=lambda msg: "task2" in msg.to_model_text().lower())
builder.add_edge(agent_a, agent_b3, condition=lambda msg: "task3" in msg.to_model_text().lower())

# B1, B2, B3 → C (synchronization point)
# Using "all" activation condition means C waits for ALL activated B agents
builder.add_edge(
    agent_b1, agent_c, 
    activation_group="b_group", 
    activation_condition="any", 
    condition="B1_COMPLETE"
)
builder.add_edge(
    agent_b2, agent_c, 
    activation_group="b_group", 
    activation_condition="any", 
    condition="B2_COMPLETE"
)
builder.add_edge(
    agent_b3, agent_c, 
    activation_group="b_group", 
    activation_condition="any", 
    condition="B3_COMPLETE"
)

# Build and create flow
graph = builder.build()
flow = GraphFlow(participants=[agent_a, agent_b1, agent_b2, agent_b3, agent_c], graph=graph)

print("=== Parallel Processing with Synchronization ===")
print("C will wait until ALL activated B agents complete their tasks")
# await Console(flow.run_stream(task="Start tasks 1 and 2"))  # Will activate B1 and B2, C waits for both



async def main():
    await Console(flow.run_stream(task="Start a parallel processing task."))

if __name__== "__main__":
    asyncio.run(main())