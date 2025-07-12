import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_core.tools import FunctionTool
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
import os
from dotenv import load_dotenv

load_dotenv()

# Define the async tools
async def get_current_weather(location: str) -> str:
    """Get the current weather in a given location"""
    return f"The weather in {location} is sunny and 72°F"

async def get_restaurant_recommendations(location: str, cuisine: str) -> str:
    """Get restaurant recommendations in a given location"""
    return f"Top {cuisine} restaurants in {location}: 1. Great Food Place 2. Tasty Eats"

async def main():
    # Initialize the Azure OpenAI client
    client = AzureOpenAIChatCompletionClient(
        azure_deployment=os.getenv("AZURE_DEPLOYMENT"),
        model=os.getenv("MODEL"),
        api_version=os.getenv("API_VERSION"),
        azure_endpoint=os.getenv("AZURE_ENDPOINT"),
        api_key=os.getenv("API_KEY"),
    )

    # Create tool objects
    weather_tool = FunctionTool(
        func=get_current_weather,
        description="Get current weather conditions for a location",
        name="get_current_weather"
    )

    restaurant_tool = FunctionTool(
        func=get_restaurant_recommendations,
        description="Get restaurant recommendations by cuisine type",
        name="get_restaurant_recommendations"
    )

    # Create assistant agent with both tools
    assistant = AssistantAgent(
        "assistant",
        model_client=client,
        tools=[weather_tool, restaurant_tool],
        system_message="""You are a helpful AI assistant that can:
        - Get current weather information
        - Recommend restaurants
        Reply "TERMINATE" when the task is done."""
    )

    # Termination condition
    termination_condition = TextMessageTermination("assistant")

    # Create the team
    team = RoundRobinGroupChat(
        [assistant],
        termination_condition=termination_condition,
    )

    # Start the conversation
    task = "What's the weather in New York and recommend some Italian restaurants there"
    stream = team.run_stream(task=task)
    
    # Use Console to display the conversation
    await Console(stream)
    
    # Close the client when done
    await client.close()

if __name__ == "__main__":
    asyncio.run(main())