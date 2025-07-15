import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_core.tools import FunctionTool
from autogen_agentchat.ui import Console
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_core import CancellationToken
import os
from dotenv import load_dotenv

load_dotenv()

async def get_current_weather(location: str) -> str:
    """Get the current weather in a given location"""
    return f"The weather in {location} is sunny and 72°F"

async def get_restaurant_recommendations(location: str, cuisine: str) -> str:
    """Get restaurant recommendations in a given location"""
    return f"Top {cuisine} restaurants in {location}: 1. Great Food Place 2. Tasty Eats"

async def main():
    client = AzureOpenAIChatCompletionClient(
        azure_deployment=os.getenv("AZURE_DEPLOYMENT"),
        model=os.getenv("MODEL"),
        api_version=os.getenv("API_VERSION"),
        azure_endpoint=os.getenv("AZURE_ENDPOINT"),
        api_key=os.getenv("API_KEY"),
    )

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

    assistant = AssistantAgent(
    "assistant",
    model_client=client,
    tools=[weather_tool, restaurant_tool],
    system_message="""You are a helpful AI assistant that can:
        - Get current weather information
        - Recommend restaurants
        Reply "TERMINATE" when the task is done.""",
    reflect_on_tool_use= True,
    max_tool_iterations=10)

    # task
    task = """What's the weather in New York and recommend some Italian restaurants there"""
    
    initial_message = TextMessage(content=task, source="user")
    
    stream = assistant.on_messages_stream(
        messages=[initial_message],
        cancellation_token=CancellationToken()
    )
    await Console(stream)

    
    await client.close()

if __name__ == "__main__":
    asyncio.run(main())