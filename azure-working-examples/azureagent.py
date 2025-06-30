import asyncio
import os

from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from autogen_ext.agents.azure._azure_ai_agent import AzureAIAgent
from azure.ai.projects.aio import AIProjectClient
from azure.identity.aio import DefaultAzureCredential
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
import dotenv


async def bing_example():
    async with DefaultAzureCredential() as credential:
        # async with AIProjectClient(  # type: ignore
        #     credential=credential, endpoint=os.getenv("AZURE_PROJECT_ENDPOINT", "")
        # ) as project_client:

        client = AzureOpenAIChatCompletionClient(
                    azure_deployment=os.getenv("AZURE_DEPLOYMENT"),
                    model=os.getenv("MODEL"),
                    api_version=os.getenv("API_VERSION"),
                    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
                    api_key= os.getenv("API_KEY"), # For key-based authentication.
                )
            
        agent_with_bing_grounding = AzureAIAgent(
            name="bing_agent",
            description="You are An AI assistant ",
            project_client=client,
            deployment_name="gpt-4o",
            instructions="You are a helpful assistant.",

            metadata={"source": "AzureAIAgent"},
        )

        # For the bing grounding tool to return the citations, the message must contain an instruction for the model to do return them.
        # For example: "Please provide citations for the answers"

        result = await agent_with_bing_grounding.on_messages(
            messages=[
                TextMessage(
                    content="What is Microsoft\'s annual leave policy? Provide citations for your answers.",
                    source="user",
                )
            ],
            cancellation_token=CancellationToken(),
            message_limit=5,
        )
        print(result)


if __name__ == "__main__":
    dotenv.load_dotenv()
    asyncio.run(bing_example())
