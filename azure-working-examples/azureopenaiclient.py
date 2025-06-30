import asyncio
import os
from azure.core.credentials import AzureKeyCredential
from autogen_core.models import UserMessage
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from azure.identity import DefaultAzureCredential

async def main():

    client = AzureOpenAIChatCompletionClient(
    azure_deployment="o4-mini",
    model="o4-mini",
    api_version="2024-12-01-preview",
    azure_endpoint="https://aihubdev9686754233.cognitiveservices.azure.com/",
     api_key="8MTxMp4CoEETInwK1knpw25ZODX11asb1g3KgiN5QPztmcRar7uwJQQJ99BFAC5RqLJXJ3w3AAAAACOGClk5", # For key-based authentication.
    )

    result = await client.create([UserMessage(content="What is the capital of France?", source="user")])
    print(result)

    # Close the client.
    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
