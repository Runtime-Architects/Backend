import os
from dotenv import load_dotenv
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient

load_dotenv()

azure_ai_gpt_client = AzureOpenAIChatCompletionClient(
    azure_deployment=os.environ["AZURE_AI_DEPLOYMENT"],
    model=os.environ["AZURE_AI_MODEL"],
    api_version=os.environ["AZURE_AI_API_VERSION"],
    azure_endpoint=os.environ["AZURE_AI_ENDPOINT"],
    api_key=os.environ["AZURE_AI_API_KEY"],
    max_completion_tokens=1024,
)
