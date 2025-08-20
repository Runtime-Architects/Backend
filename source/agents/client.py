"""
client.py

This module creates and configures the Azure client used by Autogen Agents
to interact with Azure AI models.
"""

from autogen_ext.models.openai import AzureOpenAIChatCompletionClient


class AzureClientFactory:
    """AzureClientFactory is a factory class for creating an Azure OpenAI client.

    This class initializes with the necessary parameters to connect to the Azure OpenAI service and provides a method to retrieve the client instance.

    Attributes:
        azure_deployment (str): The name of the Azure deployment.
        model (str): The model to be used for the Azure OpenAI service.
        api_version (str): The version of the API to be used.
        azure_endpoint (str): The endpoint URL for the Azure service.
        api_key (str): The API key for authenticating with the Azure service.
        max_completion_tokens (int): The maximum number of tokens for completion (default is 1024).
        _client (AzureOpenAIChatCompletionClient or None): The client instance for Azure OpenAI, initialized lazily.

    Methods:
        get_client(): Returns the Azure OpenAI client instance, creating it if it does not already exist.
    """

    def __init__(
        self,
        *,
        azure_deployment,
        model,
        api_version,
        azure_endpoint,
        api_key,
        max_completion_tokens=1024
    ):
        self.azure_deployment = azure_deployment
        self.model = model
        self.api_version = api_version
        self.azure_endpoint = azure_endpoint
        self.api_key = api_key
        self.max_completion_tokens = max_completion_tokens
        self._client = None

    def get_client(self):
        if self._client is None:
            self._client = AzureOpenAIChatCompletionClient(
                azure_deployment=self.azure_deployment,
                model=self.model,
                api_version=self.api_version,
                azure_endpoint=self.azure_endpoint,
                api_key=self.api_key,
                max_completion_tokens=self.max_completion_tokens,
            )
        return self._client
