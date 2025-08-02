from autogen_ext.models.openai import AzureOpenAIChatCompletionClient

class AzureClientFactory:
    def __init__(self, *, azure_deployment, model, api_version, azure_endpoint, api_key, max_completion_tokens=1024):
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
