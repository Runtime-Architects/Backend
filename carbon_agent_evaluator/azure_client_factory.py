"""
Centralized Azure OpenAI client factory to eliminate duplicate client setup code.
"""

import os
from typing import Optional
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class AzureClientFactory:
    """Factory class for creating Azure OpenAI clients with consistent configuration."""
    
    _client_instance: Optional[AzureOpenAIChatCompletionClient] = None
    
    @classmethod
    def get_client(cls, max_completion_tokens: Optional[int] = None) -> AzureOpenAIChatCompletionClient:
        """
        Get a configured Azure OpenAI client instance.
        
        Args:
            max_completion_tokens: Optional maximum completion tokens override
            
        Returns:
            AzureOpenAIChatCompletionClient: Configured client instance
            
        Raises:
            ValueError: If required environment variables are missing
        """
        # Check for required environment variables
        required_vars = [
            "AZURE_AI_ENDPOINT",
            "AZURE_AI_MODEL", 
            "AZURE_AI_DEPLOYMENT",
            "AZURE_AI_API_KEY",
            "AZURE_AI_API_VERSION"
        ]
        
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        # Create client configuration
        client_config = {
            "azure_deployment": os.getenv("AZURE_AI_DEPLOYMENT"),
            "model": os.getenv("AZURE_AI_MODEL"),
            "api_version": os.getenv("AZURE_AI_API_VERSION"),
            "azure_endpoint": os.getenv("AZURE_AI_ENDPOINT"),
            "api_key": os.getenv("AZURE_AI_API_KEY")
        }
        
        # Add max_completion_tokens if specified
        if max_completion_tokens is not None:
            client_config["max_completion_tokens"] = max_completion_tokens
            
        return AzureOpenAIChatCompletionClient(**client_config)
    
    @classmethod
    def get_shared_client(cls, max_completion_tokens: Optional[int] = None) -> AzureOpenAIChatCompletionClient:
        """
        Get a shared Azure OpenAI client instance (singleton pattern).
        
        Args:
            max_completion_tokens: Optional maximum completion tokens override
            
        Returns:
            AzureOpenAIChatCompletionClient: Shared client instance
        """
        if cls._client_instance is None:
            cls._client_instance = cls.get_client(max_completion_tokens)
        return cls._client_instance
    
    @classmethod
    def reset_shared_client(cls):
        """Reset the shared client instance (useful for testing)."""
        cls._client_instance = None


# Convenience functions for backward compatibility
def create_azure_client(max_completion_tokens: Optional[int] = None) -> AzureOpenAIChatCompletionClient:
    """
    Create a new Azure OpenAI client instance.
    
    Args:
        max_completion_tokens: Optional maximum completion tokens
        
    Returns:
        AzureOpenAIChatCompletionClient: New client instance
    """
    return AzureClientFactory.get_client(max_completion_tokens)


def get_shared_azure_client(max_completion_tokens: Optional[int] = None) -> AzureOpenAIChatCompletionClient:
    """
    Get the shared Azure OpenAI client instance.
    
    Args:
        max_completion_tokens: Optional maximum completion tokens
        
    Returns:
        AzureOpenAIChatCompletionClient: Shared client instance
    """
    return AzureClientFactory.get_shared_client(max_completion_tokens)