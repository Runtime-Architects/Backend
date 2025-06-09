from typing import List, Dict, Optional
from dataclasses import dataclass
import os
from cerebras.cloud.sdk import Cerebras  # Import Cerebras SDK

@dataclass
class CerebrasResponse:
    choices: List[Dict]
    model: str
    usage: Optional[Dict] = None

class CustomModelClient:
    def __init__(self, config: Dict, **kwargs):
        """
        Initialize the client.
        
        Args:
            config: Configuration dictionary with:
                - api_key: API key 
                - model: Model name to use
        """        

        if "api_key" in config:
            self.api_key = config.get("api_key")
        else:
            self.api_key = None

        # if not self.api_key:
        #     raise ValueError("Cerebras API key not provided in config_list")

        self.model_name = config.get("model", "llama-4-scout-17b-16e-instruct")
        self.client = Cerebras(api_key=self.api_key)
        print(f"Initialized Cerebras client for model: {self.model_name}")

    def create(self, params: Dict) -> CerebrasResponse:
        """
        Create a chat completion.
        
        Args:
            params: Dictionary containing:
                - messages: List of message dictionaries
                - other optional parameters like temperature, max_tokens, etc.
                
        Returns:
            CerebrasResponse object
        """

        #print("Incoming params:", params)

        try:
            # Merge instance params with call-specific params
            request_params = {
                "model": self.model_name,
                "messages": params["messages"],
                # add any necessary parameters
            }
            
            # Remove None values
            request_params = {k: v for k, v in request_params.items() if v is not None}
            
            # Make the API call
            response = self.client.chat.completions.create(**request_params)
            
            # Convert to standardized response format
            return CerebrasResponse(
                choices=[{
                    "message": {
                        "content": choice.message.content,
                        "role": choice.message.role,
                    },
                    "finish_reason": choice.finish_reason,
                } for choice in response.choices],
                model=response.model,
                usage=getattr(response, "usage", None),
            )
            
        except Exception as e:
            raise RuntimeError(f"Cerebras API error: {str(e)}") from e
            
    def message_retrieval(self, response: CerebrasResponse) -> List[str]:
        """Retrieve messages from the response."""
        # print("Response:", response)
        return [choice["message"]["content"] for choice in response.choices]
        
    def cost(self, response: CerebrasResponse) -> float:
        """Calculate the cost of the response."""
        # Implement your cost calculation based on Cerebras pricing
        return 0  
        
    @staticmethod
    def get_usage(response: CerebrasResponse) -> Dict:
        """Get usage statistics if available."""
        return response.usage or {}