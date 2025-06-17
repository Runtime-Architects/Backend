import autogen
from autogen import AssistantAgent, UserProxyAgent
from typing import Optional, Union, Dict, Any
from src.client import CustomModelClient
import os
from pathlib import Path


class ChatAgent:
    """
    A class to manage conversational agents using AutoGen with custom model clients.
    
    Attributes:
        config_list_custom (list): Configuration list for the custom model
        assistant (AssistantAgent): The assistant agent
        user_proxy (UserProxyAgent): The user proxy agent
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the ChatAgent with custom configuration.
        
        Args:
            config_path: Path to the configuration JSON file. If None, looks in package directory.
        """

        # Resolve the config file path
        if config_path is None:
            # Look in package directory
            config_file = Path(__file__).parent / "OAI_CONFIG_LIST"
        else:
            config_file = Path(config_path)

        # Verify file exists
        if not config_file.exists():
            raise FileNotFoundError(
                f"Config file not found at: {config_file.absolute()}\n"
                f"Current working directory: {os.getcwd()}\n"
                f"Package directory: {Path(__file__).parent}"
            )

        # Read config - convert Path to string for AutoGen
        self.config_list_custom = autogen.config_list_from_json(
            str(config_file),  
            filter_dict={"model_client_cls": ["CustomModelClient"]}
        )
        
        self.assistant = self._create_assistant()
        self.user_proxy = self._create_user_proxy()
        self._register_client()
    
    def _create_assistant(self) -> AssistantAgent:
        """Create and configure the assistant agent."""
        return AssistantAgent(
            "assistant",
            llm_config={"config_list": self.config_list_custom}
        )
    
    def _create_user_proxy(self) -> UserProxyAgent:
        """Create and configure the user proxy agent."""
        return UserProxyAgent(
            "user_proxy",
            human_input_mode="NEVER",
            is_termination_msg=lambda x: "TERMINATE" in x.get("content", ""),
            code_execution_config={
                "work_dir": "coding",
                "use_docker": False,
            }
        )
    
    def _register_client(self) -> None:
        """Register the custom model client with the assistant."""
        self.assistant.register_model_client(model_client_cls=CustomModelClient)
    
    def chat(self, message: str, summary_method: str = "last_msg") -> Optional[Dict[str, Any]]:
        """
        Initiate a chat conversation and return the result.
        
        Args:
            message (str): The input message/task for the agent
            summary_method (str): Method to summarize the chat ("last_msg" or "reflection_with_llm")
            
        Returns:
            dict or None: The chat result summary or None if failed
        """
        try:
            chat_result = self.user_proxy.initiate_chat(
                self.assistant,
                message=message,
                summary_method=summary_method
            )
            return chat_result
        except Exception as e:
            print(f"Error during chat: {e}")
            return None
    
    def get_final_response(self, message: str) -> str:
        """
        Get the final response from a chat conversation.
        
        Args:
            message (str): The input message/task for the agent
            
        Returns:
            str: The final response content
        """
        result = self.chat(message)
        # if result and hasattr(result, 'summary'):
        #     return str(result.summary)
        # return "No response generated"