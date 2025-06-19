import os
import time
from dotenv import load_dotenv
from autogen import AssistantAgent, UserProxyAgent

class PolicyAgent:
    def __init__(self):
        self.config = None
        self.assistant = None
        self.user_proxy = None

    def initialize(self):
        load_dotenv()
        self.config = self.load_config()
        print("Agent initialized with configuration:", self.config)

        # Define LLM config
        llm_config = {
            "model": "gpt-4o",
            "api_key": os.getenv("OPENAI_API_KEY")
        }

        # Create assistant agent
        self.assistant = AssistantAgent(name="PolicyAssistant", llm_config=llm_config)

        # Create user proxy agent with safe auto-reply limit
        self.user_proxy = UserProxyAgent(
            name="User",
            human_input_mode="NEVER",
            code_execution_config={"use_docker": False},
            max_consecutive_auto_reply=1
        )

    def run(self):
        print("Agent is running...")

        task = "What is the capital of France?"

        # Start a conversation and get the final message only
        chat_result = self.user_proxy.initiate_chat(
            self.assistant,
            message=task,
            summary_method="last_msg"  # avoids full chat loop
        )

        # Explicitly print the final message
        if chat_result:
            print("\nFinal response:\n", chat_result.summary)

    def shutdown(self):
        print("Shutting down the agent...")

    def load_config(self):
        return {"setting1": "value1", "setting2": "value2"}
    
if __name__ == "__main__":
    agent = PolicyAgent()
    agent.initialize()
    agent.run()