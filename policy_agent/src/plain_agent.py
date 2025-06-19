import os
from dotenv import load_dotenv
from openai import OpenAI  # v1+ SDK

class PolicyAgent:
    def __init__(self):
        self.config = None

    def initialize(self):
        load_dotenv()
        self.config = self.load_config()
        print("Agent initialized with configuration:", self.config)

    def run(self):
        print("Agent is running...")
        user_input = "What is the capital of France?" 
        response = self.query_openai(user_input)
        print("OpenAI Response:", response)

    def shutdown(self):
        print("Shutting down the agent...")

    def load_config(self):
        return {"setting1": "value1", "setting2": "value2"}

    def query_openai(self, prompt):
        print("Querying OpenAI...")
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100
        )
        return response.choices[0].message.content.strip()

if __name__ == "__main__":
    agent = PolicyAgent()
    agent.initialize()
    agent.run()