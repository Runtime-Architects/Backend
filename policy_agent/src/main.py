import os
from dotenv import load_dotenv
from autogen import AssistantAgent, UserProxyAgent
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any

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
            summary_method="last_msg"
        )

        # Explicitly print the final message
        if chat_result:
            print("\nFinal response:\n", chat_result.summary)

    def shutdown(self):
        print("Shutting down the agent...")

    def load_config(self):
        return {"setting1": "value1", "setting2": "value2"}

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify your frontend URL, e.g., ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = PolicyAgent()
agent.initialize()

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    message: Dict[str, Any]
    context: Any

def normalize_role(msg):
    if msg.get("role") == "assistant" and msg.get("name") == "User":
        return "user"
    elif msg.get("role") == "user" and msg.get("name") == "PolicyAssistant":
        return "assistant"
    else:
        return msg.get("role", "")

@app.post("/ask", response_model=QueryResponse)
def ask_agent(request: QueryRequest):
    try:
        chat_result = agent.user_proxy.initiate_chat(
            agent.assistant,
            message=request.question,
            summary_method=None
        )
        context = []
        message = {"content": "No response.", "role": ""}
        if chat_result and hasattr(chat_result, "chat_history"):
            for msg in chat_result.chat_history:
                norm_role = normalize_role(msg)
                context.append({
                    "content": msg.get("content", ""),
                    "role": norm_role
                })
            # Find the first non-empty assistant reply in the context
            for msg in context:
                if msg["role"] == "assistant" and msg["content"].strip():
                    message = msg
                    break
        return QueryResponse(message=message, context=context)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    agent.run()