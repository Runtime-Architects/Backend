import asyncio
from dataclasses import dataclass
from autogen_core import AgentId, MessageContext, RoutedAgent, message_handler, SingleThreadedAgentRuntime

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
import os

os.environ["OPENAI_API_KEY"] = "OPENAI_KEY"

@dataclass
class MyMessageType:
    content: str


class MyAssistant(RoutedAgent):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")
        self._delegate = AssistantAgent(name, model_client=model_client)

    @message_handler
    async def handle_my_message_type(self, message: MyMessageType, ctx: MessageContext) -> None:
        print(f"{self.id.type} received message: {message.content}")
        response = await self._delegate.on_messages(
            [TextMessage(content=message.content, source="user")],
            ctx.cancellation_token
        )
        print(f"{self.id.type} responded: {response.chat_message}")


class MyAgent(RoutedAgent):
    def __init__(self) -> None:
        super().__init__("MyAgent")

    @message_handler
    async def handle_my_message_type(self, message: MyMessageType, ctx: MessageContext) -> None:
        print(f"{self.id.type} received message: {message.content}")


async def main():
    runtime = SingleThreadedAgentRuntime()
    
    # Register agents
    await MyAgent.register(runtime, "my_agent", lambda: MyAgent())
    await MyAssistant.register(runtime, "my_assistant", lambda: MyAssistant("my_assistant"))

    # Start runtime in background
    runtime.start()

    # Send messages
    await runtime.send_message(MyMessageType("Hello, World!"), AgentId("my_agent", "default"))
    await runtime.send_message(MyMessageType("Hello, Assistant!"), AgentId("my_assistant", "default"))

    await runtime.stop()


if __name__ == "__main__":
    asyncio.run(main())
