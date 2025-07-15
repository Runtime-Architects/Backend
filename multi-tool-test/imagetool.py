from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import MultiModalMessage
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_agentchat.messages import TextMessage
from autogen_core.tools import FunctionTool
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken
import asyncio
import os
from dotenv import load_dotenv
load_dotenv()
import pytesseract
from PIL import Image
import io
import requests

pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# Register OCR tool for the agent
async def extract_text_from_image(image_url: str) -> str:
    """Extract text from an image URL using OCR."""
    try:
        # Download image
        response = requests.get(image_url)
        img = Image.open(io.BytesIO(response.content))
        
        # Use pytesseract to extract text
        text = pytesseract.image_to_string(img)
        return text
    except Exception as e:
        return f"Error: {str(e)}"
    
image_tool = FunctionTool(
        func= extract_text_from_image,
        description="Extracts texts from image.",
        name="extract_text_from_image"
    )
    
async def main():
    client = AzureOpenAIChatCompletionClient(
            azure_deployment=os.getenv("AZURE_DEPLOYMENT"),
            model=os.getenv("MODEL"),
            api_version=os.getenv("API_VERSION"),
            azure_endpoint=os.getenv("AZURE_ENDPOINT"),
            api_key=os.getenv("API_KEY"),
        )

    # Initialize agents

    assistant = AssistantAgent(
        name="Bill_Parser",
        model_client=client,
        tools=[image_tool],
        system_message="You extract and analyze data from electricity bills. Return structured JSON.",
        reflect_on_tool_use= True,
        max_tool_iterations=5)
    

    # Example: Process an electricity bill image
    image_url = "https://www.reddit.com/media?url=https%3A%2F%2Fi.redd.it%2Fneed-help-understanding-my-electricity-bill-in-dublin-v0-oxl9llwdg1rb1.jpg%3Fwidth%3D2603%26format%3Dpjpg%26auto%3Dwebp%26s%3Dd8b97e548315dd5801c8f7cb7bc714d28abe06f1"
    task = f"""
    Extract the following from this electricity bill image ({image_url}):
    1. Total kWh usage
    2. Billing period (start/end dates)
    3. Total amount due
    Return as JSON.
    """
    
    initial_message = TextMessage(content=task, source="user")
    
    stream = assistant.on_messages_stream(
        messages=[initial_message],
        cancellation_token=CancellationToken()
    )
    await Console(stream)



if __name__ == "__main__":
    asyncio.run(main())
