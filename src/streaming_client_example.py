#!/usr/bin/env python3
"""
Example client for testing the streaming endpoint
"""

import asyncio
import aiohttp
import json
import sys
from typing import AsyncGenerator


async def stream_question(question: str, base_url: str = "http://localhost:8000") -> AsyncGenerator[dict, None]:
    """
    Stream a question to the API and yield events as they arrive
    """
    url = f"{base_url}/ask-stream"
    payload = {"question": question}
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as response:
            if response.status != 200:
                raise Exception(f"HTTP {response.status}: {await response.text()}")
            
            async for line in response.content:
                line = line.decode('utf-8').strip()
                if line.startswith('data: '):
                    data = line[6:]  # Remove 'data: ' prefix
                    try:
                        event = json.loads(data)
                        yield event
                    except json.JSONDecodeError:
                        print(f"Failed to parse JSON: {data}")


async def test_streaming():
    """
    Test the streaming endpoint with a sample question
    """
    question = "Generate a monthly business report for our sales performance and carbon footprint analysis"
    
    print(f"🚀 Starting streaming request...")
    print(f"📝 Question: {question}")
    print("=" * 80)
    
    try:
        async for event_data in stream_question(question):
            event = event_data.get('event', {})
            event_type = event.get('event_type', 'unknown')
            timestamp = event.get('timestamp', '')
            agent_name = event.get('agent_name', 'System')
            message = event.get('message', '')
            data = event.get('data', {})
            
            # Format timestamp for display
            time_display = timestamp.split('T')[1][:8] if 'T' in timestamp else timestamp
            
            # Color coding based on event type
            if event_type == 'started':
                print(f"🟢 [{time_display}] {agent_name}: {message}")
            elif event_type == 'agent_thinking':
                print(f"🟡 [{time_display}] {agent_name}: {message}")
                if 'progress' in data:
                    print(f"   📊 Progress: {data['progress']}%")
            elif event_type == 'agent_response':
                print(f"🔵 [{time_display}] {agent_name}: {message}")
                if 'progress' in data:
                    print(f"   📊 Progress: {data['progress']}%")
            elif event_type == 'tool_execution':
                print(f"🔧 [{time_display}] {agent_name}: {message}")
            elif event_type == 'completed':
                print(f"✅ [{time_display}] {agent_name}: {message}")
                if 'final_response' in data:
                    print("\n" + "=" * 80)
                    print("📄 FINAL RESPONSE:")
                    print("=" * 80)
                    print(data['final_response'][:500] + "..." if len(data['final_response']) > 500 else data['final_response'])
                    print("=" * 80)
                if 'response_length' in data:
                    print(f"📏 Response length: {data['response_length']} characters")
                if 'total_messages' in data:
                    print(f"💬 Total messages: {data['total_messages']}")
            elif event_type == 'error':
                print(f"❌ [{time_display}] ERROR: {message}")
                if 'error' in data:
                    print(f"   🔍 Details: {data['error']}")
            else:
                print(f"❓ [{time_display}] {agent_name}: {message} (Type: {event_type})")
            
            print()  # Empty line for readability
            
    except Exception as e:
        print(f"❌ Error during streaming: {e}")


async def test_simple_stream():
    """
    Test the simple test stream endpoint
    """
    print("🧪 Testing simple stream endpoint...")
    print("=" * 80)
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:8000/test-stream") as response:
                if response.status != 200:
                    raise Exception(f"HTTP {response.status}: {await response.text()}")
                
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if line.startswith('data: '):
                        data = line[6:]  # Remove 'data: ' prefix
                        try:
                            event_data = json.loads(data)
                            event = event_data.get('event', {})
                            print(f"📡 {event.get('agent_name', 'System')}: {event.get('message', 'No message')}")
                        except json.JSONDecodeError:
                            print(f"Failed to parse JSON: {data}")
                            
    except Exception as e:
        print(f"❌ Error during test streaming: {e}")


async def main():
    """
    Main function to run the streaming client
    """
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            await test_simple_stream()
        else:
            question = " ".join(sys.argv[1:])
            await stream_question_demo(question)
    else:
        await test_streaming()


async def stream_question_demo(question: str):
    """
    Demo streaming with a custom question
    """
    print(f"🚀 Streaming custom question...")
    print(f"📝 Question: {question}")
    print("=" * 80)
    
    async for event_data in stream_question(question):
        event = event_data.get('event', {})
        print(f"📡 {event.get('agent_name', 'System')}: {event.get('message', 'No message')}")


if __name__ == "__main__":
    asyncio.run(main())
