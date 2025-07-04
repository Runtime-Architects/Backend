# Streaming AutoGen API

This document explains how to use the new streaming functionality in the AutoGen Business Insights API.

## Overview

The streaming API provides real-time updates during the agent processing workflow, allowing clients to see progress and intermediate results before the final response is ready.

## New Endpoints

### 1. `/ask-stream` (POST)
**Streaming endpoint** that provides real-time updates during agent processing.

**Request:**
```json
{
    "question": "Generate a monthly business report for our sales performance"
}
```

**Response:** Server-Sent Events (SSE) stream with the following event types:

- `started` - Task initialization
- `agent_thinking` - Agent is processing/thinking
- `agent_response` - Agent has provided a response
- `tool_execution` - Tool is being executed
- `completed` - Task completed with final response
- `error` - Error occurred during processing

### 2. `/test-stream` (GET)
**Test endpoint** for verifying streaming functionality with mock events.

### 3. `/client` (GET)
**Web client** - Interactive HTML interface for testing the streaming API.

## Event Structure

Each streaming event follows this structure:

```json
{
    "event": {
        "event_type": "agent_thinking",
        "timestamp": "2025-06-29T10:30:00",
        "agent_name": "PlanningAgent",
        "message": "Planning task breakdown...",
        "data": {
            "progress": 25,
            "additional_info": "..."
        }
    }
}
```

## Event Types

| Event Type | Description | Data Fields |
|------------|-------------|-------------|
| `started` | Task initialization | `question`, `progress` |
| `agent_thinking` | Agent processing | `progress` |
| `agent_response` | Agent completed subtask | `progress` |
| `tool_execution` | External tool running | `tool_name`, `parameters` |
| `completed` | Final response ready | `final_response`, `total_messages`, `response_length` |
| `error` | Error occurred | `error`, `progress` |

## Logging

The streaming system includes comprehensive logging:

### Stream Event Logger
- **Logger Name:** `stream_events`
- **Format:** `%(asctime)s - STREAM - %(levelname)s - %(message)s`
- **Purpose:** Tracks all streaming events for debugging

### Application Logger
- **Logger Name:** `__main__`
- **Format:** `%(asctime)s - %(name)s - %(levelname)s - %(message)s`
- **Purpose:** General application logging

## Usage Examples

### 1. Python Client (using aiohttp)

```python
import asyncio
import aiohttp
import json

async def stream_question(question: str):
    url = "http://localhost:8000/ask-stream"
    payload = {"question": question}
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as response:
            async for line in response.content:
                line = line.decode('utf-8').strip()
                if line.startswith('data: '):
                    event = json.loads(line[6:])
                    print(f"{event['event']['agent_name']}: {event['event']['message']}")

# Run the client
asyncio.run(stream_question("Generate a sales report"))
```

### 2. JavaScript/Browser Client

```javascript
async function streamQuestion(question) {
    const response = await fetch('/ask-stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question })
    });
    
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    
    while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');
        
        for (const line of lines) {
            if (line.startsWith('data: ')) {
                const event = JSON.parse(line.slice(6));
                console.log(`${event.event.agent_name}: ${event.event.message}`);
            }
        }
    }
}
```

### 3. curl Command

```bash
curl -N -X POST http://localhost:8000/ask-stream \
     -H "Content-Type: application/json" \
     -d '{"question": "Generate a business report"}'
```

## Running the Examples

### 1. Start the Server
```bash
cd src
python main.py
```

### 2. Use the Web Client
Open http://localhost:8000/client in your browser

### 3. Run Python Client
```bash
python streaming_client_example.py
```

### 4. Test Simple Stream
```bash
python streaming_client_example.py test
```

## Configuration

### Environment Variables
- `OPENAI_API_KEY` - Your OpenAI API key
- `LOG_LEVEL` - Logging level (default: INFO)

### Streaming Settings
- **Max Messages:** 40 (configurable in AutoGen termination condition)
- **Progress Steps:** 4 main workflow steps
- **Event Buffer:** Events are streamed in real-time without buffering

## Error Handling

The streaming API includes comprehensive error handling:

1. **Network Errors:** Automatically retry connection
2. **JSON Parse Errors:** Log and continue processing
3. **Agent Errors:** Emit error events with details
4. **Timeout Handling:** Built-in timeout for long-running tasks

## Performance Considerations

- **Memory Usage:** Events are not stored permanently, only temporarily for processing
- **Connection Limits:** Each stream maintains one persistent connection
- **Concurrent Streams:** Multiple clients can stream simultaneously
- **Resource Cleanup:** Connections are automatically cleaned up on completion or error

## Troubleshooting

### Common Issues

1. **Connection Drops**
   ```
   Solution: Check network stability and server logs
   ```

2. **No Events Received**
   ```
   Solution: Verify Content-Type headers and SSE client implementation
   ```

3. **JSON Parse Errors**
   ```
   Solution: Check event format and encoding
   ```

### Debug Mode

Enable debug logging:
```python
import logging
logging.getLogger('stream_events').setLevel(logging.DEBUG)
```

### Monitoring

Check server logs for streaming activity:
```bash
tail -f server.log | grep "STREAM"
```

## Security Considerations

- **CORS:** Configured to allow all origins (update for production)
- **Rate Limiting:** Not implemented (add for production)
- **Authentication:** Not implemented (add as needed)
- **Input Validation:** Basic validation on question length and content

## Future Enhancements

1. **WebSocket Support:** For bidirectional communication
2. **Event Replay:** Store and replay event streams
3. **Filtering:** Client-side event filtering options
4. **Compression:** Gzip compression for large responses
5. **Metrics:** Detailed performance metrics and monitoring
