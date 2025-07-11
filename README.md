# Backend Repository
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/f58da717b7b94463aa05b2e01fb437ff)](https://app.codacy.com/gh/Runtime-Architects/Backend/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)

This repository contains the core backend logic for our AI-powered business insights platform. It leverages AutoGen agents to generate comprehensive business reports with real-time streaming capabilities and WebAuthn passwordless authentication.

## 🐍 Requirements

- Python 3.10+
- OpenAI API Key
- SQLite (for local development)

## 📁 Architecture Overview

The backend is built using **FastAPI** with a multi-agent architecture powered by **AutoGen** and **OpenAI Assistants**. It provides:

- **Passwordless Authentication** using WebAuthn/FIDO2
- **Real-time Streaming** of agent processing via Server-Sent Events
- **Multi-Agent Workflow** for business intelligence generation
- **Carbon Footprint Analysis** with synthetic data generation

### Core Components

- **[`main.py`](src/main.py)** - FastAPI application with AutoGen agent orchestration
- **[`auth_routes.py`](src/auth_routes.py)** - WebAuthn authentication endpoints
- **[`models.py`](src/models.py)** - SQLModel database models
- **[`streamer.py`](src/streamer.py)** - Server-Sent Events streaming infrastructure
- **[`webauthn_service.py`](src/webauthn_service.py)** - WebAuthn service implementation
- **[`db.py`](src/db.py)** - Database connection and session management

## 🤖 Agent Architecture

The system uses four specialized AutoGen agents:

1. **PlanningAgent** - Orchestrates the workflow and breaks down complex tasks
2. **CarbonAgent** - Generates synthetic data tables and carbon footprint estimates
3. **DataAnalysisAgent** - Analyzes raw data and generates insights
4. **ReportAgent** - Creates ASCII dashboard reports with visualizations

## 🔐 Authentication System

### WebAuthn Passwordless Authentication

The system uses **WebAuthn/FIDO2** for secure, passwordless authentication with biometric support.

#### Authentication Flow:
1. **Registration**: Users register with email and create a passkey
2. **Login**: Users authenticate using their passkey (biometrics/PIN)
3. **JWT Tokens**: Successful authentication returns a JWT access token
4. **Session Management**: Tokens can be refreshed and blacklisted on logout

#### Database Models:
- **[`User`](src/models.py)** - User accounts with email
- **[`Credential`](src/models.py)** - WebAuthn credentials linked to users

## 🚀 API Endpoints

### Authentication Endpoints (`/auth`)

#### Registration
```http
POST /auth/register/begin
Content-Type: application/json

{
  "email": "user@example.com"
}
```

```http
POST /auth/register/complete
Content-Type: application/json

{
  "id": "credential_id",
  "response": {
    "clientDataJSON": "...",
    "attestationObject": "..."
  }
}
```

#### Login
```http
POST /auth/login/begin
Content-Type: application/json

{
  "email": "user@example.com"
}
```

```http
POST /auth/login/complete
Content-Type: application/json

{
  "id": "credential_id",
  "response": {
    "clientDataJSON": "...",
    "authenticatorData": "...",
    "signature": "..."
  }
}
```

#### Session Management
```http
POST /auth/logout
Authorization: Bearer <token>
```

```http
GET /auth/me
Authorization: Bearer <token>
```

```http
GET /auth/refresh-token
Authorization: Bearer <token>
```

### Business Intelligence Endpoints

#### Standard Request
```http
POST /ask
Content-Type: application/json

{
  "question": "Generate a monthly business report for our sales performance"
}
```

#### Streaming Request (Real-time)
```http
POST /ask-stream
Content-Type: application/json

{
  "question": "Analyze our carbon footprint and generate insights"
}
```

#### Health Check
```http
GET /health
```

Returns system status including:
- Agent initialization status
- OpenAI API connectivity
- Database status
- Configuration validation

#### Test Endpoints
```http
GET /test-stream
```

```http
GET /client
```

## 📊 Streaming Events

The streaming API provides real-time updates via Server-Sent Events:

### Event Types
- **`started`** - Task initialization
- **`agent_thinking`** - Agent processing
- **`agent_response`** - Agent completed subtask
- **`tool_execution`** - External tool execution
- **`completed`** - Final response ready
- **`error`** - Error occurred

### Event Structure
```json
{
  "event": {
    "event_type": "agent_thinking",
    "timestamp": "2025-01-27T10:30:00",
    "agent_name": "PlanningAgent",
    "message": "Planning task breakdown...",
    "data": {
      "progress": 25,
      "additional_info": "..."
    }
  }
}
```

## 🛠️ Setup and Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Backend
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r src/requirements.txt
   ```

4. **Environment Configuration**:
   Create a `.env` file in the root directory:
   ```env
   OPENAI_API_KEY=sk-your-openai-api-key
   OPENAI_MODEL=gpt-4o-mini
   OPENAI_CARBON_ASSISTANT_ID=asst_your_carbon_assistant_id
   OPENAI_ANALYSIS_ASSISTANT_ID=asst_your_analysis_assistant_id
   OPENAI_REPORT_ASSISTANT_ID=asst_your_report_assistant_id
   ```

5. **Run the application**:
   ```bash
   cd src
   python main.py
   ```

The API will be available at `http://localhost:8000`

## 🧪 Testing

### Web Client
Visit `http://localhost:8000/client` for an interactive testing interface.

### API Testing
```bash
# Test health endpoint
curl http://localhost:8000/health

# Test streaming endpoint
curl http://localhost:8000/test-stream
```

### Authentication Testing
The WebAuthn implementation requires HTTPS in production. For development, use:
- **Origin**: `http://localhost:5001`
- **RP ID**: `localhost`

## 📝 Configuration

### WebAuthn Settings
- **RP ID**: `localhost` (development)
- **RP Name**: `Sustainable City AI App`
- **Origin**: `http://localhost:5001`
- **Supported Algorithms**: ECDSA_SHA_256, RSASSA_PKCS1_v1_5_SHA_256

### Agent Configuration
- **Max Messages**: 40 per conversation
- **Temperature**: 0 (deterministic responses)
- **Model**: `gpt-4o-mini`
- **Tools**: Code interpreter, Carbon footprint estimator

## 📚 API Documentation

Once running, visit:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## 🔒 Security Features

- **Passwordless Authentication** with WebAuthn/FIDO2
- **JWT Token Management** with blacklisting
- **CORS Configuration** for cross-origin requests
- **Input Validation** with Pydantic models
- **Secure Database** with SQLModel ORM

## 📈 Performance

- **Streaming Response Times**: Real-time event delivery
- **Database**: SQLite for development, easily scalable to PostgreSQL
- **Connection Pooling**: Automatic session management
- **Error Handling**: Comprehensive error logging and user feedback

## 🤝 Contributing

We follow strict contribution guidelines including Conventional Commits and branching conventions.

👉 See the [CONTRIBUTING.md](./CONTRIBUTING.md) file for full details.

## 📦 Production Deployment

For production deployment:

1. **Environment Variables**:
   - Set secure `OPENAI_API_KEY`
   - Configure production database URL
   - Set appropriate CORS origins

2. **Database Migration**:
   ```bash
   # The app automatically creates tables on startup
   ```

3. **HTTPS Configuration**:
   - WebAuthn requires HTTPS in production
   - Update RP ID and origin accordingly

4. **Monitoring**:
   - Check `/health` endpoint for system status
   - Monitor streaming event logs
   - Set up proper logging infrastructure

## 🔧 Troubleshooting

### Common Issues

1. **WebAuthn Registration Fails**:
   - Check browser compatibility
   - Verify HTTPS configuration
   - Ensure proper origin settings

2. **Agent Initialization Errors**:
   - Verify OpenAI API key
   - Check assistant IDs configuration
   - Monitor `/health` endpoint

3. **Streaming Connection Issues**:
   - Check CORS configuration
   - Verify event stream client implementation
   - Monitor network connectivity
