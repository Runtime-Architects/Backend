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

- **[`main.py`](source/main.py)** - FastAPI application with AutoGen agent orchestration
- **[`auth_routes.py`](source/api/auth_routes.py)** - WebAuthn authentication endpoints
- **[`models.py`](source/api/models.py)** - SQLModel database models
- **[`streamer.py`](source/api/streamer.py)** - Server-Sent Events streaming infrastructure
- **[`webauthn_service.py`](source/api/webauthn_service.py)** - WebAuthn service implementation
- **[`db.py`](source/api/db.py)** - Database connection and session management

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
- **[`User`](source/api/models.py)** - User accounts with email
- **[`Credential`](source/api/models.py)** - WebAuthn credentials linked to users

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

#### Basic Request (Real-time & Protected)
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
   pip install -r requirements.txt
   ```

4. **Environment Configuration**:
   Create a `.env` file in the root directory:
   ```env
   # Azure OpenAI Configuration
    AZURE_AI_DEPLOYMENT=gpt-4o
    AZURE_AI_MODEL=gpt-4o
    AZURE_AI_API_VERSION=2024-12-01-preview
    AZURE_AI_ENDPOINT=azure-endpoint
    AZURE_AI_API_KEY=azure-key
    # Database Configuration
    DATABASE_URL=cosmos-db-instante

    # Application Configuration
    APP_SECRET_KEY=your-secret-key-for-jwt-tokens-change-this-in-production
    APP_DEBUG=True
    APP_HOST=0.0.0.0
    APP_PORT=8000

    # WebAuthn Configuration
    WEBAUTHN_RP_ID=localhost
    WEBAUTHN_RP_NAME=Sustainable Development
    WEBAUTHN_ORIGIN=http://localhost:5001

    # CORS Configuration
    CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000

    # Security Configuration
    JWT_SECRET_KEY=your-jwt-secret-key-change-this-in-production
    JWT_ALGORITHM=HS256
    JWT_EXPIRATION_HOURS=24
   ```

5. **Run the application**:
   ```bash
   cd source
   python main.py
   ```

The API will be available by default at `http://localhost:8000`

## 🧪 Testing
```bash
# Test health endpoint
curl http://localhost:8000/health

### Authentication Testing
The WebAuthn implementation requires HTTPS in production. For development, use:
- **Origin**: `http://localhost:3000`
- **RP ID**: `localhost`

## 📝 Configuration

### WebAuthn Settings
- **RP ID**: `localhost` (development)
- **RP Name**: `Sustainable City AI App`
- **Origin**: `http://localhost:5001`
- **Supported Algorithms**: ECDSA_SHA_256, RSASSA_PKCS1_v1_5_SHA_256

You will need a Frontend instance to test and use WebAuthn features.

### Agent Configuration
- **Max Messages**: 40 per conversation
- **Temperature**: 0 (deterministic responses)
- **Model**: `gpt-4o-mini`
- **Tools**: Code interpreter, Carbon footprint estimator

## 📚 API Documentation

Once running, visit:
- **Swagger UI**: `http://localhost:8000/docs`

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
   - Set secure `AZURE` keys
   - Configure production database URL (Postgres Instance)
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
   - Set up proper logging infrastructure and inspect the Frontend logs for any inconsistencies.

## 🔧 Troubleshooting

### Common Issues

1. **WebAuthn Registration Fails**:
   - Check browser compatibility
   - Verify HTTPS configuration
   - Ensure proper origin settings and WebAuthn configuration

2. **Agent Initialization Errors**:
   - Verify Azure API key
   - Check assistant IDs configuration
   - Monitor `/health` endpoint

3. **Streaming Connection Issues**:
   - Check CORS configuration
   - Verify event stream client implementation
   - Monitor network connectivity
