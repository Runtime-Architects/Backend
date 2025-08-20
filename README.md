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

## 🚀 Quick Start

### Step 1: Clone and Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Backend
   ```

2. **Set up environment configurations**:
   
   Create `.env` files in both configuration directories with your actual values:

   **For Local Development** - Create `configs/local/.env`:
   ```env
   # Azure OpenAI Configuration
   AZURE_AI_DEPLOYMENT=YOUR_DEPLOYMENT
   AZURE_AI_MODEL=YOUR_MODEL
   AZURE_AI_API_VERSION=YOUR_API_VERSION
   AZURE_AI_ENDPOINT=YOUR_ENDPOINT
   AZURE_AI_API_KEY=YOUR_KEY
   
   # Policy Search Configuration
   POLICY_SEARCH_INDEX_NAME=YOUR_INDEX_NAME
   POLICY_SEARCH_API_KEY=YOUR_KEY
   POLICY_SEARCH_API_VERSION=YOUR_API_VERSION
   POLICY_SEARCH_ENDPOINT=YOUR_ENDPOINT
   
   # Database Configuration
   DATABASE_URL=YOUR_DATABASE_URL
   
   # Application Configuration
   APP_SECRET_KEY=YOUR_SECRET_KEY
   APP_DEBUG=true
   APP_HOST=YOUR_HOST
   APP_PORT=YOUR_PORT
   
   # WebAuthn Configuration
   WEBAUTHN_RP_ID=YOUR_RP_ID
   WEBAUTHN_RP_NAME=YOUR_RP_NAME
   WEBAUTHN_ORIGIN=YOUR_ORIGIN
   
   # CORS Configuration
   CORS_ORIGINS=YOUR_ORIGINS
   
   # Security Configuration
   JWT_SECRET_KEY=YOUR_JWT_SECRET_KEY
   JWT_ALGORITHM=YOUR_ALGORITHM
   JWT_EXPIRATION_HOURS=YOUR_EXPIRATION_HOURS
   ```

   **For Production Deployment** - Create `configs/deployment/.env` with production values.

3. **Configure production SSL (Production only)**:
   
   If deploying to production, update the email address in `configs/deployment/start.sh`:
   ```bash
   # Find this line and update the email:
   --email your-email@example.com \
   ```
   Replace `your-email@example.com` with your actual email address for Let's Encrypt certificate notifications.

### Step 3: Deploy Configuration

Use the deployment scripts to set up your environment:

**Windows:**
```powershell
# Deploy local development configuration
.\deploy-config.bat local

# Deploy production configuration
.\deploy-config.bat deployment
```

**Linux/macOS:**
```bash
# Make script executable
chmod +x deploy-config.sh

# Deploy local development configuration
./deploy-config.sh local

# Deploy production configuration
./deploy-config.sh deployment
```

### Step 4: Run the Application

After deploying the local configuration, you have two options:

**Option A: Docker (Recommended)**
```bash
docker build -t sustainable-city-backend .
docker run -p 8000:8000 -p 6379:6379 sustainable-city-backend
```

**Option B: Local Python**
```bash
# Windows
python -m venv env
env\Scripts\activate
pip install -r requirements.txt
python source\main.py

# Linux/macOS
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
python3 source/main.py
```

The API will be available at `http://localhost:8000`

### What the Deployment Script Does

The deployment script will automatically:
- Copy the appropriate `.env` file to `source/`
- Copy other configuration files to the root directory
- For **local**: Remove deployment-specific files (nginx.conf, SSL certificates) and show run commands
- For **deployment**: Set up nginx, SSL certificates, and startup scripts

## 📊 API Endpoints

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

## 🧪 Testing

```bash
# Test health endpoint
curl http://localhost:8000/health
```

### Authentication Testing
The WebAuthn implementation requires HTTPS in production. For development, use:
- **Origin**: `http://localhost:3000`
- **RP ID**: `localhost`

## 📝 Configuration

### Environment Structure

The repository uses a `configs/` directory structure:

```
Backend/
├── configs/
│   ├── deployment/
│   │   ├── .env
│   │   ├── dockerfile
│   │   ├── nginx.conf
│   │   ├── selfsigned.crt
│   │   ├── selfsigned.key
│   │   └── start.sh
│   └── local/
│       ├── .env
│       └── dockerfile
├── deploy-config.bat    # Windows deployment script
├── deploy-config.sh     # Linux/macOS deployment script
├── requirements.txt
└── source/             # Application source code
    ├── main.py
    ├── agents/
    ├── api/
    ├── eirgridscraper/
    └── policy-scraper/
```

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

## 📦 Production Deployment

For production deployment, use the deployment configuration:

```bash
# Deploy production configuration
./deploy-config.sh deployment  # Linux/macOS
.\deploy-config.bat deployment  # Windows
```

This will set up:
- Production environment variables
- Nginx configuration
- SSL certificates
- Startup scripts

Additional production considerations:
1. **HTTPS Configuration**: WebAuthn requires HTTPS in production
2. **Database Migration**: Configure production database URL
3. **Monitoring**: Set up proper logging infrastructure
4. **Security**: Update all secret keys and API keys

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

4. **Configuration Deployment Issues**:
   - Ensure `configs/` directory structure is correct
   - Check file permissions on deployment scripts
   - Verify target environment (deployment/local) exists

## 🤝 Contributing

We follow strict contribution guidelines including Conventional Commits and branching conventions.

👉 See the [CONTRIBUTING.md](./CONTRIBUTING.md) file for full details.