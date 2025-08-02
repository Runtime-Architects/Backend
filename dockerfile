# Use Python 3.11 slim image
FROM python:3.11-slim

# Install Redis
RUN apt-get update && apt-get install -y redis-server && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy source code
COPY source/ ./source/

# Expose ports
EXPOSE 8000 6379

# Start Redis in background and run the Python app
CMD redis-server --daemonize yes && python source/main.py