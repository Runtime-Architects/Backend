# Use an official Python 3.10 base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies (optional: clean build environment)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy requirement installation first (to leverage Docker cache)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of your code
COPY . .

# Command to run your application (replace with your main script)
CMD ["python", "main.py"]
