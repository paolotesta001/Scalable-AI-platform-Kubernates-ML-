FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Make entrypoint executable
RUN chmod +x entrypoint.sh

# Expose ports: 8000 (orchestrator), 8001-8004 (agents in microservice mode)
EXPOSE 8000 8001 8002 8003 8004

# Default entrypoint: monolith mode via entrypoint.sh
# Override CMD in docker-compose / K8s for microservice mode
ENTRYPOINT ["./entrypoint.sh"]
