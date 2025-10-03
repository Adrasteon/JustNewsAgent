FROM python:3.11-slim

# Install minimal system deps for Python builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Install Python deps
WORKDIR /workspace
COPY kafka/requirements.txt /workspace/
RUN pip install --no-cache-dir -r requirements.txt

# Default entrypoint
ENTRYPOINT ["/bin/bash", "-lc"]
