#!/bin/bash
# justnews-start-agent.sh - Startup script for JustNews agents
# This script is called by systemd to start individual JustNews agents

set -euo pipefail

AGENT_NAME="$1"

if [[ -z "$AGENT_NAME" ]]; then
    echo "Error: Agent name not provided"
    exit 1
fi

# Set up environment
export PYTHONPATH="${PYTHONPATH:-}/home/adra/justnewsagent/JustNewsAgent"
export PATH="/home/adra/miniconda3/envs/justnews-v2-py312/bin:$PATH"

# Change to project directory
cd /home/adra/justnewsagent/JustNewsAgent

# Use direct Python path instead of conda
export PATH="/home/adra/miniconda3/envs/justnews-v2-py312/bin:$PATH"

# Set PYTHONPATH to include the project root
export PYTHONPATH="/home/adra/justnewsagent/JustNewsAgent:$PYTHONPATH"

# Start the agent
echo "Starting JustNews agent: $AGENT_NAME"
exec /home/adra/miniconda3/envs/justnews-v2-py312/bin/python -m agents."$AGENT_NAME".main 2>&1
