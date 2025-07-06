#!/bin/bash
# MCP Server startup script
# Ensures proper environment and path setup

# Change to script directory
cd "$(dirname "$0")"

# Set Python path
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Use uv to run with proper virtual environment
exec /Users/rebecca/.local/bin/uv run python src/email_mcp_server.py
