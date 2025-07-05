#!/bin/bash
# Quick wrapper script for email summarization

echo "🚀 Starting Email Summarization..."
echo ""

# Check if Ollama is running
if ! pgrep -f ollama > /dev/null; then
    echo "⚠️  Warning: Ollama doesn't appear to be running"
    echo "   Please start Ollama first: ollama serve"
    echo ""
fi

# Run the summarization script
uv run python summarize_unread_emails.py "$@"