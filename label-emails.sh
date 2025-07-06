#!/bin/bash
"""
Wrapper script for the email labeling CLI.
Usage: ./label-emails.sh [command] [options]
"""

cd "$(dirname "$0")"
uv run python label_emails_cli.py "$@"