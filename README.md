# Email Categorization Agent

An AI-powered email categorization system using local LLMs and multi-agent architecture, designed to automatically organize Gmail emails with privacy-first local processing.

## Features

- **Local AI Processing**: Uses Ollama models for privacy-preserving email categorization
- **Multi-Agent Architecture**: Specialized agents for different tasks (retrieval, analysis, labeling)
- **MCP Integration**: Seamless integration with Claude via Model Context Protocol
- **Interactive Mode**: Smart fallback to user interaction for ambiguous cases
- **Real-time Processing**: Push notifications for immediate email handling
- **Privacy First**: All processing happens locally, no email content sent to external services

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    MCP Server Interface                      │
├─────────────────────────────────────────────────────────────┤
│                  Orchestration Agent (LangGraph)             │
├─────────────┬───────────┬────────────┬─────────────────────┤
│  Email       │  Category  │  Gmail    │  User Interaction  │
│  Retrieval   │  Analysis  │  API      │  Agent             │
│  Agent       │  Agent     │  Agent    │                     │
├─────────────┴───────────┴────────────┴─────────────────────┤
│                 Event Bus (SQLite)                          │
├─────────────────────────────────────────────────────────────┤
│         State Management (SQLite)                           │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (fast Python package manager)
- [Ollama](https://ollama.ai) installed and running
- Gmail API credentials
- Claude Desktop (for MCP integration)

### Installation

1. Install uv (if not already installed):
```bash
# macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or with pip
pip install uv
```

2. Clone the repository:
```bash
git clone <repository-url>
cd email-agents
```

3. Set up project with uv:
```bash
uv sync
```

This will automatically create a virtual environment and install all dependencies.

4. Set up Ollama models:
```bash
ollama pull gemma2:3b
ollama pull llama3.2:3b
```

5. Configure Gmail API:
   - Create a Google Cloud project
   - Enable Gmail API
   - Download credentials.json
   - Place in project root

6. Configure environment:
```bash
cp .env.example .env
# Edit .env with your settings
```

### Usage

#### With Claude Desktop (MCP)

Add to your Claude Desktop configuration:

```json
{
  "email-categorization": {
    "command": "python",
    "args": ["-m", "src.mcp.server"],
    "cwd": "/path/to/email-agents",
    "env": {
      "GMAIL_CREDENTIALS_PATH": "./credentials.json"
    }
  }
}
```

#### Direct Usage

```bash
# Activate the virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Start the MCP server
uv run python -m src.mcp.server

# Or use the CLI interface
uv run email-agent categorize --mode interactive

# Or run directly with uv
uv run python -m src.cli categorize --mode interactive
```

## Configuration

Configuration is managed through YAML files and environment variables:

- `config/default.yaml`: Main configuration
- `.env`: Environment-specific settings
- `credentials.json`: Gmail API credentials

## Development

### Project Structure

```
email-agents/
├── src/
│   ├── agents/         # Individual agent implementations
│   ├── core/          # Core system components
│   ├── integrations/  # External service integrations
│   ├── mcp/           # MCP server implementation
│   └── utils/         # Utility functions
├── tests/             # Test suite
├── config/            # Configuration files
├── data/              # Database and state files
└── logs/              # Log files
```

### Running Tests

```bash
# Run tests with uv
uv run pytest tests/

# Run with coverage
uv run pytest tests/ --cov=src --cov-report=html
```

### Code Quality

```bash
# Format code
uv run black src/ tests/

# Type checking
uv run mypy src/

# Linting
uv run flake8 src/ tests/

# Run all quality checks
uv run pre-commit run --all-files
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run the test suite
6. Submit a pull request

## License

MIT License - see LICENSE file for details

## Security

This project prioritizes privacy and security:
- All email processing happens locally
- Credentials are stored encrypted
- No data is sent to external services
- Minimal API permissions requested

## Support

For issues and questions:
- Check the [troubleshooting guide](docs/troubleshooting.md)
- Review existing [issues](https://github.com/user/email-agents/issues)
- Create a new issue if needed

## Roadmap

- [ ] Multi-account support
- [ ] Advanced ML features with fine-tuning
- [ ] Attachment analysis
- [ ] Export/import functionality
- [ ] Cloud deployment options