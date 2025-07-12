# Email Categorization Agent (Coded using AI)

An AI-powered email categorization system using local LLMs and multi-agent architecture, designed to automatically organize Gmail emails with privacy-first local processing.

## Features

- **Local AI Processing**: Uses Ollama models for privacy-preserving email categorization
- **Multi-Agent Architecture**: Specialized agents for different tasks (retrieval, analysis, labeling)
- **Unified CLI**: Comprehensive command-line interface for email classification and management
- **MCP Integration**: Seamless integration with Claude via Model Context Protocol
- **Interactive Mode**: Smart fallback to user interaction for ambiguous cases
- **Real-time Processing**: Push notifications for immediate email handling
- **Privacy First**: All processing happens locally, no email content sent to external services

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│          CLI Interface + MCP Server Interface               │
├─────────────────────────────────────────────────────────────┤
│                  Orchestration Agent (LangGraph)             │
├─────────────┬───────────┬────────────┬─────────────────────┤
│  Email       │  Category  │  Gmail    │  User Interaction  │
│  Retrieval   │  Analysis  │  API      │  Agent             │
│  Agent       │  Agent     │  Agent    │                     │
├─────────────┴───────────┴────────────┴─────────────────────┤
│    Classification Services (Priority, Marketing, Receipt,   │
│                 Notifications, Custom)                      │
├─────────────────────────────────────────────────────────────┤
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

#### Direct CLI Usage

```bash
# Run any CLI command with uv
uv run email-agent <command>

# Examples:
uv run email-agent label all unread --limit 20
uv run email-agent label priority unread --dry-run
uv run email-agent label notifications recent 7

# Or activate the virtual environment first
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
email-agent label all unread

# Start the MCP server
uv run python -m src.mcp.server
```

## CLI Usage

The unified CLI provides comprehensive email classification and labeling capabilities through a tiered command structure.

### Basic Command Structure

```bash
email-agent <command_group> <command> [options]
```

### Available Commands

#### Email Labeling Commands (`email-agent label`)

| Command | Description | Gmail Labels Created |
|---------|-------------|---------------------|
| `email-agent label all` | Apply all classifiers in one pass | All label types below |
| `email-agent label priority` | Classify emails by priority level | `Priority/Critical`, `Priority/High`, `Priority/Medium`, `Priority/Low` |
| `email-agent label marketing` | Detect and categorize marketing emails | `Marketing/Promotional`, `Marketing/Newsletter`, `Marketing/Hybrid`, `Marketing/General` |
| `email-agent label receipt` | Identify and categorize receipts | `Receipts/Purchase`, `Receipts/Service`, `Receipts/Other` |
| `email-agent label notifications` | Classify system and service notifications | `Notifications/System`, `Notifications/Update`, `Notifications/Alert`, `Notifications/Reminder`, `Notifications/Security` |

#### Target Options

All labeling commands support these target options:

- **`unread`** - Process unread emails (default)
- **`recent N`** - Process emails from the last N days
- **`query "search"`** - Use custom Gmail search query

#### Common Options

| Option | Description | Default |
|--------|-------------|---------|
| `--dry-run` | Preview labels without applying them | Apply labels |
| `--limit N` | Maximum number of emails to process | 50 |
| `--confidence-threshold N` | Minimum confidence for labeling (0.0-1.0) | 0.7 |
| `--detailed` | Show detailed analysis results | False |

### Usage Examples

#### Basic Classification

```bash
# Apply all classifiers to unread emails
email-agent label all unread

# Classify priority for recent emails (dry run)
email-agent label priority recent 7 --dry-run

# Detect marketing emails with high confidence
email-agent label marketing unread --confidence-threshold 0.8

# Process receipts from specific sender
email-agent label receipt query "from:amazon.com"

# Classify notifications with detailed output
email-agent label notifications unread --detailed
```

#### Advanced Usage

```bash
# Combined classification with custom confidence thresholds
email-agent label all unread \
  --priority-confidence 0.8 \
  --marketing-confidence 0.7 \
  --receipt-confidence 0.9 \
  --notifications-confidence 0.6

# Process specific time range
email-agent label all recent 30 --limit 100

# Custom search with multiple criteria
email-agent label priority query "is:important OR from:boss@company.com" --limit 20

# Preview mode for testing
email-agent label all unread --dry-run --detailed
```

### Classification Types

#### Priority Classification
Analyzes email importance and urgency:
- **Critical**: Urgent emails requiring immediate attention
- **High**: Important emails that should be prioritized
- **Medium**: Standard importance emails
- **Low**: Less important, informational emails

#### Marketing Classification
Detects commercial and promotional emails:
- **Promotional**: Sales, offers, discounts
- **Newsletter**: Regular newsletters and updates
- **Hybrid**: Mixed content (promotional + informational)
- **General**: Other marketing content

#### Receipt Classification
Identifies transaction confirmations and receipts:
- **Purchase**: Product purchases, orders, deliveries
- **Service**: Service payments, subscriptions, utilities
- **Other**: Other transaction types

#### Notification Classification
Categorizes system and service notifications:
- **System**: Platform maintenance, service status
- **Update**: Software updates, new features
- **Alert**: Warnings, important notices
- **Reminder**: Appointments, deadlines, tasks
- **Security**: Account security, login alerts

### Gmail Label Structure

The CLI creates a hierarchical label structure in Gmail:

```
Priority/
├── Critical
├── High
├── Medium
└── Low

Marketing/
├── Promotional
├── Newsletter
├── Hybrid
└── General

Receipts/
├── Purchase
├── Service
└── Other

Notifications/
├── System
├── Update
├── Alert
├── Reminder
└── Security
```

### Confidence Thresholds

All classifiers use confidence scores (0.0 to 1.0) to determine label application:

- **0.9-1.0**: Very high confidence
- **0.8-0.9**: High confidence
- **0.7-0.8**: Good confidence (default threshold)
- **0.6-0.7**: Moderate confidence
- **Below 0.6**: Low confidence (typically not labeled)

### Performance Tips

1. **Use `--dry-run` first** to preview results before applying labels
2. **Set appropriate limits** with `--limit` to avoid processing too many emails at once
3. **Adjust confidence thresholds** based on your accuracy preferences
4. **Use specific queries** to target relevant emails more efficiently
5. **Process in batches** for large email volumes

### System Commands

Additional system management commands are available:

```bash
# Show system status
email-agent status

# Test Gmail connection
email-agent test-gmail

# Configure settings
email-agent config

# Show CLI information
email-agent info
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
│   ├── cli/           # Command-line interface
│   │   ├── commands/  # CLI command implementations
│   │   │   └── label/ # Email labeling commands
│   │   └── base.py    # Base CLI classes
│   ├── services/      # Classification services
│   │   ├── email_prioritizer.py
│   │   ├── marketing_classifier.py
│   │   ├── receipt_classifier.py
│   │   └── notifications_classifier.py
│   ├── agents/        # Individual agent implementations
│   ├── core/          # Core system components
│   ├── integrations/  # External service integrations
│   ├── mcp/           # MCP server implementation
│   └── models/        # Data models
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