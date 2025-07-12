"""Main CLI entry point with tiered command structure."""

import typer
from rich.console import Console
from rich.panel import Panel

# Import command groups
from .commands.label import priority, marketing, receipt, notifications, all, custom
from .commands.system import install

# Import legacy system commands from the main CLI
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from cli_main import (
    config, init, status, test_gmail, test_database, test_ollama, 
    models, gmail, test_error_recovery, monitor, validate_config
)

app = typer.Typer(help="Email Categorization Agent - Unified CLI")
console = Console()

# Create command groups
label_app = typer.Typer(help="Email labeling operations")
manage_app = typer.Typer(help="Label and category management") 
system_app = typer.Typer(help="System operations and monitoring")

# Mount label subcommands
label_app.add_typer(all.app, name="all", help="Combined classification (priority + marketing + receipt + notifications)")
label_app.add_typer(priority.app, name="priority", help="Priority classification")
label_app.add_typer(marketing.app, name="marketing", help="Marketing classification")
label_app.add_typer(receipt.app, name="receipt", help="Receipt classification")
label_app.add_typer(notifications.app, name="notifications", help="Notification classification")
label_app.add_typer(custom.app, name="custom", help="Custom category classification with AI-powered search terms")

# Mount system subcommands
system_app.add_typer(install.app, name="install", help="Install and configure the email-agents system")

# Add command groups to main app
app.add_typer(label_app, name="label", help="Email labeling operations")
app.add_typer(manage_app, name="manage", help="Management operations")
app.add_typer(system_app, name="system", help="System operations")

# Add legacy system commands directly to main app
app.command()(config)
app.command()(init)
app.command()(status)
app.command()(test_gmail)
app.command()(test_database)
app.command()(test_ollama)
app.command()(models)
app.command()(gmail)
app.command()(test_error_recovery)
app.command()(monitor)
app.command()(validate_config)

@app.command()
def info():
    """Show information about the unified CLI system"""
    console.print(Panel(
        """[bold green]Email Agent Unified CLI[/bold green]

This is the unified command-line interface for email classification and labeling.

[bold]Available Commands:[/bold]
• [cyan]email-agent label all[/cyan] - Apply all classifiers (priority + marketing + receipt + notifications)
• [cyan]email-agent label priority[/cyan] - Classify emails by priority (Critical/High/Medium/Low)
• [cyan]email-agent label marketing[/cyan] - Detect marketing emails (Promotional/Newsletter/etc)
• [cyan]email-agent label receipt[/cyan] - Identify receipts (Purchase/Service/Other)
• [cyan]email-agent label notifications[/cyan] - Classify notifications (System/Update/Alert/Reminder/Security)

[bold]Target Options:[/bold]
• [yellow]unread[/yellow] - Process unread emails
• [yellow]recent N[/yellow] - Process emails from last N days  
• [yellow]query "search"[/yellow] - Use custom Gmail search

[bold]Processing Options:[/bold]
• [yellow]--dry-run[/yellow] - Preview labels without applying them
• [yellow]--limit N[/yellow] - Process maximum N emails (default: 50)
• [yellow]--detailed[/yellow] - Show detailed analysis results

[bold]Examples:[/bold]
[dim]email-agent label all unread --limit 20
email-agent label priority unread
email-agent label marketing recent 7 --dry-run
email-agent label receipt query "from:amazon.com" --limit 10
email-agent label notifications unread --detailed[/dim]

[bold yellow]Note:[/bold yellow] Commands apply labels by default. Use --dry-run to preview first.
        """,
        title="📧 Email Agent CLI",
        border_style="blue"
    ))


def main():
    """Main entry point."""
    app()

if __name__ == "__main__":
    main()