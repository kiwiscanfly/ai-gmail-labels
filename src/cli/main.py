"""Main CLI entry point with tiered command structure."""

import typer
from rich.console import Console
from rich.panel import Panel

# Import command groups
from .commands.label import priority, marketing, receipt, notifications, all, custom
from .commands.system import install
from .commands.manage import labels, categories, stats

# Import legacy system commands from the main CLI
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from cli_main import (
    config, init, status, test_gmail, test_database, test_ollama, 
    models, gmail, test_error_recovery, monitor, validate_config
)

app = typer.Typer(
    help="Email Categorization Agent - Unified CLI",
    invoke_without_command=True,
    no_args_is_help=False
)
console = Console()

# Create command groups with auto-help
label_app = typer.Typer(help="Email labeling operations", invoke_without_command=True)
manage_app = typer.Typer(help="Label and category management", invoke_without_command=True) 
system_app = typer.Typer(help="System operations and monitoring", invoke_without_command=True)

# Mount label subcommands
label_app.add_typer(all.app, name="all", help="Combined classification (priority + marketing + receipt + notifications)")
label_app.add_typer(priority.app, name="priority", help="Priority classification")
label_app.add_typer(marketing.app, name="marketing", help="Marketing classification")
label_app.add_typer(receipt.app, name="receipt", help="Receipt classification")
label_app.add_typer(notifications.app, name="notifications", help="Notification classification")
label_app.add_typer(custom.app, name="custom", help="Custom category classification with AI-powered search terms")

# Mount system subcommands
system_app.add_typer(install.app, name="install", help="Install and configure the email-agents system")

# Mount management subcommands
manage_app.add_typer(labels.app, name="labels", help="Gmail label management (list, create, delete, analyze)")
manage_app.add_typer(categories.app, name="categories", help="Custom category management (create, train, export/import)")
manage_app.add_typer(stats.app, name="stats", help="Statistics and analytics (classification, senders, usage)")

# Add callback functions for auto-help
@label_app.callback()
def label_callback(ctx: typer.Context):
    """Email labeling operations - classify emails with priority, marketing, receipt, and notification labels."""
    if ctx.invoked_subcommand is None:
        console.print()
        console.print("[bold blue]Email Labeling Commands[/bold blue]")
        console.print("━" * 60)
        console.print()
        console.print("[bold]Available Commands:[/bold]")
        console.print()
        console.print("  [bold cyan]all[/bold cyan]           Apply all classifiers in one pass")
        console.print("                • Priority + Marketing + Receipt + Notifications")
        console.print("                • Most efficient for processing multiple emails")
        console.print()
        console.print("  [bold cyan]priority[/bold cyan]      Classify emails by importance and urgency")
        console.print("                • Labels: Critical, High, Medium, Low")
        console.print("                • Based on content, sender, and context")
        console.print()
        console.print("  [bold cyan]marketing[/bold cyan]     Detect promotional and commercial emails")
        console.print("                • Labels: Promotional, Newsletter, Hybrid, General")
        console.print("                • Identifies sales, newsletters, and marketing content")
        console.print()
        console.print("  [bold cyan]receipt[/bold cyan]       Identify purchase confirmations and invoices")
        console.print("                • Labels: Purchase, Service, Other")
        console.print("                • Extracts vendor and transaction details")
        console.print()
        console.print("  [bold cyan]notifications[/bold cyan] Classify automated system notifications")
        console.print("                • Labels: System, Update, Alert, Reminder, Security")
        console.print("                • Categorizes service notifications and alerts")
        console.print()
        console.print("  [bold cyan]custom[/bold cyan]        Create and apply custom categories")
        console.print("                • AI-powered search term generation")
        console.print("                • Train categories from existing labels")
        console.print()
        console.print("[bold]Common Usage Patterns:[/bold]")
        console.print()
        console.print("  [dim]# Process all unread emails with all classifiers[/dim]")
        console.print("  $ email-agent label all unread --limit 50")
        console.print()
        console.print("  [dim]# Preview priority classification without applying labels[/dim]")
        console.print("  $ email-agent label priority unread --dry-run")
        console.print()
        console.print("  [dim]# Process recent marketing emails with high confidence[/dim]")
        console.print("  $ email-agent label marketing recent 7 --confidence-threshold 0.8")
        console.print()
        console.print("  [dim]# Create a custom category with AI-generated search terms[/dim]")
        console.print("  $ email-agent label custom create 'project updates' --generate-terms")
        console.print()
        console.print("[bold]Target Options:[/bold]")
        console.print()
        console.print("  • unread             Process only unread emails")
        console.print("  • recent N           Process emails from last N days")
        console.print("  • query \"search\"     Use custom Gmail search query")
        console.print()
        console.print("Use [bold]email-agent label COMMAND --help[/bold] for detailed command options")

@manage_app.callback()
def manage_callback(ctx: typer.Context):
    """Management operations for labels, categories, and analytics."""
    if ctx.invoked_subcommand is None:
        console.print()
        console.print("[bold green]Management Commands[/bold green]")
        console.print("━" * 60)
        console.print()
        console.print("[bold]Available Commands:[/bold]")
        console.print()
        console.print("  [bold cyan]labels[/bold cyan]      Manage Gmail labels")
        console.print("              • list    - Show all labels with usage statistics")
        console.print("              • create  - Create new Gmail labels")
        console.print("              • delete  - Remove unused labels")
        console.print("              • analyze - Analyze label usage patterns")
        console.print()
        console.print("  [bold cyan]categories[/bold cyan]  Manage custom categories")
        console.print("              • list    - Show all custom categories")
        console.print("              • create  - Create new category with AI")
        console.print("              • train   - Train from existing labels")
        console.print("              • export  - Export categories to file")
        console.print("              • import  - Import categories from file")
        console.print()
        console.print("  [bold cyan]stats[/bold cyan]       View analytics and insights")
        console.print("              • classification - Classification performance")
        console.print("              • senders        - Top senders analysis")
        console.print("              • labels         - Label distribution")
        console.print("              • usage          - System usage metrics")
        console.print()
        console.print("[bold]Common Usage Patterns:[/bold]")
        console.print()
        console.print("  [dim]# List all Gmail labels with statistics[/dim]")
        console.print("  $ email-agent manage labels list")
        console.print()
        console.print("  [dim]# Create a custom category with AI assistance[/dim]")
        console.print("  $ email-agent manage categories create 'financial' --generate-terms")
        console.print()
        console.print("  [dim]# View classification performance over time[/dim]")
        console.print("  $ email-agent manage stats classification --last-30-days")
        console.print()
        console.print("Use [bold]email-agent manage COMMAND --help[/bold] for detailed command options")

@system_app.callback()
def system_callback(ctx: typer.Context):
    """System operations and monitoring."""
    if ctx.invoked_subcommand is None:
        console.print()
        console.print("[bold yellow]System Commands[/bold yellow]")
        console.print("━" * 60)
        console.print()
        console.print("[bold]Available Commands:[/bold]")
        console.print()
        console.print("  [bold cyan]install[/bold cyan]     Initial setup and configuration")
        console.print("              • Interactive configuration wizard")
        console.print("              • Gmail OAuth authentication")
        console.print("              • Ollama model installation")
        console.print("              • System validation")
        console.print()
        console.print("[bold]Note:[/bold] Additional system commands (status, config, monitor)")
        console.print("are available at the top level:")
        console.print()
        console.print("  $ email-agent status     # Check system health")
        console.print("  $ email-agent config     # Manage configuration")
        console.print("  $ email-agent monitor    # Real-time monitoring")
        console.print()
        console.print("[bold]Common Usage Patterns:[/bold]")
        console.print()
        console.print("  [dim]# First-time setup[/dim]")
        console.print("  $ email-agent system install")
        console.print()
        console.print("  [dim]# Check system status[/dim]")
        console.print("  $ email-agent status")
        console.print()
        console.print("Use [bold]email-agent system COMMAND --help[/bold] for detailed command options")

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

@app.callback()
def main_callback(ctx: typer.Context):
    """Show main help when no command is specified."""
    if ctx.invoked_subcommand is None:
        console.print()
        console.print("[bold blue]Email Categorization Agent[/bold blue] - AI-powered email organization")
        console.print()
        console.print("[bold]Usage:[/bold] email-agent [OPTIONS] COMMAND [ARGS]...")
        console.print()
        console.print("[bold]Available Commands:[/bold]")
        console.print()
        
        # Command groups
        console.print("  [bold cyan]label[/bold cyan]      Classify and label emails")
        console.print("             • all, priority, marketing, receipt, notifications, custom")
        console.print()
        console.print("  [bold cyan]manage[/bold cyan]     Manage labels, categories, and analytics")
        console.print("             • labels, categories, stats")
        console.print()
        console.print("  [bold cyan]system[/bold cyan]     System configuration and monitoring")
        console.print("             • install, status, config")
        console.print()
        
        # Other commands
        console.print("  [bold cyan]info[/bold cyan]       Show detailed information about the CLI")
        console.print("  [bold cyan]status[/bold cyan]     Check system health and connections")
        console.print("  [bold cyan]config[/bold cyan]     Manage configuration settings")
        console.print()
        
        console.print("[bold]Quick Start Examples:[/bold]")
        console.print()
        console.print("  [dim]# Apply all classifiers to unread emails[/dim]")
        console.print("  $ email-agent label all unread --limit 10")
        console.print()
        console.print("  [dim]# Check priority of recent emails (preview mode)[/dim]")
        console.print("  $ email-agent label priority recent 7 --dry-run")
        console.print()
        console.print("  [dim]# Show system status[/dim]")
        console.print("  $ email-agent status")
        console.print()
        
        console.print("[bold]Getting Help:[/bold]")
        console.print()
        console.print("  $ email-agent --help              # Show all commands")
        console.print("  $ email-agent COMMAND --help      # Show help for a command")
        console.print("  $ email-agent label --help        # Show label subcommands")
        console.print()
        console.print("For more information, visit: https://github.com/your-username/email-agents")
        console.print()

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