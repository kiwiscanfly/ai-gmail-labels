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
        console.print("📖 [bold cyan]Email Labeling Command Guide[/bold cyan]\n")
        
        console.print("The [bold]email-agent label[/bold] commands help you automatically classify and organize your emails using AI-powered analysis.\n")
        
        console.print("🎯 [bold yellow]Available Commands:[/bold yellow]\n")
        
        # 1. All command
        console.print("1. [bold green]Apply All Classifiers[/bold green] ([cyan]all[/cyan])\n")
        console.print("   Most efficient - applies all classifiers in one pass:\n")
        console.print("   [dim]# Process unread emails with all classifiers[/dim]")
        console.print("   [bold]uv run email-agent label all[/bold]  [dim]# Uses defaults: unread, limit 50[/dim]\n")
        console.print("   [dim]# Process recent emails[/dim]")
        console.print("   [bold]uv run email-agent label all recent --days 7 --limit 100[/bold]\n")
        console.print("   [dim]# Preview mode (dry run)[/dim]")
        console.print("   [bold]uv run email-agent label all --dry-run --detailed[/bold]\n")
        
        # 2. Priority command
        console.print("2. [bold green]Priority Classification[/bold green] ([cyan]priority[/cyan])\n")
        console.print("   Classify emails by importance and urgency:\n")
        console.print("   [yellow]Labels:[/yellow] Critical, High, Medium, Low")
        console.print("   [yellow]Features:[/yellow] Content analysis, sender reputation, urgency detection\n")
        console.print("   [dim]# Classify unread emails by priority[/dim]")
        console.print("   [bold]uv run email-agent label priority unread[/bold]\n")
        console.print("   [dim]# Preview priority without applying labels[/dim]")
        console.print("   [bold]uv run email-agent label priority recent 7 --dry-run[/bold]\n")
        
        # 3. Marketing command
        console.print("3. [bold green]Marketing Detection[/bold green] ([cyan]marketing[/cyan])\n")
        console.print("   Detect promotional and commercial emails:\n")
        console.print("   [yellow]Labels:[/yellow] Promotional, Newsletter, Hybrid, General")
        console.print("   [yellow]Features:[/yellow] Unsubscribe detection, sender patterns, content analysis\n")
        console.print("   [dim]# Detect marketing emails[/dim]")
        console.print("   [bold]uv run email-agent label marketing unread[/bold]\n")
        console.print("   [dim]# High confidence marketing detection[/dim]")
        console.print("   [bold]uv run email-agent label marketing recent 30 --confidence-threshold 0.8[/bold]\n")
        
        # 4. Receipt command
        console.print("4. [bold green]Receipt Identification[/bold green] ([cyan]receipt[/cyan])\n")
        console.print("   Identify purchase confirmations and invoices:\n")
        console.print("   [yellow]Labels:[/yellow] Purchase, Service, Other")
        console.print("   [yellow]Features:[/yellow] Vendor extraction, amount detection, transaction details\n")
        console.print("   [dim]# Find receipt emails[/dim]")
        console.print("   [bold]uv run email-agent label receipt unread[/bold]\n")
        console.print("   [dim]# Search for Amazon receipts[/dim]")
        console.print("   [bold]uv run email-agent label receipt query \"from:amazon.com\"[/bold]\n")
        
        # 5. Notifications command
        console.print("5. [bold green]Notification Classification[/bold green] ([cyan]notifications[/cyan])\n")
        console.print("   Classify automated system notifications:\n")
        console.print("   [yellow]Labels:[/yellow] System, Update, Alert, Reminder, Security")
        console.print("   [yellow]Features:[/yellow] Service detection, notification type analysis\n")
        console.print("   [dim]# Classify notifications[/dim]")
        console.print("   [bold]uv run email-agent label notifications unread[/bold]\n")
        
        # 6. Custom command
        console.print("6. [bold green]Custom Categories[/bold green] ([cyan]custom[/cyan])\n")
        console.print("   Create and apply custom categories with AI:\n")
        console.print("   [yellow]Features:[/yellow] AI-powered search terms, custom labels, hierarchical organization\n")
        console.print("   [dim]# Create custom category[/dim]")
        console.print("   [bold]uv run email-agent label custom create \"programming\"[/bold]\n")
        console.print("   [dim]# See detailed custom command help[/dim]")
        console.print("   [bold]uv run email-agent label custom[/bold]  [dim]# Shows comprehensive guide[/dim]\n")
        
        # Target options
        console.print("📧 [bold yellow]Target Options:[/bold yellow]\n")
        console.print("   • [cyan]unread[/cyan]           Process only unread emails (default)")
        console.print("   • [cyan]recent N[/cyan]         Process emails from last N days")
        console.print("   • [cyan]query \"search\"[/cyan]   Use custom Gmail search query\n")
        
        # Common options
        console.print("⚙️ [bold yellow]Common Options:[/bold yellow]\n")
        console.print("   • [cyan]--dry-run[/cyan]        Preview labels without applying them")
        console.print("   • [cyan]--limit N[/cyan]        Process maximum N emails (default: 50)")
        console.print("   • [cyan]--detailed[/cyan]       Show detailed analysis results")
        console.print("   • [cyan]--verbose[/cyan]        Show technical information")
        console.print("   • [cyan]--confidence-threshold[/cyan]  Minimum confidence for labeling (0.0-1.0)\n")
        
        # Quick start
        console.print("🚀 [bold yellow]Quick Start:[/bold yellow]\n")
        console.print("   1. [bold]Start with preview[/bold] - see what would be labeled:")
        console.print("      [bold]uv run email-agent label all --dry-run --limit 10[/bold]\n")
        console.print("   2. [bold]Apply labels[/bold] - remove --dry-run to actually label:")
        console.print("      [bold]uv run email-agent label all --limit 10[/bold]\n")
        console.print("   3. [bold]Process more emails[/bold] - increase limit or change target:")
        console.print("      [bold]uv run email-agent label all --limit 100[/bold]")
        console.print("      [bold]uv run email-agent label all recent --days 30[/bold]\n")
        
        # Examples section
        console.print("💡 [bold yellow]Common Usage Examples:[/bold yellow]\n")
        console.print("   [dim]# Quick overview of unread emails[/dim]")
        console.print("   [bold]uv run email-agent label all --dry-run --detailed[/bold]\n")
        console.print("   [dim]# Process all unread emails efficiently[/dim]")
        console.print("   [bold]uv run email-agent label all[/bold]\n")
        console.print("   [dim]# Focus on priority emails only[/dim]")
        console.print("   [bold]uv run email-agent label priority unread --confidence-threshold 0.8[/bold]\n")
        console.print("   [dim]# Find receipts from specific vendor[/dim]")
        console.print("   [bold]uv run email-agent label receipt query \"from:uber.com\"[/bold]\n")
        console.print("   [dim]# Create custom work category[/dim]")
        console.print("   [bold]uv run email-agent label custom create \"work\" --search-existing[/bold]\n")
        
        console.print("For detailed help on any command:")
        console.print("[bold]uv run email-agent label COMMAND --help[/bold]")

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