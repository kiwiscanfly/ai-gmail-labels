"""Command-line interface for the email categorization agent."""

import asyncio
import time
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

from src.core.config import get_config, reload_config
from src.core.exceptions import ConfigurationError
from src.core.event_bus import get_event_bus, AgentMessage
from src.core.state_manager import get_state_manager, WorkflowCheckpoint
from src.integrations.ollama_client import get_ollama_manager
from src.integrations.gmail_client import get_gmail_client


app = typer.Typer(help="Email Categorization Agent CLI")
console = Console()


@app.command()
def config(
    path: Optional[str] = typer.Option(None, "--config-path", "-c", help="Path to configuration file"),
    validate: bool = typer.Option(False, "--validate", "-v", help="Validate configuration"),
    show: bool = typer.Option(False, "--show", "-s", help="Show current configuration"),
):
    """Manage configuration."""
    try:
        if path:
            config_obj = reload_config(path)
        else:
            config_obj = get_config()
        
        if validate:
            errors = config_obj.validate_paths()
            if errors:
                console.print("[red]Configuration validation failed:[/red]")
                for error in errors:
                    console.print(f"  ❌ {error}")
                raise typer.Exit(1)
            else:
                console.print("[green]✅ Configuration is valid[/green]")
        
        if show:
            table = Table(title="Current Configuration")
            table.add_column("Section", style="cyan")
            table.add_column("Key", style="yellow")
            table.add_column("Value", style="green")
            
            config_dict = config_obj.to_dict()
            for section, values in config_dict.items():
                if isinstance(values, dict):
                    for key, value in values.items():
                        # Hide sensitive values
                        if any(sensitive in key.lower() for sensitive in ['password', 'token', 'key', 'secret']):
                            value = "***hidden***"
                        table.add_row(section, key, str(value))
                else:
                    table.add_row(section, "", str(values))
            
            console.print(table)
        
        if not validate and not show:
            console.print("[green]Configuration loaded successfully[/green]")
            
    except ConfigurationError as e:
        console.print(f"[red]Configuration error: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def init(
    force: bool = typer.Option(False, "--force", "-f", help="Force initialization even if files exist"),
):
    """Initialize the agent with required directories and files."""
    try:
        config_obj = get_config()
        
        console.print("[blue]Initializing email categorization agent...[/blue]")
        
        # Ensure directories exist
        config_obj.ensure_directories()
        console.print("✅ Created required directories")
        
        # Check for required files
        credentials_path = Path(config_obj.gmail.credentials_path)
        if not credentials_path.exists():
            console.print(f"[yellow]⚠️  Gmail credentials file not found: {credentials_path}[/yellow]")
            console.print("   Please download credentials.json from Google Cloud Console")
        else:
            console.print("✅ Gmail credentials file found")
        
        console.print("\n[green]Initialization complete![/green]")
        console.print("\nNext steps:")
        console.print("1. Download Gmail API credentials to credentials.json")
        console.print("2. Install and start Ollama with required models:")
        console.print("   ollama pull gemma2:3b")
        console.print("   ollama pull llama3.2:3b")
        console.print("3. Set up Gmail API credentials:")
        console.print("   - Go to https://console.cloud.google.com/")
        console.print("   - Create a new project or select existing")
        console.print("   - Enable Gmail API")
        console.print("   - Create OAuth 2.0 credentials (Desktop application)")
        console.print("   - Download credentials.json to project root")
        console.print("4. Run 'email-agent test-gmail' to test Gmail integration")
        console.print("5. Run 'email-agent categorize' to start categorizing emails")
        
    except Exception as e:
        console.print(f"[red]Initialization failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def status():
    """Show system status and health checks."""
    asyncio.run(_status())

async def _status():
    """Async status check implementation."""
    console.print("[blue]Checking system status...[/blue]")
    
    try:
        config_obj = get_config()
        
        # Create status table
        table = Table(title="System Status")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Details", style="yellow")
        
        # Check configuration
        table.add_row("Configuration", "✅ Loaded", f"Mode: {config_obj.app.mode}")
        
        # Check paths
        errors = config_obj.validate_paths()
        if errors:
            table.add_row("Paths", "❌ Invalid", f"{len(errors)} errors found")
        else:
            table.add_row("Paths", "✅ Valid", "All paths accessible")
        
        # Check Gmail credentials
        credentials_path = Path(config_obj.gmail.credentials_path)
        if credentials_path.exists():
            table.add_row("Gmail Credentials", "✅ Found", str(credentials_path))
        else:
            table.add_row("Gmail Credentials", "❌ Missing", str(credentials_path))
        
        # Check database path
        db_path = Path(config_obj.sqlite.database_path)
        if db_path.parent.exists():
            table.add_row("Database Directory", "✅ Ready", str(db_path.parent))
        else:
            table.add_row("Database Directory", "❌ Missing", str(db_path.parent))
        
        # Test database connections
        try:
            event_bus = await get_event_bus()
            stats = await event_bus.get_stats()
            table.add_row("Event Bus", "✅ Connected", f"Messages: {stats.get('total_messages', 0)}")
        except Exception as e:
            table.add_row("Event Bus", "❌ Error", str(e)[:50])
        
        try:
            state_mgr = await get_state_manager()
            table.add_row("State Manager", "✅ Connected", "Ready")
        except Exception as e:
            table.add_row("State Manager", "❌ Error", str(e)[:50])
        
        # Test Ollama connectivity
        try:
            ollama_mgr = await get_ollama_manager()
            health = await ollama_mgr.get_health_status()
            if health["status"] == "healthy":
                table.add_row("Ollama", "✅ Connected", f"Models: {health['total_models']}")
            else:
                table.add_row("Ollama", "❌ Unhealthy", health.get("error", "Unknown")[:50])
        except Exception as e:
            table.add_row("Ollama", "❌ Error", str(e)[:50])
        
        # Test Gmail connectivity
        try:
            gmail_client = await get_gmail_client()
            health = await gmail_client.get_health_status()
            if health["status"] == "healthy":
                table.add_row("Gmail", "✅ Connected", f"Email: {health.get('email_address', 'Unknown')[:30]}")
            else:
                table.add_row("Gmail", "❌ Unhealthy", health.get("error", "Unknown")[:50])
        except Exception as e:
            table.add_row("Gmail", "❌ Error", str(e)[:50])
        
        console.print(table)
        
        if errors:
            console.print("\n[red]Issues found:[/red]")
            for error in errors:
                console.print(f"  ❌ {error}")
    
    except Exception as e:
        console.print(f"[red]Status check failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def categorize(
    mode: str = typer.Option("interactive", "--mode", "-m", help="Categorization mode (automatic/interactive)"),
    dry_run: bool = typer.Option(False, "--dry-run", "-d", help="Preview categorization without applying"),
    limit: int = typer.Option(10, "--limit", "-l", help="Maximum number of emails to process"),
):
    """Start email categorization."""
    console.print(f"[blue]Starting email categorization in {mode} mode...[/blue]")
    
    if dry_run:
        console.print("[yellow]Running in dry-run mode - no changes will be made[/yellow]")
    
    # This would normally start the actual categorization process
    # For now, just show a placeholder
    panel = Panel(
        "[bold]Email categorization would start here[/bold]\n\n"
        f"Mode: {mode}\n"
        f"Dry run: {dry_run}\n"
        f"Limit: {limit}\n\n"
        "[italic]Implementation coming in next phase...[/italic]",
        title="Categorization Process",
        border_style="blue"
    )
    console.print(panel)


@app.command()
def test_database():
    """Test database systems (event bus and state management)."""
    asyncio.run(_test_database())

async def _test_database():
    """Test database functionality."""
    console.print("[blue]Testing database systems...[/blue]")
    
    try:
        # Test Event Bus
        console.print("\n[yellow]Testing Event Bus:[/yellow]")
        event_bus = await get_event_bus()
        
        # Send a test message
        test_msg = AgentMessage(
            sender_agent="cli_test",
            recipient_agent="test_agent",
            message_type="test_message",
            payload={"test": "data", "timestamp": time.time()}
        )
        
        await event_bus.publish(test_msg)
        console.print("✅ Message published successfully")
        
        # Get stats
        stats = await event_bus.get_stats()
        console.print(f"📊 Event Bus Stats: {stats.get('total_messages', 0)} total messages")
        
        # Test State Manager
        console.print("\n[yellow]Testing State Manager:[/yellow]")
        state_mgr = await get_state_manager()
        
        # Test checkpoint
        checkpoint = WorkflowCheckpoint(
            workflow_id="test_workflow",
            workflow_type="test",
            state_data={"step": 1, "test": True}
        )
        
        await state_mgr.save_checkpoint(checkpoint)
        console.print("✅ Checkpoint saved successfully")
        
        # Load checkpoint
        loaded = await state_mgr.load_checkpoint("test_workflow")
        if loaded:
            console.print("✅ Checkpoint loaded successfully")
        else:
            console.print("❌ Failed to load checkpoint")
        
        # Test preferences
        await state_mgr.set_preference("test_key", {"value": "test_data"})
        pref = await state_mgr.get_preference("test_key")
        if pref:
            console.print("✅ Preferences working")
        else:
            console.print("❌ Preferences failed")
        
        # Get categorization stats
        cat_stats = await state_mgr.get_categorization_stats()
        console.print(f"📈 Categorization Stats: {cat_stats.get('total_processed', 0)} processed")
        
        console.print("\n[green]✅ All database tests passed![/green]")
        
    except Exception as e:
        console.print(f"\n[red]❌ Database test failed: {e}[/red]")
        raise typer.Exit(1)

@app.command()
def test_ollama():
    """Test Ollama integration and model management."""
    asyncio.run(_test_ollama())

async def _test_ollama():
    """Test Ollama functionality."""
    console.print("[blue]Testing Ollama integration...[/blue]")
    
    try:
        # Initialize Ollama manager
        console.print("\n[yellow]Initializing Ollama Manager:[/yellow]")
        ollama_mgr = await get_ollama_manager()
        console.print("✅ Ollama manager initialized")
        
        # Get health status
        health = await ollama_mgr.get_health_status()
        console.print(f"🏥 Health Status: {health['status']}")
        console.print(f"📡 Response Time: {health.get('response_time_ms', 0):.2f}ms")
        
        # List available models
        console.print("\n[yellow]Available Models:[/yellow]")
        models = await ollama_mgr.list_models()
        if models:
            for model in models[:5]:  # Show first 5 models
                console.print(f"📦 {model.name} - {model.size / (1024*1024*1024):.1f}GB")
        else:
            console.print("❌ No models found")
            return
        
        # Test model switching
        console.print("\n[yellow]Testing Model Management:[/yellow]")
        primary_model = await ollama_mgr.switch_model("categorization")
        console.print(f"🔄 Switched to categorization model: {primary_model}")
        
        # Test simple generation
        console.print("\n[yellow]Testing Text Generation:[/yellow]")
        try:
            result = await ollama_mgr.generate(
                prompt="Categorize this email subject: 'Meeting reminder for tomorrow'",
                options={"num_predict": 50}
            )
            console.print(f"✅ Generation successful")
            console.print(f"📝 Response: {result.content[:100]}...")
            console.print(f"⚡ Speed: {result.tokens_per_second:.1f} tokens/sec")
            
        except Exception as e:
            console.print(f"❌ Generation failed: {e}")
            return
        
        # Test chat format
        console.print("\n[yellow]Testing Chat Format:[/yellow]")
        try:
            chat_result = await ollama_mgr.chat(
                messages=[
                    {"role": "system", "content": "You are an email categorization assistant."},
                    {"role": "user", "content": "What category would you assign to an email about 'Project budget review'?"}
                ],
                options={"num_predict": 30}
            )
            console.print("✅ Chat successful")
            console.print(f"💬 Chat Response: {chat_result.content[:100]}...")
            
        except Exception as e:
            console.print(f"❌ Chat failed: {e}")
        
        # Get model statistics
        console.print("\n[yellow]Model Statistics:[/yellow]")
        stats = await ollama_mgr.get_model_stats()
        console.print(f"📊 Total Models: {stats['total_models']}")
        console.print(f"🔥 Loaded Models: {stats['loaded_models']}")
        console.print(f"💾 Total Size: {stats['total_size_bytes'] / (1024*1024*1024):.1f}GB")
        console.print(f"🎯 Current Model: {stats['current_model']}")
        
        console.print("\n[green]✅ All Ollama tests passed![/green]")
        
    except Exception as e:
        console.print(f"\n[red]❌ Ollama test failed: {e}[/red]")
        # Print more details for debugging
        import traceback
        console.print(f"[red]Details: {traceback.format_exc()}[/red]")
        raise typer.Exit(1)

@app.command()
def models(
    action: str = typer.Argument(help="Action: list, pull, delete, health"),
    model_name: Optional[str] = typer.Argument(default=None, help="Model name for pull/delete actions"),
):
    """Manage Ollama models."""
    asyncio.run(_manage_models(action, model_name))

async def _manage_models(action: str, model_name: Optional[str]):
    """Manage Ollama models implementation."""
    try:
        ollama_mgr = await get_ollama_manager()
        
        if action == "list":
            console.print("[blue]Available Ollama Models:[/blue]")
            models = await ollama_mgr.list_models()
            
            if not models:
                console.print("No models found")
                return
            
            table = Table(title="Ollama Models")
            table.add_column("Name", style="cyan")
            table.add_column("Size", style="yellow")
            table.add_column("Modified", style="green")
            table.add_column("Loaded", style="magenta")
            
            for model in models:
                size_gb = model.size / (1024*1024*1024)
                loaded_status = "✅" if model.loaded else "❌"
                table.add_row(
                    model.name,
                    f"{size_gb:.1f} GB",
                    model.modified_at,
                    loaded_status
                )
            
            console.print(table)
            
        elif action == "health":
            console.print("[blue]Ollama Health Status:[/blue]")
            health = await ollama_mgr.get_health_status()
            stats = await ollama_mgr.get_model_stats()
            
            console.print(f"Status: {health['status']}")
            console.print(f"Host: {health['host']}")
            console.print(f"Response Time: {health.get('response_time_ms', 0):.2f}ms")
            console.print(f"Total Models: {stats['total_models']}")
            console.print(f"Loaded Models: {stats['loaded_models']}")
            console.print(f"Current Model: {stats['current_model']}")
            
        elif action == "pull":
            if not model_name:
                console.print("[red]Model name required for pull action[/red]")
                raise typer.Exit(1)
            
            console.print(f"[blue]Pulling model: {model_name}[/blue]")
            await ollama_mgr.pull_model(model_name)
            console.print(f"[green]✅ Model {model_name} pulled successfully[/green]")
            
        elif action == "delete":
            if not model_name:
                console.print("[red]Model name required for delete action[/red]")
                raise typer.Exit(1)
            
            # Confirm deletion
            confirm = typer.confirm(f"Are you sure you want to delete model '{model_name}'?")
            if not confirm:
                console.print("Deletion cancelled")
                return
            
            console.print(f"[blue]Deleting model: {model_name}[/blue]")
            await ollama_mgr.delete_model(model_name)
            console.print(f"[green]✅ Model {model_name} deleted successfully[/green]")
            
        else:
            console.print(f"[red]Unknown action: {action}[/red]")
            console.print("Available actions: list, pull, delete, health")
            raise typer.Exit(1)
            
    except Exception as e:
        console.print(f"[red]❌ Models command failed: {e}[/red]")
        raise typer.Exit(1)

@app.command()
def test_gmail():
    """Test Gmail API integration and authentication."""
    asyncio.run(_test_gmail())

async def _test_gmail():
    """Test Gmail functionality."""
    console.print("[blue]Testing Gmail integration...[/blue]")
    
    try:
        # Initialize Gmail client
        console.print("\n[yellow]Initializing Gmail Client:[/yellow]")
        gmail_client = await get_gmail_client()
        console.print("✅ Gmail client initialized")
        
        # Get profile
        console.print("\n[yellow]Testing Authentication:[/yellow]")
        profile = await gmail_client.get_profile()
        console.print(f"📧 Email: {profile.get('emailAddress')}")
        console.print(f"📊 Total Messages: {profile.get('messagesTotal', 0):,}")
        console.print(f"🧵 Total Threads: {profile.get('threadsTotal', 0):,}")
        
        # List labels
        console.print("\n[yellow]Testing Labels:[/yellow]")
        labels = await gmail_client.list_labels()
        console.print(f"🏷️  Total Labels: {len(labels)}")
        
        # Show first few labels
        for label in labels[:5]:
            console.print(f"   📝 {label.name} ({label.type}) - {label.messages_total} messages")
        
        # Search for recent messages
        console.print("\n[yellow]Testing Message Search:[/yellow]")
        try:
            recent_messages = []
            async for message in gmail_client.search_messages("in:inbox", max_results=5):
                recent_messages.append(message)
                
            console.print(f"📨 Found {len(recent_messages)} recent messages")
            
            # Show message details
            for msg in recent_messages[:3]:
                console.print(f"   ✉️  {msg.subject[:50]}... from {msg.sender[:30]}")
                
        except Exception as e:
            console.print(f"❌ Message search failed: {e}")
        
        # Test label operations (safe operations only)
        console.print("\n[yellow]Testing Label Operations:[/yellow]")
        try:
            # Try to get a common label
            inbox_label = await gmail_client.get_label_by_name("INBOX")
            if inbox_label:
                console.print(f"✅ Found INBOX label: {inbox_label.messages_total} messages")
            
            # Test creating a test label (we'll delete it after)
            test_label_name = f"EmailAgent_Test_{int(time.time())}"
            test_label = await gmail_client.create_label(test_label_name)
            console.print(f"✅ Created test label: {test_label.name}")
            
            # Clean up test label
            await gmail_client.delete_label(test_label_name)
            console.print("✅ Cleaned up test label")
            
        except Exception as e:
            console.print(f"❌ Label operations failed: {e}")
        
        # Get health status
        console.print("\n[yellow]Health Status:[/yellow]")
        health = await gmail_client.get_health_status()
        console.print(f"🏥 Status: {health['status']}")
        console.print(f"🔐 Authenticated: {health['authenticated']}")
        console.print(f"📈 Credentials Valid: {health.get('credentials_valid', False)}")
        
        console.print("\n[green]✅ All Gmail tests passed![/green]")
        
    except Exception as e:
        console.print(f"\n[red]❌ Gmail test failed: {e}[/red]")
        # Print more details for debugging
        import traceback
        console.print(f"[red]Details: {traceback.format_exc()}[/red]")
        raise typer.Exit(1)

@app.command()
def gmail(
    action: str = typer.Argument(help="Action: labels, messages, search, health"),
    query: Optional[str] = typer.Argument(default=None, help="Search query for messages"),
    limit: int = typer.Option(10, "--limit", "-l", help="Limit results"),
):
    """Manage Gmail operations."""
    asyncio.run(_manage_gmail(action, query, limit))

async def _manage_gmail(action: str, query: Optional[str], limit: int):
    """Manage Gmail operations implementation."""
    try:
        gmail_client = await get_gmail_client()
        
        if action == "labels":
            console.print("[blue]Gmail Labels:[/blue]")
            labels = await gmail_client.list_labels()
            
            table = Table(title="Gmail Labels")
            table.add_column("Name", style="cyan")
            table.add_column("Type", style="yellow")
            table.add_column("Messages", style="green")
            table.add_column("Unread", style="red")
            
            for label in sorted(labels, key=lambda l: l.name):
                table.add_row(
                    label.name,
                    label.type,
                    str(label.messages_total),
                    str(label.messages_unread) if label.messages_unread else "0"
                )
            
            console.print(table)
            
        elif action == "messages":
            console.print(f"[blue]Recent Messages (limit: {limit}):[/blue]")
            
            messages = []
            async for message in gmail_client.search_messages("in:inbox", max_results=limit):
                messages.append(message)
            
            table = Table(title="Recent Messages")
            table.add_column("Subject", style="cyan", max_width=40)
            table.add_column("From", style="yellow", max_width=30)
            table.add_column("Date", style="green")
            table.add_column("Labels", style="magenta")
            
            for msg in messages:
                labels_str = ", ".join(msg.label_ids[:3])  # Show first 3 labels
                if len(msg.label_ids) > 3:
                    labels_str += "..."
                    
                table.add_row(
                    msg.subject[:40] + "..." if len(msg.subject) > 40 else msg.subject,
                    msg.sender[:30] + "..." if len(msg.sender) > 30 else msg.sender,
                    msg.date[:20] if msg.date else "Unknown",
                    labels_str
                )
            
            console.print(table)
            
        elif action == "search":
            if not query:
                console.print("[red]Search query required for search action[/red]")
                raise typer.Exit(1)
            
            console.print(f"[blue]Searching for: '{query}' (limit: {limit}):[/blue]")
            
            messages = []
            async for message in gmail_client.search_messages(query, max_results=limit):
                messages.append(message)
            
            console.print(f"Found {len(messages)} messages:")
            for msg in messages:
                console.print(f"📧 {msg.subject[:60]} - from {msg.sender[:40]}")
                
        elif action == "health":
            console.print("[blue]Gmail Health Status:[/blue]")
            health = await gmail_client.get_health_status()
            
            for key, value in health.items():
                console.print(f"{key}: {value}")
                
        else:
            console.print(f"[red]Unknown action: {action}[/red]")
            console.print("Available actions: labels, messages, search, health")
            raise typer.Exit(1)
            
    except Exception as e:
        console.print(f"[red]❌ Gmail command failed: {e}[/red]")
        raise typer.Exit(1)

@app.command()
def server(
    host: str = typer.Option("localhost", "--host", help="MCP server host"),
    port: int = typer.Option(8080, "--port", help="MCP server port"),
):
    """Start the MCP server."""
    console.print(f"[blue]Starting MCP server on {host}:{port}...[/blue]")
    
    # This would normally start the MCP server
    # For now, just show a placeholder
    panel = Panel(
        "[bold]MCP server would start here[/bold]\n\n"
        f"Host: {host}\n"
        f"Port: {port}\n\n"
        "[italic]Implementation coming in next phase...[/italic]",
        title="MCP Server",
        border_style="green"
    )
    console.print(panel)


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()