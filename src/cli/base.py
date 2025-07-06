"""Base CLI command classes and utilities."""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.panel import Panel

from src.services.email_service import EmailService


class BaseCLICommand(ABC):
    """Base class for all CLI commands following the service layer pattern."""
    
    def __init__(self):
        self.console = Console()
        self.email_service = None
        self.initialized = False
    
    async def initialize(self):
        """Initialize services following the architecture pattern."""
        if not self.initialized:
            self.email_service = EmailService()
            await self.email_service.initialize()
            self.initialized = True
    
    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the command logic."""
        pass
    
    def format_output(self, results: Dict[str, Any]) -> None:
        """Format and display results using rich console."""
        pass
    
    def _parse_target(self, target: str) -> str:
        """Parse target specification into Gmail query."""
        if target == "unread":
            return "is:unread"
        elif target == "all":
            return ""
        elif target.startswith("recent:"):
            days = target.split(":")[1].replace("days", "").replace("day", "")
            return f"newer_than:{days}d"
        elif target.startswith("from:"):
            sender = target.split(":", 1)[1]
            return f"from:{sender}"
        elif target.startswith("query:"):
            return target.split(":", 1)[1]
        else:
            # Default to treating as unread if not recognized
            return "is:unread"


class BaseEmailProcessor(BaseCLICommand):
    """Base class for email processing commands."""
    
    def __init__(self):
        super().__init__()
        self.progress = None
        
    async def process_emails(
        self, 
        query: str,
        limit: Optional[int] = None,
        dry_run: bool = True
    ) -> List[Any]:
        """Process emails using the email service layer."""
        emails = []
        
        self.console.print(f"[blue]Searching for emails with query: '{query}'[/blue]")
        
        async for email_ref in self.email_service.search_emails(query, limit):
            full_email = await self.email_service.get_email_content(email_ref.email_id)
            if full_email:
                emails.append(full_email)
        
        self.console.print(f"[green]Found {len(emails)} emails to process[/green]")
        return emails
    
    def display_results_table(self, results: Dict[str, Any], columns: Dict[str, str]):
        """Display results in a formatted table."""
        console = self.console
        
        # Summary panel
        console.print("\n")
        summary_text = f"Processing Complete: {results.get('successful', 0)}/{results.get('total', len(results.get('results', [])))} emails processed successfully"
        if results.get('dry_run'):
            summary_text += " (DRY RUN)"
        
        console.print(Panel(
            summary_text,
            title="Summary",
            style="green" if results.get('failed', 0) == 0 else "yellow"
        ))
        
        if not results.get('results'):
            return
        
        # Create results table
        table = Table(title="Email Processing Results")
        for col_key, col_title in columns.items():
            table.add_column(col_title, style="cyan" if col_key == "subject" else "white")
        
        for result in results['results'][:20]:  # Show first 20
            row_data = []
            for col_key in columns.keys():
                if col_key == "subject":
                    value = result.get('subject', 'Unknown')[:50]
                    if len(result.get('subject', '')) > 50:
                        value += "..."
                elif col_key == "sender":
                    value = result.get('sender', 'Unknown')[:30]
                    if len(result.get('sender', '')) > 30:
                        value += "..."
                elif col_key == "labels":
                    if results.get('dry_run'):
                        labels = result.get('labels_would_apply', [])
                    else:
                        labels = result.get('labels_applied', [])
                    value = ", ".join(labels) if labels else "None"
                else:
                    value = str(result.get(col_key, ''))
                row_data.append(value)
            
            table.add_row(*row_data)
        
        console.print(table)
        
        if len(results.get('results', [])) > 20:
            console.print(f"[yellow]... and {len(results['results']) - 20} more emails[/yellow]")
    
    async def shutdown(self) -> None:
        """Shutdown and clean up resources."""
        try:
            self.console.print("[blue]Shutting down...[/blue]")
            
            # Clean up database connections
            try:
                from src.core.database_pool import shutdown_database_pool
                await shutdown_database_pool()
            except Exception:
                pass
                
        except Exception as e:
            self.console.print(f"[red]Error during shutdown: {e}[/red]")


def run_async_command(command_func):
    """Decorator to run async command functions."""
    def wrapper(*args, **kwargs):
        async def run():
            command = None
            try:
                result = await command_func(*args, **kwargs)
                if hasattr(result, 'shutdown'):
                    command = result
                    await command.shutdown()
                return result
            except KeyboardInterrupt:
                Console().print("\n[yellow]Operation cancelled by user.[/yellow]")
                if command and hasattr(command, 'shutdown'):
                    await command.shutdown()
            except Exception as e:
                Console().print(f"\n[red]Error: {e}[/red]")
                if command and hasattr(command, 'shutdown'):
                    await command.shutdown()
                raise
        
        asyncio.run(run())
    
    return wrapper