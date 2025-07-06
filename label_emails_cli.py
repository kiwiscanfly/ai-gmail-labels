#!/usr/bin/env python3
"""
CLI script for email labeling using LangChain and classifier services.
Uses the existing infrastructure to classify and label emails automatically.
"""

import asyncio
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich.text import Text

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.config import get_config
from src.services.email_service import EmailService
from src.services.email_prioritizer import EmailPrioritizer
from src.services.marketing_classifier import MarketingEmailClassifier
from src.services.receipt_classifier import ReceiptClassifier
from src.integrations.ollama_client import get_ollama_manager
from src.integrations.gmail_client import get_gmail_client

console = Console()
app = typer.Typer(help="Email Labeling CLI using LangChain and Classifier Services")

class EmailLabeler:
    """Main email labeling class that orchestrates all services."""
    
    def __init__(self):
        self.config = get_config()
        self.email_service = None
        self.prioritizer = None
        self.marketing_classifier = None
        self.receipt_classifier = None
        self.ollama_manager = None
        self.gmail_client = None
        
    async def initialize(self):
        """Initialize all services."""
        console.print("[blue]Initializing services...[/blue]")
        
        # Initialize services
        self.email_service = EmailService()
        await self.email_service.initialize()
        
        self.prioritizer = EmailPrioritizer()
        await self.prioritizer.initialize()
        
        self.marketing_classifier = MarketingEmailClassifier()
        await self.marketing_classifier.initialize()
        
        self.receipt_classifier = ReceiptClassifier()
        await self.receipt_classifier.initialize()
        
        self.ollama_manager = await get_ollama_manager()
        self.gmail_client = await get_gmail_client()
        
        console.print("[green]✓ All services initialized successfully[/green]")
    
    async def classify_email(self, email) -> Dict[str, Any]:
        """Classify a single email using all classifiers."""
        classifications = {}
        labels_to_apply = []
        
        try:
            # Priority classification
            if self.prioritizer:
                priority_result = await self.prioritizer.classify_email(email)
                classifications["priority"] = {
                    "level": priority_result.get("priority", "medium"),
                    "confidence": priority_result.get("confidence", 0.5),
                    "reasoning": priority_result.get("reasoning", "Priority analysis completed"),
                    "is_urgent": priority_result.get("is_urgent", False)
                }
                
                # Add priority label
                priority_level = priority_result.get("priority", "medium")
                labels_to_apply.append(f"Priority/{priority_level.title()}")
            
            # Marketing classification
            if self.marketing_classifier:
                marketing_result = await self.marketing_classifier.classify_email(email)
                is_marketing = marketing_result.get("is_marketing", False)
                
                classifications["marketing"] = {
                    "is_marketing": is_marketing,
                    "subtype": marketing_result.get("subtype", "unknown"),
                    "confidence": marketing_result.get("confidence", 0.5),
                    "reasoning": marketing_result.get("reasoning", "Marketing analysis completed")
                }
                
                if is_marketing:
                    subtype = marketing_result.get("subtype", "promotional")
                    labels_to_apply.append(f"Marketing/{subtype.title()}")
            
            # Receipt classification
            if self.receipt_classifier:
                receipt_result = await self.receipt_classifier.classify_email(email)
                is_receipt = receipt_result.get("is_receipt", False)
                
                classifications["receipt"] = {
                    "is_receipt": is_receipt,
                    "vendor": receipt_result.get("vendor", "Unknown"),
                    "confidence": receipt_result.get("confidence", 0.5),
                    "reasoning": receipt_result.get("reasoning", "Receipt analysis completed")
                }
                
                if is_receipt:
                    vendor = receipt_result.get("vendor", "Purchase")
                    labels_to_apply.append(f"Receipts/{vendor}")
            
            return {
                "email_id": email.id,
                "subject": email.subject or "No Subject",
                "sender": email.sender or "Unknown Sender",
                "classifications": classifications,
                "suggested_labels": labels_to_apply,
                "processed": True
            }
            
        except Exception as e:
            return {
                "email_id": getattr(email, 'id', 'unknown'),
                "subject": getattr(email, 'subject', 'Unknown'),
                "sender": getattr(email, 'sender', 'Unknown'),
                "error": str(e),
                "processed": False
            }
    
    async def apply_labels_to_email(self, email_id: str, labels: List[str], create_labels: bool = True) -> Dict[str, Any]:
        """Apply labels to an email."""
        results = {"applied": [], "failed": []}
        
        for label in labels:
            try:
                await self.gmail_client.apply_label_by_name(
                    email_id, 
                    label, 
                    create_if_missing=create_labels
                )
                results["applied"].append(label)
            except Exception as e:
                results["failed"].append({"label": label, "error": str(e)})
        
        return results
    
    async def process_emails(
        self, 
        query: str, 
        limit: int = 50, 
        dry_run: bool = True,
        classification_types: List[str] = None
    ) -> Dict[str, Any]:
        """Process emails with the specified query."""
        if classification_types is None:
            classification_types = ["priority", "marketing", "receipt"]
        
        results = []
        successful = 0
        failed = 0
        
        # Search for emails
        emails = []
        console.print(f"[blue]Searching for emails with query: '{query}'[/blue]")
        
        async for email_ref in self.email_service.search_emails(query, limit):
            full_email = await self.email_service.get_email_content(email_ref.email_id)
            if full_email:
                emails.append(full_email)
        
        console.print(f"[green]Found {len(emails)} emails to process[/green]")
        
        if not emails:
            return {
                "total": 0,
                "successful": 0,
                "failed": 0,
                "results": [],
                "dry_run": dry_run
            }
        
        # Process emails with progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Processing emails...", total=len(emails))
            
            for email in emails:
                try:
                    # Classify email
                    result = await self.classify_email(email)
                    
                    # Apply labels if not dry run
                    if not dry_run and result.get("processed") and result.get("suggested_labels"):
                        label_results = await self.apply_labels_to_email(
                            email.id, 
                            result["suggested_labels"]
                        )
                        result["labels_applied"] = label_results["applied"]
                        result["labels_failed"] = label_results["failed"]
                    elif dry_run:
                        result["labels_would_apply"] = result.get("suggested_labels", [])
                    
                    results.append(result)
                    
                    if result.get("processed"):
                        successful += 1
                    else:
                        failed += 1
                    
                except Exception as e:
                    failed += 1
                    results.append({
                        "email_id": getattr(email, 'id', 'unknown'),
                        "subject": getattr(email, 'subject', 'Unknown'),
                        "error": str(e),
                        "processed": False
                    })
                
                progress.advance(task)
        
        return {
            "total": len(emails),
            "successful": successful,
            "failed": failed,
            "results": results,
            "dry_run": dry_run,
            "classification_types": classification_types
        }


def display_results(results: Dict[str, Any]):
    """Display the results in a nice table format."""
    console.print("\n")
    console.print(Panel(
        f"Processing Complete: {results['successful']}/{results['total']} emails processed successfully",
        title="Summary",
        style="green" if results['failed'] == 0 else "yellow"
    ))
    
    if not results['results']:
        return
    
    # Create main results table
    table = Table(title="Email Classification Results")
    table.add_column("Subject", style="cyan", max_width=40)
    table.add_column("Sender", style="magenta", max_width=30)
    table.add_column("Priority", justify="center")
    table.add_column("Marketing", justify="center")
    table.add_column("Receipt", justify="center")
    
    if results['dry_run']:
        table.add_column("Would Apply Labels", style="yellow")
    else:
        table.add_column("Applied Labels", style="green")
        table.add_column("Failed Labels", style="red")
    
    for result in results['results'][:20]:  # Show first 20
        if not result.get('processed'):
            table.add_row(
                result.get('subject', 'Unknown'),
                result.get('sender', 'Unknown'),
                "[red]ERROR[/red]",
                "[red]ERROR[/red]",
                "[red]ERROR[/red]",
                result.get('error', 'Unknown error')
            )
            continue
        
        classifications = result.get('classifications', {})
        
        # Priority info
        priority_info = classifications.get('priority', {})
        priority_text = f"{priority_info.get('level', 'unknown')} ({priority_info.get('confidence', 0):.2f})"
        if priority_info.get('is_urgent'):
            priority_text += " ⚡"
        
        # Marketing info
        marketing_info = classifications.get('marketing', {})
        if marketing_info.get('is_marketing'):
            marketing_text = f"✓ {marketing_info.get('subtype', 'unknown')} ({marketing_info.get('confidence', 0):.2f})"
        else:
            marketing_text = "✗"
        
        # Receipt info
        receipt_info = classifications.get('receipt', {})
        if receipt_info.get('is_receipt'):
            receipt_text = f"✓ {receipt_info.get('vendor', 'unknown')} ({receipt_info.get('confidence', 0):.2f})"
        else:
            receipt_text = "✗"
        
        # Labels
        if results['dry_run']:
            labels_text = ", ".join(result.get('labels_would_apply', []))
            table.add_row(
                result.get('subject', 'Unknown')[:40],
                result.get('sender', 'Unknown')[:30],
                priority_text,
                marketing_text,
                receipt_text,
                labels_text
            )
        else:
            applied_labels = ", ".join(result.get('labels_applied', []))
            failed_labels = ", ".join([f["label"] for f in result.get('labels_failed', [])])
            table.add_row(
                result.get('subject', 'Unknown')[:40],
                result.get('sender', 'Unknown')[:30],
                priority_text,
                marketing_text,
                receipt_text,
                applied_labels,
                failed_labels
            )
    
    console.print(table)
    
    if len(results['results']) > 20:
        console.print(f"[yellow]... and {len(results['results']) - 20} more emails[/yellow]")


@app.command()
def label_unread(
    limit: int = typer.Option(50, "--limit", "-l", help="Maximum number of emails to process"),
    dry_run: bool = typer.Option(True, "--dry-run/--apply", help="Preview labels without applying them"),
    types: str = typer.Option("priority,marketing,receipt", "--types", "-t", help="Classification types (comma-separated)")
):
    """Label unread emails using AI classification."""
    classification_types = [t.strip() for t in types.split(",")]
    asyncio.run(_label_emails("is:unread", limit, dry_run, classification_types))


@app.command()
def label_recent(
    days: int = typer.Option(1, "--days", "-d", help="Number of days to look back"),
    limit: int = typer.Option(50, "--limit", "-l", help="Maximum number of emails to process"),
    dry_run: bool = typer.Option(True, "--dry-run/--apply", help="Preview labels without applying them"),
    types: str = typer.Option("priority,marketing,receipt", "--types", "-t", help="Classification types (comma-separated)")
):
    """Label recent emails using AI classification."""
    query = f"newer_than:{days}d"
    classification_types = [t.strip() for t in types.split(",")]
    asyncio.run(_label_emails(query, limit, dry_run, classification_types))


@app.command()
def label_from(
    sender: str = typer.Argument(..., help="Sender email or domain"),
    limit: int = typer.Option(50, "--limit", "-l", help="Maximum number of emails to process"),
    dry_run: bool = typer.Option(True, "--dry-run/--apply", help="Preview labels without applying them"),
    types: str = typer.Option("priority,marketing,receipt", "--types", "-t", help="Classification types (comma-separated)")
):
    """Label emails from a specific sender using AI classification."""
    query = f"from:{sender}"
    classification_types = [t.strip() for t in types.split(",")]
    asyncio.run(_label_emails(query, limit, dry_run, classification_types))


@app.command()
def label_custom(
    query: str = typer.Argument(..., help="Gmail search query"),
    limit: int = typer.Option(50, "--limit", "-l", help="Maximum number of emails to process"),
    dry_run: bool = typer.Option(True, "--dry-run/--apply", help="Preview labels without applying them"),
    types: str = typer.Option("priority,marketing,receipt", "--types", "-t", help="Classification types (comma-separated)")
):
    """Label emails matching a custom Gmail query using AI classification."""
    classification_types = [t.strip() for t in types.split(",")]
    asyncio.run(_label_emails(query, limit, dry_run, classification_types))


@app.command()
def status():
    """Check the status of all services."""
    asyncio.run(_check_status())


async def _label_emails(query: str, limit: int, dry_run: bool, classification_types: List[str]):
    """Internal function to label emails."""
    labeler = EmailLabeler()
    
    try:
        await labeler.initialize()
        
        console.print(f"\n[bold]Email Labeling Configuration:[/bold]")
        console.print(f"Query: {query}")
        console.print(f"Limit: {limit}")
        console.print(f"Mode: {'DRY RUN (preview only)' if dry_run else 'APPLY LABELS'}")
        console.print(f"Classification Types: {', '.join(classification_types)}")
        
        if not dry_run:
            confirm = typer.confirm("\n⚠️  This will apply labels to your Gmail account. Continue?")
            if not confirm:
                console.print("[yellow]Operation cancelled.[/yellow]")
                return
        
        console.print("\n[bold]Processing emails...[/bold]")
        results = await labeler.process_emails(query, limit, dry_run, classification_types)
        
        display_results(results)
        
        if dry_run and results['successful'] > 0:
            console.print(f"\n[yellow]💡 To apply these labels, run the same command with --apply[/yellow]")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user.[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")


async def _check_status():
    """Check service status."""
    labeler = EmailLabeler()
    
    console.print("[blue]Checking service status...[/blue]")
    
    try:
        await labeler.initialize()
        
        console.print("\n[green]✓ All services are operational![/green]")
        
        # Test Ollama
        if labeler.ollama_manager:
            health = await labeler.ollama_manager.get_health_status()
            console.print(f"Ollama Status: {health.get('status', 'unknown')}")
            console.print(f"Current Model: {health.get('current_model', 'unknown')}")
        
        # Test Gmail
        if labeler.gmail_client:
            gmail_health = await labeler.gmail_client.get_health_status()
            console.print(f"Gmail Status: {gmail_health.get('status', 'unknown')}")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


if __name__ == "__main__":
    app()