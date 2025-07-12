"""Gmail label management commands."""

import asyncio
from typing import Optional, Dict, Any, List
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint

from src.cli.base import run_async_command
from src.integrations.gmail_client import get_gmail_client
from src.services.email_service import EmailService

app = typer.Typer(help="Gmail label management commands")
console = Console()


class LabelManager:
    """Manager for Gmail label operations."""
    
    def __init__(self):
        self.console = Console()
        self.email_service = None
    
    async def initialize(self):
        """Initialize email service."""
        if not self.email_service:
            self.email_service = EmailService()
            await self.email_service.initialize()
    
    async def shutdown(self):
        """Cleanup resources."""
        try:
            # Clean up database connections
            from src.core.database_pool import shutdown_database_pool
            await shutdown_database_pool()
        except Exception:
            pass
    
    async def list_labels(self, show_usage: bool = False, show_nested: bool = True) -> Dict[str, Any]:
        """List all Gmail labels with optional usage statistics.
        
        Args:
            show_usage: Whether to include usage statistics
            show_nested: Whether to show nested structure
            
        Returns:
            Dictionary with label information
        """
        try:
            # Get all labels with statistics (always fetch for proper display)
            labels = await self.email_service.get_labels(include_stats=True)
            
            if not labels:
                return {"labels": [], "total_count": 0}
            
            # Sort labels alphabetically
            labels.sort(key=lambda x: x.name.lower())
            
            # Process labels for display
            label_data = []
            
            for label in labels:
                label_info = {
                    "id": label.id,
                    "name": label.name,
                    "type": label.type if hasattr(label, 'type') else "user",
                    "messages_total": getattr(label, 'messages_total', 0),
                    "messages_unread": getattr(label, 'messages_unread', 0),
                    "threads_total": getattr(label, 'threads_total', 0),
                    "threads_unread": getattr(label, 'threads_unread', 0),
                    "is_nested": "/" in label.name,
                    "parent": label.name.split("/")[0] if "/" in label.name else None,
                    "level": label.name.count("/")
                }
                
                # Get usage statistics if requested
                if show_usage:
                    try:
                        # Get recent activity for this label
                        recent_emails = []
                        async for email in self.email_service.search_emails(
                            query=f'label:"{label.name}"',
                            limit=10
                        ):
                            recent_emails.append(email)
                        label_info["recent_activity"] = len(recent_emails)
                    except Exception:
                        label_info["recent_activity"] = 0
                
                label_data.append(label_info)
            
            return {
                "labels": label_data,
                "total_count": len(label_data),
                "user_labels": len([l for l in label_data if l["type"] == "user"]),
                "system_labels": len([l for l in label_data if l["type"] == "system"]),
                "nested_labels": len([l for l in label_data if l["is_nested"]])
            }
            
        except Exception as e:
            self.console.print(f"[red]Failed to list labels: {e}[/red]")
            raise
    
    async def analyze_labels(self, unused_threshold_days: int = 30) -> Dict[str, Any]:
        """Analyze label usage and provide cleanup recommendations.
        
        Args:
            unused_threshold_days: Days threshold for considering labels unused
            
        Returns:
            Analysis results with recommendations
        """
        try:
            # Get all user labels with statistics
            all_labels = await self.email_service.get_labels(include_stats=True)
            user_labels = [l for l in all_labels if getattr(l, 'type', 'user') == 'user']
            
            analysis = {
                "total_labels": len(user_labels),
                "empty_labels": [],
                "low_usage_labels": [],
                "duplicate_candidates": [],
                "nested_structure": {},
                "recommendations": []
            }
            
            self.console.print("[blue]Analyzing label usage...[/blue]")
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            ) as progress:
                task = progress.add_task("Analyzing labels...", total=len(user_labels))
                
                for label in user_labels:
                    try:
                        # Check if label has any emails
                        emails = []
                        async for email in self.email_service.search_emails(
                            query=f'label:"{label.name}"',
                            limit=1
                        ):
                            emails.append(email)
                        
                        if not emails:
                            analysis["empty_labels"].append({
                                "name": label.name,
                                "id": label.id,
                                "messages_total": getattr(label, 'messages_total', 0)
                            })
                        elif getattr(label, 'messages_total', 0) < 5:
                            analysis["low_usage_labels"].append({
                                "name": label.name,
                                "id": label.id,
                                "messages_total": getattr(label, 'messages_total', 0)
                            })
                        
                        # Track nested structure
                        if "/" in label.name:
                            parent = label.name.split("/")[0]
                            if parent not in analysis["nested_structure"]:
                                analysis["nested_structure"][parent] = []
                            analysis["nested_structure"][parent].append(label.name)
                        
                        progress.advance(task)
                        
                    except Exception as e:
                        self.console.print(f"[yellow]Warning: Could not analyze {label.name}: {e}[/yellow]")
                        continue
            
            # Generate recommendations
            if analysis["empty_labels"]:
                analysis["recommendations"].append({
                    "type": "cleanup",
                    "priority": "high",
                    "description": f"Consider deleting {len(analysis['empty_labels'])} empty labels",
                    "labels": [l["name"] for l in analysis["empty_labels"][:5]]  # Show first 5
                })
            
            if analysis["low_usage_labels"]:
                analysis["recommendations"].append({
                    "type": "review",
                    "priority": "medium",
                    "description": f"Review {len(analysis['low_usage_labels'])} labels with low usage",
                    "labels": [l["name"] for l in analysis["low_usage_labels"][:5]]
                })
            
            # Check for potential duplicates (similar names)
            label_names = [l.name.lower() for l in user_labels]
            for i, name1 in enumerate(label_names):
                for j, name2 in enumerate(label_names[i+1:], i+1):
                    if self._are_similar_labels(name1, name2):
                        analysis["duplicate_candidates"].append({
                            "label1": user_labels[i].name,
                            "label2": user_labels[j].name,
                            "similarity": "high"
                        })
            
            if analysis["duplicate_candidates"]:
                analysis["recommendations"].append({
                    "type": "merge",
                    "priority": "low",
                    "description": f"Consider merging {len(analysis['duplicate_candidates'])} similar labels",
                    "pairs": analysis["duplicate_candidates"][:3]
                })
            
            return analysis
            
        except Exception as e:
            self.console.print(f"[red]Failed to analyze labels: {e}[/red]")
            raise
    
    def _are_similar_labels(self, name1: str, name2: str) -> bool:
        """Check if two label names are similar enough to be duplicates."""
        # Simple similarity check - can be enhanced
        if abs(len(name1) - len(name2)) > 3:
            return False
        
        # Check for common variations
        variations = [
            (name1.replace(" ", ""), name2.replace(" ", "")),
            (name1.replace("-", ""), name2.replace("-", "")),
            (name1.replace("_", ""), name2.replace("_", "")),
        ]
        
        for v1, v2 in variations:
            if v1 == v2:
                return True
        
        return False
    
    async def create_label(self, name: str, color: Optional[str] = None) -> Dict[str, Any]:
        """Create a new Gmail label.
        
        Args:
            name: Label name (can include / for nesting)
            color: Optional color for the label
            
        Returns:
            Created label information
        """
        try:
            # Check if label already exists
            existing_labels = await self.email_service.get_labels()
            existing_names = {label.name for label in existing_labels}
            
            if name in existing_names:
                return {
                    "success": False,
                    "error": f"Label '{name}' already exists",
                    "existing_label": name
                }
            
            # Create parent labels if needed for nested structure
            if "/" in name:
                parts = name.split("/")
                current_path = ""
                
                for i, part in enumerate(parts[:-1]):  # Don't create the final label yet
                    current_path = "/".join(parts[:i+1])
                    if current_path not in existing_names:
                        self.console.print(f"[blue]Creating parent label: {current_path}[/blue]")
                        await self.email_service.create_label(name=current_path)
                        existing_names.add(current_path)
            
            # Create the actual label
            self.console.print(f"[blue]Creating label: {name}[/blue]")
            label = await self.email_service.create_label(name=name)
            
            return {
                "success": True,
                "label": {
                    "id": label.id,
                    "name": label.name,
                    "created": True
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to create label '{name}': {str(e)}"
            }
    
    async def delete_label(self, name: str, force: bool = False) -> Dict[str, Any]:
        """Delete a Gmail label.
        
        Args:
            name: Label name to delete
            force: Skip confirmation prompts
            
        Returns:
            Deletion result
        """
        try:
            # Find the label
            labels = await self.email_service.get_labels()
            target_label = None
            
            for label in labels:
                if label.name == name:
                    target_label = label
                    break
            
            if not target_label:
                return {
                    "success": False,
                    "error": f"Label '{name}' not found"
                }
            
            # Check if label has emails
            emails = []
            async for email in self.email_service.search_emails(
                query=f'label:"{name}"',
                limit=1
            ):
                emails.append(email)
            
            if emails and not force:
                return {
                    "success": False,
                    "error": f"Label '{name}' contains emails. Use --force to delete anyway.",
                    "has_emails": True,
                    "email_count": getattr(target_label, 'messages_total', 0)
                }
            
            # Delete the label
            await self.email_service.delete_label(target_label.id)
            
            return {
                "success": True,
                "deleted_label": name,
                "had_emails": len(emails) > 0 if emails else False
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to delete label '{name}': {str(e)}"
            }
    
    async def rename_label(self, old_name: str, new_name: str) -> Dict[str, Any]:
        """Rename a Gmail label.
        
        Args:
            old_name: Current label name
            new_name: New label name
            
        Returns:
            Rename result
        """
        try:
            # Find the label
            labels = await self.email_service.get_labels()
            target_label = None
            existing_names = {label.name for label in labels}
            
            for label in labels:
                if label.name == old_name:
                    target_label = label
                    break
            
            if not target_label:
                return {
                    "success": False,
                    "error": f"Label '{old_name}' not found"
                }
            
            if new_name in existing_names:
                return {
                    "success": False,
                    "error": f"Label '{new_name}' already exists"
                }
            
            # Rename the label
            await self.email_service.update_label(target_label.id, name=new_name)
            
            return {
                "success": True,
                "old_name": old_name,
                "new_name": new_name,
                "label_id": target_label.id
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to rename label '{old_name}': {str(e)}"
            }

    def format_output(self, results: Dict[str, Any], command_type: str) -> None:
        """Format and display command results."""
        if command_type == "list":
            self._format_list_output(results)
        elif command_type == "analyze":
            self._format_analyze_output(results)
        elif command_type == "create":
            self._format_create_output(results)
        elif command_type == "delete":
            self._format_delete_output(results)
        elif command_type == "rename":
            self._format_rename_output(results)
    
    def _format_list_output(self, results: Dict[str, Any]) -> None:
        """Format label list output."""
        labels = results.get("labels", [])
        
        if not labels:
            self.console.print("[yellow]No labels found[/yellow]")
            return
        
        # Create table
        table = Table(title=f"Gmail Labels ({results['total_count']} total)")
        table.add_column("Name", style="cyan", no_wrap=False)
        table.add_column("Type", style="green")
        table.add_column("Messages", justify="right", style="blue")
        table.add_column("Unread", justify="right", style="yellow")
        table.add_column("Level", justify="center", style="dim")
        
        for label in labels:
            level_indicator = "  " * label["level"] + ("└─ " if label["level"] > 0 else "")
            name_display = level_indicator + label["name"].split("/")[-1] if label["is_nested"] else label["name"]
            
            table.add_row(
                name_display,
                label["type"],
                str(label["messages_total"]),
                str(label["messages_unread"]),
                str(label["level"])
            )
        
        self.console.print(table)
        
        # Summary
        summary = f"[bold]Summary:[/bold] {results['user_labels']} user labels, {results['system_labels']} system labels"
        if results["nested_labels"] > 0:
            summary += f", {results['nested_labels']} nested"
        self.console.print(summary)
    
    def _format_analyze_output(self, results: Dict[str, Any]) -> None:
        """Format label analysis output."""
        self.console.print(Panel(
            f"[bold]Label Analysis Results[/bold]\n\n"
            f"Total Labels: {results['total_labels']}\n"
            f"Empty Labels: {len(results['empty_labels'])}\n"
            f"Low Usage Labels: {len(results['low_usage_labels'])}\n"
            f"Duplicate Candidates: {len(results['duplicate_candidates'])}",
            title="📊 Analysis Summary",
            border_style="blue"
        ))
        
        # Show recommendations
        if results["recommendations"]:
            self.console.print("\n[bold yellow]🔍 Recommendations:[/bold yellow]")
            for i, rec in enumerate(results["recommendations"], 1):
                priority_color = {"high": "red", "medium": "yellow", "low": "green"}.get(rec["priority"], "white")
                self.console.print(f"{i}. [{priority_color}]{rec['description']}[/{priority_color}]")
                
                if "labels" in rec:
                    for label in rec["labels"]:
                        self.console.print(f"   • {label}")
                elif "pairs" in rec:
                    for pair in rec["pairs"]:
                        self.console.print(f"   • {pair['label1']} ↔ {pair['label2']}")
        
        # Show empty labels
        if results["empty_labels"]:
            self.console.print("\n[bold red]🗑️  Empty Labels:[/bold red]")
            for label in results["empty_labels"][:10]:  # Show first 10
                self.console.print(f"  • {label['name']}")
    
    def _format_create_output(self, results: Dict[str, Any]) -> None:
        """Format label creation output."""
        if results["success"]:
            self.console.print(f"[green]✅ Successfully created label: {results['label']['name']}[/green]")
        else:
            self.console.print(f"[red]❌ {results['error']}[/red]")
    
    def _format_delete_output(self, results: Dict[str, Any]) -> None:
        """Format label deletion output."""
        if results["success"]:
            self.console.print(f"[green]✅ Successfully deleted label: {results['deleted_label']}[/green]")
            if results.get("had_emails"):
                self.console.print("[yellow]⚠️  Warning: Label contained emails that were untagged[/yellow]")
        else:
            self.console.print(f"[red]❌ {results['error']}[/red]")
    
    def _format_rename_output(self, results: Dict[str, Any]) -> None:
        """Format label rename output."""
        if results["success"]:
            self.console.print(f"[green]✅ Successfully renamed '{results['old_name']}' to '{results['new_name']}'[/green]")
        else:
            self.console.print(f"[red]❌ {results['error']}[/red]")


@app.command()
def list(
    usage: bool = typer.Option(False, "--usage", "-u", help="Show usage statistics"),
    nested: bool = typer.Option(True, "--nested/--flat", help="Show nested structure")
):
    """List all Gmail labels with optional statistics."""
    
    @run_async_command
    async def run():
        manager = LabelManager()
        await manager.initialize()
        
        results = await manager.list_labels(show_usage=usage, show_nested=nested)
        manager.format_output(results, "list")
        return manager
    
    run()


@app.command()
def analyze(
    unused_days: int = typer.Option(30, "--unused-days", help="Days threshold for unused labels")
):
    """Analyze label usage and provide cleanup recommendations."""
    
    @run_async_command
    async def run():
        manager = LabelManager()
        await manager.initialize()
        
        results = await manager.analyze_labels(unused_threshold_days=unused_days)
        manager.format_output(results, "analyze")
        return manager
    
    run()


@app.command()
def create(
    name: str = typer.Argument(..., help="Label name (use / for nesting, e.g., 'Projects/AI')"),
    color: Optional[str] = typer.Option(None, "--color", help="Label color (optional)")
):
    """Create a new Gmail label."""
    
    @run_async_command
    async def run():
        manager = LabelManager()
        await manager.initialize()
        
        results = await manager.create_label(name, color)
        manager.format_output(results, "create")
        return manager
    
    run()


@app.command()
def delete(
    name: str = typer.Argument(..., help="Label name to delete"),
    force: bool = typer.Option(False, "--force", "-f", help="Delete even if label contains emails")
):
    """Delete a Gmail label."""
    
    @run_async_command
    async def run():
        manager = LabelManager()
        await manager.initialize()
        
        results = await manager.delete_label(name, force)
        manager.format_output(results, "delete")
        return manager
    
    run()


@app.command()
def rename(
    old_name: str = typer.Argument(..., help="Current label name"),
    new_name: str = typer.Argument(..., help="New label name")
):
    """Rename a Gmail label."""
    
    @run_async_command
    async def run():
        manager = LabelManager()
        await manager.initialize()
        
        results = await manager.rename_label(old_name, new_name)
        manager.format_output(results, "rename")
        return manager
    
    run()


@app.command()
def cleanup(
    dry_run: bool = typer.Option(True, "--dry-run/--apply", help="Preview cleanup actions without applying"),
    empty_labels: bool = typer.Option(True, "--empty-labels/--skip-empty", help="Include empty labels in cleanup"),
    unused_days: int = typer.Option(30, "--unused-days", help="Days threshold for unused labels")
):
    """Clean up unused and empty labels."""
    
    @run_async_command
    async def run():
        manager = LabelManager()
        await manager.initialize()
        
        # First analyze
        analysis = await manager.analyze_labels(unused_threshold_days=unused_days)
        
        cleanup_actions = []
        
        if empty_labels and analysis["empty_labels"]:
            cleanup_actions.extend([
                {"action": "delete", "label": label["name"], "reason": "empty"}
                for label in analysis["empty_labels"]
            ])
        
        if not cleanup_actions:
            manager.console.print("[green]✅ No cleanup actions needed![/green]")
            return manager
        
        # Display proposed actions
        manager.console.print(f"\n[bold yellow]🧹 Cleanup Plan ({len(cleanup_actions)} actions):[/bold yellow]")
        for action in cleanup_actions:
            manager.console.print(f"  • Delete '{action['label']}' (reason: {action['reason']})")
        
        if dry_run:
            manager.console.print("\n[yellow]DRY RUN MODE - No changes were made[/yellow]")
            manager.console.print("Run with --apply to execute cleanup actions")
        else:
            manager.console.print(f"\n[red]Executing {len(cleanup_actions)} cleanup actions...[/red]")
            
            for action in cleanup_actions:
                result = await manager.delete_label(action["label"], force=True)
                if result["success"]:
                    manager.console.print(f"  ✅ Deleted {action['label']}")
                else:
                    manager.console.print(f"  ❌ Failed to delete {action['label']}: {result['error']}")
        
        return manager
    
    run()


if __name__ == "__main__":
    app()