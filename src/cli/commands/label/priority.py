"""Priority-based email labeling commands."""

import asyncio
from typing import Optional, Dict, Any, List
import typer
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from src.cli.base import BaseEmailProcessor, run_async_command
from src.services.email_prioritizer import EmailPrioritizer
from src.models.email import EmailCategory

app = typer.Typer(help="Priority-based email labeling commands")


class PriorityLabelCommand(BaseEmailProcessor):
    """Priority labeling command implementation."""
    
    def __init__(self):
        super().__init__()
        self.prioritizer = None
        self.priority_labels = {
            "critical": "Priority/Critical",
            "high": "Priority/High", 
            "medium": "Priority/Medium",
            "low": "Priority/Low"
        }
    
    async def initialize(self):
        """Initialize priority labeling services."""
        await super().initialize()
        self.prioritizer = EmailPrioritizer()
        await self.prioritizer.initialize()
        
        # Ensure priority labels exist
        await self._ensure_priority_labels_exist()
    
    async def _ensure_priority_labels_exist(self) -> None:
        """Ensure all priority labels exist in Gmail with proper nesting."""
        self.console.print("[blue]Checking priority labels...[/blue]")
        
        try:
            # Get existing labels
            existing_labels = await self.email_service.get_labels()
            existing_label_names = {label.name for label in existing_labels}
            
            # First ensure the parent "Priority" label exists
            parent_label = "Priority"
            if parent_label not in existing_label_names:
                self.console.print(f"   Creating parent label: {parent_label}")
                await self.email_service.create_label(name=parent_label)
                existing_label_names.add(parent_label)
            else:
                self.console.print(f"   ✓ Parent label exists: {parent_label}")
            
            # Create missing priority sublabels
            for priority_level, label_name in self.priority_labels.items():
                if label_name not in existing_label_names:
                    self.console.print(f"   Creating nested label: {label_name}")
                    await self.email_service.create_label(name=label_name)
                else:
                    self.console.print(f"   ✓ Nested label exists: {label_name}")
                    
        except Exception as e:
            self.console.print(f"[red]Failed to ensure priority labels exist: {e}[/red]")
            raise
    
    async def execute(
        self,
        target: str = "unread",
        confidence_threshold: float = 0.7,
        dry_run: bool = True,
        limit: Optional[int] = None,
        detailed: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute priority labeling."""
        
        # Convert target to Gmail query
        query = self._parse_target(target)
        
        # Process emails
        emails = await self.process_emails(query, limit, dry_run)
        
        if not emails:
            return {"processed": 0, "results": [], "dry_run": dry_run}
        
        self.console.print(f"[blue]Processing {len(emails)} emails for priority classification...[/blue]")
        if dry_run:
            self.console.print("[yellow]DRY RUN MODE - No labels will be applied[/yellow]")
        
        results = []
        priority_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0, "error": 0}
        
        # Process emails with progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console
        ) as progress:
            task = progress.add_task("Processing emails...", total=len(emails))
            
            for i, email in enumerate(emails, 1):
                try:
                    # Analyze email priority
                    priority_score = await self.prioritizer.analyze_priority(email)
                    
                    # Get the corresponding label name
                    label_name = self.priority_labels.get(priority_score.level, "Priority/Medium")
                    
                    result = {
                        "email_id": email.id,
                        "subject": email.subject,
                        "sender": email.sender,
                        "priority_level": priority_score.level,
                        "confidence": priority_score.confidence,
                        "reasoning": priority_score.reasoning,
                        "is_genuine_urgency": priority_score.is_genuine_urgency,
                        "sender_reputation": priority_score.sender_reputation,
                        "needs_review": priority_score.needs_review,
                        "detected_tactics": priority_score.detected_tactics
                    }
                    
                    if priority_score.confidence >= confidence_threshold:
                        if not dry_run:
                            # Apply the label to the email
                            category = EmailCategory(
                                email_id=email.id,
                                suggested_labels=[label_name],
                                confidence_scores={label_name: priority_score.confidence},
                                reasoning=priority_score.reasoning
                            )
                            await self.email_service.apply_category(email.id, category, create_labels=True)
                            result["label_applied"] = label_name
                            self.console.print(f"   🏷️  Applied {label_name} to: {email.subject[:40]}...")
                        else:
                            result["label_would_apply"] = label_name
                            self.console.print(f"   🔍 Would apply {label_name} (confidence: {priority_score.confidence:.1%})")
                    else:
                        result["label_skipped"] = f"Confidence {priority_score.confidence:.1%} below threshold {confidence_threshold:.1%}"
                        if not dry_run:
                            self.console.print(f"   ⚠️  Skipped {email.subject[:40]}... (low confidence)")
                    
                    results.append(result)
                    priority_counts[priority_score.level] += 1
                    
                    # Small delay to avoid overwhelming the API
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    self.console.print(f"   ❌ Failed to process {email.subject[:40]}...: {str(e)}")
                    priority_counts["error"] += 1
                    results.append({
                        "email_id": email.id,
                        "subject": email.subject,
                        "priority_level": "error",
                        "confidence": 0.0,
                        "reasoning": f"Error: {str(e)}",
                        "error": str(e)
                    })
                
                progress.advance(task)
        
        # Print summary
        self.console.print(f"\n[green]🎉 Processing complete![/green]")
        self.console.print(f"   📊 Priority Distribution:")
        for level, count in priority_counts.items():
            if count > 0:
                self.console.print(f"      {level.title()}: {count}")
        
        # Show detailed results if requested
        if detailed:
            self._print_detailed_results(results)
        
        return {
            "processed": len(results),
            "priority_counts": priority_counts,
            "results": results,
            "dry_run": dry_run,
            "successful": len([r for r in results if r.get("priority_level") != "error"]),
            "failed": priority_counts["error"]
        }
    
    def _print_detailed_results(self, results: List[Dict[str, Any]]) -> None:
        """Print detailed results with priority analysis."""
        self.console.print(f"\n📋 Detailed Results:")
        self.console.print("=" * 80)
        
        for result in results:
            self.console.print(f"\nSubject: {result.get('subject', 'Unknown')[:60]}...")
            self.console.print(f"Priority: {result.get('priority_level', 'unknown').title()}")
            self.console.print(f"Confidence: {result.get('confidence', 0):.1%}")
            
            label_text = result.get('label_applied') or result.get('label_would_apply', 'None')
            self.console.print(f"Label: {label_text}")
            
            if result.get('reasoning'):
                self.console.print(f"Reasoning: {result.get('reasoning')}")
            
            if result.get('detected_tactics'):
                self.console.print(f"Marketing Tactics: {', '.join(result.get('detected_tactics', []))}")
            
            if result.get('needs_review'):
                self.console.print("⚠️  Needs Review: Low confidence classification")
            
            if not result.get('is_genuine_urgency', True):
                self.console.print("🚨 Marketing Manipulation Detected")
            
            self.console.print("-" * 80)
    
    def format_output(self, results: Dict[str, Any]) -> None:
        """Format and display results."""
        columns = {
            "subject": "Subject",
            "sender": "Sender", 
            "priority_level": "Priority",
            "confidence": "Confidence",
            "labels": "Labels"
        }
        self.display_results_table(results, columns)


@app.command()
def unread(
    confidence_threshold: float = typer.Option(0.7, "--confidence-threshold", "-c", help="Minimum confidence for labeling"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview labels without applying them (default: apply labels)"),
    limit: Optional[int] = typer.Option(50, "--limit", "-l", help="Maximum number of emails to process (default: 50)"),
    detailed: bool = typer.Option(False, "--detailed", help="Show detailed analysis results")
):
    """Label unread emails with priority levels."""
    
    @run_async_command
    async def run():
        command = PriorityLabelCommand()
        await command.initialize()
        
        results = await command.execute(
            target="unread",
            confidence_threshold=confidence_threshold,
            dry_run=dry_run,
            limit=limit,
            detailed=detailed
        )
        
        command.format_output(results)
        return command
    
    run()


@app.command()
def recent(
    days: int = typer.Argument(7, help="Number of days to look back"),
    confidence_threshold: float = typer.Option(0.7, "--confidence-threshold", "-c", help="Minimum confidence for labeling"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview labels without applying them (default: apply labels)"),
    limit: Optional[int] = typer.Option(50, "--limit", "-l", help="Maximum number of emails to process (default: 50)"),
    detailed: bool = typer.Option(False, "--detailed", help="Show detailed analysis results")
):
    """Label recent emails with priority levels."""
    
    @run_async_command
    async def run():
        command = PriorityLabelCommand()
        await command.initialize()
        
        results = await command.execute(
            target=f"recent:{days}days",
            confidence_threshold=confidence_threshold,
            dry_run=dry_run,
            limit=limit,
            detailed=detailed
        )
        
        command.format_output(results)
        return command
    
    run()


@app.command() 
def query(
    search_query: str = typer.Argument(..., help="Gmail search query"),
    confidence_threshold: float = typer.Option(0.7, "--confidence-threshold", "-c", help="Minimum confidence for labeling"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview labels without applying them (default: apply labels)"),
    limit: Optional[int] = typer.Option(50, "--limit", "-l", help="Maximum number of emails to process (default: 50)"),
    detailed: bool = typer.Option(False, "--detailed", help="Show detailed analysis results")
):
    """Label emails matching custom query with priority levels."""
    
    @run_async_command
    async def run():
        command = PriorityLabelCommand()
        await command.initialize()
        
        results = await command.execute(
            target=f"query:{search_query}",
            confidence_threshold=confidence_threshold,
            dry_run=dry_run,
            limit=limit,
            detailed=detailed
        )
        
        command.format_output(results)
        return command
    
    run()


if __name__ == "__main__":
    app()