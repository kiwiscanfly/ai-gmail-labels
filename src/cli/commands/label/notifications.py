"""Notification-based email labeling commands."""

import asyncio
from typing import Optional, Dict, Any, List
import typer
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from src.cli.base import BaseEmailProcessor, run_async_command
from src.services.notifications_classifier import NotificationsClassifier
from src.models.email import EmailCategory

app = typer.Typer(help="Notification-based email labeling commands")


class NotificationsLabelCommand(BaseEmailProcessor):
    """Notifications labeling command implementation."""
    
    def __init__(self):
        super().__init__()
        self.notifications_classifier = None
        self.notification_labels = {
            "system": "Notifications/System",
            "update": "Notifications/Update", 
            "alert": "Notifications/Alert",
            "reminder": "Notifications/Reminder",
            "security": "Notifications/Security"
        }
    
    async def initialize(self):
        """Initialize notification labeling services."""
        await super().initialize()
        self.notifications_classifier = NotificationsClassifier()
        await self.notifications_classifier.initialize()
        
        # Ensure notification labels exist
        await self._ensure_notification_labels_exist()
    
    async def _ensure_notification_labels_exist(self) -> None:
        """Ensure all notification labels exist in Gmail with proper nesting."""
        self.console.print("[blue]Checking notification labels...[/blue]")
        
        try:
            # Get existing labels
            existing_labels = await self.email_service.get_labels()
            existing_label_names = {label.name for label in existing_labels}
            
            # First ensure the parent "Notifications" label exists
            parent_label = "Notifications"
            if parent_label not in existing_label_names:
                self.console.print(f"   Creating parent label: {parent_label}")
                await self.email_service.create_label(name=parent_label)
                existing_label_names.add(parent_label)
            else:
                self.console.print(f"   ✓ Parent label exists: {parent_label}")
            
            # Create missing notification sublabels
            for notification_type, label_name in self.notification_labels.items():
                if label_name not in existing_label_names:
                    self.console.print(f"   Creating nested label: {label_name}")
                    await self.email_service.create_label(name=label_name)
                else:
                    self.console.print(f"   ✓ Nested label exists: {label_name}")
                    
        except Exception as e:
            self.console.print(f"[red]Failed to ensure notification labels exist: {e}[/red]")
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
        """Execute notification labeling."""
        
        # Convert target to Gmail query
        query = self._parse_target(target)
        
        # Process emails
        emails = await self.process_emails(query, limit, dry_run)
        
        if not emails:
            return {"processed": 0, "results": [], "dry_run": dry_run}
        
        self.console.print(f"[blue]Processing {len(emails)} emails for notification classification...[/blue]")
        if dry_run:
            self.console.print("[yellow]DRY RUN MODE - No labels will be applied[/yellow]")
        
        results = []
        notification_counts = {"system": 0, "update": 0, "alert": 0, "reminder": 0, "security": 0, "non_notification": 0, "error": 0}
        
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
                    # Analyze email for notifications
                    notification_result = await self.notifications_classifier.classify_notification(email)
                    
                    result = {
                        "email_id": email.id,
                        "subject": email.subject,
                        "sender": email.sender,
                        "is_notification": notification_result.is_notification,
                        "notification_type": notification_result.notification_type,
                        "confidence": notification_result.confidence,
                        "reasoning": notification_result.reasoning,
                        "sender_type": notification_result.sender_type,
                        "urgency_level": notification_result.urgency_level,
                        "action_required": notification_result.action_required,
                        "notification_indicators": notification_result.notification_indicators
                    }
                    
                    if notification_result.is_notification and notification_result.confidence >= confidence_threshold:
                        # Get the corresponding label name
                        label_name = self.notification_labels.get(notification_result.notification_type, "Notifications/Alert")
                        
                        if not dry_run:
                            # Apply the label to the email
                            category = EmailCategory(
                                email_id=email.id,
                                suggested_labels=[label_name],
                                confidence_scores={label_name: notification_result.confidence},
                                reasoning=notification_result.reasoning
                            )
                            await self.email_service.apply_category(email.id, category, create_labels=True)
                            result["label_applied"] = label_name
                            self.console.print(f"   🔔 Applied {label_name} to: {email.subject[:40]}...")
                        else:
                            result["label_would_apply"] = label_name
                            self.console.print(f"   🔍 Would apply {label_name} (confidence: {notification_result.confidence:.1%})")
                        
                        notification_counts[notification_result.notification_type] += 1
                    elif notification_result.is_notification:
                        result["label_skipped"] = f"Confidence {notification_result.confidence:.1%} below threshold {confidence_threshold:.1%}"
                        notification_counts[notification_result.notification_type] += 1
                        if not dry_run:
                            self.console.print(f"   ⚠️  Skipped {email.subject[:40]}... (low confidence)")
                    else:
                        result["label_skipped"] = "Not classified as notification"
                        notification_counts["non_notification"] += 1
                        if not dry_run:
                            self.console.print(f"   ℹ️  Not a notification: {email.subject[:40]}...")
                    
                    results.append(result)
                    
                    # Small delay to avoid overwhelming the API
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    self.console.print(f"   ❌ Failed to process {email.subject[:40]}...: {str(e)}")
                    notification_counts["error"] += 1
                    results.append({
                        "email_id": email.id,
                        "subject": email.subject,
                        "is_notification": False,
                        "notification_type": "error",
                        "confidence": 0.0,
                        "reasoning": f"Error: {str(e)}",
                        "error": str(e)
                    })
                
                progress.advance(task)
        
        # Print summary
        self.console.print(f"\n[green]🎉 Notification processing complete![/green]")
        self.console.print(f"   📊 Notification Distribution:")
        for notification_type, count in notification_counts.items():
            if count > 0:
                self.console.print(f"      {notification_type.replace('_', ' ').title()}: {count}")
        
        # Show detailed results if requested
        if detailed:
            self._print_detailed_results(results)
        
        return {
            "processed": len(results),
            "notification_counts": notification_counts,
            "results": results,
            "dry_run": dry_run,
            "successful": len([r for r in results if r.get("notification_type") != "error"]),
            "failed": notification_counts["error"]
        }
    
    def _print_detailed_results(self, results: List[Dict[str, Any]]) -> None:
        """Print detailed results with notification analysis."""
        self.console.print(f"\n📋 Detailed Notification Results:")
        self.console.print("=" * 80)
        
        for result in results:
            self.console.print(f"\nSubject: {result.get('subject', 'Unknown')[:60]}...")
            self.console.print(f"Is Notification: {result.get('is_notification', False)}")
            
            if result.get('is_notification'):
                self.console.print(f"Type: {result.get('notification_type', 'unknown').title()}")
                self.console.print(f"Confidence: {result.get('confidence', 0):.1%}")
                
                if result.get('sender_type'):
                    self.console.print(f"Sender Type: {result.get('sender_type').title()}")
                
                if result.get('urgency_level'):
                    self.console.print(f"Urgency: {result.get('urgency_level').title()}")
                
                if result.get('action_required'):
                    self.console.print("⚡ Action Required")
            
            label_text = result.get('label_applied') or result.get('label_would_apply', 'None')
            self.console.print(f"Label: {label_text}")
            
            if result.get('reasoning'):
                self.console.print(f"Reasoning: {result.get('reasoning')}")
            
            if result.get('notification_indicators'):
                self.console.print(f"Indicators: {', '.join(result.get('notification_indicators', []))}")
            
            self.console.print("-" * 80)
    
    def format_output(self, results: Dict[str, Any]) -> None:
        """Format and display results."""
        columns = {
            "subject": "Subject",
            "sender": "Sender", 
            "notification_type": "Type",
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
    """Label unread emails with notification types."""
    
    @run_async_command
    async def run():
        command = NotificationsLabelCommand()
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
    """Label recent emails with notification types."""
    
    @run_async_command
    async def run():
        command = NotificationsLabelCommand()
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
    """Label emails matching custom query with notification types."""
    
    @run_async_command
    async def run():
        command = NotificationsLabelCommand()
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