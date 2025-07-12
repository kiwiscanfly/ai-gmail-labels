"""Combined email classification commands - applies all classifiers."""

import asyncio
from typing import Optional, Dict, Any, List
import typer
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from src.cli.base import BaseEmailProcessor, run_async_command
from src.services.email_prioritizer import EmailPrioritizer
from src.services.marketing_classifier import MarketingEmailClassifier
from src.services.receipt_classifier import ReceiptClassifier
from src.services.notifications_classifier import NotificationsClassifier
from src.services.custom_classifier import CustomClassifier
from src.models.email import EmailCategory

app = typer.Typer(help="Combined email classification commands")


class AllLabelCommand(BaseEmailProcessor):
    """Combined labeling command that applies all classifiers."""
    
    def __init__(self):
        super().__init__()
        self.prioritizer = None
        self.marketing_classifier = None
        self.receipt_classifier = None
        self.notifications_classifier = None
        self.custom_classifier = None
        
        # All label mappings
        self.priority_labels = {
            "critical": "Priority/Critical",
            "high": "Priority/High", 
            "medium": "Priority/Medium",
            "low": "Priority/Low"
        }
        
        self.marketing_labels = {
            "promotional": "Marketing/Promotional",
            "newsletter": "Marketing/Newsletter", 
            "hybrid": "Marketing/Hybrid",
            "general": "Marketing/General"
        }
        
        self.receipt_labels = {
            "purchase": "Receipts/Purchase",
            "service": "Receipts/Service",
            "other": "Receipts/Other"
        }
        
        self.notification_labels = {
            "system": "Notifications/System",
            "update": "Notifications/Update", 
            "alert": "Notifications/Alert",
            "reminder": "Notifications/Reminder",
            "security": "Notifications/Security"
        }
        
        # Non-marketing emails won't get marketing labels
        self.skip_marketing_subtypes = ["personal", "transactional"]
    
    async def initialize(self):
        """Initialize all classification services."""
        await super().initialize()
        
        # Initialize all classifiers
        self.prioritizer = EmailPrioritizer()
        await self.prioritizer.initialize()
        
        self.marketing_classifier = MarketingEmailClassifier()
        await self.marketing_classifier.initialize()
        
        self.receipt_classifier = ReceiptClassifier()
        await self.receipt_classifier.initialize()
        
        self.notifications_classifier = NotificationsClassifier()
        await self.notifications_classifier.initialize()
        
        self.custom_classifier = CustomClassifier()
        await self.custom_classifier.initialize()
        
        # Ensure all labels exist
        await self._ensure_all_labels_exist()
    
    async def _ensure_all_labels_exist(self) -> None:
        """Ensure all classification labels exist in Gmail with proper nesting."""
        self.console.print("[blue]Checking all classification labels...[/blue]")
        
        try:
            # Get existing labels
            existing_labels = await self.email_service.get_labels()
            existing_label_names = {label.name for label in existing_labels}
            
            # Parent labels to create
            parent_labels = ["Priority", "Marketing", "Receipts", "Notifications"]
            
            # Create parent labels if needed
            for parent_label in parent_labels:
                if parent_label not in existing_label_names:
                    self.console.print(f"   Creating parent label: {parent_label}")
                    await self.email_service.create_label(name=parent_label)
                    existing_label_names.add(parent_label)
                else:
                    self.console.print(f"   ✓ Parent label exists: {parent_label}")
            
            # Create all sublabels
            all_labels = {**self.priority_labels, **self.marketing_labels, **self.receipt_labels, **self.notification_labels}
            for subtype, label_name in all_labels.items():
                if label_name not in existing_label_names:
                    self.console.print(f"   Creating nested label: {label_name}")
                    await self.email_service.create_label(name=label_name)
                else:
                    self.console.print(f"   ✓ Nested label exists: {label_name}")
                    
        except Exception as e:
            self.console.print(f"[red]Failed to ensure labels exist: {e}[/red]")
            raise
    
    async def classify_email(self, email) -> Dict[str, Any]:
        """Classify a single email using all classifiers."""
        classifications = {}
        labels_to_apply = []
        
        try:
            # Priority classification
            if self.prioritizer:
                priority_result = await self.prioritizer.analyze_priority(email)
                classifications["priority"] = {
                    "level": priority_result.level,
                    "confidence": priority_result.confidence,
                    "reasoning": priority_result.reasoning,
                    "is_genuine_urgency": priority_result.is_genuine_urgency,
                    "sender_reputation": priority_result.sender_reputation,
                    "needs_review": priority_result.needs_review,
                    "detected_tactics": priority_result.detected_tactics
                }
                
                # Add priority label
                priority_label = self.priority_labels.get(priority_result.level, "Priority/Medium")
                labels_to_apply.append(priority_label)
            
            # Marketing classification
            if self.marketing_classifier:
                marketing_result = await self.marketing_classifier.classify_email(email)
                classifications["marketing"] = {
                    "is_marketing": marketing_result.is_marketing,
                    "subtype": marketing_result.subtype,
                    "confidence": marketing_result.confidence,
                    "reasoning": marketing_result.reasoning,
                    "marketing_indicators": marketing_result.marketing_indicators,
                    "unsubscribe_detected": marketing_result.unsubscribe_detected,
                    "bulk_sending_indicators": marketing_result.bulk_sending_indicators,
                    "sender_reputation": marketing_result.sender_reputation
                }
                
                # Add marketing label if it's actually marketing and not excluded
                if (marketing_result.is_marketing and 
                    marketing_result.subtype not in self.skip_marketing_subtypes):
                    marketing_label = self.marketing_labels.get(marketing_result.subtype, "Marketing/General")
                    labels_to_apply.append(marketing_label)
            
            # Receipt classification
            if self.receipt_classifier:
                receipt_result = await self.receipt_classifier.classify_receipt(email)
                classifications["receipt"] = {
                    "is_receipt": receipt_result.is_receipt,
                    "receipt_type": receipt_result.receipt_type,
                    "confidence": receipt_result.confidence,
                    "reasoning": receipt_result.reasoning,
                    "vendor": receipt_result.vendor,
                    "amount": receipt_result.amount,
                    "currency": receipt_result.currency,
                    "order_number": receipt_result.order_number,
                    "transaction_date": receipt_result.transaction_date,
                    "payment_method": receipt_result.payment_method,
                    "receipt_indicators": receipt_result.receipt_indicators
                }
                
                # Add receipt label if it's actually a receipt
                if receipt_result.is_receipt:
                    receipt_label = self.receipt_labels.get(receipt_result.receipt_type, "Receipts/Other")
                    labels_to_apply.append(receipt_label)
            
            # Notifications classification
            if self.notifications_classifier:
                notification_result = await self.notifications_classifier.classify_notification(email)
                classifications["notifications"] = {
                    "is_notification": notification_result.is_notification,
                    "notification_type": notification_result.notification_type,
                    "confidence": notification_result.confidence,
                    "reasoning": notification_result.reasoning,
                    "sender_type": notification_result.sender_type,
                    "urgency_level": notification_result.urgency_level,
                    "action_required": notification_result.action_required,
                    "notification_indicators": notification_result.notification_indicators
                }
                
                # Add notification label if it's actually a notification
                if notification_result.is_notification:
                    notification_label = self.notification_labels.get(notification_result.notification_type, "Notifications/Alert")
                    labels_to_apply.append(notification_label)
            
            # Custom classification (Note: Custom categories are handled separately via custom command)
            # This is included for completeness but custom labels are typically applied independently
            if self.custom_classifier:
                # Get available custom categories
                custom_categories = await self.custom_classifier.get_categories()
                custom_results = {}
                
                # Apply each custom category
                for category in custom_categories[:3]:  # Limit to first 3 categories for combined processing
                    try:
                        custom_result = await self.custom_classifier.classify_email(
                            email, category.name, category.search_terms, category.confidence_threshold
                        )
                        custom_results[category.name] = {
                            "is_match": custom_result.is_match,
                            "confidence": custom_result.confidence,
                            "reasoning": custom_result.reasoning,
                            "suggested_label": custom_result.suggested_label
                        }
                        
                        # Add custom label if it matches
                        if custom_result.is_match and custom_result.confidence >= category.confidence_threshold:
                            custom_label = custom_result.suggested_label or category.name
                            labels_to_apply.append(custom_label)
                            
                    except Exception as e:
                        custom_results[category.name] = {"error": str(e)}
                
                if custom_results:
                    classifications["custom"] = custom_results
            
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
    
    async def execute(
        self,
        target: str = "unread",
        priority_confidence: float = 0.7,
        marketing_confidence: float = 0.7,
        receipt_confidence: float = 0.7,
        notifications_confidence: float = 0.7,
        custom_confidence: float = 0.7,
        dry_run: bool = True,
        limit: Optional[int] = None,
        exclude_personal: bool = True,
        detailed: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute combined classification."""
        
        # Convert target to Gmail query
        query = self._parse_target(target)
        
        # Process emails
        emails = await self.process_emails(query, limit, dry_run)
        
        if not emails:
            return {"processed": 0, "results": [], "dry_run": dry_run}
        
        self.console.print(f"[blue]Processing {len(emails)} emails with all classifiers...[/blue]")
        if dry_run:
            self.console.print("[yellow]DRY RUN MODE - No labels will be applied[/yellow]")
        
        results = []
        counts = {
            "priority": {"critical": 0, "high": 0, "medium": 0, "low": 0},
            "marketing": {"promotional": 0, "newsletter": 0, "hybrid": 0, "transactional": 0, "personal": 0, "general": 0},
            "receipt": {"purchase": 0, "service": 0, "other": 0, "non_receipt": 0},
            "notifications": {"system": 0, "update": 0, "alert": 0, "reminder": 0, "security": 0, "non_notification": 0},
            "errors": 0
        }
        
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
                    # Classify email with all classifiers
                    result = await self.classify_email(email)
                    
                    if not result.get("processed"):
                        results.append(result)
                        counts["errors"] += 1
                        continue
                    
                    # Filter labels based on confidence thresholds
                    labels_to_apply = []
                    classifications = result.get("classifications", {})
                    
                    # Priority label (always apply if meets threshold)
                    priority_info = classifications.get("priority", {})
                    if priority_info.get("confidence", 0) >= priority_confidence:
                        priority_label = self.priority_labels.get(priority_info.get("level"), "Priority/Medium")
                        labels_to_apply.append(priority_label)
                    
                    # Marketing label (only if marketing and meets criteria)
                    marketing_info = classifications.get("marketing", {})
                    if (marketing_info.get("is_marketing") and 
                        marketing_info.get("subtype") not in self.skip_marketing_subtypes and
                        marketing_info.get("confidence", 0) >= marketing_confidence):
                        if not exclude_personal or marketing_info.get("subtype") != "personal":
                            marketing_label = self.marketing_labels.get(marketing_info.get("subtype"), "Marketing/General")
                            labels_to_apply.append(marketing_label)
                    
                    # Receipt label (only if receipt and meets threshold)
                    receipt_info = classifications.get("receipt", {})
                    if (receipt_info.get("is_receipt") and 
                        receipt_info.get("confidence", 0) >= receipt_confidence):
                        receipt_label = self.receipt_labels.get(receipt_info.get("receipt_type"), "Receipts/Other")
                        labels_to_apply.append(receipt_label)
                    
                    # Notifications label (only if notification and meets threshold)
                    notifications_info = classifications.get("notifications", {})
                    if (notifications_info.get("is_notification") and 
                        notifications_info.get("confidence", 0) >= notifications_confidence):
                        notification_label = self.notification_labels.get(notifications_info.get("notification_type"), "Notifications/Alert")
                        labels_to_apply.append(notification_label)
                    
                    # Custom labels (only if matches and meets threshold)
                    custom_info = classifications.get("custom", {})
                    for category_name, custom_result in custom_info.items():
                        if (custom_result.get("is_match") and 
                            custom_result.get("confidence", 0) >= custom_confidence):
                            custom_label = custom_result.get("suggested_label") or category_name
                            labels_to_apply.append(custom_label)
                    
                    # Apply labels if not dry run
                    if labels_to_apply and not dry_run:
                        category = EmailCategory(
                            email_id=email.id,
                            suggested_labels=labels_to_apply,
                            confidence_scores={label: 0.8 for label in labels_to_apply},  # Average confidence
                            reasoning="Combined classification result"
                        )
                        await self.email_service.apply_category(email.id, category, create_labels=True)
                        result["labels_applied"] = labels_to_apply
                        self.console.print(f"   🏷️  Applied {len(labels_to_apply)} labels to: {email.subject[:40]}...")
                    elif labels_to_apply:
                        result["labels_would_apply"] = labels_to_apply
                        self.console.print(f"   🔍 Would apply {len(labels_to_apply)} labels: {', '.join(labels_to_apply)}")
                    else:
                        result["labels_skipped"] = "No labels met confidence thresholds"
                        self.console.print(f"   ⚠️  No labels applied to: {email.subject[:40]}... (low confidence)")
                    
                    result["final_labels"] = labels_to_apply
                    results.append(result)
                    
                    # Update counts
                    if priority_info:
                        counts["priority"][priority_info.get("level", "medium")] += 1
                    if marketing_info:
                        counts["marketing"][marketing_info.get("subtype", "general")] += 1
                    if receipt_info:
                        if receipt_info.get("is_receipt"):
                            counts["receipt"][receipt_info.get("receipt_type", "other")] += 1
                        else:
                            counts["receipt"]["non_receipt"] += 1
                    if notifications_info:
                        if notifications_info.get("is_notification"):
                            counts["notifications"][notifications_info.get("notification_type", "alert")] += 1
                        else:
                            counts["notifications"]["non_notification"] += 1
                    
                    # Small delay to avoid overwhelming the API
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    self.console.print(f"   ❌ Failed to process {email.subject[:40]}...: {str(e)}")
                    counts["errors"] += 1
                    results.append({
                        "email_id": email.id,
                        "subject": email.subject,
                        "error": str(e),
                        "processed": False
                    })
                
                progress.advance(task)
        
        # Print summary
        self.console.print(f"\n[green]🎉 Combined processing complete![/green]")
        self.console.print(f"   📊 Classification Summary:")
        
        self.console.print(f"      Priority Distribution:")
        for level, count in counts["priority"].items():
            if count > 0:
                self.console.print(f"        {level.title()}: {count}")
        
        self.console.print(f"      Marketing Distribution:")
        for subtype, count in counts["marketing"].items():
            if count > 0:
                self.console.print(f"        {subtype.title()}: {count}")
        
        self.console.print(f"      Receipt Distribution:")
        for receipt_type, count in counts["receipt"].items():
            if count > 0:
                self.console.print(f"        {receipt_type.replace('_', ' ').title()}: {count}")
        
        self.console.print(f"      Notifications Distribution:")
        for notification_type, count in counts["notifications"].items():
            if count > 0:
                self.console.print(f"        {notification_type.replace('_', ' ').title()}: {count}")
        
        if counts["errors"] > 0:
            self.console.print(f"      Errors: {counts['errors']}")
        
        # Show detailed results if requested
        if detailed:
            self._print_detailed_results(results)
        
        return {
            "processed": len(results),
            "counts": counts,
            "results": results,
            "dry_run": dry_run,
            "successful": len([r for r in results if r.get("processed")]),
            "failed": counts["errors"]
        }
    
    def _print_detailed_results(self, results: List[Dict[str, Any]]) -> None:
        """Print detailed results with combined analysis."""
        self.console.print(f"\n📋 Detailed Combined Analysis:")
        self.console.print("=" * 80)
        
        for result in results:
            if not result.get("processed"):
                continue
                
            self.console.print(f"\nSubject: {result.get('subject', 'Unknown')[:60]}...")
            
            classifications = result.get('classifications', {})
            
            # Priority details
            priority_info = classifications.get('priority', {})
            if priority_info:
                self.console.print(f"Priority: {priority_info.get('level', 'unknown').title()} (confidence: {priority_info.get('confidence', 0):.1%})")
                if priority_info.get('reasoning'):
                    self.console.print(f"  Reasoning: {priority_info.get('reasoning')}")
            
            # Marketing details
            marketing_info = classifications.get('marketing', {})
            if marketing_info:
                marketing_status = "Yes" if marketing_info.get('is_marketing') else "No"
                self.console.print(f"Marketing: {marketing_status} - {marketing_info.get('subtype', 'unknown').title()} (confidence: {marketing_info.get('confidence', 0):.1%})")
                if marketing_info.get('marketing_indicators'):
                    self.console.print(f"  Indicators: {', '.join(marketing_info.get('marketing_indicators', []))}")
            
            # Receipt details
            receipt_info = classifications.get('receipt', {})
            if receipt_info:
                receipt_status = "Yes" if receipt_info.get('is_receipt') else "No"
                self.console.print(f"Receipt: {receipt_status} - {receipt_info.get('receipt_type', 'unknown').title()} (confidence: {receipt_info.get('confidence', 0):.1%})")
                if receipt_info.get('vendor'):
                    self.console.print(f"  Vendor: {receipt_info.get('vendor')}")
                if receipt_info.get('amount'):
                    amount_str = receipt_info.get('amount')
                    if receipt_info.get('currency'):
                        amount_str = f"{receipt_info.get('currency')} {amount_str}"
                    self.console.print(f"  Amount: {amount_str}")
            
            # Notifications details
            notifications_info = classifications.get('notifications', {})
            if notifications_info:
                notification_status = "Yes" if notifications_info.get('is_notification') else "No"
                self.console.print(f"Notification: {notification_status} - {notifications_info.get('notification_type', 'unknown').title()} (confidence: {notifications_info.get('confidence', 0):.1%})")
                if notifications_info.get('sender_type'):
                    self.console.print(f"  Sender Type: {notifications_info.get('sender_type').title()}")
                if notifications_info.get('urgency_level'):
                    self.console.print(f"  Urgency: {notifications_info.get('urgency_level').title()}")
                if notifications_info.get('action_required'):
                    self.console.print("  ⚡ Action Required")
            
            # Final labels
            final_labels = result.get('final_labels', [])
            labels_text = ', '.join(final_labels) if final_labels else 'None'
            self.console.print(f"Final Labels: {labels_text}")
            
            self.console.print("-" * 80)
    
    def format_output(self, results: Dict[str, Any]) -> None:
        """Format and display results."""
        columns = {
            "subject": "Subject",
            "sender": "Sender",
            "final_labels": "Applied Labels"
        }
        self.display_results_table(results, columns)


@app.command()
def unread(
    priority_confidence: float = typer.Option(0.7, "--priority-confidence", help="Minimum confidence for priority labeling"),
    marketing_confidence: float = typer.Option(0.7, "--marketing-confidence", help="Minimum confidence for marketing labeling"),
    receipt_confidence: float = typer.Option(0.7, "--receipt-confidence", help="Minimum confidence for receipt labeling"),
    notifications_confidence: float = typer.Option(0.7, "--notifications-confidence", help="Minimum confidence for notifications labeling"),
    custom_confidence: float = typer.Option(0.7, "--custom-confidence", help="Minimum confidence for custom category labeling"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview labels without applying them (default: apply labels)"),
    limit: Optional[int] = typer.Option(50, "--limit", "-l", help="Maximum number of emails to process (default: 50)"),
    exclude_personal: bool = typer.Option(True, "--exclude-personal/--include-personal", help="Exclude personal emails from marketing labeling"),
    detailed: bool = typer.Option(False, "--detailed", help="Show detailed analysis results")
):
    """Label unread emails with all classifiers (priority + marketing + receipt + notifications)."""
    
    @run_async_command
    async def run():
        command = AllLabelCommand()
        await command.initialize()
        
        results = await command.execute(
            target="unread",
            priority_confidence=priority_confidence,
            marketing_confidence=marketing_confidence,
            receipt_confidence=receipt_confidence,
            notifications_confidence=notifications_confidence,
            custom_confidence=custom_confidence,
            dry_run=dry_run,
            limit=limit,
            exclude_personal=exclude_personal,
            detailed=detailed
        )
        
        command.format_output(results)
        return command
    
    run()


@app.command()
def recent(
    days: int = typer.Argument(7, help="Number of days to look back"),
    priority_confidence: float = typer.Option(0.7, "--priority-confidence", help="Minimum confidence for priority labeling"),
    marketing_confidence: float = typer.Option(0.7, "--marketing-confidence", help="Minimum confidence for marketing labeling"),
    receipt_confidence: float = typer.Option(0.7, "--receipt-confidence", help="Minimum confidence for receipt labeling"),
    notifications_confidence: float = typer.Option(0.7, "--notifications-confidence", help="Minimum confidence for notifications labeling"),
    custom_confidence: float = typer.Option(0.7, "--custom-confidence", help="Minimum confidence for custom category labeling"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview labels without applying them (default: apply labels)"),
    limit: Optional[int] = typer.Option(50, "--limit", "-l", help="Maximum number of emails to process (default: 50)"),
    exclude_personal: bool = typer.Option(True, "--exclude-personal/--include-personal", help="Exclude personal emails from marketing labeling"),
    detailed: bool = typer.Option(False, "--detailed", help="Show detailed analysis results")
):
    """Label recent emails with all classifiers (priority + marketing + receipt + notifications)."""
    
    @run_async_command
    async def run():
        command = AllLabelCommand()
        await command.initialize()
        
        results = await command.execute(
            target=f"recent:{days}days",
            priority_confidence=priority_confidence,
            marketing_confidence=marketing_confidence,
            receipt_confidence=receipt_confidence,
            notifications_confidence=notifications_confidence,
            custom_confidence=custom_confidence,
            dry_run=dry_run,
            limit=limit,
            exclude_personal=exclude_personal,
            detailed=detailed
        )
        
        command.format_output(results)
        return command
    
    run()


@app.command()
def query(
    search_query: str = typer.Argument(..., help="Gmail search query"),
    priority_confidence: float = typer.Option(0.7, "--priority-confidence", help="Minimum confidence for priority labeling"),
    marketing_confidence: float = typer.Option(0.7, "--marketing-confidence", help="Minimum confidence for marketing labeling"),
    receipt_confidence: float = typer.Option(0.7, "--receipt-confidence", help="Minimum confidence for receipt labeling"),
    notifications_confidence: float = typer.Option(0.7, "--notifications-confidence", help="Minimum confidence for notifications labeling"),
    custom_confidence: float = typer.Option(0.7, "--custom-confidence", help="Minimum confidence for custom category labeling"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview labels without applying them (default: apply labels)"),
    limit: Optional[int] = typer.Option(50, "--limit", "-l", help="Maximum number of emails to process (default: 50)"),
    exclude_personal: bool = typer.Option(True, "--exclude-personal/--include-personal", help="Exclude personal emails from marketing labeling"),
    detailed: bool = typer.Option(False, "--detailed", help="Show detailed analysis results")
):
    """Label emails matching custom query with all classifiers (priority + marketing + receipt + notifications)."""
    
    @run_async_command
    async def run():
        command = AllLabelCommand()
        await command.initialize()
        
        results = await command.execute(
            target=f"query:{search_query}",
            priority_confidence=priority_confidence,
            marketing_confidence=marketing_confidence,
            receipt_confidence=receipt_confidence,
            notifications_confidence=notifications_confidence,
            custom_confidence=custom_confidence,
            dry_run=dry_run,
            limit=limit,
            exclude_personal=exclude_personal,
            detailed=detailed
        )
        
        command.format_output(results)
        return command
    
    run()


if __name__ == "__main__":
    app()