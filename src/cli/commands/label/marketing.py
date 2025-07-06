"""Marketing email classification commands."""

import asyncio
from typing import Optional, Dict, Any, List
import typer
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from src.cli.base import BaseEmailProcessor, run_async_command
from src.services.marketing_classifier import MarketingEmailClassifier
from src.models.email import EmailCategory

app = typer.Typer(help="Marketing email classification commands")


class MarketingLabelCommand(BaseEmailProcessor):
    """Marketing labeling command implementation."""
    
    def __init__(self):
        super().__init__()
        self.marketing_classifier = None
        
        # Marketing label mapping - only for actual marketing emails
        self.marketing_labels = {
            "promotional": "Marketing/Promotional",
            "newsletter": "Marketing/Newsletter", 
            "hybrid": "Marketing/Hybrid",
            "general": "Marketing/General"
        }
        
        # Non-marketing emails won't get marketing labels
        self.skip_label_subtypes = ["personal", "transactional"]
    
    async def initialize(self):
        """Initialize marketing classification services."""
        await super().initialize()
        self.marketing_classifier = MarketingEmailClassifier()
        await self.marketing_classifier.initialize()
        
        # Ensure marketing labels exist
        await self._ensure_marketing_labels_exist()
    
    async def _ensure_marketing_labels_exist(self) -> None:
        """Ensure all marketing labels exist in Gmail with proper nesting."""
        self.console.print("[blue]Checking marketing labels...[/blue]")
        
        try:
            # Get existing labels
            existing_labels = await self.email_service.get_labels()
            existing_label_names = {label.name for label in existing_labels}
            
            # First ensure the parent "Marketing" label exists
            parent_label = "Marketing"
            if parent_label not in existing_label_names:
                self.console.print(f"   Creating parent label: {parent_label}")
                await self.email_service.create_label(name=parent_label)
                existing_label_names.add(parent_label)
            else:
                self.console.print(f"   ✓ Parent label exists: {parent_label}")
            
            # Create missing marketing sublabels (only for marketing types)
            for subtype, label_name in self.marketing_labels.items():
                if label_name not in existing_label_names:
                    self.console.print(f"   Creating nested label: {label_name}")
                    await self.email_service.create_label(name=label_name)
                else:
                    self.console.print(f"   ✓ Nested label exists: {label_name}")
                    
        except Exception as e:
            self.console.print(f"[red]Failed to ensure marketing labels exist: {e}[/red]")
            raise
    
    async def execute(
        self,
        target: str = "unread",
        confidence_threshold: float = 0.7,
        dry_run: bool = True,
        limit: Optional[int] = None,
        types: Optional[List[str]] = None,
        exclude_personal: bool = True,
        sender_analysis: bool = False,
        detailed: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute marketing classification."""
        
        # Convert target to Gmail query
        query = self._parse_target(target)
        
        # Process emails
        emails = await self.process_emails(query, limit, dry_run)
        
        if not emails:
            return {"processed": 0, "results": [], "dry_run": dry_run}
        
        self.console.print(f"[blue]Processing {len(emails)} emails for marketing classification...[/blue]")
        if dry_run:
            self.console.print("[yellow]DRY RUN MODE - No labels will be applied[/yellow]")
        
        results = []
        marketing_counts = {
            "promotional": 0, "newsletter": 0, "hybrid": 0, 
            "transactional": 0, "personal": 0, "general": 0, "error": 0
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
                    # Analyze marketing classification
                    marketing_result = await self.marketing_classifier.classify_email(email)
                    
                    result = {
                        "email_id": email.id,
                        "subject": email.subject,
                        "sender": email.sender,
                        "is_marketing": marketing_result.is_marketing,
                        "marketing_subtype": marketing_result.subtype,
                        "confidence": marketing_result.confidence,
                        "reasoning": marketing_result.reasoning,
                        "marketing_indicators": marketing_result.marketing_indicators,
                        "unsubscribe_detected": marketing_result.unsubscribe_detected,
                        "bulk_sending_indicators": marketing_result.bulk_sending_indicators,
                        "sender_reputation": marketing_result.sender_reputation
                    }
                    
                    # Check if we should apply labels
                    should_label = (
                        marketing_result.is_marketing and 
                        marketing_result.subtype not in self.skip_label_subtypes and
                        marketing_result.confidence >= confidence_threshold
                    )
                    
                    # Filter by types if specified
                    if types and marketing_result.subtype not in types:
                        should_label = False
                    
                    # Exclude personal emails if requested
                    if exclude_personal and marketing_result.subtype == "personal":
                        should_label = False
                    
                    if should_label:
                        # Get the corresponding label name
                        label_name = self.marketing_labels.get(marketing_result.subtype, "Marketing/General")
                        
                        if not dry_run:
                            # Apply the label to the email
                            category = EmailCategory(
                                email_id=email.id,
                                suggested_labels=[label_name],
                                confidence_scores={label_name: marketing_result.confidence},
                                reasoning=marketing_result.reasoning
                            )
                            await self.email_service.apply_category(email.id, category, create_labels=True)
                            result["label_applied"] = label_name
                            self.console.print(f"   🏷️  Applied {label_name} to: {email.subject[:40]}...")
                        else:
                            result["label_would_apply"] = label_name
                            self.console.print(f"   🔍 Would apply {label_name} (confidence: {marketing_result.confidence:.1%})")
                    else:
                        # Non-marketing email or filtered out
                        if not marketing_result.is_marketing:
                            result["label_skipped"] = "Non-marketing email"
                            self.console.print(f"   ✉️  Non-marketing email: {email.subject[:40]}...")
                        elif marketing_result.confidence < confidence_threshold:
                            result["label_skipped"] = f"Confidence {marketing_result.confidence:.1%} below threshold"
                            self.console.print(f"   ⚠️  Skipped {email.subject[:40]}... (low confidence)")
                        else:
                            result["label_skipped"] = f"Filtered out ({marketing_result.subtype})"
                            self.console.print(f"   🚫 Filtered out {email.subject[:40]}... ({marketing_result.subtype})")
                    
                    results.append(result)
                    marketing_counts[marketing_result.subtype] += 1
                    
                    # Show marketing indicators if found
                    if marketing_result.is_marketing and marketing_result.marketing_indicators and detailed:
                        self.console.print(f"       Marketing indicators: {', '.join(marketing_result.marketing_indicators)}")
                    
                    # Small delay to avoid overwhelming the API
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    self.console.print(f"   ❌ Failed to process {email.subject[:40]}...: {str(e)}")
                    marketing_counts["error"] += 1
                    results.append({
                        "email_id": email.id,
                        "subject": email.subject,
                        "is_marketing": False,
                        "marketing_subtype": "error",
                        "confidence": 0.0,
                        "reasoning": f"Error: {str(e)}",
                        "error": str(e)
                    })
                
                progress.advance(task)
        
        # Print summary
        self.console.print(f"\n[green]🎉 Processing complete![/green]")
        self.console.print(f"   📊 Marketing Classification Distribution:")
        for subtype, count in marketing_counts.items():
            if count > 0:
                self.console.print(f"      {subtype.title()}: {count}")
        
        # Show sender statistics if requested
        if sender_analysis:
            self._print_sender_statistics()
        
        # Show detailed results if requested
        if detailed:
            self._print_detailed_results(results)
        
        return {
            "processed": len(results),
            "marketing_counts": marketing_counts,
            "results": results,
            "dry_run": dry_run,
            "successful": len([r for r in results if r.get("marketing_subtype") != "error"]),
            "failed": marketing_counts["error"]
        }
    
    def _print_sender_statistics(self) -> None:
        """Print sender marketing statistics."""
        stats = self.marketing_classifier.get_sender_statistics()
        
        self.console.print(f"\n📈 Sender Statistics:")
        self.console.print(f"   Total Senders: {stats.get('total_senders', 0)}")
        self.console.print(f"   Marketing Senders: {stats.get('marketing_senders', 0)}")
        self.console.print(f"   Individual Senders: {stats.get('individual_senders', 0)}")
        self.console.print(f"   Average Marketing Rate: {stats.get('average_marketing_rate', 0):.1%}")
    
    def _print_detailed_results(self, results: List[Dict[str, Any]]) -> None:
        """Print detailed results with marketing analysis."""
        self.console.print(f"\n📋 Detailed Marketing Analysis:")
        self.console.print("=" * 80)
        
        for result in results:
            self.console.print(f"\nSubject: {result.get('subject', 'Unknown')[:60]}...")
            self.console.print(f"Marketing Type: {result.get('marketing_subtype', 'unknown').title()}")
            self.console.print(f"Is Marketing: {'Yes' if result.get('is_marketing') else 'No'}")
            self.console.print(f"Confidence: {result.get('confidence', 0):.1%}")
            
            label_text = result.get('label_applied') or result.get('label_would_apply', 'None')
            self.console.print(f"Label: {label_text}")
            
            if result.get('reasoning'):
                self.console.print(f"Reasoning: {result.get('reasoning')}")
            
            if result.get('marketing_indicators'):
                self.console.print(f"Marketing Indicators: {', '.join(result.get('marketing_indicators', []))}")
            
            if result.get('unsubscribe_detected'):
                self.console.print("📧 Unsubscribe Link Detected")
            
            if result.get('bulk_sending_indicators'):
                self.console.print("📤 Bulk Sending Indicators Found")
            
            sender_rep = result.get('sender_reputation', 0)
            if sender_rep > 0:
                self.console.print(f"📊 Sender Reputation: {sender_rep:.1%}")
            
            self.console.print("-" * 80)
    
    def format_output(self, results: Dict[str, Any]) -> None:
        """Format and display results."""
        columns = {
            "subject": "Subject",
            "sender": "Sender",
            "marketing_subtype": "Type",
            "confidence": "Confidence",
            "labels": "Labels"
        }
        self.display_results_table(results, columns)


@app.command()
def unread(
    confidence_threshold: float = typer.Option(0.7, "--confidence-threshold", "-c", help="Minimum confidence for labeling"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview labels without applying them (default: apply labels)"),
    limit: Optional[int] = typer.Option(50, "--limit", "-l", help="Maximum number of emails to process (default: 50)"),
    types: Optional[str] = typer.Option(None, "--types", help="Comma-separated marketing types to include"),
    exclude_personal: bool = typer.Option(True, "--exclude-personal/--include-personal", help="Exclude personal emails from labeling"),
    sender_analysis: bool = typer.Option(False, "--sender-analysis", help="Include sender statistics"),
    detailed: bool = typer.Option(False, "--detailed", help="Show detailed analysis results")
):
    """Label unread emails with marketing classification."""
    
    @run_async_command
    async def run():
        command = MarketingLabelCommand()
        await command.initialize()
        
        # Parse types if provided
        types_list = None
        if types:
            types_list = [t.strip() for t in types.split(',')]
        
        results = await command.execute(
            target="unread",
            confidence_threshold=confidence_threshold,
            dry_run=dry_run,
            limit=limit,
            types=types_list,
            exclude_personal=exclude_personal,
            sender_analysis=sender_analysis,
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
    types: Optional[str] = typer.Option(None, "--types", help="Comma-separated marketing types to include"),
    exclude_personal: bool = typer.Option(True, "--exclude-personal/--include-personal", help="Exclude personal emails from labeling"),
    sender_analysis: bool = typer.Option(False, "--sender-analysis", help="Include sender statistics"),
    detailed: bool = typer.Option(False, "--detailed", help="Show detailed analysis results")
):
    """Label recent emails with marketing classification."""
    
    @run_async_command
    async def run():
        command = MarketingLabelCommand()
        await command.initialize()
        
        # Parse types if provided
        types_list = None
        if types:
            types_list = [t.strip() for t in types.split(',')]
        
        results = await command.execute(
            target=f"recent:{days}days",
            confidence_threshold=confidence_threshold,
            dry_run=dry_run,
            limit=limit,
            types=types_list,
            exclude_personal=exclude_personal,
            sender_analysis=sender_analysis,
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
    types: Optional[str] = typer.Option(None, "--types", help="Comma-separated marketing types to include"),
    exclude_personal: bool = typer.Option(True, "--exclude-personal/--include-personal", help="Exclude personal emails from labeling"),
    sender_analysis: bool = typer.Option(False, "--sender-analysis", help="Include sender statistics"),
    detailed: bool = typer.Option(False, "--detailed", help="Show detailed analysis results")
):
    """Label emails matching custom query with marketing classification."""
    
    @run_async_command
    async def run():
        command = MarketingLabelCommand()
        await command.initialize()
        
        # Parse types if provided
        types_list = None
        if types:
            types_list = [t.strip() for t in types.split(',')]
        
        results = await command.execute(
            target=f"query:{search_query}",
            confidence_threshold=confidence_threshold,
            dry_run=dry_run,
            limit=limit,
            types=types_list,
            exclude_personal=exclude_personal,
            sender_analysis=sender_analysis,
            detailed=detailed
        )
        
        command.format_output(results)
        return command
    
    run()


if __name__ == "__main__":
    app()