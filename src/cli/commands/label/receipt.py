"""Receipt email classification commands."""

import asyncio
from typing import Optional, Dict, Any, List
import typer
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from src.cli.base import BaseEmailProcessor, run_async_command
from src.services.receipt_classifier import ReceiptClassifier
from src.models.email import EmailCategory

app = typer.Typer(help="Receipt email classification commands")


class ReceiptLabelCommand(BaseEmailProcessor):
    """Receipt labeling command implementation."""
    
    def __init__(self):
        super().__init__()
        self.receipt_classifier = None
        
        # Receipt label mapping - simplified to three types
        self.receipt_labels = {
            "purchase": "Receipts/Purchase",
            "service": "Receipts/Service", 
            "other": "Receipts/Other"
        }
    
    async def initialize(self):
        """Initialize receipt classification services."""
        await super().initialize()
        self.receipt_classifier = ReceiptClassifier()
        await self.receipt_classifier.initialize()
        
        # Ensure receipt labels exist
        await self._ensure_receipt_labels_exist()
    
    async def _ensure_receipt_labels_exist(self) -> None:
        """Ensure all receipt labels exist in Gmail with proper nesting."""
        self.console.print("[blue]Checking receipt labels...[/blue]")
        
        try:
            # Get existing labels
            existing_labels = await self.email_service.get_labels()
            existing_label_names = {label.name for label in existing_labels}
            
            # First ensure the parent "Receipts" label exists
            parent_label = "Receipts"
            if parent_label not in existing_label_names:
                self.console.print(f"   Creating parent label: {parent_label}")
                await self.email_service.create_label(name=parent_label)
                existing_label_names.add(parent_label)
            else:
                self.console.print(f"   ✓ Parent label exists: {parent_label}")
            
            # Create missing receipt sublabels
            for receipt_type, label_name in self.receipt_labels.items():
                if label_name not in existing_label_names:
                    self.console.print(f"   Creating nested label: {label_name}")
                    await self.email_service.create_label(name=label_name)
                else:
                    self.console.print(f"   ✓ Nested label exists: {label_name}")
                    
        except Exception as e:
            self.console.print(f"[red]Failed to ensure receipt labels exist: {e}[/red]")
            raise
    
    async def execute(
        self,
        target: str = "unread",
        confidence_threshold: float = 0.7,
        dry_run: bool = True,
        limit: Optional[int] = None,
        types: Optional[List[str]] = None,
        extract_details: bool = False,
        vendor_stats: bool = False,
        detailed: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute receipt classification."""
        
        # Convert target to Gmail query
        query = self._parse_target(target)
        
        # Process emails
        emails = await self.process_emails(query, limit, dry_run)
        
        if not emails:
            return {"processed": 0, "results": [], "dry_run": dry_run}
        
        self.console.print(f"[blue]Processing {len(emails)} emails for receipt classification...[/blue]")
        if dry_run:
            self.console.print("[yellow]DRY RUN MODE - No labels will be applied[/yellow]")
        
        results = []
        receipt_counts = {
            "purchase": 0, "service": 0, "other": 0, 
            "non_receipt": 0, "error": 0
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
                    # Analyze receipt classification
                    receipt_result = await self.receipt_classifier.classify_receipt(email)
                    
                    result = {
                        "email_id": email.id,
                        "subject": email.subject,
                        "sender": email.sender,
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
                    
                    # Check if we should apply labels
                    should_label = (
                        receipt_result.is_receipt and 
                        receipt_result.confidence >= confidence_threshold
                    )
                    
                    # Filter by types if specified
                    if types and receipt_result.receipt_type not in types:
                        should_label = False
                    
                    if should_label:
                        # Get the corresponding label name
                        label_name = self.receipt_labels.get(receipt_result.receipt_type, "Receipts/Other")
                        
                        if not dry_run:
                            # Apply the label to the email
                            category = EmailCategory(
                                email_id=email.id,
                                suggested_labels=[label_name],
                                confidence_scores={label_name: receipt_result.confidence},
                                reasoning=receipt_result.reasoning
                            )
                            await self.email_service.apply_category(email.id, category, create_labels=True)
                            result["labels_applied"] = [label_name]
                            self.console.print(f"   🧾 Applied {label_name} to: {email.subject[:40]}...")
                            
                            # Print receipt details if available and extract_details is True
                            if extract_details:
                                if receipt_result.vendor:
                                    self.console.print(f"      Vendor: {receipt_result.vendor}")
                                if receipt_result.amount:
                                    self.console.print(f"      Amount: {receipt_result.amount}")
                                if receipt_result.order_number:
                                    self.console.print(f"      Order #: {receipt_result.order_number}")
                        else:
                            result["labels_would_apply"] = [label_name]
                            self.console.print(f"   🔍 Would apply {label_name} (confidence: {receipt_result.confidence:.1%})")
                            
                            # Print receipt details in dry run if requested
                            if extract_details:
                                if receipt_result.vendor:
                                    self.console.print(f"      Vendor: {receipt_result.vendor}")
                                if receipt_result.amount:
                                    self.console.print(f"      Amount: {receipt_result.amount}")
                                if receipt_result.order_number:
                                    self.console.print(f"      Order #: {receipt_result.order_number}")
                    else:
                        # Non-receipt email or filtered out
                        if not receipt_result.is_receipt:
                            result["label_skipped"] = "Non-receipt email"
                            self.console.print(f"   📧 Non-receipt email: {email.subject[:40]}...")
                        elif receipt_result.confidence < confidence_threshold:
                            result["label_skipped"] = f"Confidence {receipt_result.confidence:.1%} below threshold"
                            self.console.print(f"   ⚠️  Skipped {email.subject[:40]}... (low confidence)")
                        else:
                            result["label_skipped"] = f"Filtered out ({receipt_result.receipt_type})"
                            self.console.print(f"   🚫 Filtered out {email.subject[:40]}... ({receipt_result.receipt_type})")
                    
                    results.append(result)
                    
                    # Update counts
                    if receipt_result.is_receipt:
                        receipt_counts[receipt_result.receipt_type] += 1
                    else:
                        receipt_counts["non_receipt"] += 1
                    
                    # Show receipt indicators if found
                    if receipt_result.is_receipt and receipt_result.receipt_indicators and detailed:
                        self.console.print(f"       Receipt indicators: {', '.join(receipt_result.receipt_indicators)}")
                    
                    # Small delay to avoid overwhelming the API
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    self.console.print(f"   ❌ Failed to process {email.subject[:40]}...: {str(e)}")
                    receipt_counts["error"] += 1
                    results.append({
                        "email_id": email.id,
                        "subject": email.subject,
                        "is_receipt": False,
                        "receipt_type": "error",
                        "confidence": 0.0,
                        "reasoning": f"Error: {str(e)}",
                        "error": str(e)
                    })
                
                progress.advance(task)
        
        # Print summary
        self.console.print(f"\n[green]🎉 Processing complete![/green]")
        self.console.print(f"   📊 Receipt Classification Distribution:")
        for receipt_type, count in receipt_counts.items():
            if count > 0:
                self.console.print(f"      {receipt_type.replace('_', ' ').title()}: {count}")
        
        # Show vendor statistics if requested
        if vendor_stats:
            self._print_vendor_statistics()
        
        # Show detailed results if requested
        if detailed:
            self._print_detailed_results(results)
        
        return {
            "processed": len(results),
            "receipt_counts": receipt_counts,
            "results": results,
            "dry_run": dry_run,
            "successful": len([r for r in results if r.get("receipt_type") != "error"]),
            "failed": receipt_counts["error"]
        }
    
    def _print_vendor_statistics(self) -> None:
        """Print vendor receipt statistics."""
        stats = self.receipt_classifier.get_vendor_statistics()
        
        self.console.print(f"\n📈 Vendor Receipt Statistics:")
        self.console.print(f"   Total Vendors: {stats.get('total_vendors', 0)}")
        self.console.print(f"   Receipt Vendors: {stats.get('receipt_vendors', 0)}")
        self.console.print(f"   Average Receipt Rate: {stats.get('average_receipt_rate', 0):.1%}")
        
        top_vendors = stats.get('top_receipt_vendors', [])
        if top_vendors:
            self.console.print(f"\n   Top Receipt Vendors:")
            for vendor_name, receipt_rate, total_emails in top_vendors:
                self.console.print(f"      {vendor_name}: {receipt_rate:.1%} ({total_emails} emails)")
    
    def _print_detailed_results(self, results: List[Dict[str, Any]]) -> None:
        """Print detailed results with receipt analysis."""
        self.console.print(f"\n📋 Detailed Receipt Analysis:")
        self.console.print("=" * 80)
        
        for result in results:
            self.console.print(f"\nSubject: {result.get('subject', 'Unknown')[:60]}...")
            self.console.print(f"Receipt Type: {result.get('receipt_type', 'unknown').replace('_', ' ').title()}")
            self.console.print(f"Is Receipt: {'Yes' if result.get('is_receipt') else 'No'}")
            self.console.print(f"Confidence: {result.get('confidence', 0):.1%}")
            
            label_text = result.get('label_applied') or result.get('label_would_apply', 'None')
            self.console.print(f"Label: {label_text}")
            
            if result.get('reasoning'):
                self.console.print(f"Reasoning: {result.get('reasoning')}")
            
            # Receipt details
            if result.get('is_receipt'):
                if result.get('vendor'):
                    self.console.print(f"Vendor: {result.get('vendor')}")
                if result.get('amount'):
                    amount_str = result.get('amount')
                    if result.get('currency'):
                        amount_str = f"{result.get('currency')} {amount_str}"
                    self.console.print(f"Amount: {amount_str}")
                if result.get('order_number'):
                    self.console.print(f"Order Number: {result.get('order_number')}")
                if result.get('transaction_date'):
                    self.console.print(f"Transaction Date: {result.get('transaction_date')}")
                if result.get('payment_method'):
                    self.console.print(f"Payment Method: {result.get('payment_method')}")
            
            if result.get('receipt_indicators'):
                self.console.print(f"Receipt Indicators: {', '.join(result.get('receipt_indicators', []))}")
            
            self.console.print("-" * 80)
    
    def format_output(self, results: Dict[str, Any]) -> None:
        """Format and display results."""
        columns = {
            "subject": "Subject",
            "sender": "Sender",
            "receipt_type": "Type",
            "confidence": "Confidence",
            "labels": "Labels"
        }
        self.display_results_table(results, columns)


@app.command()
def unread(
    confidence_threshold: float = typer.Option(0.7, "--confidence-threshold", "-c", help="Minimum confidence for labeling"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview labels without applying them (default: apply labels)"),
    limit: Optional[int] = typer.Option(50, "--limit", "-l", help="Maximum number of emails to process (default: 50)"),
    types: Optional[str] = typer.Option(None, "--types", help="Comma-separated receipt types to include"),
    extract_details: bool = typer.Option(False, "--extract-details", help="Extract and display transaction details"),
    vendor_stats: bool = typer.Option(False, "--vendor-stats", help="Include vendor statistics"),
    detailed: bool = typer.Option(False, "--detailed", help="Show detailed analysis results")
):
    """Label unread emails with receipt classification."""
    
    @run_async_command
    async def run():
        command = ReceiptLabelCommand()
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
            extract_details=extract_details,
            vendor_stats=vendor_stats,
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
    types: Optional[str] = typer.Option(None, "--types", help="Comma-separated receipt types to include"),
    extract_details: bool = typer.Option(False, "--extract-details", help="Extract and display transaction details"),
    vendor_stats: bool = typer.Option(False, "--vendor-stats", help="Include vendor statistics"),
    detailed: bool = typer.Option(False, "--detailed", help="Show detailed analysis results")
):
    """Label recent emails with receipt classification."""
    
    @run_async_command
    async def run():
        command = ReceiptLabelCommand()
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
            extract_details=extract_details,
            vendor_stats=vendor_stats,
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
    types: Optional[str] = typer.Option(None, "--types", help="Comma-separated receipt types to include"),
    extract_details: bool = typer.Option(False, "--extract-details", help="Extract and display transaction details"),
    vendor_stats: bool = typer.Option(False, "--vendor-stats", help="Include vendor statistics"),
    detailed: bool = typer.Option(False, "--detailed", help="Show detailed analysis results")
):
    """Label emails matching custom query with receipt classification."""
    
    @run_async_command
    async def run():
        command = ReceiptLabelCommand()
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
            extract_details=extract_details,
            vendor_stats=vendor_stats,
            detailed=detailed
        )
        
        command.format_output(results)
        return command
    
    run()


@app.command()
def search(
    query: str = typer.Option(..., "--query", "-q", help="Gmail search query"),
    confidence_threshold: float = typer.Option(0.7, "--confidence-threshold", "-c", help="Minimum confidence for labeling"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview labels without applying them (default: apply labels)"),
    limit: Optional[int] = typer.Option(50, "--limit", "-l", help="Maximum number of emails to process (default: 50)"),
    types: Optional[str] = typer.Option(None, "--types", help="Comma-separated receipt types to include"),
    extract_details: bool = typer.Option(False, "--extract-details", help="Extract and display transaction details"),
    vendor_stats: bool = typer.Option(False, "--vendor-stats", help="Include vendor statistics"),
    detailed: bool = typer.Option(False, "--detailed", help="Show detailed analysis results")
):
    """Label emails using --query option with receipt classification."""
    
    @run_async_command
    async def run():
        command = ReceiptLabelCommand()
        await command.initialize()
        
        # Parse types if provided
        types_list = None
        if types:
            types_list = [t.strip() for t in types.split(',')]
        
        results = await command.execute(
            target=f"query:{query}",
            confidence_threshold=confidence_threshold,
            dry_run=dry_run,
            limit=limit,
            types=types_list,
            extract_details=extract_details,
            vendor_stats=vendor_stats,
            detailed=detailed
        )
        
        command.format_output(results)
        return command
    
    run()


if __name__ == "__main__":
    app()