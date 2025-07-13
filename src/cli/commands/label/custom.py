"""Custom email labeling commands with AI-generated search terms."""

import asyncio
from typing import Optional, Dict, Any, List
import typer
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.panel import Panel

from src.cli.base import BaseEmailProcessor, run_async_command
from src.services.custom_classifier import CustomClassifier
from src.cli.langchain.agents import EmailAnalysisAgent
from src.models.email import EmailCategory
from src.integrations.ollama_client import get_ollama_manager

app = typer.Typer(help="Custom email labeling commands with AI-powered categorization", invoke_without_command=True)


@app.callback()
def main_callback(ctx: typer.Context):
    """Custom category classification with AI-powered search terms."""
    # If no subcommand is invoked, show helpful guidance
    if ctx.invoked_subcommand is None:
        console = typer.get_app_dir("email-agent")  # Get console for consistent formatting
        from rich.console import Console
        console = Console()
        
        # Display comprehensive usage guide
        console.print("📖 [bold cyan]Custom Label Command Guide[/bold cyan]\n")
        
        console.print("The [bold]email-agent label custom[/bold] command helps you create AI-powered custom categories for your emails.\n")
        
        console.print("🎯 [bold yellow]Available Commands:[/bold yellow]\n")
        
        # 1. Create command
        console.print("1. [bold green]Create Custom Categories[/bold green] ([cyan]create[/cyan])\n")
        console.print("   Create and apply custom labels with AI-generated search terms:\n")
        
        console.print("   [dim]# Basic usage - will generate AI search terms for 'programming'[/dim]")
        console.print("   [bold]uv run email-agent label custom create \"programming\"[/bold]\n")
        
        console.print("   [dim]# Create with specific search terms[/dim]")
        console.print("   [bold]uv run email-agent label custom create \"work\" --search-terms \"project,meeting,deadline,task\"[/bold]\n")
        
        console.print("   [dim]# Advanced usage with parent label[/dim]")
        console.print("   [bold]uv run email-agent label custom create \"travel\" \\[/bold]")
        console.print("   [bold]  --parent-label \"Personal/Travel\" \\[/bold]")
        console.print("   [bold]  --search-existing \\[/bold]")
        console.print("   [bold]  --confidence-threshold 0.8[/bold]\n")
        
        console.print("   [dim]# Preview mode (dry run)[/dim]")
        console.print("   [bold]uv run email-agent label custom create \"finance\" --dry-run --detailed[/bold]\n")
        
        console.print("   [yellow]Examples:[/yellow]")
        console.print("   [dim]# AI-powered programming email classification[/dim]")
        console.print("   [bold]uv run email-agent label custom create \"programming\" --limit 100[/bold]\n")
        
        console.print("   [dim]# Work emails with custom terms[/dim]")
        console.print("   [bold]uv run email-agent label custom create \"work\" \\[/bold]")
        console.print("   [bold]    --search-terms \"standup,sprint,jira,github,pull request\" \\[/bold]")
        console.print("   [bold]    --target \"recent:30days\"[/bold]\n")
        
        console.print("   [dim]# Personal finance tracking[/dim]")
        console.print("   [bold]uv run email-agent label custom create \"finance\" \\[/bold]")
        console.print("   [bold]    --parent-label \"Personal/Finance\" \\[/bold]")
        console.print("   [bold]    --search-existing[/bold]\n")
        
        # 2. Generate terms command
        console.print("2. [bold green]Generate Search Terms[/bold green] ([cyan]generate-terms[/cyan])\n")
        console.print("   Generate AI-powered search terms without classifying emails:\n")
        
        console.print("   [dim]# Generate terms for a category[/dim]")
        console.print("   [bold]uv run email-agent label custom generate-terms \"programming\"[/bold]\n")
        
        console.print("   [dim]# Generate with additional context[/dim]")
        console.print("   [bold]uv run email-agent label custom generate-terms \"travel\" \\[/bold]")
        console.print("   [bold]    --context \"vacation planning, flight bookings, hotel reservations\"[/bold]\n")
        
        # 3. Analyze command
        console.print("3. [bold green]Analyze Categories[/bold green] ([cyan]analyze[/cyan])\n")
        console.print("   Analyze existing labeled emails for insights:\n")
        
        console.print("   [dim]# Analyze existing labels[/dim]")
        console.print("   [bold]uv run email-agent label custom analyze \"programming\" --sample-size 50[/bold]\n")
        
        # 4. List command
        console.print("4. [bold green]List Categories[/bold green] ([cyan]list-categories[/cyan])\n")
        console.print("   View all custom categories and their statistics:\n")
        
        console.print("   [bold]uv run email-agent label custom list-categories[/bold]\n")
        
        # Common use cases
        console.print("🎯 [bold yellow]Common Use Cases:[/bold yellow]\n")
        
        console.print("   [bold green]Programming/Tech Emails[/bold green]")
        console.print("   [bold]uv run email-agent label custom create \"programming\" --search-existing --limit 200[/bold]\n")
        
        console.print("   [bold green]Work Project Organization[/bold green]")
        console.print("   [bold]uv run email-agent label custom create \"project-alpha\" \\[/bold]")
        console.print("   [bold]     --search-terms \"alpha project,milestone,deliverable\" \\[/bold]")
        console.print("   [bold]     --parent-label \"Work/Projects\"[/bold]\n")
        
        console.print("   [bold green]Personal Finance Tracking[/bold green]")
        console.print("   [bold]uv run email-agent label custom create \"finance\" \\[/bold]")
        console.print("   [bold]     --parent-label \"Personal\" \\[/bold]")
        console.print("   [bold]     --search-existing \\[/bold]")
        console.print("   [bold]     --detailed[/bold]\n")
        
        console.print("   [bold green]Travel Planning[/bold green]")
        console.print("   [bold]uv run email-agent label custom create \"travel\" \\[/bold]")
        console.print("   [bold]     --search-terms \"booking,flight,hotel,airbnb,vacation\" \\[/bold]")
        console.print("   [bold]     --confidence-threshold 0.7[/bold]\n")
        
        # Key options
        console.print("⚙️ [bold yellow]Key Options Explained:[/bold yellow]\n")
        console.print("   • [cyan]--target[/cyan]: Which emails to process (unread, recent:7days, etc.)")
        console.print("   • [cyan]--search-terms[/cyan]: Manual search terms (if not provided, AI generates them)")
        console.print("   • [cyan]--search-existing[/cyan]: Search your entire mailbox with generated terms")
        console.print("   • [cyan]--parent-label[/cyan]: Create hierarchical labels like \"Work/Projects/Alpha\"")
        console.print("   • [cyan]--confidence-threshold[/cyan]: Minimum confidence (0.0-1.0) for applying labels")
        console.print("   • [cyan]--dry-run[/cyan]: Preview what would be labeled without actually applying labels")
        console.print("   • [cyan]--detailed[/cyan]: Show classification reasoning and confidence scores\n")
        
        # Quick start
        console.print("🚀 [bold yellow]Quick Start:[/bold yellow]\n")
        console.print("   1. [bold]Start simple[/bold] - let AI generate terms:")
        console.print("      [bold]uv run email-agent label custom create \"programming\" --dry-run[/bold]\n")
        console.print("   2. [bold]Review the results[/bold] and adjust confidence threshold if needed\n")
        console.print("   3. [bold]Apply for real[/bold]:")
        console.print("      [bold]uv run email-agent label custom create \"programming\" --limit 100[/bold]\n")
        console.print("   4. [bold]Expand to your mailbox[/bold]:")
        console.print("      [bold]uv run email-agent label custom create \"programming\" --search-existing[/bold]\n")
        
        console.print("The AI will automatically generate relevant search terms based on your category name and then classify emails accordingly!\n")
        
        console.print("💡 [bold yellow]For detailed help on any command:[/bold yellow]")
        console.print("   [bold]uv run email-agent label custom [command] --help[/bold]")
        
        raise typer.Exit()


class CustomLabelCommand(BaseEmailProcessor):
    """Custom labeling command with AI-generated search terms and classification."""
    
    def __init__(self):
        super().__init__()
        self.custom_classifier = None
        self.analysis_agent = None
        self.ollama_manager = None
    
    async def initialize(self):
        """Initialize custom labeling services."""
        await super().initialize()
        self.custom_classifier = CustomClassifier()
        await self.custom_classifier.initialize()
        
        # Initialize analysis agent for insights
        self.ollama_manager = await get_ollama_manager()
        self.analysis_agent = EmailAnalysisAgent(self.ollama_manager)
        
        self.console.print("[blue]Custom labeling services initialized[/blue]")
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute custom classification - wrapper for execute_classification."""
        return await self.execute_classification(**kwargs)
    
    async def execute_classification(
        self,
        category: str,
        target: str = "unread",
        search_terms: Optional[List[str]] = None,
        generate_terms: bool = True,
        parent_label: Optional[str] = None,
        confidence_threshold: float = 0.7,
        dry_run: bool = True,
        limit: Optional[int] = None,
        search_existing: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute custom email classification for a specific category.
        
        Args:
            category: Category name to classify emails for
            target: Email target (unread/recent/query)
            search_terms: Optional predefined search terms
            generate_terms: Whether to generate AI-powered search terms
            parent_label: Optional parent label for organization
            confidence_threshold: Minimum confidence for labeling
            dry_run: Whether to preview without applying labels
            limit: Maximum number of emails to process
            search_existing: Whether to search entire mailbox or just target
            
        Returns:
            Classification results
        """
        # Generate or use provided search terms
        if generate_terms and not search_terms:
            self.console.print(f"[blue]Generating AI-powered search terms for '{category}'...[/blue]")
            search_terms = await self.custom_classifier.label_chain.generate_search_terms(category)
            self.console.print(f"[green]Generated terms: {', '.join(search_terms)}[/green]")
        elif search_terms:
            self.console.print(f"[blue]Using provided search terms: {', '.join(search_terms)}[/blue]")
        else:
            search_terms = [category.lower()]
            self.console.print(f"[yellow]Using default search term: {category.lower()}[/yellow]")
        
        # Determine search strategy
        if search_existing:
            # Search entire mailbox using generated terms
            query_parts = [f'"{term}"' for term in search_terms]
            query = " OR ".join(query_parts)
            self.console.print(f"[blue]Searching entire mailbox with terms: {query}[/blue]")
        else:
            # Combine target specification with search terms
            base_query = self._parse_target(target)
            # Create search query with terms
            term_queries = [f'"{term}"' for term in search_terms[:5]]  # Use top 5 terms to avoid query length issues
            terms_query = f"({' OR '.join(term_queries)})"
            
            # Combine with target
            if base_query:
                query = f"{base_query} {terms_query}"
            else:
                query = terms_query
            
            self.console.print(f"[blue]Searching {target} emails that match: {', '.join(search_terms[:5])}[/blue]")
        
        # Process emails
        emails = await self.process_emails(query, limit, dry_run)
        
        if not emails:
            return {"processed": 0, "results": [], "category": category, "search_terms": search_terms, "dry_run": dry_run}
        
        self.console.print(f"[blue]Processing {len(emails)} emails for custom category '{category}'...[/blue]")
        if dry_run:
            self.console.print("[yellow]DRY RUN MODE - No labels will be applied[/yellow]")
        
        results = []
        match_count = 0
        
        # Determine label name
        label_name = parent_label or category
        
        # Process emails with progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console
        ) as progress:
            task = progress.add_task(f"Classifying for '{category}'...", total=len(emails))
            
            for i, email in enumerate(emails, 1):
                try:
                    # Classify email for the custom category
                    classification_result = await self.custom_classifier.classify_email(
                        email, category, search_terms, confidence_threshold
                    )
                    
                    result = {
                        "email_id": email.id,
                        "subject": email.subject,
                        "sender": email.sender,
                        "category": category,
                        "is_match": classification_result.is_match,
                        "confidence": classification_result.confidence,
                        "reasoning": classification_result.reasoning,
                        "suggested_label": classification_result.suggested_label,
                        "search_terms_used": classification_result.search_terms_used
                    }
                    
                    if classification_result.is_match and classification_result.confidence >= confidence_threshold:
                        if not dry_run:
                            # Apply the label to the email
                            category_obj = EmailCategory(
                                email_id=email.id,
                                suggested_labels=[label_name],
                                confidence_scores={label_name: classification_result.confidence},
                                reasoning=classification_result.reasoning
                            )
                            await self.email_service.apply_category(email.id, category_obj, create_labels=True)
                            result["label_applied"] = label_name
                            self.console.print(f"   🏷️  Applied '{label_name}' to: {email.subject[:40]}...")
                        else:
                            result["label_would_apply"] = label_name
                            self.console.print(f"   🔍 Would apply '{label_name}' (confidence: {classification_result.confidence:.1%})")
                        
                        match_count += 1
                    elif classification_result.is_match:
                        result["label_skipped"] = f"Confidence {classification_result.confidence:.1%} below threshold {confidence_threshold:.1%}"
                        if not dry_run:
                            self.console.print(f"   ⚠️  Skipped {email.subject[:40]}... (low confidence)")
                    else:
                        result["label_skipped"] = "Email does not match category"
                        if not dry_run:
                            self.console.print(f"   ℹ️  No match: {email.subject[:40]}...")
                    
                    results.append(result)
                    
                    # Small delay to avoid overwhelming the API
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    self.console.print(f"   ❌ Failed to process {email.subject[:40]}...: {str(e)}")
                    results.append({
                        "email_id": email.id,
                        "subject": email.subject,
                        "category": category,
                        "is_match": False,
                        "confidence": 0.0,
                        "reasoning": f"Error: {str(e)}",
                        "error": str(e)
                    })
                
                progress.advance(task)
        
        # Print summary
        self.console.print(f"\n[green]🎉 Custom classification complete![/green]")
        self.console.print(f"   📊 Category: {category}")
        self.console.print(f"   🎯 Matches found: {match_count}/{len(results)}")
        self.console.print(f"   🔍 Search terms used: {', '.join(search_terms)}")
        
        return {
            "processed": len(results),
            "matches": match_count,
            "category": category,
            "search_terms": search_terms,
            "results": results,
            "dry_run": dry_run,
            "successful": len([r for r in results if not r.get("error")]),
            "failed": len([r for r in results if r.get("error")])
        }
    
    async def analyze_category_performance(
        self,
        category: str,
        sample_size: int = 50
    ) -> Dict[str, Any]:
        """Analyze existing emails for a category to provide insights.
        
        Args:
            category: Category to analyze
            sample_size: Number of emails to analyze
            
        Returns:
            Analysis results and recommendations
        """
        # Search for emails with the category label
        query = f'label:"{category}"'
        emails = await self.process_emails(query, sample_size, dry_run=True)
        
        if not emails:
            return {
                "error": f"No emails found with label '{category}'",
                "category": category,
                "email_count": 0
            }
        
        self.console.print(f"[blue]Analyzing {len(emails)} emails for category '{category}'...[/blue]")
        
        # Use the analysis agent to provide insights
        analysis = await self.analysis_agent.analyze_email_collection(emails)
        analysis["category"] = category
        analysis["analyzed_email_count"] = len(emails)
        
        return analysis
    
    def format_output(self, results: Dict[str, Any]) -> None:
        """Format and display results."""
        if results.get("error"):
            self.console.print(f"[red]Error: {results['error']}[/red]")
            return
        
        # Summary panel
        summary_text = f"Custom Classification Results for '{results.get('category', 'Unknown')}'\n"
        summary_text += f"Matches: {results.get('matches', 0)}/{results.get('processed', 0)} emails"
        if results.get('dry_run'):
            summary_text += " (DRY RUN)"
        
        self.console.print(Panel(
            summary_text,
            title="Custom Classification Summary",
            style="green" if results.get('matches', 0) > 0 else "yellow"
        ))
        
        # Results table
        if results.get('results'):
            table = Table(title=f"Classification Results for '{results.get('category')}'")
            table.add_column("Subject", style="cyan", width=40)
            table.add_column("Sender", style="white", width=25)
            table.add_column("Match", style="green")
            table.add_column("Confidence", style="yellow")
            table.add_column("Label", style="blue")
            
            for result in results['results'][:20]:  # Show first 20
                subject = result.get('subject', 'Unknown')[:37]
                if len(result.get('subject', '')) > 37:
                    subject += "..."
                
                sender = result.get('sender', 'Unknown')[:22]
                if len(result.get('sender', '')) > 22:
                    sender += "..."
                
                match_status = "✅ Yes" if result.get('is_match') else "❌ No"
                confidence = f"{result.get('confidence', 0):.1%}"
                
                if results.get('dry_run'):
                    label = result.get('label_would_apply', 'None')
                else:
                    label = result.get('label_applied', 'None')
                
                table.add_row(subject, sender, match_status, confidence, label)
            
            self.console.print(table)
            
            if len(results.get('results', [])) > 20:
                self.console.print(f"[yellow]... and {len(results['results']) - 20} more emails[/yellow]")


@app.command()
def create(
    category: str = typer.Argument(..., help="Category name for custom labeling"),
    target: str = typer.Option("unread", "--target", "-t", help="Email target: unread, all, recent:7days, from:sender, query:..."),
    search_terms: Optional[str] = typer.Option(None, "--search-terms", "-s", help="Comma-separated search terms (will generate if not provided)"),
    parent_label: Optional[str] = typer.Option(None, "--parent-label", "-p", help="Parent label for hierarchical organization"),
    search_existing: bool = typer.Option(False, "--search-existing", help="Search entire mailbox with generated terms (ignores target)"),
    confidence_threshold: float = typer.Option(0.6, "--confidence-threshold", "-c", help="Minimum confidence for labeling"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview labels without applying them (default: apply labels)"),
    limit: Optional[int] = typer.Option(50, "--limit", "-l", help="Maximum number of emails to process (default: 50)"),
    detailed: bool = typer.Option(False, "--detailed", help="Show detailed classification results")
):
    """Create and apply custom category labels with AI-generated search terms."""
    
    @run_async_command
    async def run():
        command = CustomLabelCommand()
        await command.initialize()
        
        # Parse search terms if provided
        terms_list = None
        if search_terms:
            terms_list = [term.strip() for term in search_terms.split(',')]
        
        results = await command.execute_classification(
            category=category,
            target=target if not search_existing else "all",
            search_terms=terms_list,
            generate_terms=not search_terms,  # Generate if not provided
            parent_label=parent_label,
            confidence_threshold=confidence_threshold,
            dry_run=dry_run,
            limit=limit,
            search_existing=search_existing
        )
        
        command.format_output(results)
        
        if detailed and results.get('results'):
            command.console.print("\n[bold]📋 Detailed Results:[/bold]")
            for result in results['results'][:10]:  # Show first 10 detailed
                if result.get('reasoning'):
                    command.console.print(f"\n📧 {result.get('subject', 'Unknown')[:50]}...")
                    command.console.print(f"   Confidence: {result.get('confidence', 0):.1%}")
                    command.console.print(f"   Reasoning: {result.get('reasoning')}")
        
        return command
    
    run()


@app.command()
def generate_terms(
    category: str = typer.Argument(..., help="Category to generate search terms for"),
    context: Optional[str] = typer.Option(None, "--context", "-c", help="Additional context for term generation")
):
    """Generate AI-powered search terms for a category without classifying emails."""
    
    @run_async_command
    async def run():
        command = CustomLabelCommand()
        await command.initialize()
        
        command.console.print(f"[blue]Generating search terms for category '{category}'...[/blue]")
        
        search_terms = await command.custom_classifier.label_chain.generate_search_terms(
            category, context
        )
        
        command.console.print(f"\n[green]📝 Generated Search Terms for '{category}':[/green]")
        for i, term in enumerate(search_terms, 1):
            command.console.print(f"   {i}. {term}")
        
        command.console.print(f"\n[blue]💡 Usage Example:[/blue]")
        command.console.print(f"   email-agent label custom \"{category}\" --search-terms \"{','.join(search_terms)}\"")
        
        return command
    
    run()


@app.command()
def analyze(
    category: str = typer.Argument(..., help="Category to analyze"),
    sample_size: int = typer.Option(50, "--sample-size", "-s", help="Number of emails to analyze")
):
    """Analyze existing labeled emails to provide insights and recommendations."""
    
    @run_async_command
    async def run():
        command = CustomLabelCommand()
        await command.initialize()
        
        analysis = await command.analyze_category_performance(category, sample_size)
        
        if analysis.get("error"):
            command.console.print(f"[red]Error: {analysis['error']}[/red]")
            return command
        
        # Display analysis results
        command.console.print(f"\n[green]📊 Analysis Results for '{category}'[/green]")
        
        # Basic statistics
        if analysis.get("email_count"):
            command.console.print(f"   📧 Analyzed emails: {analysis['email_count']}")
        
        # Sender statistics
        if analysis.get("sender_statistics"):
            stats = analysis["sender_statistics"]
            command.console.print(f"   👥 Unique senders: {stats.get('total_unique_senders', 0)}")
            command.console.print(f"   🌐 Unique domains: {stats.get('total_unique_domains', 0)}")
            
            if stats.get("top_senders"):
                command.console.print("\n   🔝 Top Senders:")
                for sender, count in stats["top_senders"][:5]:
                    command.console.print(f"      • {sender[:40]}... ({count} emails)")
        
        # Raw analysis from LLM
        if analysis.get("raw_analysis"):
            command.console.print(f"\n[blue]🤖 AI Analysis:[/blue]")
            command.console.print(analysis["raw_analysis"])
        
        return command
    
    run()


@app.command()
def list_categories():
    """List all custom categories and their statistics."""
    
    @run_async_command
    async def run():
        command = CustomLabelCommand()
        await command.initialize()
        
        categories = await command.custom_classifier.get_categories()
        stats = await command.custom_classifier.get_category_statistics()
        
        if not categories:
            command.console.print("[yellow]No custom categories found.[/yellow]")
            command.console.print("\n[blue]💡 Create a new category with:[/blue]")
            command.console.print("   email-agent label custom create \"your-category\"")
            return command
        
        # Display overview statistics
        command.console.print(f"\n[green]📊 Custom Categories Overview[/green]")
        command.console.print(f"   Total categories: {stats.get('total_categories', 0)}")
        command.console.print(f"   Active categories: {stats.get('active_categories', 0)}")
        command.console.print(f"   Total usage: {stats.get('total_usage', 0)} classifications")
        
        # Display categories table
        table = Table(title="Custom Categories")
        table.add_column("Name", style="cyan")
        table.add_column("Description", style="white", width=30)
        table.add_column("Usage Count", style="yellow")
        table.add_column("Search Terms", style="blue", width=25)
        table.add_column("Created", style="green")
        
        for category in sorted(categories, key=lambda c: c.usage_count, reverse=True):
            # Format creation date
            from datetime import datetime
            created_date = datetime.fromtimestamp(category.created_at).strftime("%Y-%m-%d")
            
            # Format search terms
            terms_display = ", ".join(category.search_terms[:3])
            if len(category.search_terms) > 3:
                terms_display += f" (+{len(category.search_terms) - 3})"
            
            table.add_row(
                category.name,
                category.description[:27] + "..." if len(category.description) > 30 else category.description,
                str(category.usage_count),
                terms_display,
                created_date
            )
        
        command.console.print(table)
        
        return command
    
    run()


if __name__ == "__main__":
    app()