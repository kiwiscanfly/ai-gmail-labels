"""Statistics and analytics commands."""

import asyncio
import json
import csv
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.bar import Bar
from pathlib import Path

from src.cli.base import run_async_command
from src.services.email_service import EmailService

app = typer.Typer(help="Statistics and analytics commands")
console = Console()


class StatsAnalyzer:
    """Analyzer for email statistics and performance metrics."""
    
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
    
    async def classification_performance(self, days: int = 30) -> Dict[str, Any]:
        """Analyze classification performance metrics.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Classification performance statistics
        """
        try:
            # Get labels for each classification type
            priority_labels = [
                "Priority/Critical", "Priority/High", "Priority/Medium", "Priority/Low"
            ]
            marketing_labels = [
                "Marketing/Promotional", "Marketing/Newsletter", "Marketing/Hybrid", "Marketing/General"
            ]
            receipt_labels = [
                "Receipts/Purchase", "Receipts/Service", "Receipts/Other"
            ]
            notification_labels = [
                "Notifications/System", "Notifications/Update", "Notifications/Alert", 
                "Notifications/Reminder", "Notifications/Security"
            ]
            
            all_classification_labels = priority_labels + marketing_labels + receipt_labels + notification_labels
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            stats = {
                "analysis_period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "days": days
                },
                "priority": {},
                "marketing": {},
                "receipts": {},
                "notifications": {},
                "summary": {}
            }
            
            self.console.print(f"[blue]Analyzing classification performance for last {days} days...[/blue]")
            
            total_classified = 0
            
            # Analyze each classification type
            for category, labels in [
                ("priority", priority_labels),
                ("marketing", marketing_labels), 
                ("receipts", receipt_labels),
                ("notifications", notification_labels)
            ]:
                category_stats = {
                    "total_emails": 0,
                    "labels": {},
                    "top_label": None,
                    "distribution": {}
                }
                
                for label in labels:
                    try:
                        # Get emails with this label from the time period
                        query = f'label:"{label}" after:{start_date.strftime("%Y/%m/%d")}'
                        emails = []
                        async for email in self.email_service.search_emails(query=query, limit=1000):
                            emails.append(email)
                        
                        count = len(emails)
                        category_stats["labels"][label] = count
                        category_stats["total_emails"] += count
                        
                    except Exception as e:
                        self.console.print(f"[yellow]Warning: Could not analyze {label}: {e}[/yellow]")
                        category_stats["labels"][label] = 0
                
                # Calculate distribution
                if category_stats["total_emails"] > 0:
                    for label, count in category_stats["labels"].items():
                        percentage = (count / category_stats["total_emails"]) * 100
                        category_stats["distribution"][label] = {
                            "count": count,
                            "percentage": percentage
                        }
                    
                    # Find top label
                    category_stats["top_label"] = max(
                        category_stats["labels"].items(), 
                        key=lambda x: x[1]
                    )[0]
                
                stats[category] = category_stats
                total_classified += category_stats["total_emails"]
            
            # Calculate summary statistics
            stats["summary"] = {
                "total_classified_emails": total_classified,
                "avg_per_day": total_classified / days if days > 0 else 0,
                "most_active_category": max(
                    ["priority", "marketing", "receipts", "notifications"],
                    key=lambda cat: stats[cat]["total_emails"]
                ),
                "classification_breakdown": {
                    cat: stats[cat]["total_emails"] for cat in ["priority", "marketing", "receipts", "notifications"]
                }
            }
            
            return stats
            
        except Exception as e:
            self.console.print(f"[red]Failed to analyze classification performance: {e}[/red]")
            raise
    
    async def sender_analysis(self, limit: int = 50, days: int = 30) -> Dict[str, Any]:
        """Analyze sender patterns and statistics.
        
        Args:
            limit: Number of top senders to analyze
            days: Number of days to analyze
            
        Returns:
            Sender analysis results
        """
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            self.console.print(f"[blue]Analyzing sender patterns for last {days} days...[/blue]")
            
            # Get recent emails
            query = f'after:{start_date.strftime("%Y/%m/%d")}'
            emails = []
            async for email in self.email_service.search_emails(query=query, limit=5000):
                emails.append(email)
            
            if not emails:
                return {
                    "analysis_period": {"days": days},
                    "total_emails": 0,
                    "senders": []
                }
            
            # Analyze sender patterns
            sender_stats = {}
            domain_stats = {}
            
            for email in emails:
                sender = email.sender or "Unknown"
                
                # Extract domain
                domain = "unknown"
                if "@" in sender:
                    domain = sender.split("@")[-1].lower()
                
                # Count sender occurrences
                if sender not in sender_stats:
                    sender_stats[sender] = {
                        "count": 0,
                        "domain": domain,
                        "subjects": set(),
                        "first_seen": email.date,
                        "last_seen": email.date
                    }
                
                sender_stats[sender]["count"] += 1
                sender_stats[sender]["subjects"].add(email.subject or "No Subject")
                
                if email.date:
                    if email.date < sender_stats[sender]["first_seen"]:
                        sender_stats[sender]["first_seen"] = email.date
                    if email.date > sender_stats[sender]["last_seen"]:
                        sender_stats[sender]["last_seen"] = email.date
                
                # Count domain occurrences
                domain_stats[domain] = domain_stats.get(domain, 0) + 1
            
            # Sort senders by email count
            top_senders = sorted(
                sender_stats.items(),
                key=lambda x: x[1]["count"],
                reverse=True
            )[:limit]
            
            # Sort domains by email count
            top_domains = sorted(
                domain_stats.items(),
                key=lambda x: x[1],
                reverse=True
            )[:20]
            
            # Identify potential patterns
            automated_senders = []
            high_volume_senders = []
            
            for sender, stats in top_senders:
                # High volume threshold
                if stats["count"] > len(emails) * 0.05:  # More than 5% of emails
                    high_volume_senders.append({
                        "sender": sender,
                        "count": stats["count"],
                        "percentage": (stats["count"] / len(emails)) * 100
                    })
                
                # Automated sender patterns
                automated_keywords = ["noreply", "no-reply", "donotreply", "automated", "system"]
                if any(keyword in sender.lower() for keyword in automated_keywords):
                    automated_senders.append({
                        "sender": sender,
                        "count": stats["count"]
                    })
            
            return {
                "analysis_period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "days": days
                },
                "total_emails": len(emails),
                "unique_senders": len(sender_stats),
                "unique_domains": len(domain_stats),
                "top_senders": [
                    {
                        "sender": sender,
                        "count": stats["count"],
                        "domain": stats["domain"],
                        "unique_subjects": len(stats["subjects"]),
                        "percentage": (stats["count"] / len(emails)) * 100
                    }
                    for sender, stats in top_senders
                ],
                "top_domains": [
                    {
                        "domain": domain,
                        "count": count,
                        "percentage": (count / len(emails)) * 100
                    }
                    for domain, count in top_domains
                ],
                "patterns": {
                    "high_volume_senders": high_volume_senders,
                    "automated_senders": automated_senders[:10]
                }
            }
            
        except Exception as e:
            self.console.print(f"[red]Failed to analyze senders: {e}[/red]")
            raise
    
    async def usage_statistics(self, days: int = 7) -> Dict[str, Any]:
        """Analyze system usage patterns.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Usage statistics
        """
        try:
            # This would typically analyze command usage logs
            # For now, we'll analyze email processing patterns
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            self.console.print(f"[blue]Analyzing system usage for last {days} days...[/blue]")
            
            # Get all labels to understand label usage
            all_labels = await self.email_service.get_labels()
            user_labels = [label for label in all_labels if getattr(label, 'type', 'user') == 'user']
            
            # Analyze label usage
            label_usage = []
            for label in user_labels:
                try:
                    # Check recent activity
                    query = f'label:"{label.name}" after:{start_date.strftime("%Y/%m/%d")}'
                    recent_emails = []
                    async for email in self.email_service.search_emails(query=query, limit=10):
                        recent_emails.append(email)
                    
                    label_usage.append({
                        "label": label.name,
                        "total_messages": getattr(label, 'messages_total', 0),
                        "total_unread": getattr(label, 'messages_unread', 0),
                        "recent_activity": len(recent_emails)
                    })
                except Exception:
                    continue
            
            # Sort by recent activity
            label_usage.sort(key=lambda x: x["recent_activity"], reverse=True)
            
            # Calculate statistics
            total_labels = len(user_labels)
            active_labels = len([l for l in label_usage if l["recent_activity"] > 0])
            
            return {
                "analysis_period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "days": days
                },
                "label_statistics": {
                    "total_labels": total_labels,
                    "active_labels": active_labels,
                    "inactive_labels": total_labels - active_labels,
                    "activity_rate": (active_labels / total_labels * 100) if total_labels > 0 else 0
                },
                "most_active_labels": label_usage[:10],
                "summary": {
                    "total_user_labels": total_labels,
                    "labels_with_recent_activity": active_labels,
                    "average_messages_per_label": sum(l["total_messages"] for l in label_usage) / len(label_usage) if label_usage else 0
                }
            }
            
        except Exception as e:
            self.console.print(f"[red]Failed to analyze usage statistics: {e}[/red]")
            raise
    
    async def export_stats(
        self, 
        file_path: str, 
        stats_type: str, 
        format: str = "csv",
        **kwargs
    ) -> Dict[str, Any]:
        """Export statistics to a file.
        
        Args:
            file_path: Export file path
            stats_type: Type of statistics (classification, senders, usage)
            format: Export format (csv, json)
            **kwargs: Additional arguments for stats generation
            
        Returns:
            Export result
        """
        try:
            # Generate statistics based on type
            if stats_type == "classification":
                stats = await self.classification_performance(**kwargs)
            elif stats_type == "senders":
                stats = await self.sender_analysis(**kwargs)
            elif stats_type == "usage":
                stats = await self.usage_statistics(**kwargs)
            else:
                return {
                    "success": False,
                    "error": f"Unknown stats type: {stats_type}"
                }
            
            # Prepare export path
            export_path = Path(file_path)
            export_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format.lower() == "csv":
                # Export to CSV format
                self._export_to_csv(stats, export_path, stats_type)
            else:  # json
                with open(export_path, 'w') as f:
                    json.dump(stats, f, indent=2, default=str)
            
            return {
                "success": True,
                "exported_file": str(export_path),
                "format": format,
                "stats_type": stats_type
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to export statistics: {str(e)}"
            }
    
    def _export_to_csv(self, stats: Dict[str, Any], file_path: Path, stats_type: str):
        """Export statistics to CSV format."""
        with open(file_path, 'w', newline='') as csvfile:
            if stats_type == "classification":
                writer = csv.writer(csvfile)
                writer.writerow(["Category", "Label", "Count", "Percentage"])
                
                for category in ["priority", "marketing", "receipts", "notifications"]:
                    if category in stats and "distribution" in stats[category]:
                        for label, data in stats[category]["distribution"].items():
                            writer.writerow([
                                category.title(),
                                label,
                                data["count"],
                                f"{data['percentage']:.2f}%"
                            ])
            
            elif stats_type == "senders":
                writer = csv.writer(csvfile)
                writer.writerow(["Sender", "Count", "Domain", "Percentage", "Unique_Subjects"])
                
                for sender_data in stats.get("top_senders", []):
                    writer.writerow([
                        sender_data["sender"],
                        sender_data["count"],
                        sender_data["domain"],
                        f"{sender_data['percentage']:.2f}%",
                        sender_data["unique_subjects"]
                    ])
            
            elif stats_type == "usage":
                writer = csv.writer(csvfile)
                writer.writerow(["Label", "Total_Messages", "Unread_Messages", "Recent_Activity"])
                
                for label_data in stats.get("most_active_labels", []):
                    writer.writerow([
                        label_data["label"],
                        label_data["total_messages"],
                        label_data["total_unread"],
                        label_data["recent_activity"]
                    ])

    def format_output(self, results: Dict[str, Any], command_type: str) -> None:
        """Format and display command results."""
        if command_type == "classification":
            self._format_classification_output(results)
        elif command_type == "senders":
            self._format_senders_output(results)
        elif command_type == "usage":
            self._format_usage_output(results)
        elif command_type == "export":
            self._format_export_output(results)
    
    def _format_classification_output(self, results: Dict[str, Any]) -> None:
        """Format classification performance output."""
        period = results["analysis_period"]
        summary = results["summary"]
        
        # Summary panel
        self.console.print(Panel(
            f"[bold]Classification Performance Analysis[/bold]\n\n"
            f"Analysis Period: {period['days']} days\n"
            f"Total Classified Emails: {summary['total_classified_emails']:,}\n"
            f"Average per Day: {summary['avg_per_day']:.1f}\n"
            f"Most Active Category: {summary['most_active_category'].title()}",
            title="📊 Summary",
            border_style="blue"
        ))
        
        # Category breakdown
        for category in ["priority", "marketing", "receipts", "notifications"]:
            if category in results and results[category]["total_emails"] > 0:
                cat_data = results[category]
                
                self.console.print(f"\n[bold cyan]{category.title()} Classification[/bold cyan]")
                
                table = Table()
                table.add_column("Label", style="white")
                table.add_column("Count", justify="right", style="blue")
                table.add_column("Percentage", justify="right", style="green")
                table.add_column("Bar", style="yellow")
                
                for label, dist_data in cat_data.get("distribution", {}).items():
                    count = dist_data["count"]
                    percentage = dist_data["percentage"]
                    
                    # Create a simple bar representation
                    bar_length = int(percentage / 5)  # Scale down for display
                    bar = "█" * bar_length
                    
                    table.add_row(
                        label.split("/")[-1],  # Show only the label part after /
                        f"{count:,}",
                        f"{percentage:.1f}%",
                        bar
                    )
                
                self.console.print(table)
    
    def _format_senders_output(self, results: Dict[str, Any]) -> None:
        """Format sender analysis output."""
        period = results["analysis_period"]
        
        # Summary panel
        self.console.print(Panel(
            f"[bold]Sender Analysis[/bold]\n\n"
            f"Analysis Period: {period['days']} days\n"
            f"Total Emails: {results['total_emails']:,}\n"
            f"Unique Senders: {results['unique_senders']:,}\n"
            f"Unique Domains: {results['unique_domains']:,}",
            title="📫 Sender Statistics",
            border_style="blue"
        ))
        
        # Top senders table
        if results.get("top_senders"):
            self.console.print("\n[bold cyan]Top Senders[/bold cyan]")
            table = Table()
            table.add_column("Sender", style="white", no_wrap=False)
            table.add_column("Count", justify="right", style="blue")
            table.add_column("Percentage", justify="right", style="green")
            table.add_column("Subjects", justify="right", style="yellow")
            
            for sender in results["top_senders"][:15]:
                table.add_row(
                    sender["sender"][:50] + "..." if len(sender["sender"]) > 50 else sender["sender"],
                    f"{sender['count']:,}",
                    f"{sender['percentage']:.1f}%",
                    str(sender["unique_subjects"])
                )
            
            self.console.print(table)
        
        # High volume senders
        if results["patterns"]["high_volume_senders"]:
            self.console.print("\n[bold red]🚨 High Volume Senders[/bold red]")
            for sender in results["patterns"]["high_volume_senders"][:5]:
                self.console.print(f"  • {sender['sender']}: {sender['count']} emails ({sender['percentage']:.1f}%)")
        
        # Top domains
        if results.get("top_domains"):
            self.console.print("\n[bold cyan]Top Domains[/bold cyan]")
            for domain in results["top_domains"][:10]:
                self.console.print(f"  • {domain['domain']}: {domain['count']} emails ({domain['percentage']:.1f}%)")
    
    def _format_usage_output(self, results: Dict[str, Any]) -> None:
        """Format usage statistics output."""
        period = results["analysis_period"]
        label_stats = results["label_statistics"]
        
        # Summary panel
        self.console.print(Panel(
            f"[bold]System Usage Statistics[/bold]\n\n"
            f"Analysis Period: {period['days']} days\n"
            f"Total Labels: {label_stats['total_labels']}\n"
            f"Active Labels: {label_stats['active_labels']}\n"
            f"Activity Rate: {label_stats['activity_rate']:.1f}%",
            title="📈 Usage Summary",
            border_style="blue"
        ))
        
        # Most active labels
        if results.get("most_active_labels"):
            self.console.print("\n[bold cyan]Most Active Labels[/bold cyan]")
            table = Table()
            table.add_column("Label", style="white")
            table.add_column("Total Messages", justify="right", style="blue")
            table.add_column("Unread", justify="right", style="yellow")
            table.add_column("Recent Activity", justify="right", style="green")
            
            for label in results["most_active_labels"]:
                if label["recent_activity"] > 0:  # Only show labels with recent activity
                    table.add_row(
                        label["label"],
                        f"{label['total_messages']:,}",
                        f"{label['total_unread']:,}",
                        str(label["recent_activity"])
                    )
            
            self.console.print(table)
    
    def _format_export_output(self, results: Dict[str, Any]) -> None:
        """Format export results output."""
        if results["success"]:
            self.console.print(f"[green]✅ Successfully exported {results['stats_type']} statistics[/green]")
            self.console.print(f"   File: {results['exported_file']}")
            self.console.print(f"   Format: {results['format']}")
        else:
            self.console.print(f"[red]❌ {results['error']}[/red]")


@app.command()
def classification(
    days: int = typer.Option(30, "--days", "-d", help="Number of days to analyze")
):
    """Analyze classification performance metrics."""
    
    @run_async_command
    async def run():
        analyzer = StatsAnalyzer()
        await analyzer.initialize()
        
        results = await analyzer.classification_performance(days=days)
        analyzer.format_output(results, "classification")
        return analyzer
    
    run()


@app.command()
def senders(
    limit: int = typer.Option(50, "--limit", "-l", help="Number of top senders to show"),
    days: int = typer.Option(30, "--days", "-d", help="Number of days to analyze")
):
    """Analyze sender patterns and statistics."""
    
    @run_async_command
    async def run():
        analyzer = StatsAnalyzer()
        await analyzer.initialize()
        
        results = await analyzer.sender_analysis(limit=limit, days=days)
        analyzer.format_output(results, "senders")
        return analyzer
    
    run()


@app.command()
def usage(
    days: int = typer.Option(7, "--days", "-d", help="Number of days to analyze")
):
    """Analyze system usage patterns."""
    
    @run_async_command
    async def run():
        analyzer = StatsAnalyzer()
        await analyzer.initialize()
        
        results = await analyzer.usage_statistics(days=days)
        analyzer.format_output(results, "usage")
        return analyzer
    
    run()


@app.command()
def export(
    stats_type: str = typer.Argument(..., help="Type of statistics (classification, senders, usage)"),
    file_path: str = typer.Argument(..., help="Export file path"),
    format: str = typer.Option("csv", "--format", "-f", help="Export format (csv or json)"),
    days: int = typer.Option(30, "--days", "-d", help="Number of days to analyze"),
    limit: int = typer.Option(50, "--limit", "-l", help="Limit for results (senders only)")
):
    """Export statistics to a file."""
    
    @run_async_command
    async def run():
        analyzer = StatsAnalyzer()
        await analyzer.initialize()
        
        # Prepare kwargs based on stats type
        kwargs = {"days": days}
        if stats_type == "senders":
            kwargs["limit"] = limit
        
        results = await analyzer.export_stats(
            file_path=file_path,
            stats_type=stats_type,
            format=format,
            **kwargs
        )
        analyzer.format_output(results, "export")
        return analyzer
    
    run()


if __name__ == "__main__":
    app()