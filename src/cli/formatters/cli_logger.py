"""
Awesome CLI logging helper that's MCP-aware.

This module provides enhanced CLI logging that automatically adjusts based on the environment:
- Rich console output for interactive CLI usage
- MCP-safe stderr logging when running as MCP server
- Smart progress indicators and user feedback
"""

import os
import sys
import time
import random
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.live import Live

# Remove py-spinners dependency - using Rich's built-in spinners instead

from src.core.mcp_logger import get_mcp_logger, mcp_print


class CLILogger:
    """
    Smart CLI logger that adapts to MCP vs interactive environments.
    """
    
    def __init__(self, verbose: bool = False):
        self.is_mcp_mode = self._detect_mcp_mode()
        self.console = Console(file=sys.stderr, force_terminal=not self.is_mcp_mode)
        self.mcp_logger = get_mcp_logger("cli") if self.is_mcp_mode else None
        self._progress_context = None
        self._current_progress = None
        self.verbose = verbose
        
    def _detect_mcp_mode(self) -> bool:
        """Detect if we're running in MCP server mode."""
        # Check for common MCP indicators
        return (
            os.getenv("MCP_SERVER_MODE") == "true" or
            "--mcp" in sys.argv or
            "mcp" in sys.argv[0] or
            not sys.stdout.isatty()  # Often indicates piped/server mode
        )
    
    def _get_random_spinner(self) -> str:
        """Get a random spinner from Rich's built-in spinners."""
        # Basic Rich spinners that are guaranteed to work in all versions
        rich_spinners = [
            "dots", "dots2", "dots3", "dots4", "dots5", "dots6", "dots7", "dots8", "dots9", "dots10", "dots11", "dots12",
            "line", "line2", "pipe", "simpleDots", "simpleDotsScrolling", "star", "star2", "flip", "hamburger", "growVertical",
            "growHorizontal", "balloon", "balloon2", "noise", "bounce", "boxBounce", "boxBounce2", "triangle", "arc", "circle",
            "squareCorners", "circleQuarters", "circleHalves", "squish", "toggle", "toggle2", "toggle3", "toggle4", "toggle5",
            "toggle6", "toggle7", "toggle8", "toggle9", "toggle10", "toggle11", "toggle12", "toggle13", "arrow", "arrow2",
            "arrow3", "bouncingBar", "bouncingBall"
        ]
        
        return random.choice(rich_spinners)
    
    def info(self, message: str, **kwargs) -> None:
        """Display info message."""
        if self.is_mcp_mode:
            self.mcp_logger.info(message, **kwargs)
        else:
            self.console.print(f"[blue]{message}[/blue]")
    
    def verbose_info(self, message: str, **kwargs) -> None:
        """Display info message only in verbose mode."""
        if self.verbose:
            self.info(message, **kwargs)
    
    def success(self, message: str, **kwargs) -> None:
        """Display success message."""
        if self.is_mcp_mode:
            self.mcp_logger.info(f"✓ {message}", **kwargs)
        else:
            self.console.print(f"[green]✓ {message}[/green]")
    
    def warning(self, message: str, **kwargs) -> None:
        """Display warning message."""
        if self.is_mcp_mode:
            self.mcp_logger.warning(message, **kwargs)
        else:
            self.console.print(f"[yellow]⚠️  {message}[/yellow]")
    
    def error(self, message: str, **kwargs) -> None:
        """Display error message."""
        if self.is_mcp_mode:
            self.mcp_logger.error(message, **kwargs)
        else:
            self.console.print(f"[red]❌ {message}[/red]")
    
    def step(self, message: str, step: int = None, total: int = None) -> None:
        """Display a step in a process."""
        if step and total:
            prefix = f"[{step}/{total}]"
        else:
            prefix = "▶"
        
        if self.is_mcp_mode:
            self.mcp_logger.info(f"{prefix} {message}")
        else:
            self.console.print(f"[cyan]{prefix}[/cyan] {message}")
    
    def llm_status(self, model: str, operation: str, duration: float = None, tokens: int = None) -> None:
        """Display LLM operation status in a user-friendly way."""
        if not self.verbose:
            return  # Skip technical LLM details in non-verbose mode
            
        if duration and tokens:
            rate = tokens / duration if duration > 0 else 0
            status = f"🤖 {model}: {operation} ({duration:.1f}s, {rate:.0f} tok/s)"
        elif duration:
            status = f"🤖 {model}: {operation} ({duration:.1f}s)"
        else:
            status = f"🤖 {model}: {operation}"
        
        if self.is_mcp_mode:
            self.mcp_logger.info(status)
        else:
            self.console.print(f"[dim]{status}[/dim]")
    
    def user_friendly_status(self, message: str) -> None:
        """Display a user-friendly status message (always shown)."""
        if self.is_mcp_mode:
            self.mcp_logger.info(message)
        else:
            self.console.print(f"[cyan]● {message}[/cyan]")
    
    @contextmanager
    def status_with_spinner(self, message: str, success_message: str = None):
        """Display a status message with spinner animation until completion."""
        if self.is_mcp_mode:
            self.mcp_logger.info(message)
            yield
            if success_message:
                self.mcp_logger.info(success_message)
        else:
            # Use a random spinner for visual variety
            spinner_name = self._get_random_spinner()
            
            # Create a progress with just spinner and text
            progress = Progress(
                SpinnerColumn(spinner_name),
                TextColumn(f"[cyan]{message}[/cyan]"),
                console=self.console,
                transient=True  # Remove when done
            )
            
            with progress:
                task_id = progress.add_task("", total=None)
                yield
                
            # Show completion message
            if success_message:
                self.console.print(f"[green]✓ {success_message}[/green]")
            else:
                self.console.print(f"[green]✓ {message}[/green]")
    
    def classification_result(self, email_subject: str, labels: List[str], dry_run: bool = False) -> None:
        """Display classification result."""
        subject_preview = email_subject[:40] + "..." if len(email_subject) > 40 else email_subject
        
        if not labels:
            message = f"⚠️  No labels for: {subject_preview} (low confidence)"
            if self.is_mcp_mode:
                self.mcp_logger.warning(message)
            else:
                self.console.print(f"[yellow]{message}[/yellow]")
        else:
            prefix = "🔍 Would apply" if dry_run else "🏷️  Applied"
            label_text = ", ".join(labels)
            message = f"{prefix} {len(labels)} labels to: {subject_preview} -> {label_text}"
            
            if self.is_mcp_mode:
                self.mcp_logger.info(message)
            else:
                color = "blue" if dry_run else "green"
                self.console.print(f"[{color}]{message}[/{color}]")
    
    @contextmanager
    def progress(self, description: str, total: Optional[int] = None):
        """Create a progress context manager."""
        if self.is_mcp_mode:
            # Simple text-based progress for MCP mode
            self.mcp_logger.info(f"Starting: {description}")
            yield MCPProgressTracker(self.mcp_logger, description, total)
            self.mcp_logger.info(f"Completed: {description}")
        else:
            # Rich progress for interactive mode with random spinner
            spinner_name = self._get_random_spinner()
            progress = Progress(
                SpinnerColumn(spinner_name),
                TextColumn("[progress.description]{task.description}"),
                BarColumn() if total else TextColumn(""),
                TaskProgressColumn() if total else TextColumn(""),
                TimeElapsedColumn(),
                console=self.console,
                transient=False
            )
            
            with progress:
                task_id = progress.add_task(description, total=total)
                yield RichProgressTracker(progress, task_id)
    
    def show_summary_panel(self, title: str, content: str, style: str = "blue") -> None:
        """Show a summary panel."""
        if self.is_mcp_mode:
            self.mcp_logger.info(f"{title}: {content}")
        else:
            self.console.print(Panel(content, title=title, style=style))
    
    def show_results_table(self, headers: List[str], rows: List[List[str]], title: str = "Results") -> None:
        """Show results in a table format."""
        if self.is_mcp_mode:
            # Simple text table for MCP mode
            self.mcp_logger.info(f"{title}:")
            for row in rows:
                self.mcp_logger.info("  " + " | ".join(row))
        else:
            # Rich table for interactive mode
            table = Table(title=title)
            for header in headers:
                table.add_column(header)
            for row in rows:
                table.add_row(*row)
            self.console.print(table)


class ProgressTracker:
    """Base progress tracker interface."""
    
    def __init__(self, description: str, total: Optional[int] = None):
        self.description = description
        self.total = total
        self.completed = 0
    
    def update(self, advance: int = 1) -> None:
        """Update progress."""
        pass
    
    def set_description(self, description: str) -> None:
        """Update description."""
        pass


class MCPProgressTracker(ProgressTracker):
    """Simple progress tracker for MCP mode."""
    
    def __init__(self, logger, description: str, total: Optional[int] = None):
        super().__init__(description, total)
        self.logger = logger
        self._last_percent = -1
    
    def update(self, advance: int = 1) -> None:
        """Update progress with periodic logging."""
        self.completed += advance
        
        if self.total:
            percent = int((self.completed / self.total) * 100)
            # Only log every 20% to avoid spam
            if percent >= self._last_percent + 20:
                self.logger.info(f"{self.description}: {percent}% ({self.completed}/{self.total})")
                self._last_percent = percent
    
    def set_description(self, description: str) -> None:
        """Update description."""
        self.description = description


class RichProgressTracker(ProgressTracker):
    """Rich progress tracker for interactive mode."""
    
    def __init__(self, progress: Progress, task_id):
        self.progress = progress
        self.task_id = task_id
    
    def update(self, advance: int = 1) -> None:
        """Update rich progress."""
        self.progress.update(self.task_id, advance=advance)
    
    def set_description(self, description: str) -> None:
        """Update description."""
        self.progress.update(self.task_id, description=description)


# Global instance for easy access
cli_logger = CLILogger()


def set_verbose_mode(verbose: bool = True) -> None:
    """Set verbose mode for the global logger and configure global logging level."""
    import logging
    import structlog
    
    global cli_logger
    cli_logger.verbose = verbose
    
    # Configure global logging level based on verbose flag
    if verbose:
        # Show all logs in verbose mode
        logging.basicConfig(level=logging.DEBUG)
        structlog.configure(
            wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG),
            cache_logger_on_first_use=True,
        )
    else:
        # Only show WARNING and above in non-verbose mode
        logging.basicConfig(level=logging.WARNING)
        structlog.configure(
            wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING), 
            cache_logger_on_first_use=True,
        )
        
        # Suppress third-party library logs in non-verbose mode
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("googleapiclient").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)


# Convenience functions
def info(message: str, **kwargs) -> None:
    """Log info message."""
    cli_logger.info(message, **kwargs)


def verbose_info(message: str, **kwargs) -> None:
    """Log info message only in verbose mode."""
    cli_logger.verbose_info(message, **kwargs)


def user_friendly_status(message: str) -> None:
    """Display a user-friendly status message (always shown)."""
    cli_logger.user_friendly_status(message)

def status_with_spinner(message: str, success_message: str = None):
    """Display a status message with spinner animation until completion."""
    return cli_logger.status_with_spinner(message, success_message)


def success(message: str, **kwargs) -> None:
    """Log success message."""
    cli_logger.success(message, **kwargs)


def warning(message: str, **kwargs) -> None:
    """Log warning message."""
    cli_logger.warning(message, **kwargs)


def error(message: str, **kwargs) -> None:
    """Log error message."""
    cli_logger.error(message, **kwargs)


def step(message: str, step: int = None, total: int = None) -> None:
    """Log step message."""
    cli_logger.step(message, step, total)


def llm_status(model: str, operation: str, duration: float = None, tokens: int = None) -> None:
    """Log LLM status."""
    cli_logger.llm_status(model, operation, duration, tokens)


def classification_result(email_subject: str, labels: List[str], dry_run: bool = False) -> None:
    """Log classification result."""
    cli_logger.classification_result(email_subject, labels, dry_run)


def progress(description: str, total: Optional[int] = None):
    """Create progress context."""
    return cli_logger.progress(description, total)


def show_summary_panel(title: str, content: str, style: str = "blue") -> None:
    """Show summary panel."""
    cli_logger.show_summary_panel(title, content, style)


def show_results_table(headers: List[str], rows: List[List[str]], title: str = "Results") -> None:
    """Show results table."""
    cli_logger.show_results_table(headers, rows, title)