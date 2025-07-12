"""System installation command for email-agents project."""

import asyncio
import json
import os
import sys
import time
import webbrowser
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich import print as rprint
from dotenv import load_dotenv, set_key
import structlog

from src.core.config import get_config
from src.core.exceptions import ConfigurationError
from src.integrations.gmail_client import get_gmail_client
from src.integrations.ollama_client import get_ollama_manager

logger = structlog.get_logger(__name__)

app = typer.Typer(help="System installation and setup")
console = Console()


class SystemInstaller:
    """Handles the complete system installation process."""
    
    def __init__(self, force: bool = False, skip_ollama: bool = False, 
                 skip_gmail: bool = False, config_path: str = ".env"):
        self.force = force
        self.skip_ollama = skip_ollama
        self.skip_gmail = skip_gmail
        self.config_path = Path(config_path)
        self.env_example_path = Path(".env.example")
        self.console = console
        self.errors = []
        self.warnings = []
        
    async def run(self) -> bool:
        """Run the complete installation process."""
        try:
            self._print_header()
            
            # Phase 1: System Requirements
            if not await self._check_system_requirements():
                return False
                
            # Phase 2: Environment Configuration
            if not await self._configure_environment():
                return False
                
            # Phase 3: Gmail Authentication
            if not self.skip_gmail:
                if not await self._setup_gmail_authentication():
                    self.warnings.append("Gmail authentication skipped or failed")
                    
            # Phase 4: Ollama Models
            if not self.skip_ollama:
                if not await self._install_ollama_models():
                    self.warnings.append("Ollama model installation skipped or failed")
                    
            # Phase 5: System Validation
            validation_results = await self._validate_installation()
            
            # Display final summary
            self._display_summary(validation_results)
            
            return len(self.errors) == 0
            
        except Exception as e:
            self.console.print(f"[red]Installation failed: {e}[/red]")
            logger.error("Installation failed", error=str(e))
            return False
    
    def _print_header(self):
        """Print installation header."""
        self.console.print(Panel(
            "[bold blue]Email Agent System Installation[/bold blue]\n\n"
            "This wizard will guide you through setting up the email categorization system.\n"
            "Press Enter to accept default values or type custom values.",
            title="📧 Welcome",
            border_style="blue"
        ))
    
    async def _check_system_requirements(self) -> bool:
        """Check and validate system requirements."""
        self.console.print("\n[bold yellow]Phase 1: Checking System Requirements[/bold yellow]")
        
        requirements = []
        
        # Check Python version
        python_version = sys.version_info
        if python_version >= (3, 13):
            requirements.append(("Python Version", f"{python_version.major}.{python_version.minor}.{python_version.micro}", True))
        else:
            requirements.append(("Python Version", f"{python_version.major}.{python_version.minor} (3.13+ required)", False))
            self.errors.append("Python 3.13 or higher is required")
        
        # Check uv installation
        try:
            import subprocess
            result = subprocess.run(["uv", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                uv_version = result.stdout.strip()
                requirements.append(("uv Package Manager", uv_version, True))
            else:
                requirements.append(("uv Package Manager", "Not found", False))
                self.errors.append("uv is not installed. Install from https://docs.astral.sh/uv/")
        except FileNotFoundError:
            requirements.append(("uv Package Manager", "Not found", False))
            self.errors.append("uv is not installed. Install from https://docs.astral.sh/uv/")
        
        # Check for required files
        if self.env_example_path.exists():
            requirements.append((".env.example", "Found", True))
        else:
            requirements.append((".env.example", "Missing", False))
            self.errors.append(".env.example file not found")
        
        # Display requirements table
        table = Table(title="System Requirements")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="white")
        table.add_column("Check", style="green")
        
        for component, status, passed in requirements:
            check_mark = "✅" if passed else "❌"
            table.add_row(component, status, check_mark)
        
        self.console.print(table)
        
        if self.errors:
            self.console.print("\n[red]❌ System requirements not met. Please fix the issues above.[/red]")
            return False
        
        self.console.print("\n[green]✅ All system requirements met![/green]")
        return True
    
    async def _configure_environment(self) -> bool:
        """Configure environment variables interactively."""
        self.console.print("\n[bold yellow]Phase 2: Environment Configuration[/bold yellow]")
        
        # Load defaults from .env.example
        defaults = self._load_env_defaults()
        
        # Check if .env already exists
        if self.config_path.exists() and not self.force:
            if not Confirm.ask(f"\n{self.config_path} already exists. Overwrite?", default=False):
                self.console.print("[yellow]Using existing configuration[/yellow]")
                load_dotenv(self.config_path)
                return True
        
        # Configuration sections
        sections = [
            ("Ollama Configuration", [
                ("OLLAMA__HOST", "Ollama server URL"),
                ("OLLAMA__MODELS__PRIMARY", "Primary model for classification"),
                ("OLLAMA__MODELS__FALLBACK", "Fallback model"),
                ("OLLAMA__TIMEOUT", "Request timeout (seconds)"),
                ("OLLAMA__MAX_RETRIES", "Maximum retry attempts"),
            ]),
            ("LLM Parameters", [
                ("LLM__TEMPERATURE", "Generation temperature (0.0-1.0)"),
                ("LLM__TOP_P", "Top-p sampling parameter"),
                ("LLM__NUM_PREDICT", "Maximum tokens to generate"),
            ]),
            ("Gmail Settings", [
                ("GMAIL__CREDENTIALS_PATH", "Path to credentials.json"),
                ("GMAIL__TOKEN_PATH", "Path to store tokens"),
                ("GMAIL__BATCH_SIZE", "Email batch size"),
            ]),
            ("Email Processing", [
                ("EMAIL__DEFAULT_LIMIT", "Default processing limit"),
                ("EMAIL__MAX_RESULTS", "Maximum search results"),
                ("EMAIL__CACHE_TTL", "Cache TTL in seconds"),
            ]),
            ("Classification Thresholds", [
                ("CLASSIFICATION__PRIORITY__HIGH_CONFIDENCE", "High priority threshold"),
                ("CLASSIFICATION__MARKETING__THRESHOLD", "Marketing detection threshold"),
                ("CLASSIFICATION__RECEIPT__THRESHOLD", "Receipt detection threshold"),
                ("CLASSIFICATION__CUSTOM__THRESHOLD", "Custom category threshold"),
            ]),
        ]
        
        # Collect configuration values
        config_values = {}
        
        for section_name, prompts in sections:
            self.console.print(f"\n[bold cyan]🔧 {section_name}[/bold cyan]")
            
            for key, description in prompts:
                default = defaults.get(key, "")
                value = Prompt.ask(
                    f"  {description}",
                    default=str(default),
                    show_default=True
                )
                config_values[key] = value
        
        # Write configuration to file
        self.console.print(f"\n[blue]Writing configuration to {self.config_path}...[/blue]")
        
        try:
            # Create config file with header
            with open(self.config_path, 'w') as f:
                f.write("# Email Agent Configuration\n")
                f.write(f"# Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("# Run 'uv run email-agent system install' to reconfigure\n\n")
            
            # Write each configuration value
            for key, value in config_values.items():
                set_key(self.config_path, key, value)
            
            # Also copy any missing values from .env.example
            for key, value in defaults.items():
                if key not in config_values:
                    set_key(self.config_path, key, str(value))
            
            self.console.print("[green]✅ Configuration saved successfully![/green]")
            
            # Load the configuration
            load_dotenv(self.config_path)
            return True
            
        except Exception as e:
            self.console.print(f"[red]Failed to save configuration: {e}[/red]")
            self.errors.append(f"Configuration save failed: {e}")
            return False
    
    def _load_env_defaults(self) -> Dict[str, str]:
        """Load default values from .env.example."""
        defaults = {}
        
        if self.env_example_path.exists():
            with open(self.env_example_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        defaults[key.strip()] = value.strip().strip('"').strip("'")
        
        return defaults
    
    async def _setup_gmail_authentication(self) -> bool:
        """Set up Gmail OAuth authentication."""
        self.console.print("\n[bold yellow]Phase 3: Gmail Authentication Setup[/bold yellow]")
        
        # Check for credentials.json
        creds_path = Path(os.getenv("GMAIL__CREDENTIALS_PATH", "./credentials.json"))
        
        if not creds_path.exists():
            self.console.print("\n[red]❌ credentials.json not found![/red]")
            self.console.print("\n[yellow]To set up Gmail authentication:[/yellow]")
            self.console.print("1. Go to https://console.cloud.google.com/")
            self.console.print("2. Create a new project or select existing")
            self.console.print("3. Enable Gmail API")
            self.console.print("4. Create OAuth 2.0 credentials (Desktop application)")
            self.console.print("5. Download and save as credentials.json in project root")
            
            if not Confirm.ask("\nDo you have credentials.json ready?", default=False):
                return False
        
        # Test Gmail authentication
        try:
            self.console.print("\n[blue]Testing Gmail authentication...[/blue]")
            self.console.print("[yellow]A browser window will open for authentication[/yellow]")
            
            gmail_client = await get_gmail_client()
            profile = await gmail_client.get_profile()
            
            self.console.print(f"\n[green]✅ Gmail authenticated successfully![/green]")
            self.console.print(f"   Email: {profile.get('emailAddress')}")
            self.console.print(f"   Messages: {profile.get('messagesTotal')}")
            
            return True
            
        except Exception as e:
            self.console.print(f"\n[red]Gmail authentication failed: {e}[/red]")
            self.errors.append(f"Gmail auth failed: {e}")
            return False
    
    async def _install_ollama_models(self) -> bool:
        """Install required Ollama models."""
        self.console.print("\n[bold yellow]Phase 4: Ollama Model Installation[/bold yellow]")
        
        try:
            # Check Ollama connection
            ollama_manager = await get_ollama_manager()
            health = await ollama_manager.get_health_status()
            
            if health["status"] != "healthy":
                self.console.print("[red]❌ Ollama is not running![/red]")
                self.console.print("Please start Ollama: https://ollama.ai")
                return False
            
            self.console.print(f"[green]✅ Connected to Ollama at {health['host']}[/green]")
            
            # Get required models from config
            models_to_install = [
                os.getenv("OLLAMA__MODELS__PRIMARY", "gemma3:4b"),
                os.getenv("OLLAMA__MODELS__FALLBACK", "llama3.2:3b"),
            ]
            
            # Remove duplicates
            models_to_install = list(set(models_to_install))
            
            # Check existing models
            existing_models = await ollama_manager.list_models()
            existing_names = [m.name for m in existing_models]
            
            # Install missing models
            for model in models_to_install:
                if model in existing_names:
                    self.console.print(f"   ✅ Model already installed: {model}")
                else:
                    self.console.print(f"   📥 Installing model: {model}")
                    
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        BarColumn(),
                        TaskProgressColumn(),
                        console=self.console
                    ) as progress:
                        task = progress.add_task(f"Pulling {model}...", total=None)
                        
                        try:
                            await ollama_manager.pull_model(model)
                            progress.update(task, completed=100)
                            self.console.print(f"   ✅ Successfully installed: {model}")
                        except Exception as e:
                            self.console.print(f"   ❌ Failed to install {model}: {e}")
                            self.warnings.append(f"Model {model} installation failed")
            
            # Test model inference
            self.console.print("\n[blue]Testing model inference...[/blue]")
            test_result = await ollama_manager.generate(
                prompt="Hello, this is a test.",
                options={"num_predict": 10}
            )
            
            if test_result.content:
                self.console.print("[green]✅ Model inference test successful![/green]")
                return True
            else:
                self.console.print("[red]❌ Model inference test failed[/red]")
                return False
                
        except Exception as e:
            self.console.print(f"[red]Ollama setup failed: {e}[/red]")
            self.errors.append(f"Ollama setup failed: {e}")
            return False
    
    async def _validate_installation(self) -> Dict[str, Any]:
        """Validate the complete installation."""
        self.console.print("\n[bold yellow]Phase 5: System Validation[/bold yellow]")
        
        validation_results = {
            "python": {"status": "✅", "version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"},
            "configuration": {"status": "❓", "details": ""},
            "ollama": {"status": "❓", "details": ""},
            "gmail": {"status": "❓", "details": ""},
            "database": {"status": "❓", "details": ""},
        }
        
        # Validate configuration
        try:
            config = get_config()
            validation_results["configuration"]["status"] = "✅"
            validation_results["configuration"]["details"] = "Loaded successfully"
        except Exception as e:
            validation_results["configuration"]["status"] = "❌"
            validation_results["configuration"]["details"] = str(e)
        
        # Validate Ollama
        if not self.skip_ollama:
            try:
                ollama_manager = await get_ollama_manager()
                models = await ollama_manager.list_models()
                validation_results["ollama"]["status"] = "✅"
                validation_results["ollama"]["details"] = f"{len(models)} models available"
            except Exception as e:
                validation_results["ollama"]["status"] = "❌"
                validation_results["ollama"]["details"] = str(e)
        else:
            validation_results["ollama"]["status"] = "⏭️"
            validation_results["ollama"]["details"] = "Skipped"
        
        # Validate Gmail
        if not self.skip_gmail:
            try:
                gmail_client = await get_gmail_client()
                profile = await gmail_client.get_profile()
                validation_results["gmail"]["status"] = "✅"
                validation_results["gmail"]["details"] = profile.get('emailAddress', 'Unknown')
            except Exception as e:
                validation_results["gmail"]["status"] = "❌"
                validation_results["gmail"]["details"] = str(e)
        else:
            validation_results["gmail"]["status"] = "⏭️"
            validation_results["gmail"]["details"] = "Skipped"
        
        # Validate database
        try:
            from src.core.state_manager import get_state_manager
            state_mgr = await get_state_manager()
            validation_results["database"]["status"] = "✅"
            validation_results["database"]["details"] = "Initialized"
        except Exception as e:
            validation_results["database"]["status"] = "❌"
            validation_results["database"]["details"] = str(e)
        
        return validation_results
    
    def _display_summary(self, validation_results: Dict[str, Any]):
        """Display installation summary."""
        # Create summary table
        table = Table(title="Installation Summary", show_header=False)
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="white")
        table.add_column("Details", style="yellow")
        
        table.add_row("System Requirements", "", "")
        table.add_row("  Python", validation_results["python"]["status"], validation_results["python"]["version"])
        table.add_row("  Configuration", validation_results["configuration"]["status"], validation_results["configuration"]["details"])
        
        table.add_row("", "", "")
        table.add_row("External Services", "", "")
        table.add_row("  Ollama", validation_results["ollama"]["status"], validation_results["ollama"]["details"])
        table.add_row("  Gmail", validation_results["gmail"]["status"], validation_results["gmail"]["details"])
        
        table.add_row("", "", "")
        table.add_row("Database", "", "")
        table.add_row("  SQLite", validation_results["database"]["status"], validation_results["database"]["details"])
        
        self.console.print("\n")
        self.console.print(table)
        
        # Show warnings and errors
        if self.warnings:
            self.console.print("\n[yellow]⚠️  Warnings:[/yellow]")
            for warning in self.warnings:
                self.console.print(f"   - {warning}")
        
        if self.errors:
            self.console.print("\n[red]❌ Errors:[/red]")
            for error in self.errors:
                self.console.print(f"   - {error}")
            self.console.print("\n[red]Installation completed with errors. Please fix the issues above.[/red]")
        else:
            self.console.print("\n[green]✅ Installation completed successfully![/green]")
            self.console.print("\n[bold]🚀 Ready to use! Try these commands:[/bold]")
            self.console.print("\n  # Check system status")
            self.console.print("  uv run email-agent status")
            self.console.print("\n  # Label priority emails")
            self.console.print("  uv run email-agent label priority --limit 10")
            self.console.print("\n  # Create custom category")
            self.console.print("  uv run email-agent label custom create \"work\"")


@app.command()
def install(
    force: bool = typer.Option(False, "--force", "-f", help="Force reinstall/reconfigure"),
    skip_ollama: bool = typer.Option(False, "--skip-ollama", help="Skip Ollama setup"),
    skip_gmail: bool = typer.Option(False, "--skip-gmail", help="Skip Gmail setup"),
    config_path: str = typer.Option(".env", "--config-path", "-c", help="Configuration file path"),
):
    """Install and configure the email-agents system."""
    installer = SystemInstaller(
        force=force,
        skip_ollama=skip_ollama,
        skip_gmail=skip_gmail,
        config_path=config_path
    )
    
    # Run the installer
    success = asyncio.run(installer.run())
    
    if not success:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()