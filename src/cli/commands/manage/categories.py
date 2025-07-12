"""Custom category management commands."""

import asyncio
import json
import yaml
from pathlib import Path
from typing import Optional, Dict, Any, List
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.cli.base import run_async_command
from src.services.custom_classifier import CustomClassifier
from src.services.email_service import EmailService
from src.cli.langchain.chains import CustomLabelChain
from src.integrations.ollama_client import get_ollama_manager

app = typer.Typer(help="Custom category management commands")
console = Console()


class CategoryManager:
    """Manager for custom email categories."""
    
    def __init__(self):
        self.console = Console()
        self.email_service = None
        self.custom_classifier = None
        self.label_chain = None
    
    async def initialize(self):
        """Initialize category management services."""
        self.email_service = EmailService()
        await self.email_service.initialize()
        
        self.custom_classifier = CustomClassifier()
        await self.custom_classifier.initialize()
        
        ollama_manager = await get_ollama_manager()
        self.label_chain = CustomLabelChain(ollama_manager)
    
    async def shutdown(self):
        """Cleanup resources."""
        try:
            # Clean up database connections
            from src.core.database_pool import shutdown_database_pool
            await shutdown_database_pool()
        except Exception:
            pass
    
    async def list_categories(self) -> Dict[str, Any]:
        """List all custom categories.
        
        Returns:
            Dictionary with category information
        """
        try:
            categories = await self.custom_classifier.get_categories()
            
            category_data = []
            for category in categories:
                category_info = {
                    "name": category.name,
                    "description": getattr(category, 'description', ''),
                    "search_terms": category.search_terms,
                    "confidence_threshold": category.confidence_threshold,
                    "created_at": getattr(category, 'created_at', None),
                    "last_used": getattr(category, 'last_used', None),
                    "email_count": getattr(category, 'email_count', 0)
                }
                category_data.append(category_info)
            
            return {
                "categories": category_data,
                "total_count": len(category_data)
            }
            
        except Exception as e:
            self.console.print(f"[red]Failed to list categories: {e}[/red]")
            raise
    
    async def create_category(
        self, 
        name: str, 
        description: str = "", 
        search_terms: Optional[List[str]] = None,
        confidence_threshold: float = 0.7,
        generate_terms: bool = True
    ) -> Dict[str, Any]:
        """Create a new custom category.
        
        Args:
            name: Category name
            description: Category description
            search_terms: Manual search terms (optional)
            confidence_threshold: Classification confidence threshold
            generate_terms: Whether to generate additional search terms with AI
            
        Returns:
            Creation result
        """
        try:
            # Check if category already exists
            existing_categories = await self.custom_classifier.get_categories()
            existing_names = {cat.name.lower() for cat in existing_categories}
            
            if name.lower() in existing_names:
                return {
                    "success": False,
                    "error": f"Category '{name}' already exists"
                }
            
            # Generate search terms if requested
            final_search_terms = search_terms or []
            
            if generate_terms:
                self.console.print(f"[blue]Generating search terms for '{name}'...[/blue]")
                generated_terms = await self.label_chain.generate_search_terms(name, description)
                
                # Combine manual and generated terms
                final_search_terms.extend(generated_terms)
                
                # Remove duplicates while preserving order
                seen = set()
                final_search_terms = [term for term in final_search_terms 
                                    if term.lower() not in seen and not seen.add(term.lower())]
                
                self.console.print(f"[green]Generated {len(generated_terms)} terms, total: {len(final_search_terms)}[/green]")
            
            if not final_search_terms:
                final_search_terms = [name.lower()]
            
            # Create the category
            result = await self.custom_classifier.create_category(
                name=name,
                description=description,
                search_terms=final_search_terms,
                confidence_threshold=confidence_threshold
            )
            
            return {
                "success": True,
                "category": {
                    "name": name,
                    "description": description,
                    "search_terms": final_search_terms,
                    "confidence_threshold": confidence_threshold,
                    "generated_terms": len(generated_terms) if generate_terms else 0
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to create category '{name}': {str(e)}"
            }
    
    async def delete_category(self, name: str) -> Dict[str, Any]:
        """Delete a custom category.
        
        Args:
            name: Category name to delete
            
        Returns:
            Deletion result
        """
        try:
            # Check if category exists
            categories = await self.custom_classifier.get_categories()
            target_category = None
            
            for category in categories:
                if category.name.lower() == name.lower():
                    target_category = category
                    break
            
            if not target_category:
                return {
                    "success": False,
                    "error": f"Category '{name}' not found"
                }
            
            # Delete the category
            await self.custom_classifier.delete_category(name)
            
            return {
                "success": True,
                "deleted_category": name
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to delete category '{name}': {str(e)}"
            }
    
    async def update_category(
        self,
        name: str,
        description: Optional[str] = None,
        search_terms: Optional[List[str]] = None,
        confidence_threshold: Optional[float] = None,
        regenerate_terms: bool = False
    ) -> Dict[str, Any]:
        """Update an existing category.
        
        Args:
            name: Category name to update
            description: New description (optional)
            search_terms: New search terms (optional)
            confidence_threshold: New confidence threshold (optional)
            regenerate_terms: Whether to regenerate search terms
            
        Returns:
            Update result
        """
        try:
            # Check if category exists
            categories = await self.custom_classifier.get_categories()
            target_category = None
            
            for category in categories:
                if category.name.lower() == name.lower():
                    target_category = category
                    break
            
            if not target_category:
                return {
                    "success": False,
                    "error": f"Category '{name}' not found"
                }
            
            # Prepare updates
            updates = {}
            if description is not None:
                updates["description"] = description
            if confidence_threshold is not None:
                updates["confidence_threshold"] = confidence_threshold
            
            # Handle search terms
            final_search_terms = search_terms
            if regenerate_terms:
                self.console.print(f"[blue]Regenerating search terms for '{name}'...[/blue]")
                new_description = description or getattr(target_category, 'description', '')
                generated_terms = await self.label_chain.generate_search_terms(name, new_description)
                
                if search_terms:
                    # Combine provided and generated terms
                    final_search_terms = list(search_terms) + generated_terms
                else:
                    # Use existing terms + generated terms
                    existing_terms = target_category.search_terms or []
                    final_search_terms = existing_terms + generated_terms
                
                # Remove duplicates
                seen = set()
                final_search_terms = [term for term in final_search_terms 
                                    if term.lower() not in seen and not seen.add(term.lower())]
                
                self.console.print(f"[green]Regenerated terms, total: {len(final_search_terms)}[/green]")
            
            if final_search_terms is not None:
                updates["search_terms"] = final_search_terms
            
            # Update the category
            await self.custom_classifier.update_category(name, **updates)
            
            return {
                "success": True,
                "updated_category": name,
                "updates": updates
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to update category '{name}': {str(e)}"
            }
    
    async def train_category(
        self,
        name: str,
        source_label: str,
        min_confidence: float = 0.8,
        sample_size: int = 50
    ) -> Dict[str, Any]:
        """Train a category from existing Gmail labels.
        
        Args:
            name: Category name to train
            source_label: Gmail label to learn from
            min_confidence: Minimum confidence for training samples
            sample_size: Number of emails to analyze for training
            
        Returns:
            Training result
        """
        try:
            # Get emails with the source label
            emails = []
            async for email in self.email_service.search_emails(
                query=f'label:"{source_label}"',
                limit=sample_size
            ):
                emails.append(email)
            
            if not emails:
                return {
                    "success": False,
                    "error": f"No emails found with label '{source_label}'"
                }
            
            self.console.print(f"[blue]Training category '{name}' from {len(emails)} emails with label '{source_label}'...[/blue]")
            
            # Analyze email patterns to extract terms
            all_subjects = [email.subject or "" for email in emails]
            all_content = []
            
            for email in emails:
                content = ""
                if email.body_text:
                    content += email.body_text[:200]  # First 200 chars
                all_content.append(content)
            
            # Use AI to extract common patterns and generate search terms
            combined_text = " ".join(all_subjects + all_content)
            context = f"Training from {len(emails)} emails. Common patterns in subjects and content."
            
            trained_terms = await self.label_chain.generate_search_terms(name, context)
            
            # Update or create the category
            categories = await self.custom_classifier.get_categories()
            existing_category = None
            
            for category in categories:
                if category.name.lower() == name.lower():
                    existing_category = category
                    break
            
            if existing_category:
                # Update existing category with trained terms
                existing_terms = existing_category.search_terms or []
                combined_terms = list(set(existing_terms + trained_terms))
                
                await self.custom_classifier.update_category(
                    name,
                    search_terms=combined_terms,
                    description=f"Trained from {len(emails)} emails with label '{source_label}'"
                )
            else:
                # Create new category
                await self.custom_classifier.create_category(
                    name=name,
                    description=f"Trained from {len(emails)} emails with label '{source_label}'",
                    search_terms=trained_terms,
                    confidence_threshold=min_confidence
                )
            
            return {
                "success": True,
                "trained_category": name,
                "source_label": source_label,
                "training_emails": len(emails),
                "extracted_terms": len(trained_terms),
                "terms": trained_terms
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to train category '{name}': {str(e)}"
            }
    
    async def export_categories(self, file_path: str, format: str = "yaml") -> Dict[str, Any]:
        """Export categories to a file.
        
        Args:
            file_path: Path to export file
            format: Export format (yaml or json)
            
        Returns:
            Export result
        """
        try:
            categories = await self.custom_classifier.get_categories()
            
            export_data = {
                "categories": [],
                "export_timestamp": str(asyncio.get_event_loop().time()),
                "total_categories": len(categories)
            }
            
            for category in categories:
                category_data = {
                    "name": category.name,
                    "description": getattr(category, 'description', ''),
                    "search_terms": category.search_terms,
                    "confidence_threshold": category.confidence_threshold
                }
                export_data["categories"].append(category_data)
            
            # Write to file
            export_path = Path(file_path)
            export_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format.lower() == "json":
                with open(export_path, 'w') as f:
                    json.dump(export_data, f, indent=2)
            else:  # yaml
                with open(export_path, 'w') as f:
                    yaml.dump(export_data, f, default_flow_style=False, indent=2)
            
            return {
                "success": True,
                "exported_file": str(export_path),
                "format": format,
                "categories_exported": len(categories)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to export categories: {str(e)}"
            }
    
    async def import_categories(
        self, 
        file_path: str, 
        overwrite: bool = False,
        dry_run: bool = True
    ) -> Dict[str, Any]:
        """Import categories from a file.
        
        Args:
            file_path: Path to import file
            overwrite: Whether to overwrite existing categories
            dry_run: Whether to preview import without applying
            
        Returns:
            Import result
        """
        try:
            import_path = Path(file_path)
            if not import_path.exists():
                return {
                    "success": False,
                    "error": f"Import file not found: {file_path}"
                }
            
            # Load import data
            if import_path.suffix.lower() == '.json':
                with open(import_path, 'r') as f:
                    import_data = json.load(f)
            else:  # yaml
                with open(import_path, 'r') as f:
                    import_data = yaml.safe_load(f)
            
            if "categories" not in import_data:
                return {
                    "success": False,
                    "error": "Invalid import file format - missing 'categories' key"
                }
            
            # Get existing categories
            existing_categories = await self.custom_classifier.get_categories()
            existing_names = {cat.name.lower() for cat in existing_categories}
            
            # Plan import actions
            import_actions = []
            conflicts = []
            
            for category_data in import_data["categories"]:
                name = category_data.get("name", "")
                if not name:
                    continue
                
                if name.lower() in existing_names:
                    if overwrite:
                        import_actions.append({
                            "action": "update",
                            "name": name,
                            "data": category_data
                        })
                    else:
                        conflicts.append(name)
                else:
                    import_actions.append({
                        "action": "create",
                        "name": name,
                        "data": category_data
                    })
            
            # Display import plan
            result = {
                "success": True,
                "import_file": file_path,
                "total_categories": len(import_data["categories"]),
                "import_actions": len(import_actions),
                "conflicts": len(conflicts),
                "actions": import_actions,
                "conflict_names": conflicts,
                "dry_run": dry_run
            }
            
            if dry_run:
                return result
            
            # Execute import
            successful_imports = 0
            failed_imports = []
            
            for action in import_actions:
                try:
                    if action["action"] == "create":
                        await self.custom_classifier.create_category(
                            name=action["data"]["name"],
                            description=action["data"].get("description", ""),
                            search_terms=action["data"].get("search_terms", []),
                            confidence_threshold=action["data"].get("confidence_threshold", 0.7)
                        )
                    else:  # update
                        await self.custom_classifier.update_category(
                            action["data"]["name"],
                            description=action["data"].get("description"),
                            search_terms=action["data"].get("search_terms"),
                            confidence_threshold=action["data"].get("confidence_threshold")
                        )
                    successful_imports += 1
                except Exception as e:
                    failed_imports.append({
                        "name": action["name"],
                        "error": str(e)
                    })
            
            result.update({
                "successful_imports": successful_imports,
                "failed_imports": len(failed_imports),
                "failures": failed_imports
            })
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to import categories: {str(e)}"
            }

    def format_output(self, results: Dict[str, Any], command_type: str) -> None:
        """Format and display command results."""
        if command_type == "list":
            self._format_list_output(results)
        elif command_type == "create":
            self._format_create_output(results)
        elif command_type == "delete":
            self._format_delete_output(results)
        elif command_type == "update":
            self._format_update_output(results)
        elif command_type == "train":
            self._format_train_output(results)
        elif command_type == "export":
            self._format_export_output(results)
        elif command_type == "import":
            self._format_import_output(results)
    
    def _format_list_output(self, results: Dict[str, Any]) -> None:
        """Format category list output."""
        categories = results.get("categories", [])
        
        if not categories:
            self.console.print("[yellow]No custom categories found[/yellow]")
            self.console.print("Create your first category with: [cyan]email-agent manage categories create[/cyan]")
            return
        
        # Create table
        table = Table(title=f"Custom Categories ({results['total_count']} total)")
        table.add_column("Name", style="cyan")
        table.add_column("Description", style="white")
        table.add_column("Search Terms", style="green")
        table.add_column("Threshold", justify="center", style="yellow")
        table.add_column("Emails", justify="right", style="blue")
        
        for category in categories:
            terms_display = ", ".join(category["search_terms"][:3])
            if len(category["search_terms"]) > 3:
                terms_display += f" (+{len(category['search_terms']) - 3} more)"
            
            table.add_row(
                category["name"],
                category["description"][:50] + "..." if len(category["description"]) > 50 else category["description"],
                terms_display,
                f"{category['confidence_threshold']:.1f}",
                str(category["email_count"])
            )
        
        self.console.print(table)
    
    def _format_create_output(self, results: Dict[str, Any]) -> None:
        """Format category creation output."""
        if results["success"]:
            category = results["category"]
            self.console.print(f"[green]✅ Successfully created category: {category['name']}[/green]")
            self.console.print(f"   Description: {category['description']}")
            self.console.print(f"   Search Terms: {', '.join(category['search_terms'])}")
            self.console.print(f"   Confidence Threshold: {category['confidence_threshold']}")
            if category.get("generated_terms", 0) > 0:
                self.console.print(f"   Generated Terms: {category['generated_terms']}")
        else:
            self.console.print(f"[red]❌ {results['error']}[/red]")
    
    def _format_delete_output(self, results: Dict[str, Any]) -> None:
        """Format category deletion output."""
        if results["success"]:
            self.console.print(f"[green]✅ Successfully deleted category: {results['deleted_category']}[/green]")
        else:
            self.console.print(f"[red]❌ {results['error']}[/red]")
    
    def _format_update_output(self, results: Dict[str, Any]) -> None:
        """Format category update output."""
        if results["success"]:
            self.console.print(f"[green]✅ Successfully updated category: {results['updated_category']}[/green]")
            for key, value in results["updates"].items():
                if key == "search_terms":
                    self.console.print(f"   {key}: {', '.join(value)}")
                else:
                    self.console.print(f"   {key}: {value}")
        else:
            self.console.print(f"[red]❌ {results['error']}[/red]")
    
    def _format_train_output(self, results: Dict[str, Any]) -> None:
        """Format category training output."""
        if results["success"]:
            self.console.print(f"[green]✅ Successfully trained category: {results['trained_category']}[/green]")
            self.console.print(f"   Source Label: {results['source_label']}")
            self.console.print(f"   Training Emails: {results['training_emails']}")
            self.console.print(f"   Extracted Terms: {results['extracted_terms']}")
            self.console.print(f"   Terms: {', '.join(results['terms'])}")
        else:
            self.console.print(f"[red]❌ {results['error']}[/red]")
    
    def _format_export_output(self, results: Dict[str, Any]) -> None:
        """Format category export output."""
        if results["success"]:
            self.console.print(f"[green]✅ Successfully exported {results['categories_exported']} categories[/green]")
            self.console.print(f"   File: {results['exported_file']}")
            self.console.print(f"   Format: {results['format']}")
        else:
            self.console.print(f"[red]❌ {results['error']}[/red]")
    
    def _format_import_output(self, results: Dict[str, Any]) -> None:
        """Format category import output."""
        if results["success"]:
            if results["dry_run"]:
                self.console.print("[yellow]📋 Import Preview (DRY RUN)[/yellow]")
                self.console.print(f"   File: {results['import_file']}")
                self.console.print(f"   Total Categories: {results['total_categories']}")
                self.console.print(f"   Import Actions: {results['import_actions']}")
                if results["conflicts"]:
                    self.console.print(f"   Conflicts: {results['conflicts']} (use --overwrite to replace)")
                    for name in results["conflict_names"][:5]:
                        self.console.print(f"     • {name}")
                self.console.print("\n[blue]Run with --apply to execute import[/blue]")
            else:
                self.console.print(f"[green]✅ Successfully imported categories[/green]")
                self.console.print(f"   Successful: {results['successful_imports']}")
                if results["failed_imports"] > 0:
                    self.console.print(f"   Failed: {results['failed_imports']}")
                    for failure in results["failures"][:3]:
                        self.console.print(f"     • {failure['name']}: {failure['error']}")
        else:
            self.console.print(f"[red]❌ {results['error']}[/red]")


@app.command()
def list():
    """List all custom categories."""
    
    @run_async_command
    async def run():
        manager = CategoryManager()
        await manager.initialize()
        
        results = await manager.list_categories()
        manager.format_output(results, "list")
        return manager
    
    run()


@app.command()
def create(
    name: str = typer.Argument(..., help="Category name"),
    description: str = typer.Option("", "--description", "-d", help="Category description"),
    search_terms: Optional[str] = typer.Option(None, "--terms", "-t", help="Comma-separated search terms"),
    confidence: float = typer.Option(0.7, "--confidence", "-c", help="Classification confidence threshold"),
    no_generate: bool = typer.Option(False, "--no-generate", help="Skip AI term generation")
):
    """Create a new custom category."""
    
    @run_async_command
    async def run():
        manager = CategoryManager()
        await manager.initialize()
        
        terms_list = None
        if search_terms:
            terms_list = [term.strip() for term in search_terms.split(',')]
        
        results = await manager.create_category(
            name=name,
            description=description,
            search_terms=terms_list,
            confidence_threshold=confidence,
            generate_terms=not no_generate
        )
        manager.format_output(results, "create")
        return manager
    
    run()


@app.command()
def delete(
    name: str = typer.Argument(..., help="Category name to delete")
):
    """Delete a custom category."""
    
    @run_async_command
    async def run():
        manager = CategoryManager()
        await manager.initialize()
        
        results = await manager.delete_category(name)
        manager.format_output(results, "delete")
        return manager
    
    run()


@app.command()
def update(
    name: str = typer.Argument(..., help="Category name to update"),
    description: Optional[str] = typer.Option(None, "--description", "-d", help="New description"),
    search_terms: Optional[str] = typer.Option(None, "--terms", "-t", help="New search terms (comma-separated)"),
    confidence: Optional[float] = typer.Option(None, "--confidence", "-c", help="New confidence threshold"),
    regenerate_terms: bool = typer.Option(False, "--regenerate", help="Regenerate search terms with AI")
):
    """Update an existing category."""
    
    @run_async_command
    async def run():
        manager = CategoryManager()
        await manager.initialize()
        
        terms_list = None
        if search_terms:
            terms_list = [term.strip() for term in search_terms.split(',')]
        
        results = await manager.update_category(
            name=name,
            description=description,
            search_terms=terms_list,
            confidence_threshold=confidence,
            regenerate_terms=regenerate_terms
        )
        manager.format_output(results, "update")
        return manager
    
    run()


@app.command()
def train(
    name: str = typer.Argument(..., help="Category name to train"),
    source_label: str = typer.Argument(..., help="Gmail label to learn from"),
    confidence: float = typer.Option(0.8, "--confidence", "-c", help="Minimum confidence for training"),
    sample_size: int = typer.Option(50, "--sample-size", "-s", help="Number of emails to analyze")
):
    """Train a category from existing Gmail labels."""
    
    @run_async_command
    async def run():
        manager = CategoryManager()
        await manager.initialize()
        
        results = await manager.train_category(
            name=name,
            source_label=source_label,
            min_confidence=confidence,
            sample_size=sample_size
        )
        manager.format_output(results, "train")
        return manager
    
    run()


@app.command()
def export(
    file_path: str = typer.Argument(..., help="Export file path"),
    format: str = typer.Option("yaml", "--format", "-f", help="Export format (yaml or json)")
):
    """Export categories to a file."""
    
    @run_async_command
    async def run():
        manager = CategoryManager()
        await manager.initialize()
        
        results = await manager.export_categories(file_path, format)
        manager.format_output(results, "export")
        return manager
    
    run()


@app.command(name="import")
def import_categories(
    file_path: str = typer.Argument(..., help="Import file path"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing categories"),
    dry_run: bool = typer.Option(True, "--dry-run/--apply", help="Preview import without applying")
):
    """Import categories from a file."""
    
    @run_async_command
    async def run():
        manager = CategoryManager()
        await manager.initialize()
        
        results = await manager.import_categories(
            file_path=file_path,
            overwrite=overwrite,
            dry_run=dry_run
        )
        manager.format_output(results, "import")
        return manager
    
    run()


if __name__ == "__main__":
    app()