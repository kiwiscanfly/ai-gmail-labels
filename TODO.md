# Unified Email CLI Tool Development Plan

## Overview
Create a comprehensive, unified CLI tool that consolidates all email labeling functionality from the existing scripts (`label_emails_cli.py`, `label_marketing_emails.py`, `label_priority_emails.py`, and `label_receipt_emails.py`) into a single, intelligent system using LangChain with a tiered command structure.

## Architecture Integration

### Current System Analysis
Based on ARCHITECTURE.md, the system follows a layered architecture:
- **Models Layer**: Domain models and data structures
- **Core Layer**: Infrastructure (config, database, event bus, state management)
- **Services Layer**: Business logic (EmailService, EmailPrioritizer, MarketingClassifier, ReceiptClassifier)
- **Integrations Layer**: External APIs (GmailClient, OllamaClient)
- **Agents Layer**: Multi-agent orchestration using LangGraph

### New CLI Architecture
The unified CLI will add a dedicated CLI layer following the existing architecture patterns:

```
src/
├── cli/                     # NEW: CLI-specific code
│   ├── __init__.py         # CLI module exports
│   ├── base.py             # Base CLI components and utilities
│   ├── commands/           # Command implementations
│   │   ├── __init__.py     # Command exports
│   │   ├── label/          # Label command group
│   │   │   ├── __init__.py
│   │   │   ├── base.py     # Base label commands
│   │   │   ├── custom.py   # Custom labeling commands
│   │   │   ├── marketing.py # Marketing labeling commands
│   │   │   ├── priority.py # Priority labeling commands
│   │   │   ├── receipt.py  # Receipt labeling commands
│   │   │   └── unified.py  # Unified labeling commands
│   │   ├── manage/         # Management command group
│   │   │   ├── __init__.py
│   │   │   ├── labels.py   # Label management
│   │   │   ├── categories.py # Category management
│   │   │   └── stats.py    # Statistics and analytics
│   │   └── system/         # System command group
│   │       ├── __init__.py
│   │       ├── status.py   # System status checks
│   │       ├── config.py   # Configuration management
│   │       └── monitor.py  # Monitoring commands
│   ├── parsers/            # Command line argument parsers
│   │   ├── __init__.py
│   │   ├── base.py         # Base parser utilities
│   │   ├── email_selectors.py # Email selection parsing
│   │   └── label_options.py   # Label option parsing
│   ├── formatters/         # Output formatting
│   │   ├── __init__.py
│   │   ├── tables.py       # Table formatting
│   │   ├── progress.py     # Progress indicators
│   │   └── reports.py      # Report generation
│   └── langchain/          # LangChain integration
│       ├── __init__.py
│       ├── chains.py       # Custom LangChain chains
│       ├── agents.py       # CLI-specific agents
│       └── prompts.py      # CLI-specific prompts
├── models/                 # [EXISTING] Domain models
├── core/                   # [EXISTING] Core infrastructure
├── services/               # [EXISTING] Business logic
├── integrations/           # [EXISTING] External APIs
├── agents/                 # [EXISTING] Multi-agent orchestration
└── cli.py                  # [MODIFIED] Main CLI entry point
```

## Tiered CLI Command Structure

### Main Command: `email-agent`

#### Tier 1: Primary Command Groups
```bash
email-agent label      # Email labeling operations
email-agent manage     # Label and category management
email-agent system     # System operations and monitoring
```

#### Tier 2: Command Categories

##### Label Commands (`email-agent label`)
```bash
# Specific classification types
email-agent label priority    # Priority classification only
email-agent label marketing   # Marketing classification only  
email-agent label receipt     # Receipt classification only
email-agent label custom      # Custom label classification

# Unified operations
email-agent label all         # All classification types
email-agent label auto        # Intelligent auto-classification
```

##### Management Commands (`email-agent manage`)
```bash
email-agent manage labels     # Gmail label management
email-agent manage categories # Custom category management
email-agent manage stats      # Analytics and reporting
```

##### System Commands (`email-agent system`)
```bash
email-agent system status     # Health checks and status
email-agent system config     # Configuration management
email-agent system monitor    # Real-time monitoring
```

#### Tier 3: Specific Operations and Options

##### Priority Labeling (`email-agent label priority`)
```bash
# Basic usage
email-agent label priority --unread              # Process unread emails
email-agent label priority --recent 7days        # Last 7 days

# Advanced options
email-agent label priority --query "is:important" --dry-run
email-agent label priority --batch-size 20 --parallel 3
```

##### Marketing Labeling (`email-agent label marketing`)
```bash
# Basic usage
email-agent label marketing --unread             # Process unread emails
email-agent label marketing --all                # All emails (use cautiously)

# Subtype specific
email-agent label marketing --types promotional,newsletter
```

##### Receipt Labeling (`email-agent label receipt`)
```bash
# Basic usage
email-agent label receipt --unread               # Process unread emails
```

##### Custom Labeling (`email-agent label custom`)
```bash
# AI-generated search terms (generates terms by default)
email-agent label custom "programming"
email-agent label custom "finance"

# Manual search terms
email-agent label custom "work" --search-terms "project,meeting,deadline"

# Advanced custom labeling
email-agent label custom "travel" \
  --generate-terms \
  --search-existing \
  --parent-label "Personal/Travel" \
  --confidence-threshold 0.8
```

##### Unified Operations (`email-agent label all`)
```bash
# Process all classification types
email-agent label all --unread                   # All types on unread emails
email-agent label all --recent 30days

# Intelligent processing
email-agent label auto --unread                  # AI decides which classifiers to use
email-agent label auto --learn-from-existing     # Learn from current labels
```

##### Label Management (`email-agent manage labels`)
```bash
# List and analyze labels
email-agent manage labels list                   # All labels with stats
email-agent manage labels analyze --usage        # Usage statistics
email-agent manage labels cleanup --dry-run      # Find unused labels

# Create and modify labels
email-agent manage labels create "Projects/AI Research"
email-agent manage labels rename "Old Label" "New Label"
email-agent manage labels delete "Unused Label" --force
```

##### Category Management (`email-agent manage categories`)
```bash
# Custom category operations
email-agent manage categories list               # All custom categories
email-agent manage categories create "programming" \
  --description "Software development emails" \
  --generate-terms

# Train from existing data
email-agent manage categories train "work" \
  --from-label "Projects/*" \
  --confidence-threshold 0.8

# Export/import categories
email-agent manage categories export categories.yaml
email-agent manage categories import categories.yaml
```

##### Analytics (`email-agent manage stats`)
```bash
# Performance and usage statistics
email-agent manage stats classification --last-30-days
email-agent manage stats labels --distribution
email-agent manage stats performance --export-csv results.csv

# Sender analysis
email-agent manage stats senders --marketing-rate
email-agent manage stats senders --top-volume --limit 20
```

## Command Implementation Architecture

### Base CLI Framework (`src/cli/base.py`)

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import typer
from rich.console import Console
from langchain.schema import BaseMessage

class BaseCLICommand(ABC):
    """Base class for all CLI commands following the service layer pattern"""
    
    def __init__(self):
        self.console = Console()
        self.email_service = None
        self.initialized = False
    
    async def initialize(self):
        """Initialize services following the architecture pattern"""
        if not self.initialized:
            self.email_service = EmailService()
            await self.email_service.initialize()
            self.initialized = True
    
    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the command logic"""
        pass
    
    def format_output(self, results: Dict[str, Any]) -> None:
        """Format and display results using rich console"""
        pass

class BaseEmailProcessor(BaseCLICommand):
    """Base class for email processing commands"""
    
    def __init__(self):
        super().__init__()
        self.progress = None
        
    async def process_emails(
        self, 
        target: str,
        limit: Optional[int] = None,
        dry_run: bool = True
    ) -> List[Any]:
        """Process emails using the email service layer"""
        pass
```

### LangChain Integration (`src/cli/langchain/`)

#### Custom CLI Chains (`src/cli/langchain/chains.py`)

```python
from langchain.chains import LLMChain
from langchain.schema import BasePromptTemplate
from langchain.callbacks.base import BaseCallbackHandler

class EmailRouterChain(LLMChain):
    """Routes emails to appropriate classification services"""
    
    async def route_email(self, email: EmailMessage) -> List[str]:
        """Determines which classification services should process this email"""
        routing_prompt = """
        Analyze this email and determine which classification services should process it:
        
        Email Subject: {subject}
        Email Sender: {sender}
        Email Snippet: {snippet}
        
        Available services:
        - priority: Assigns priority levels (critical/high/medium/low)
        - marketing: Detects marketing emails (promotional/newsletter/hybrid)
        - receipt: Identifies receipts (purchase/subscription/service)
        - custom: Custom user-defined categories
        
        Return only the service names that should process this email, separated by commas.
        """
        
        result = await self.arun(
            subject=email.subject,
            sender=email.sender,
            snippet=email.snippet[:200]
        )
        
        return [service.strip() for service in result.split(',')]

class CustomLabelChain(LLMChain):
    """Generates search terms and applies custom labels"""
    
    async def generate_search_terms(self, category: str, context: str = None) -> List[str]:
        """Generates relevant search terms for a custom category"""
        generation_prompt = """
        Generate 5-10 relevant search terms for finding emails related to "{category}".
        
        Category: {category}
        Context: {context}
        
        Consider:
        - Synonyms and related terms
        - Common email patterns
        - Professional and casual language
        - Acronyms and abbreviations
        
        Return terms separated by commas, most relevant first.
        """
        
        result = await self.arun(category=category, context=context or "")
        return [term.strip() for term in result.split(',')]
    
    async def classify_for_category(
        self, 
        email: EmailMessage, 
        category: str, 
        search_terms: List[str]
    ) -> Dict[str, Any]:
        """Classifies email for a specific custom category"""
        classification_prompt = """
        Determine if this email belongs to the "{category}" category.
        
        Email Subject: {subject}
        Email Sender: {sender}
        Email Content: {content}
        
        Category: {category}
        Related Terms: {search_terms}
        
        Provide:
        1. Classification: yes/no
        2. Confidence: 0.0-1.0
        3. Reasoning: brief explanation
        4. Suggested labels: specific labels to apply
        
        Format as JSON.
        """
        
        result = await self.arun(
            category=category,
            subject=email.subject,
            sender=email.sender,
            content=email.content[:500],
            search_terms=", ".join(search_terms)
        )
        
        return json.loads(result)
```

#### CLI-Specific Agents (`src/cli/langchain/agents.py`)

```python
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.tools import Tool

class EmailAnalysisAgent:
    """Agent for analyzing email patterns and providing insights"""
    
    def __init__(self, llm):
        self.llm = llm
        self.tools = self._create_tools()
        self.agent = create_structured_chat_agent(llm, self.tools)
        self.executor = AgentExecutor(agent=self.agent, tools=self.tools)
    
    def _create_tools(self) -> List[Tool]:
        """Create tools for email analysis"""
        return [
            Tool(
                name="analyze_sender_patterns",
                description="Analyze email patterns from a specific sender",
                func=self._analyze_sender_patterns
            ),
            Tool(
                name="suggest_custom_categories",
                description="Suggest custom categories based on email content",
                func=self._suggest_custom_categories
            ),
            Tool(
                name="optimize_search_query",
                description="Optimize Gmail search queries for better results",
                func=self._optimize_search_query
            )
        ]
    
    async def analyze_email_collection(self, emails: List[EmailMessage]) -> Dict[str, Any]:
        """Analyze a collection of emails and provide insights"""
        analysis_query = f"""
        Analyze these {len(emails)} emails and provide insights about:
        1. Common themes and categories
        2. Sender patterns and behavior
        3. Suggested custom labels
        4. Optimization recommendations
        
        Email subjects: {[email.subject for email in emails[:10]]}
        """
        
        result = await self.executor.arun(analysis_query)
        return result
```

### Command Implementations

#### Priority Labeling (`src/cli/commands/label/priority.py`)

```python
import typer
from typing import Optional
from src.cli.base import BaseEmailProcessor
from src.services.email_prioritizer import EmailPrioritizer

app = typer.Typer(help="Priority-based email labeling commands")

class PriorityLabelCommand(BaseEmailProcessor):
    """Priority labeling command implementation"""
    
    def __init__(self):
        super().__init__()
        self.prioritizer = None
    
    async def initialize(self):
        """Initialize priority labeling services"""
        await super().initialize()
        self.prioritizer = EmailPrioritizer()
        await self.prioritizer.initialize()
    
    async def execute(
        self,
        target: str = "unread",
        confidence_threshold: float = 0.7,
        dry_run: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute priority labeling"""
        
        # Convert target to Gmail query
        query = self._parse_target(target)
        
        # Process emails
        emails = await self.process_emails(query, kwargs.get('limit'), dry_run)
        
        results = []
        for email in emails:
            priority_result = await self.prioritizer.analyze_priority(email)
            
            if priority_result.confidence >= confidence_threshold:
                label_name = f"Priority/{priority_result.level.title()}"
                
                if not dry_run:
                    await self.email_service.apply_label_by_name(
                        email.id, label_name, create_if_missing=True
                    )
                
                results.append({
                    "email_id": email.id,
                    "subject": email.subject,
                    "priority": priority_result.level,
                    "confidence": priority_result.confidence,
                    "label_applied": label_name if not dry_run else None,
                    "label_would_apply": label_name if dry_run else None
                })
        
        return {
            "processed": len(results),
            "results": results,
            "dry_run": dry_run
        }

@app.command()
def unread(
    confidence_threshold: float = typer.Option(0.7, "--confidence-threshold", "-c"),
    dry_run: bool = typer.Option(True, "--dry-run/--apply"),
    limit: Optional[int] = typer.Option(None, "--limit", "-l"),
    batch_size: int = typer.Option(10, "--batch-size"),
    parallel: int = typer.Option(3, "--parallel")
):
    """Label unread emails with priority levels"""
    import asyncio
    
    async def run():
        command = PriorityLabelCommand()
        await command.initialize()
        
        results = await command.execute(
            target="unread",
            confidence_threshold=confidence_threshold,
            dry_run=dry_run,
            limit=limit,
            batch_size=batch_size,
            parallel=parallel
        )
        
        command.format_output(results)
    
    asyncio.run(run())

@app.command()
def recent(
    days: int = typer.Argument(7, help="Number of days to look back"),
    confidence_threshold: float = typer.Option(0.7, "--confidence-threshold", "-c"),
    dry_run: bool = typer.Option(True, "--dry-run/--apply"),
    limit: Optional[int] = typer.Option(None, "--limit", "-l")
):
    """Label recent emails with priority levels"""
    import asyncio
    
    async def run():
        command = PriorityLabelCommand()
        await command.initialize()
        
        results = await command.execute(
            target=f"recent:{days}days",
            confidence_threshold=confidence_threshold,
            dry_run=dry_run,
            limit=limit
        )
        
        command.format_output(results)
    
    asyncio.run(run())

@app.command()
def custom_query(
    query: str = typer.Argument(..., help="Gmail search query"),
    confidence_threshold: float = typer.Option(0.7, "--confidence-threshold", "-c"),
    dry_run: bool = typer.Option(True, "--dry-run/--apply"),
    limit: Optional[int] = typer.Option(None, "--limit", "-l")
):
    """Label emails matching custom query with priority levels"""
    import asyncio
    
    async def run():
        command = PriorityLabelCommand()
        await command.initialize()
        
        results = await command.execute(
            target=f"query:{query}",
            confidence_threshold=confidence_threshold,
            dry_run=dry_run,
            limit=limit
        )
        
        command.format_output(results)
    
    asyncio.run(run())
```

#### Custom Labeling (`src/cli/commands/label/custom.py`)

```python
import typer
from typing import Optional, List
from src.cli.base import BaseEmailProcessor
from src.cli.langchain.chains import CustomLabelChain
from src.integrations.ollama_client import get_ollama_manager

app = typer.Typer(help="Custom label classification commands")

class CustomLabelCommand(BaseEmailProcessor):
    """Custom labeling command with AI-generated search terms"""
    
    def __init__(self):
        super().__init__()
        self.label_chain = None
        self.ollama_manager = None
    
    async def initialize(self):
        """Initialize custom labeling services"""
        await super().initialize()
        self.ollama_manager = await get_ollama_manager()
        self.label_chain = CustomLabelChain(llm=self.ollama_manager)
    
    async def execute(
        self,
        category: str,
        generate_terms: bool = False,
        search_terms: Optional[List[str]] = None,
        search_existing: bool = False,
        confidence_threshold: float = 0.8,
        parent_label: Optional[str] = None,
        dry_run: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute custom labeling"""
        
        # Generate or use provided search terms
        if generate_terms:
            self.console.print(f"[blue]Generating search terms for '{category}'...[/blue]")
            terms = await self.label_chain.generate_search_terms(category)
            self.console.print(f"[green]Generated terms: {', '.join(terms)}[/green]")
        else:
            terms = search_terms or [category]
        
        # Search for emails
        if search_existing:
            # Search entire mailbox with generated terms
            query_parts = [f'"{term}"' for term in terms]
            query = " OR ".join(query_parts)
        else:
            # Default to unread emails
            query = "is:unread"
        
        # Process emails
        emails = await self.process_emails(query, kwargs.get('limit'), dry_run)
        
        results = []
        label_name = parent_label or category
        
        for email in emails:
            classification = await self.label_chain.classify_for_category(
                email, category, terms
            )
            
            if classification.get('confidence', 0) >= confidence_threshold:
                if not dry_run:
                    await self.email_service.apply_label_by_name(
                        email.id, label_name, create_if_missing=True
                    )
                
                results.append({
                    "email_id": email.id,
                    "subject": email.subject,
                    "category": category,
                    "confidence": classification.get('confidence'),
                    "reasoning": classification.get('reasoning'),
                    "suggested_labels": classification.get('suggested_labels', []),
                    "label_applied": label_name if not dry_run else None,
                    "label_would_apply": label_name if dry_run else None
                })
        
        return {
            "category": category,
            "search_terms": terms,
            "processed": len(results),
            "results": results,
            "dry_run": dry_run
        }

@app.command()
def create(
    category: str = typer.Argument(..., help="Category name for custom labeling"),
    search_terms: Optional[str] = typer.Option(None, "--search-terms", "-s", help="Comma-separated search terms"),
    search_existing: bool = typer.Option(False, "--search-existing", help="Search entire mailbox, not just unread"),
    parent_label: Optional[str] = typer.Option(None, "--parent-label", "-p"),
    dry_run: bool = typer.Option(True, "--dry-run/--apply"),
    limit: Optional[int] = typer.Option(None, "--limit", "-l")
):
    """Create and apply custom category labels"""
    import asyncio
    
    async def run():
        command = CustomLabelCommand()
        await command.initialize()
        
        # Parse search terms if provided
        terms_list = None
        if search_terms:
            terms_list = [term.strip() for term in search_terms.split(',')]
        
        results = await command.execute(
            category=category,
            search_terms=terms_list,
            search_existing=search_existing,
            confidence_threshold=confidence_threshold,
            parent_label=parent_label,
            dry_run=dry_run,
            limit=limit
        )
        
        command.format_output(results)
    
    asyncio.run(run())

@app.command()
def analyze(
    category: str = typer.Argument(..., help="Category to analyze"),
    sample_size: int = typer.Option(50, "--sample-size", "-s", help="Number of emails to analyze")
):
    """Analyze emails to suggest improvements for custom category"""
    import asyncio
    
    async def run():
        command = CustomLabelCommand()
        await command.initialize()
        
        # Search for emails with existing label
        query = f"label:{category}"
        emails = await command.process_emails(query, sample_size, dry_run=True)
        
        if not emails:
            command.console.print(f"[yellow]No emails found with label '{category}'[/yellow]")
            return
        
        # Analyze patterns
        command.console.print(f"[blue]Analyzing {len(emails)} emails for category '{category}'...[/blue]")
        
        # Use LangChain agent for analysis
        from src.cli.langchain.agents import EmailAnalysisAgent
        analysis_agent = EmailAnalysisAgent(command.ollama_manager)
        
        insights = await analysis_agent.analyze_email_collection(emails)
        
        command.console.print("[green]Analysis Results:[/green]")
        command.console.print(insights)
    
    asyncio.run(run())
```

#### Unified Operations (`src/cli/commands/label/unified.py`)

```python
import typer
from typing import Optional, List
from src.cli.base import BaseEmailProcessor
from src.cli.langchain.chains import EmailRouterChain
from src.services.email_prioritizer import EmailPrioritizer
from src.services.marketing_classifier import MarketingEmailClassifier  
from src.services.receipt_classifier import ReceiptClassifier

app = typer.Typer(help="Unified email labeling commands")

class UnifiedLabelCommand(BaseEmailProcessor):
    """Unified labeling that intelligently applies all classification types"""
    
    def __init__(self):
        super().__init__()
        self.router_chain = None
        self.prioritizer = None
        self.marketing_classifier = None
        self.receipt_classifier = None
    
    async def initialize(self):
        """Initialize all classification services"""
        await super().initialize()
        
        # Initialize classifiers
        self.prioritizer = EmailPrioritizer()
        await self.prioritizer.initialize()
        
        self.marketing_classifier = MarketingEmailClassifier()
        await self.marketing_classifier.initialize()
        
        self.receipt_classifier = ReceiptClassifier()
        await self.receipt_classifier.initialize()
        
        # Initialize router chain
        from src.integrations.ollama_client import get_ollama_manager
        ollama_manager = await get_ollama_manager()
        self.router_chain = EmailRouterChain(llm=ollama_manager)
    
    async def execute(
        self,
        target: str = "unread",
        classification_types: Optional[List[str]] = None,
        intelligent_routing: bool = True,
        confidence_threshold: float = 0.7,
        dry_run: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute unified classification"""
        
        # Default to all types if not specified
        if classification_types is None:
            classification_types = ["priority", "marketing", "receipt"]
        
        # Convert target to Gmail query
        query = self._parse_target(target)
        
        # Process emails
        emails = await self.process_emails(query, kwargs.get('limit'), dry_run)
        
        results = []
        
        for email in emails:
            email_result = {
                "email_id": email.id,
                "subject": email.subject,
                "sender": email.sender,
                "classifications": {},
                "labels_applied": [],
                "labels_would_apply": []
            }
            
            # Determine which classifiers to use
            if intelligent_routing:
                applicable_types = await self.router_chain.route_email(email)
                # Intersect with requested types
                types_to_use = list(set(applicable_types) & set(classification_types))
            else:
                types_to_use = classification_types
            
            # Apply each classification type
            if "priority" in types_to_use:
                priority_result = await self.prioritizer.analyze_priority(email)
                email_result["classifications"]["priority"] = {
                    "level": priority_result.level,
                    "confidence": priority_result.confidence,
                    "reasoning": priority_result.reasoning
                }
                
                if priority_result.confidence >= confidence_threshold:
                    label = f"Priority/{priority_result.level.title()}"
                    if not dry_run:
                        await self.email_service.apply_label_by_name(
                            email.id, label, create_if_missing=True
                        )
                        email_result["labels_applied"].append(label)
                    else:
                        email_result["labels_would_apply"].append(label)
            
            if "marketing" in types_to_use:
                marketing_result = await self.marketing_classifier.classify_email(email)
                email_result["classifications"]["marketing"] = {
                    "is_marketing": marketing_result.is_marketing,
                    "subtype": marketing_result.subtype,
                    "confidence": marketing_result.confidence,
                    "reasoning": marketing_result.reasoning
                }
                
                if marketing_result.is_marketing and marketing_result.confidence >= confidence_threshold:
                    label = f"Marketing/{marketing_result.subtype.title()}"
                    if not dry_run:
                        await self.email_service.apply_label_by_name(
                            email.id, label, create_if_missing=True
                        )
                        email_result["labels_applied"].append(label)
                    else:
                        email_result["labels_would_apply"].append(label)
            
            if "receipt" in types_to_use:
                receipt_result = await self.receipt_classifier.classify_receipt(email)
                email_result["classifications"]["receipt"] = {
                    "is_receipt": receipt_result.is_receipt,
                    "receipt_type": receipt_result.receipt_type,
                    "confidence": receipt_result.confidence,
                    "vendor": receipt_result.vendor
                }
                
                if receipt_result.is_receipt and receipt_result.confidence >= confidence_threshold:
                    label = f"Receipts/{receipt_result.receipt_type.title()}"
                    if not dry_run:
                        await self.email_service.apply_label_by_name(
                            email.id, label, create_if_missing=True
                        )
                        email_result["labels_applied"].append(label)
                    else:
                        email_result["labels_would_apply"].append(label)
            
            results.append(email_result)
        
        return {
            "processed": len(results),
            "classification_types": types_to_use,
            "intelligent_routing": intelligent_routing,
            "results": results,
            "dry_run": dry_run
        }

@app.command()
def all(
    target: str = typer.Option("unread", "--target", "-t", help="Email target (unread/all/recent:Ndays)"),
    types: Optional[str] = typer.Option(None, "--types", help="Classification types (comma-separated)"),
    confidence_threshold: float = typer.Option(0.7, "--confidence-threshold", "-c"),
    dry_run: bool = typer.Option(True, "--dry-run/--apply"),
    limit: Optional[int] = typer.Option(None, "--limit", "-l")
):
    """Apply all classification types to emails"""
    import asyncio
    
    async def run():
        command = UnifiedLabelCommand()
        await command.initialize()
        
        # Parse types
        classification_types = None
        if types:
            classification_types = [t.strip() for t in types.split(',')]
        
        results = await command.execute(
            target=target,
            classification_types=classification_types,
            intelligent_routing=False,
            confidence_threshold=confidence_threshold,
            dry_run=dry_run,
            limit=limit
        )
        
        command.format_output(results)
    
    asyncio.run(run())

@app.command()
def auto(
    target: str = typer.Option("unread", "--target", "-t"),
    confidence_threshold: float = typer.Option(0.7, "--confidence-threshold", "-c"),
    learn_from_existing: bool = typer.Option(False, "--learn-from-existing"),
    dry_run: bool = typer.Option(True, "--dry-run/--apply"),
    limit: Optional[int] = typer.Option(None, "--limit", "-l")
):
    """Intelligent auto-classification with AI routing"""
    import asyncio
    
    async def run():
        command = UnifiedLabelCommand()
        await command.initialize()
        
        results = await command.execute(
            target=target,
            classification_types=["priority", "marketing", "receipt"],
            intelligent_routing=True,
            confidence_threshold=confidence_threshold,
            dry_run=dry_run,
            limit=limit
        )
        
        command.format_output(results)
    
    asyncio.run(run())
```

## Main CLI Entry Point Modifications

### Updated `src/cli.py`

```python
import typer
from rich.console import Console

# Import command groups
from src.cli.commands.label import priority, marketing, receipt, custom, unified
from src.cli.commands.manage import labels, categories, stats  
from src.cli.commands.system import status, config, monitor

app = typer.Typer(help="Email Categorization Agent - Unified CLI")
console = Console()

# Add command groups
label_app = typer.Typer(help="Email labeling operations")
manage_app = typer.Typer(help="Label and category management")
system_app = typer.Typer(help="System operations and monitoring")

# Mount label subcommands
label_app.add_typer(priority.app, name="priority", help="Priority classification")
label_app.add_typer(marketing.app, name="marketing", help="Marketing classification")
label_app.add_typer(receipt.app, name="receipt", help="Receipt classification")
label_app.add_typer(custom.app, name="custom", help="Custom labeling")
label_app.add_typer(unified.app, name="all", help="Unified operations")
label_app.add_typer(unified.app, name="auto", help="Intelligent auto-classification")

# Mount management subcommands
manage_app.add_typer(labels.app, name="labels", help="Gmail label management")
manage_app.add_typer(categories.app, name="categories", help="Custom category management")
manage_app.add_typer(stats.app, name="stats", help="Analytics and reporting")

# Mount system subcommands
system_app.add_typer(status.app, name="status", help="System status and health")
system_app.add_typer(config.app, name="config", help="Configuration management")
system_app.add_typer(monitor.app, name="monitor", help="Real-time monitoring")

# Add to main app
app.add_typer(label_app, name="label", help="Email labeling operations")
app.add_typer(manage_app, name="manage", help="Management operations")
app.add_typer(system_app, name="system", help="System operations")

# Legacy command compatibility
@app.command(hidden=True)
def label_emails():
    """Legacy compatibility - redirects to new unified command"""
    console.print("[yellow]This command has been replaced by 'email-agent label all'[/yellow]")
    console.print("Use 'email-agent label --help' for available options")

def main():
    """Main entry point"""
    app()

if __name__ == "__main__":
    main()
```

## Migration and Legacy Compatibility

Ensure the MCP server continues to work without errors.

### Legacy Script Mapping

```bash
# Old command -> New command
python label_priority_emails.py --dry-run
# becomes
email-agent label priority --unread --dry-run

python label_marketing_emails.py --limit 100 --detailed
# becomes  
email-agent label marketing --unread --limit 100 --verbose

python label_receipt_emails.py --vendor-stats
# becomes
email-agent label receipt --unread --vendor-stats

python label_emails_cli.py label-unread --types priority,marketing --apply
# becomes
email-agent label all --target unread --types priority,marketing --apply
```

## Success Metrics and Testing

### Functionality Metrics
- [ ] All existing CLI functionality replicated
- [ ] Tiered command structure implemented
- [ ] LangChain integration working
- [ ] Custom label generation functional
- [ ] Intelligent routing operational

### Performance Metrics  
- [ ] Command execution time < 2 seconds for startup
- [ ] Email processing speed matches existing tools
- [ ] Memory usage optimized with service layer
- [ ] Parallel processing implemented

### User Experience Metrics
- [ ] Intuitive command hierarchy
- [ ] Comprehensive help documentation
- [ ] Rich console output with progress indicators
- [ ] Error handling and recovery
- [ ] Backward compatibility maintained

This unified CLI tool will provide a comprehensive, intelligent, and user-friendly interface for email management while maintaining the architectural principles and performance characteristics of the existing system.