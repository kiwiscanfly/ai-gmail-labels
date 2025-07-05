# Product Requirements Document: Agent-Based Email Categorization Workflow System

## Executive Summary

This PRD defines a Python-based agent workflow system for Gmail email categorization using local Ollama models. The system employs a multi-agent architecture with specialized agents coordinated by an orchestration agent, providing both automatic and interactive modes for email categorization. Built on modern agent frameworks, the system emphasizes privacy through local LLM processing, modular design for easy model swapping, and seamless integration through the Model Context Protocol (MCP) for interaction with Claude and other MCP-compatible clients.

## 1. System Overview

### 1.1 Core Specifications
- **Language**: Python 3.13+ with type hints throughout
- **LLM Models**: Local Ollama deployment (gemma2:3b, llama3.2:3b)
- **Interface**: Model Context Protocol (MCP) server for integration with Claude
- **Operation Modes**: 
  - Automatic mode: Autonomous categorization and labeling
  - Interactive mode: User confirmation for ambiguous cases
- **Integration**: Native Gmail API integration, MCP server implementation
- **Architecture**: Event-driven multi-agent system with persistent state management

### 1.2 Key Features
- Leverages existing Gmail labels for categorization
- Smart categorization with fallback to user interaction
- Local-first approach for privacy and cost efficiency
- Long-running task capability with state persistence
- Modular agent design enabling easy component swapping
- Real-time email monitoring with push notifications

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CLI Interface (Typer + Rich)              │
├─────────────────────────────────────────────────────────────┤
│                  Orchestration Agent (LangGraph)             │
├─────────────┬───────────┬────────────┬─────────────────────┤
│  Email       │  Category  │  Gmail    │  User Interaction  │
│  Retrieval   │  Analysis  │  API      │  Agent             │
│  Agent       │  Agent     │  Agent    │                     │
├─────────────┴───────────┴────────────┴─────────────────────┤
│                 Event Bus (Redis/In-Memory)                  │
├─────────────────────────────────────────────────────────────┤
│         State Management (Redis + SQLite)                    │
├─────────────────────────────────────────────────────────────┤
│           Ollama Model Manager (gemma2:3b, llama3.2:3b)     │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Agent Roles and Responsibilities

**Orchestration Agent**
- Manages workflow execution and agent coordination
- Handles state transitions and error recovery
- Routes tasks to appropriate specialized agents
- Manages user interaction modes (automatic/interactive)

**Email Retrieval Agent**
- Monitors Gmail for new emails via push notifications
- Fetches email content and metadata efficiently
- Manages pagination and batch processing
- Handles attachment processing when needed

**Category Analysis Agent**  
- Analyzes email content using local LLMs
- Determines appropriate categories/labels
- Calculates confidence scores for categorization
- Triggers user interaction for low-confidence cases

**Gmail API Agent**
- Handles all Gmail API interactions
- Applies labels to emails
- Manages API quotas and rate limiting
- Implements retry logic with exponential backoff

**User Interaction Agent**
- Handles interaction flows through MCP protocol
- Presents categorization options for ambiguous emails
- Learns from user feedback to improve categorization
- Maintains conversation context and history
- Provides real-time status updates via MCP resources

### 2.3 Communication Architecture

**Event-Driven Message Passing with SQLite**
```python
import aiosqlite
import json
import asyncio
from typing import Dict, List, Callable, Optional
from datetime import datetime
from dataclasses import dataclass, field
import uuid

@dataclass
class AgentMessage:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender_agent: str
    recipient_agent: str
    message_type: str
    payload: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    correlation_id: Optional[str] = None
    status: str = "pending"  # pending, processing, completed, failed

class SQLiteEventBus:
    def __init__(self, db_path: str = "./data/eventbus.db"):
        self.db_path = db_path
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._polling_tasks = {}
        
    async def initialize(self):
        """Initialize the event bus database schema"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    sender_agent TEXT NOT NULL,
                    recipient_agent TEXT NOT NULL,
                    message_type TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    correlation_id TEXT,
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_recipient_status 
                ON messages(recipient_agent, status)
            """)
            
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_correlation 
                ON messages(correlation_id)
            """)
            
            await db.commit()
    
    async def publish(self, message: AgentMessage):
        """Publish a message to the event bus"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO messages 
                (id, sender_agent, recipient_agent, message_type, 
                 payload, timestamp, correlation_id, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                message.id,
                message.sender_agent,
                message.recipient_agent,
                message.message_type,
                json.dumps(message.payload),
                message.timestamp,
                message.correlation_id,
                message.status
            ))
            await db.commit()
        
        # Notify local subscribers immediately
        if message.recipient_agent in self.subscribers:
            for handler in self.subscribers[message.recipient_agent]:
                asyncio.create_task(handler(message))
    
    async def subscribe(self, agent_id: str, handler: Callable):
        """Subscribe to messages for a specific agent"""
        self.subscribers[agent_id].append(handler)
        
        # Start polling task for this agent if not already running
        if agent_id not in self._polling_tasks:
            task = asyncio.create_task(self._poll_messages(agent_id))
            self._polling_tasks[agent_id] = task
    
    async def _poll_messages(self, agent_id: str):
        """Poll for new messages for a specific agent"""
        while True:
            try:
                async with aiosqlite.connect(self.db_path) as db:
                    db.row_factory = aiosqlite.Row
                    
                    # Fetch pending messages
                    async with db.execute("""
                        SELECT * FROM messages 
                        WHERE recipient_agent = ? AND status = 'pending'
                        ORDER BY timestamp ASC
                        LIMIT 10
                    """, (agent_id,)) as cursor:
                        rows = await cursor.fetchall()
                    
                    for row in rows:
                        # Update status to processing
                        await db.execute("""
                            UPDATE messages 
                            SET status = 'processing', 
                                updated_at = CURRENT_TIMESTAMP
                            WHERE id = ?
                        """, (row['id'],))
                        await db.commit()
                        
                        # Reconstruct message
                        message = AgentMessage(
                            id=row['id'],
                            sender_agent=row['sender_agent'],
                            recipient_agent=row['recipient_agent'],
                            message_type=row['message_type'],
                            payload=json.loads(row['payload']),
                            timestamp=row['timestamp'],
                            correlation_id=row['correlation_id'],
                            status='processing'
                        )
                        
                        # Process message
                        for handler in self.subscribers[agent_id]:
                            try:
                                await handler(message)
                                
                                # Mark as completed
                                await db.execute("""
                                    UPDATE messages 
                                    SET status = 'completed', 
                                        updated_at = CURRENT_TIMESTAMP
                                    WHERE id = ?
                                """, (row['id'],))
                            except Exception as e:
                                # Mark as failed
                                await db.execute("""
                                    UPDATE messages 
                                    SET status = 'failed', 
                                        updated_at = CURRENT_TIMESTAMP
                                    WHERE id = ?
                                """, (row['id'],))
                                logger.error(f"Handler error: {e}")
                        
                        await db.commit()
                
                # Small delay between polls
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Polling error for agent {agent_id}: {e}")
                await asyncio.sleep(1)
    
    async def get_message_history(self, 
                                  agent_id: Optional[str] = None,
                                  correlation_id: Optional[str] = None,
                                  limit: int = 100) -> List[AgentMessage]:
        """Retrieve message history"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            
            query = "SELECT * FROM messages WHERE 1=1"
            params = []
            
            if agent_id:
                query += " AND (sender_agent = ? OR recipient_agent = ?)"
                params.extend([agent_id, agent_id])
            
            if correlation_id:
                query += " AND correlation_id = ?"
                params.append(correlation_id)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
            
            return [
                AgentMessage(
                    id=row['id'],
                    sender_agent=row['sender_agent'],
                    recipient_agent=row['recipient_agent'],
                    message_type=row['message_type'],
                    payload=json.loads(row['payload']),
                    timestamp=row['timestamp'],
                    correlation_id=row['correlation_id'],
                    status=row['status']
                )
                for row in rows
            ]
    
    async def cleanup_old_messages(self, days: int = 30):
        """Clean up old completed messages"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                DELETE FROM messages 
                WHERE status IN ('completed', 'failed') 
                AND created_at < datetime('now', '-' || ? || ' days')
            """, (days,))
            await db.commit()
```

## 3. Technical Stack

### 3.1 Core Dependencies

**Agent Framework**: LangGraph
- Stateful multi-agent workflows
- Built-in persistence and checkpointing
- Graph-based execution control
- Production-ready with tracing support

**LLM Integration**: Ollama Python Client
```python
# Model configuration
MODELS = {
    "categorization": "gemma2:3b",      # Primary categorization model
    "reasoning": "llama3.2:3b",         # Complex reasoning tasks
    "fallback": "gemma2:2b"             # Fast fallback model
}

# Dynamic model management
class OllamaModelManager:
    def __init__(self):
        self.client = ollama.AsyncClient()
        self.current_model = MODELS["categorization"]
    
    async def switch_model(self, task_type: str):
        self.current_model = MODELS.get(task_type, MODELS["categorization"])
        # Preload model for faster inference
        await self.client.generate(
            model=self.current_model,
            prompt="",
            keep_alive=True
        )
```

**MCP Integration**: Python MCP SDK
- Implements MCP server for tool exposure
- Resource providers for email status
- Streaming responses for real-time updates
- Session management for stateful interactions

**Gmail Integration**: Google API Python Client
- OAuth 2.0 authentication
- Push notifications via Cloud Pub/Sub
- Batch operations for efficiency
- Comprehensive error handling

**State Management**: SQLite
- Single SQLite database for all persistence needs
- Message queue implementation using SQLite tables
- Transaction support for consistency
- LangGraph checkpointing for workflow persistence

### 3.2 Additional Components

- **asyncio**: Concurrent task execution
- **pydantic**: Data validation and settings
- **python-dotenv**: Environment configuration
- **tenacity**: Retry mechanisms
- **structlog**: Structured logging
- **mcp**: Model Context Protocol Python SDK
- **uvicorn**: ASGI server for MCP
- **aiosqlite**: Async SQLite support

## 4. Implementation Specifications

### 4.1 Project Structure

```
email-categorization-agent/
├── src/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── orchestrator.py
│   │   ├── email_retrieval.py
│   │   ├── category_analysis.py
│   │   ├── gmail_api.py
│   │   └── user_interaction.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── event_bus.py
│   │   ├── state_manager.py
│   │   ├── config.py
│   │   └── exceptions.py
│   ├── integrations/
│   │   ├── __init__.py
│   │   ├── ollama_client.py
│   │   ├── gmail_client.py
│   │   └── mcp_server.py
│   ├── mcp/
│   │   ├── __init__.py
│   │   ├── server.py
│   │   ├── tools.py
│   │   ├── resources.py
│   │   └── handlers.py
│   └── utils/
│       ├── __init__.py
│       ├── logging.py
│       └── helpers.py
├── tests/
├── config/
│   ├── default.yaml
│   └── credentials/
├── data/
│   ├── state.db
│   └── categories.json
├── pyproject.toml
├── README.md
├── mcp.json
└── .env.example
```

### 4.2 Orchestration Agent Implementation

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from typing import TypedDict, List, Annotated
import operator
import aiosqlite

class EmailWorkflowState(TypedDict):
    email_id: str
    email_content: dict
    suggested_categories: List[str]
    confidence_score: float
    user_decision: Optional[str]
    applied_labels: List[str]
    error: Optional[str]

class OrchestrationAgent:
    def __init__(self, config: Config):
        self.config = config
        self.workflow = self._build_workflow()
        self.db_path = config.sqlite.database_path
    
    def _build_workflow(self) -> StateGraph:
        workflow = StateGraph(EmailWorkflowState)
        
        # Add nodes
        workflow.add_node("fetch_email", self.fetch_email_node)
        workflow.add_node("analyze_category", self.analyze_category_node)
        workflow.add_node("check_confidence", self.check_confidence_node)
        workflow.add_node("user_interaction", self.user_interaction_node)
        workflow.add_node("apply_labels", self.apply_labels_node)
        
        # Add edges
        workflow.add_edge("fetch_email", "analyze_category")
        workflow.add_edge("analyze_category", "check_confidence")
        
        # Conditional routing based on confidence
        workflow.add_conditional_edges(
            "check_confidence",
            self.route_by_confidence,
            {
                "high_confidence": "apply_labels",
                "low_confidence": "user_interaction"
            }
        )
        
        workflow.add_edge("user_interaction", "apply_labels")
        workflow.add_edge("apply_labels", END)
        
        # Use SQLite checkpointer
        checkpointer = SqliteSaver.from_conn_string(
            f"sqlite:///{self.db_path}"
        )
        
        return workflow.compile(checkpointer=checkpointer)
    
    def route_by_confidence(self, state: EmailWorkflowState) -> str:
        if state["confidence_score"] >= self.config.confidence_threshold:
            return "high_confidence"
        return "low_confidence"
```

### 4.3 Category Analysis Agent

```python
class CategoryAnalysisAgent:
    def __init__(self, ollama_manager: OllamaModelManager):
        self.ollama = ollama_manager
        self.prompt_template = PromptTemplate(
            template="""Analyze the following email and suggest appropriate categories/labels.

Email Subject: {subject}
Email Content: {content}
Sender: {sender}

Current available labels: {available_labels}

Instructions:
1. Analyze the email content, subject, and sender
2. Suggest 1-3 most appropriate labels from the available list
3. If no existing label fits well, suggest a new label with prefix "SUGGESTED:"
4. Provide a confidence score (0-1) for your categorization

Output format:
{{
    "suggested_labels": ["label1", "label2"],
    "confidence": 0.85,
    "reasoning": "Brief explanation"
}}
"""
        )
    
    async def analyze_email(self, email_data: dict, available_labels: List[str]) -> dict:
        prompt = self.prompt_template.format(
            subject=email_data.get("subject", ""),
            content=email_data.get("content", ""),
            sender=email_data.get("from", ""),
            available_labels=", ".join(available_labels)
        )
        
        response = await self.ollama.client.chat(
            model=self.ollama.current_model,
            messages=[
                {"role": "system", "content": "You are an email categorization expert."},
                {"role": "user", "content": prompt}
            ],
            format="json"
        )
        
        return json.loads(response['message']['content'])
```

### 4.4 MCP Server Implementation

```python
from mcp import Server, Tool, Resource
from mcp.types import TextContent, ImageContent, ToolResponse
from typing import Any, Dict, List, Optional
import asyncio
import json

class EmailCategorizationMCPServer:
    def __init__(self, agent_system: AgentSystem):
        self.agent_system = agent_system
        self.server = Server("email-categorization-agent")
        self._setup_tools()
        self._setup_resources()
        
    def _setup_tools(self):
        """Register MCP tools"""
        
        @self.server.tool()
        async def categorize_emails(
            mode: str = "interactive",
            email_ids: Optional[List[str]] = None,
            dry_run: bool = False
        ) -> ToolResponse:
            """Categorize emails using AI agents
            
            Args:
                mode: Operation mode - 'automatic' or 'interactive'
                email_ids: Specific email IDs to categorize (None for all uncategorized)
                dry_run: Preview categorization without applying labels
            """
            try:
                self.agent_system.set_mode(mode)
                
                if email_ids:
                    results = []
                    for email_id in email_ids:
                        result = await self.agent_system.categorize_email(
                            email_id, 
                            dry_run=dry_run
                        )
                        results.append(result)
                else:
                    results = await self.agent_system.categorize_uncategorized(
                        dry_run=dry_run
                    )
                
                return ToolResponse(
                    content=[
                        TextContent(
                            type="text",
                            text=json.dumps(results, indent=2)
                        )
                    ]
                )
            except Exception as e:
                return ToolResponse(
                    content=[
                        TextContent(
                            type="text",
                            text=f"Error: {str(e)}"
                        )
                    ],
                    is_error=True
                )
        
        @self.server.tool()
        async def list_uncategorized_emails(
            limit: int = 50,
            include_preview: bool = True
        ) -> ToolResponse:
            """List emails that need categorization
            
            Args:
                limit: Maximum number of emails to return
                include_preview: Include email preview/snippet
            """
            emails = await self.agent_system.get_uncategorized_emails(
                limit=limit,
                include_preview=include_preview
            )
            
            return ToolResponse(
                content=[
                    TextContent(
                        type="text",
                        text=json.dumps(emails, indent=2)
                    )
                ]
            )
        
        @self.server.tool()
        async def get_email_details(email_id: str) -> ToolResponse:
            """Get detailed information about a specific email
            
            Args:
                email_id: Gmail message ID
            """
            details = await self.agent_system.get_email_details(email_id)
            
            return ToolResponse(
                content=[
                    TextContent(
                        type="text",
                        text=json.dumps(details, indent=2)
                    )
                ]
            )
        
        @self.server.tool()
        async def apply_label_to_email(
            email_id: str,
            label: str,
            create_if_missing: bool = True
        ) -> ToolResponse:
            """Manually apply a label to an email
            
            Args:
                email_id: Gmail message ID
                label: Label name to apply
                create_if_missing: Create label if it doesn't exist
            """
            result = await self.agent_system.apply_label(
                email_id,
                label,
                create_if_missing
            )
            
            return ToolResponse(
                content=[
                    TextContent(
                        type="text",
                        text=f"Label '{label}' applied successfully to email {email_id}"
                    )
                ]
            )
        
        @self.server.tool()
        async def get_categorization_stats() -> ToolResponse:
            """Get statistics about email categorization"""
            stats = await self.agent_system.get_stats()
            
            return ToolResponse(
                content=[
                    TextContent(
                        type="text",
                        text=json.dumps(stats, indent=2)
                    )
                ]
            )
        
        @self.server.tool()
        async def train_on_feedback(
            email_id: str,
            correct_label: str,
            incorrect_label: Optional[str] = None
        ) -> ToolResponse:
            """Provide feedback to improve categorization
            
            Args:
                email_id: Email that was categorized
                correct_label: The correct label for this email
                incorrect_label: The incorrect label that was applied
            """
            await self.agent_system.record_feedback(
                email_id,
                correct_label,
                incorrect_label
            )
            
            return ToolResponse(
                content=[
                    TextContent(
                        type="text",
                        text="Feedback recorded successfully"
                    )
                ]
            )
        
        @self.server.tool()
        async def manage_labels(
            action: str,
            label_name: str,
            new_name: Optional[str] = None
        ) -> ToolResponse:
            """Manage Gmail labels
            
            Args:
                action: 'create', 'delete', 'rename', or 'list'
                label_name: Label to act on (ignored for 'list')
                new_name: New name for rename action
            """
            if action == "list":
                labels = await self.agent_system.list_labels()
                return ToolResponse(
                    content=[
                        TextContent(
                            type="text",
                            text=json.dumps(labels, indent=2)
                        )
                    ]
                )
            elif action == "create":
                await self.agent_system.create_label(label_name)
                message = f"Label '{label_name}' created"
            elif action == "delete":
                await self.agent_system.delete_label(label_name)
                message = f"Label '{label_name}' deleted"
            elif action == "rename":
                await self.agent_system.rename_label(label_name, new_name)
                message = f"Label '{label_name}' renamed to '{new_name}'"
            else:
                return ToolResponse(
                    content=[
                        TextContent(
                            type="text",
                            text=f"Unknown action: {action}"
                        )
                    ],
                    is_error=True
                )
            
            return ToolResponse(
                content=[TextContent(type="text", text=message)]
            )
    
    def _setup_resources(self):
        """Register MCP resources for monitoring"""
        
        @self.server.resource("categorization://status")
        async def get_system_status() -> Resource:
            """Current system status and health"""
            status = await self.agent_system.get_health_status()
            
            return Resource(
                uri="categorization://status",
                name="System Status",
                description="Current health and status of the email categorization system",
                mime_type="application/json",
                content=json.dumps(status, indent=2)
            )
        
        @self.server.resource("categorization://queue")
        async def get_processing_queue() -> Resource:
            """Emails currently being processed"""
            queue = await self.agent_system.get_processing_queue()
            
            return Resource(
                uri="categorization://queue",
                name="Processing Queue",
                description="Emails currently in the processing queue",
                mime_type="application/json",
                content=json.dumps(queue, indent=2)
            )
        
        @self.server.resource("categorization://config")
        async def get_configuration() -> Resource:
            """Current system configuration"""
            config = self.agent_system.get_config()
            
            return Resource(
                uri="categorization://config",
                name="Configuration",
                description="Current system configuration and settings",
                mime_type="application/json",
                content=json.dumps(config.dict(), indent=2)
            )
    
    async def handle_user_interaction(self, email_data: dict, suggestions: List[str]) -> str:
        """Handle interactive categorization through MCP"""
        # Create a unique interaction ID
        interaction_id = str(uuid4())
        
        # Store the interaction state
        await self.agent_system.store_interaction(interaction_id, {
            'email_data': email_data,
            'suggestions': suggestions,
            'status': 'pending'
        })
        
        # Return interaction details for the MCP client
        return json.dumps({
            'interaction_id': interaction_id,
            'email': {
                'id': email_data['id'],
                'from': email_data['from'],
                'subject': email_data['subject'],
                'preview': email_data.get('snippet', '')[:200]
            },
            'suggested_categories': suggestions,
            'instructions': "Use the 'respond_to_categorization' tool with your choice"
        })
    
    async def start(self, host: str = "localhost", port: int = 8080):
        """Start the MCP server"""
        await self.server.start(host=host, port=port)
        logger.info(f"MCP server started on {host}:{port}")

# Additional tool for handling user responses
@server.tool()
async def respond_to_categorization(
    interaction_id: str,
    choice: str
) -> ToolResponse:
    """Respond to a categorization request
    
    Args:
        interaction_id: ID of the categorization interaction
        choice: Either a number (1-based index), 'skip', or a custom label
    """
    interaction = await agent_system.get_interaction(interaction_id)
    
    if not interaction:
        return ToolResponse(
            content=[TextContent(type="text", text="Invalid interaction ID")],
            is_error=True
        )
    
    email_id = interaction['email_data']['id']
    suggestions = interaction['suggestions']
    
    if choice.lower() == 'skip':
        label = None
    elif choice.isdigit() and 1 <= int(choice) <= len(suggestions):
        label = suggestions[int(choice) - 1]
    else:
        label = choice  # Custom label
    
    if label:
        await agent_system.apply_label(email_id, label)
        message = f"Applied label '{label}' to email"
    else:
        message = "Skipped email categorization"
    
    # Update interaction status
    await agent_system.update_interaction(interaction_id, {'status': 'completed'})
    
    return ToolResponse(
        content=[TextContent(type="text", text=message)]
    )
```

### 4.5 MCP Configuration File

```json
{
  "name": "email-categorization-agent",
  "version": "1.0.0",
  "description": "AI-powered email categorization using local LLMs",
  "author": "Your Name",
  "license": "MIT",
  "main": "src/mcp/server.py",
  "tools": [
    {
      "name": "categorize_emails",
      "description": "Categorize emails using AI agents"
    },
    {
      "name": "list_uncategorized_emails",
      "description": "List emails that need categorization"
    },
    {
      "name": "get_email_details",
      "description": "Get detailed information about an email"
    },
    {
      "name": "apply_label_to_email",
      "description": "Manually apply a label to an email"
    },
    {
      "name": "get_categorization_stats",
      "description": "Get email categorization statistics"
    },
    {
      "name": "train_on_feedback",
      "description": "Provide feedback to improve categorization"
    },
    {
      "name": "manage_labels",
      "description": "Manage Gmail labels"
    },
    {
      "name": "respond_to_categorization",
      "description": "Respond to interactive categorization requests"
    }
  ],
  "resources": [
    "categorization://status",
    "categorization://queue",
    "categorization://config"
  ],
  "capabilities": {
    "streaming": true,
    "stateful": true,
    "interactive": true
  },
  "requirements": {
    "python": ">=3.9",
    "ollama": "required",
    "gmail_api": "required"
  }
}
```

### 4.6 SQLite State Manager Implementation

```python
import aiosqlite
from typing import Dict, Any, Optional, List
from datetime import datetime
import json

class SQLiteStateManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        
    async def initialize(self):
        """Initialize all required tables"""
        async with aiosqlite.connect(self.db_path) as db:
            # Enable WAL mode for better concurrency
            await db.execute("PRAGMA journal_mode=WAL")
            
            # User preferences table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS user_preferences (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Conversation history table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS conversation_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    interaction_id TEXT UNIQUE,
                    email_id TEXT NOT NULL,
                    user_input TEXT,
                    agent_response TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Categorization feedback table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS categorization_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email_id TEXT NOT NULL,
                    suggested_label TEXT,
                    correct_label TEXT,
                    confidence_score REAL,
                    feedback_type TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Email processing queue
            await db.execute("""
                CREATE TABLE IF NOT EXISTS email_queue (
                    email_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL DEFAULT 'pending',
                    priority INTEGER DEFAULT 5,
                    retry_count INTEGER DEFAULT 0,
                    last_error TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_email_queue_status 
                ON email_queue(status, priority DESC)
            """)
            
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_feedback_email 
                ON categorization_feedback(email_id)
            """)
            
            await db.commit()
    
    async def store_interaction(self, 
                               interaction_id: str,
                               interaction_data: Dict[str, Any]):
        """Store user interaction data"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO conversation_history 
                (interaction_id, email_id, user_input, agent_response)
                VALUES (?, ?, ?, ?)
            """, (
                interaction_id,
                interaction_data['email_data']['id'],
                json.dumps(interaction_data.get('user_input', {})),
                json.dumps(interaction_data.get('agent_response', {}))
            ))
            await db.commit()
    
    async def get_interaction(self, interaction_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve interaction data"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            
            async with db.execute("""
                SELECT * FROM conversation_history 
                WHERE interaction_id = ?
            """, (interaction_id,)) as cursor:
                row = await cursor.fetchone()
            
            if row:
                return {
                    'interaction_id': row['interaction_id'],
                    'email_id': row['email_id'],
                    'user_input': json.loads(row['user_input']),
                    'agent_response': json.loads(row['agent_response']),
                    'timestamp': row['timestamp']
                }
            
            return None
    
    async def add_to_queue(self, email_id: str, priority: int = 5):
        """Add email to processing queue"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT OR IGNORE INTO email_queue 
                (email_id, priority)
                VALUES (?, ?)
            """, (email_id, priority))
            await db.commit()
    
    async def get_next_from_queue(self) -> Optional[str]:
        """Get next email from queue"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            
            async with db.execute("""
                SELECT email_id FROM email_queue 
                WHERE status = 'pending'
                ORDER BY priority DESC, created_at ASC
                LIMIT 1
            """) as cursor:
                row = await cursor.fetchone()
            
            if row:
                # Update status to processing
                await db.execute("""
                    UPDATE email_queue 
                    SET status = 'processing', 
                        updated_at = CURRENT_TIMESTAMP
                    WHERE email_id = ?
                """, (row['email_id'],))
                await db.commit()
                
                return row['email_id']
            
            return None
    
    async def record_feedback(self,
                             email_id: str,
                             suggested_label: str,
                             correct_label: str,
                             confidence_score: float,
                             feedback_type: str = "correction"):
        """Record categorization feedback"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO categorization_feedback 
                (email_id, suggested_label, correct_label, 
                 confidence_score, feedback_type)
                VALUES (?, ?, ?, ?, ?)
            """, (
                email_id,
                suggested_label,
                correct_label,
                confidence_score,
                feedback_type
            ))
            await db.commit()
    
    async def get_categorization_stats(self) -> Dict[str, Any]:
        """Get categorization statistics"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            
            # Overall accuracy
            async with db.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN suggested_label = correct_label THEN 1 ELSE 0 END) as correct,
                    AVG(confidence_score) as avg_confidence
                FROM categorization_feedback
                WHERE feedback_type = 'correction'
            """) as cursor:
                stats = await cursor.fetchone()
            
            # Label accuracy
            async with db.execute("""
                SELECT 
                    suggested_label,
                    COUNT(*) as count,
                    SUM(CASE WHEN suggested_label = correct_label THEN 1 ELSE 0 END) as correct
                FROM categorization_feedback
                WHERE feedback_type = 'correction'
                GROUP BY suggested_label
            """) as cursor:
                label_stats = await cursor.fetchall()
            
            return {
                'overall_accuracy': stats['correct'] / stats['total'] if stats['total'] > 0 else 0,
                'total_processed': stats['total'],
                'average_confidence': stats['avg_confidence'],
                'label_accuracy': [
                    {
                        'label': row['suggested_label'],
                        'accuracy': row['correct'] / row['count'] if row['count'] > 0 else 0,
                        'count': row['count']
                    }
                    for row in label_stats
                ]
            }
```

```python
class GmailAPIAgent:
    def __init__(self, credentials_path: str):
        self.service = self._authenticate(credentials_path)
        self.quota_manager = QuotaManager()
        self.batch_size = 50
    
    def _authenticate(self, credentials_path: str) -> Resource:
        """Handle OAuth2 authentication"""
        creds = None
        token_path = 'token.json'
        
        if os.path.exists(token_path):
            creds = Credentials.from_authorized_user_file(token_path, SCOPES)
        
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    credentials_path, SCOPES)
                creds = flow.run_local_server(port=0)
            
            with open(token_path, 'w') as token:
                token.write(creds.to_json())
        
        return build('gmail', 'v1', credentials=creds)
    
    async def setup_push_notifications(self, topic_name: str):
        """Configure Gmail push notifications"""
        request = {
            'labelIds': ['INBOX'],
            'topicName': topic_name,
            'labelFilterBehavior': 'INCLUDE'
        }
        
        watch_response = self.service.users().watch(
            userId='me', 
            body=request
        ).execute()
        
        return watch_response['historyId']
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(HttpError)
    )
    async def apply_labels(self, message_id: str, label_ids: List[str]):
        """Apply labels with retry logic"""
        modify_request = {
            'addLabelIds': label_ids,
            'removeLabelIds': []
        }
        
        return self.service.users().messages().modify(
            userId='me',
            id=message_id,
            body=modify_request
        ).execute()
```

## 5. Configuration Management

### 5.1 Configuration Schema

```yaml
# config/default.yaml
app:
  name: "Email Categorization Agent"
  version: "1.0.0"
  mode: "interactive"  # automatic | interactive

ollama:
  host: "http://localhost:11434"
  models:
    primary: "gemma2:3b"
    fallback: "llama3.2:3b"
    reasoning: "llama3.2:3b"
  timeout: 60
  max_retries: 3

gmail:
  credentials_path: "./credentials.json"
  token_path: "./token.json"
  scopes:
    - "https://www.googleapis.com/auth/gmail.modify"
    - "https://www.googleapis.com/auth/gmail.labels"
  batch_size: 50
  rate_limit:
    requests_per_second: 10
    quota_per_user_per_second: 250

categorization:
  confidence_threshold: 0.75
  max_suggestions: 3
  enable_new_label_creation: true
  
sqlite:
  database_path: "./data/agent_state.db"
  connection_pool_size: 5
  busy_timeout: 30000  # milliseconds
  journal_mode: "WAL"  # Write-Ahead Logging for better concurrency
  
state:
  checkpoint_interval: 60  # seconds

logging:
  level: "INFO"
  format: "json"
  output: "./logs/agent.log"
```

### 5.2 Environment Variables

```bash
# .env.example
# Ollama Configuration
OLLAMA_HOST=http://localhost:11434
OLLAMA_NUM_THREADS=8
OLLAMA_MAX_LOADED_MODELS=2

# Gmail Configuration  
GMAIL_CREDENTIALS_PATH=./credentials.json
GMAIL_TOKEN_PATH=./token.json

# SQLite Configuration
SQLITE_DATABASE_PATH=./data/agent_state.db
SQLITE_BUSY_TIMEOUT=30000
SQLITE_JOURNAL_MODE=WAL

# Application Settings
APP_MODE=interactive
LOG_LEVEL=INFO
CONFIDENCE_THRESHOLD=0.75

# MCP Server Settings
MCP_HOST=localhost
MCP_PORT=8080
MCP_AUTH_TOKEN=your-auth-token-here
```

## 6. Error Handling and Recovery

### 6.1 Error Handling Strategy

```python
class ErrorHandler:
    def __init__(self):
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60
        )
        self.retry_policy = RetryPolicy(
            max_attempts=3,
            backoff_factor=2
        )
    
    async def handle_agent_error(self, error: Exception, context: dict):
        """Centralized error handling for agent failures"""
        if isinstance(error, RateLimitError):
            await self._handle_rate_limit(error, context)
        elif isinstance(error, AuthenticationError):
            await self._handle_auth_error(error, context)
        elif isinstance(error, ModelError):
            await self._handle_model_error(error, context)
        else:
            await self._handle_generic_error(error, context)
    
    async def _handle_model_error(self, error: ModelError, context: dict):
        """Handle Ollama model errors with fallback"""
        logger.warning(f"Model error: {error}, attempting fallback")
        
        # Try fallback model
        fallback_model = context.get('fallback_model')
        if fallback_model:
            context['model'] = fallback_model
            return await self.retry_policy.execute(
                context['operation'],
                context
            )
        
        raise error
```

### 6.2 State Recovery Mechanisms

```python
import aiosqlite
from typing import Optional, Dict, Any

class StateRecoveryManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.recovery_strategies = {
            'email_processing': self._recover_email_processing,
            'batch_operation': self._recover_batch_operation,
            'user_interaction': self._recover_user_interaction
        }
        
    async def initialize(self):
        """Initialize recovery tables"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS workflow_state (
                    workflow_id TEXT PRIMARY KEY,
                    workflow_type TEXT NOT NULL,
                    state_data TEXT NOT NULL,
                    checkpoint_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT NOT NULL,
                    error_message TEXT
                )
            """)
            
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_workflow_status 
                ON workflow_state(status, workflow_type)
            """)
            
            await db.commit()
    
    async def save_checkpoint(self, 
                             workflow_id: str, 
                             workflow_type: str,
                             state_data: Dict[str, Any],
                             status: str = "in_progress"):
        """Save workflow checkpoint"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO workflow_state
                (workflow_id, workflow_type, state_data, status)
                VALUES (?, ?, ?, ?)
            """, (
                workflow_id,
                workflow_type,
                json.dumps(state_data),
                status
            ))
            await db.commit()
    
    async def recover_workflow(self, workflow_id: str, workflow_type: str):
        """Recover interrupted workflow from last checkpoint"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            
            async with db.execute("""
                SELECT * FROM workflow_state
                WHERE workflow_id = ? AND workflow_type = ?
                ORDER BY checkpoint_time DESC
                LIMIT 1
            """, (workflow_id, workflow_type)) as cursor:
                row = await cursor.fetchone()
            
            if not row:
                logger.error(f"No checkpoint found for workflow {workflow_id}")
                return None
            
            state_data = json.loads(row['state_data'])
            recovery_strategy = self.recovery_strategies.get(workflow_type)
            
            if recovery_strategy:
                return await recovery_strategy(state_data)
            
            logger.error(f"No recovery strategy for workflow type {workflow_type}")
            return None
    
    async def cleanup_completed_workflows(self, days: int = 7):
        """Clean up old completed workflows"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                DELETE FROM workflow_state
                WHERE status IN ('completed', 'failed')
                AND checkpoint_time < datetime('now', '-' || ? || ' days')
            """, (days,))
            await db.commit()
```

## 7. Security Considerations

### 7.1 Authentication and Authorization
- OAuth 2.0 tokens stored encrypted using system keyring
- Minimal Gmail API scopes requested
- Token refresh handled automatically with rotation
- Credentials never logged or exposed

### 7.2 Data Privacy
- All email processing done locally via Ollama
- No email content sent to external services
- User preferences and history stored locally
- Optional encryption for state database

### 7.3 Security Best Practices
```python
class SecurityManager:
    def __init__(self):
        self.keyring = keyring.get_keyring()
        self.fernet = Fernet(self._get_or_create_key())
    
    def store_credential(self, service: str, username: str, credential: str):
        """Securely store credentials"""
        encrypted = self.fernet.encrypt(credential.encode())
        self.keyring.set_password(service, username, encrypted.decode())
    
    def get_credential(self, service: str, username: str) -> Optional[str]:
        """Retrieve and decrypt credentials"""
        encrypted = self.keyring.get_password(service, username)
        if encrypted:
            return self.fernet.decrypt(encrypted.encode()).decode()
        return None
```

## 8. Performance Optimization

### 8.1 Optimization Strategies

**Model Performance**
- Preload models on startup
- Use model-specific optimizations (context length, temperature)
- Implement response caching for similar emails
- Batch processing for multiple emails

**API Efficiency**
- Batch Gmail API requests (max 50 per batch)
- Use partial response fields to reduce data transfer
- Implement intelligent pagination
- Cache label mappings

**System Performance**
- Async/await throughout for non-blocking operations
- SQLite connection pooling with WAL mode for concurrency
- Lazy loading of components
- Memory-efficient email content handling
- SQLite query optimization with proper indexing

### 8.2 Performance Monitoring

```python
import aiosqlite
import psutil
import time
from contextlib import asynccontextmanager
from typing import Dict, List

class PerformanceMonitor:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.metrics = {
            'email_processing_time': [],
            'model_inference_time': [],
            'api_response_time': [],
            'memory_usage': []
        }
    
    async def initialize(self):
        """Initialize performance tracking tables"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    duration REAL NOT NULL,
                    memory_delta INTEGER,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    context TEXT
                )
            """)
            
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_metric_timestamp 
                ON performance_metrics(metric_name, timestamp)
            """)
            
            await db.commit()
    
    @asynccontextmanager
    async def measure(self, metric_name: str, context: Optional[str] = None):
        """Context manager for performance measurement"""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        
        yield
        
        elapsed_time = time.time() - start_time
        memory_delta = psutil.Process().memory_info().rss - start_memory
        
        # Store in SQLite
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO performance_metrics 
                (metric_name, duration, memory_delta, context)
                VALUES (?, ?, ?, ?)
            """, (metric_name, elapsed_time, memory_delta, context))
            await db.commit()
        
        # Also keep in memory for quick access
        self.metrics[metric_name].append({
            'duration': elapsed_time,
            'memory_delta': memory_delta,
            'timestamp': time.time()
        })
    
    async def get_metrics_summary(self, 
                                  metric_name: str, 
                                  hours: int = 24) -> Dict[str, float]:
        """Get performance metrics summary"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            
            async with db.execute("""
                SELECT 
                    AVG(duration) as avg_duration,
                    MIN(duration) as min_duration,
                    MAX(duration) as max_duration,
                    COUNT(*) as count,
                    AVG(memory_delta) as avg_memory_delta
                FROM performance_metrics
                WHERE metric_name = ?
                AND timestamp > datetime('now', '-' || ? || ' hours')
            """, (metric_name, hours)) as cursor:
                row = await cursor.fetchone()
            
            if row:
                return {
                    'avg_duration': row['avg_duration'],
                    'min_duration': row['min_duration'],
                    'max_duration': row['max_duration'],
                    'count': row['count'],
                    'avg_memory_delta': row['avg_memory_delta']
                }
            
            return {}
```

## 9. Testing Strategy

### 9.1 Test Coverage Requirements
- Unit tests: 80% minimum coverage
- Integration tests: Gmail API and Ollama integration
- End-to-end tests: Complete workflow scenarios
- Performance tests: Load and stress testing
- Security tests: Authentication and data protection

### 9.2 Testing Implementation

```python
# tests/test_category_agent.py
import pytest
from unittest.mock import Mock, patch

class TestCategoryAnalysisAgent:
    @pytest.fixture
    def agent(self):
        mock_ollama = Mock()
        return CategoryAnalysisAgent(mock_ollama)
    
    @pytest.mark.asyncio
    async def test_email_categorization(self, agent):
        # Test email data
        email_data = {
            "subject": "Invoice for Project X",
            "content": "Please find attached the invoice...",
            "from": "billing@company.com"
        }
        
        # Mock Ollama response
        agent.ollama.client.chat.return_value = {
            'message': {
                'content': '{"suggested_labels": ["Finance", "Invoices"], "confidence": 0.92}'
            }
        }
        
        result = await agent.analyze_email(email_data, ["Finance", "Personal", "Work"])
        
        assert result['confidence'] == 0.92
        assert "Finance" in result['suggested_labels']
```

## 10. Deployment and Operations

### 10.1 Deployment Options

**Local Development**
```bash
# Setup with uv (recommended)
uv sync
ollama pull gemma2:3b
ollama pull llama3.2:3b

# Run MCP Server
uv run python -m src.mcp.server

# Or with uvicorn for production
uv run uvicorn src.mcp.server:app --host 0.0.0.0 --port 8080

# Alternative with pip
pip install -e .
python -m src.mcp.server
```

**Claude Desktop Configuration**
```json
// Add to Claude Desktop's MCP configuration
{
  "email-categorization": {
    "command": "python",
    "args": ["-m", "src.mcp.server"],
    "cwd": "/path/to/email-categorization-agent",
    "env": {
      "GMAIL_CREDENTIALS_PATH": "./credentials.json",
      "OLLAMA_HOST": "http://localhost:11434"
    }
  }
}
```

**Production Deployment**
- Docker containerization with multi-stage builds
- Docker Compose for service orchestration
- Kubernetes manifests for scalable deployment
- SystemD service for Linux servers

### 10.2 Monitoring and Observability

**Logging Strategy**
- Structured JSON logging with correlation IDs
- Log aggregation to centralized system
- Error tracking with Sentry integration
- Performance metrics to Prometheus

**Health Checks**
```python
@server.resource("categorization://health")
async def health_check() -> Resource:
    """Comprehensive health check endpoint"""
    checks = {
        "ollama": await check_ollama_health(),
        "gmail": await check_gmail_health(),
        "sqlite": await check_sqlite_health(),
    }
    
    status = "healthy" if all(checks.values()) else "unhealthy"
    
    return Resource(
        uri="categorization://health",
        name="Health Status",
        description="System health check results",
        mime_type="application/json",
        content=json.dumps({
            "status": status,
            "checks": checks,
            "timestamp": datetime.now().isoformat()
        })
    )

async def check_sqlite_health() -> bool:
    """Check SQLite database health"""
    try:
        async with aiosqlite.connect(db_path) as db:
            async with db.execute("SELECT 1") as cursor:
                await cursor.fetchone()
        return True
    except Exception as e:
        logger.error(f"SQLite health check failed: {e}")
        return False
```

## 11. Future Enhancements

### 11.1 Planned Features
1. **Multi-account Support**: Handle multiple Gmail accounts
2. **Advanced ML Features**: Custom model fine-tuning on user data
3. **Smart Scheduling**: Time-based email processing rules
4. **Attachment Analysis**: Process and categorize based on attachments
5. **Export/Import**: Backup and restore categorization rules
6. **MCP Extensions**: Custom prompts and templates for different email types

### 11.2 Scalability Roadmap
1. **Horizontal Scaling**: Distribute agents across multiple nodes
2. **Queue Integration**: Add Celery/RabbitMQ for better task distribution
3. **Cloud Integration**: Optional cloud model fallback
4. **API Layer**: REST/GraphQL API for third-party integrations

## 12. Success Metrics

### 12.1 Key Performance Indicators
- **Categorization Accuracy**: >90% correct categorization
- **Processing Speed**: <5 seconds per email
- **User Intervention Rate**: <10% of emails require user input
- **System Uptime**: 99.9% availability
- **Resource Usage**: <2GB RAM, <20% CPU average

### 12.2 User Success Metrics
- Time saved per day on email management
- Reduction in missed important emails
- User satisfaction score
- Feature adoption rate
- MCP integration effectiveness

## Conclusion

This PRD defines a comprehensive agent-based email categorization system that leverages modern Python frameworks, local LLMs, and robust architectural patterns. The system's integration with the Model Context Protocol (MCP) enables seamless interaction through Claude and other MCP-compatible clients, providing a natural conversational interface for email management. The modular design ensures flexibility and maintainability, while the focus on privacy and performance makes it suitable for both personal and enterprise use. The system's event-driven architecture and comprehensive error handling ensure reliability, while the MCP interface provides an excellent user experience for email management tasks.