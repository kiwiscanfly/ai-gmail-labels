"""SQLite-based event bus for inter-agent communication."""

import aiosqlite
import json
import asyncio
import uuid
import time
from typing import Dict, List, Callable, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field
from collections import defaultdict
import structlog

from src.core.config import get_config
from src.core.exceptions import DatabaseError, StateError

logger = structlog.get_logger(__name__)


@dataclass
class AgentMessage:
    """Message passed between agents."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender_agent: str = ""
    recipient_agent: str = ""
    message_type: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    correlation_id: Optional[str] = None
    status: str = "pending"  # pending, processing, completed, failed
    priority: int = 5  # 1-10, lower is higher priority
    retry_count: int = 0
    max_retries: int = 3

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "sender_agent": self.sender_agent,
            "recipient_agent": self.recipient_agent,
            "message_type": self.message_type,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "correlation_id": self.correlation_id,
            "status": self.status,
            "priority": self.priority,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentMessage":
        """Create message from dictionary."""
        return cls(**data)


class SQLiteEventBus:
    """SQLite-based event bus for agent communication."""

    def __init__(self, db_path: Optional[str] = None):
        """Initialize the event bus.
        
        Args:
            db_path: Path to SQLite database. If None, uses config default.
        """
        config = get_config()
        self.db_path = db_path or config.sqlite.database_path
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._polling_tasks: Dict[str, asyncio.Task] = {}
        self._running = False
        self._poll_interval = 0.1  # 100ms polling interval
        
    async def initialize(self) -> None:
        """Initialize the event bus database schema."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Enable WAL mode for better concurrency
                await db.execute("PRAGMA journal_mode=WAL")
                await db.execute("PRAGMA busy_timeout=30000")
                
                # Create messages table
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
                        priority INTEGER NOT NULL DEFAULT 5,
                        retry_count INTEGER NOT NULL DEFAULT 0,
                        max_retries INTEGER NOT NULL DEFAULT 3,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create indexes for performance
                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_recipient_status_priority 
                    ON messages(recipient_agent, status, priority ASC, timestamp ASC)
                """)
                
                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_correlation 
                    ON messages(correlation_id)
                """)
                
                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_timestamp 
                    ON messages(timestamp)
                """)
                
                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_status_timestamp 
                    ON messages(status, timestamp)
                """)
                
                await db.commit()
                
            logger.info("Event bus initialized", db_path=self.db_path)
            
        except Exception as e:
            logger.error("Failed to initialize event bus", error=str(e))
            raise DatabaseError(f"Failed to initialize event bus: {e}")

    async def start(self) -> None:
        """Start the event bus processing."""
        if self._running:
            return
            
        self._running = True
        logger.info("Event bus started")

    async def stop(self) -> None:
        """Stop the event bus processing."""
        self._running = False
        
        # Cancel all polling tasks
        for task in self._polling_tasks.values():
            task.cancel()
            
        # Wait for tasks to complete
        if self._polling_tasks:
            await asyncio.gather(*self._polling_tasks.values(), return_exceptions=True)
            
        self._polling_tasks.clear()
        logger.info("Event bus stopped")

    async def publish(self, message: AgentMessage) -> None:
        """Publish a message to the event bus.
        
        Args:
            message: The message to publish.
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT INTO messages 
                    (id, sender_agent, recipient_agent, message_type, 
                     payload, timestamp, correlation_id, status, priority,
                     retry_count, max_retries)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    message.id,
                    message.sender_agent,
                    message.recipient_agent,
                    message.message_type,
                    json.dumps(message.payload),
                    message.timestamp,
                    message.correlation_id,
                    message.status,
                    message.priority,
                    message.retry_count,
                    message.max_retries
                ))
                await db.commit()
                
            logger.debug(
                "Message published",
                message_id=message.id,
                sender=message.sender_agent,
                recipient=message.recipient_agent,
                type=message.message_type
            )
            
            # Notify local subscribers immediately if they exist
            if message.recipient_agent in self.subscribers:
                for handler in self.subscribers[message.recipient_agent]:
                    try:
                        await handler(message)
                    except Exception as e:
                        logger.error(
                            "Error in local message handler",
                            error=str(e),
                            message_id=message.id
                        )
                        
        except Exception as e:
            logger.error("Failed to publish message", error=str(e), message_id=message.id)
            raise DatabaseError(f"Failed to publish message: {e}")

    async def subscribe(self, agent_id: str, handler: Callable[[AgentMessage], Any]) -> None:
        """Subscribe to messages for a specific agent.
        
        Args:
            agent_id: The agent ID to subscribe to.
            handler: Async function to handle incoming messages.
        """
        self.subscribers[agent_id].append(handler)
        
        # Start polling task for this agent if not already running
        if agent_id not in self._polling_tasks and self._running:
            task = asyncio.create_task(self._poll_messages(agent_id))
            self._polling_tasks[agent_id] = task
            
        logger.info("Agent subscribed", agent_id=agent_id)

    async def unsubscribe(self, agent_id: str, handler: Optional[Callable] = None) -> None:
        """Unsubscribe from messages.
        
        Args:
            agent_id: The agent ID to unsubscribe.
            handler: Specific handler to remove. If None, removes all handlers.
        """
        if handler:
            if agent_id in self.subscribers and handler in self.subscribers[agent_id]:
                self.subscribers[agent_id].remove(handler)
        else:
            self.subscribers[agent_id].clear()
            
        # Stop polling if no more handlers
        if not self.subscribers[agent_id] and agent_id in self._polling_tasks:
            self._polling_tasks[agent_id].cancel()
            del self._polling_tasks[agent_id]
            
        logger.info("Agent unsubscribed", agent_id=agent_id)

    async def _poll_messages(self, agent_id: str) -> None:
        """Poll for new messages for a specific agent.
        
        Args:
            agent_id: The agent ID to poll for.
        """
        logger.debug("Started polling for agent", agent_id=agent_id)
        
        while self._running and agent_id in self.subscribers:
            try:
                messages = await self._fetch_pending_messages(agent_id, limit=10)
                
                for message in messages:
                    await self._process_message(message, agent_id)
                    
                # Small delay between polls
                await asyncio.sleep(self._poll_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "Polling error for agent",
                    agent_id=agent_id,
                    error=str(e)
                )
                await asyncio.sleep(1)  # Longer delay on error

    async def _fetch_pending_messages(self, agent_id: str, limit: int = 10) -> List[AgentMessage]:
        """Fetch pending messages for an agent.
        
        Args:
            agent_id: The agent ID to fetch messages for.
            limit: Maximum number of messages to fetch.
            
        Returns:
            List of pending messages.
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                
                async with db.execute("""
                    SELECT * FROM messages 
                    WHERE recipient_agent = ? AND status = 'pending'
                    ORDER BY priority ASC, timestamp ASC
                    LIMIT ?
                """, (agent_id, limit)) as cursor:
                    rows = await cursor.fetchall()
                
                messages = []
                for row in rows:
                    message = AgentMessage(
                        id=row['id'],
                        sender_agent=row['sender_agent'],
                        recipient_agent=row['recipient_agent'],
                        message_type=row['message_type'],
                        payload=json.loads(row['payload']),
                        timestamp=row['timestamp'],
                        correlation_id=row['correlation_id'],
                        status=row['status'],
                        priority=row['priority'],
                        retry_count=row['retry_count'],
                        max_retries=row['max_retries']
                    )
                    messages.append(message)
                
                return messages
                
        except Exception as e:
            logger.error("Failed to fetch messages", agent_id=agent_id, error=str(e))
            return []

    async def _process_message(self, message: AgentMessage, agent_id: str) -> None:
        """Process a single message.
        
        Args:
            message: The message to process.
            agent_id: The agent ID processing the message.
        """
        try:
            # Update status to processing
            await self._update_message_status(message.id, "processing")
            
            # Process with all handlers for this agent
            handlers = self.subscribers.get(agent_id, [])
            for handler in handlers:
                try:
                    await handler(message)
                except Exception as e:
                    logger.error(
                        "Handler error",
                        error=str(e),
                        message_id=message.id,
                        handler=str(handler)
                    )
                    raise
            
            # Mark as completed
            await self._update_message_status(message.id, "completed")
            
            logger.debug(
                "Message processed successfully",
                message_id=message.id,
                agent_id=agent_id
            )
            
        except Exception as e:
            # Handle retry logic
            if message.retry_count < message.max_retries:
                await self._retry_message(message)
            else:
                await self._update_message_status(message.id, "failed", str(e))
                
            logger.error(
                "Message processing failed",
                message_id=message.id,
                error=str(e),
                retry_count=message.retry_count
            )

    async def _update_message_status(self, message_id: str, status: str, error_message: Optional[str] = None) -> None:
        """Update message status in database.
        
        Args:
            message_id: The message ID to update.
            status: The new status.
            error_message: Optional error message for failed status.
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                if error_message:
                    await db.execute("""
                        UPDATE messages 
                        SET status = ?, updated_at = CURRENT_TIMESTAMP,
                            payload = json_patch(payload, json('{"error": "' || ? || '"}'))
                        WHERE id = ?
                    """, (status, error_message, message_id))
                else:
                    await db.execute("""
                        UPDATE messages 
                        SET status = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE id = ?
                    """, (status, message_id))
                    
                await db.commit()
                
        except Exception as e:
            logger.error("Failed to update message status", message_id=message_id, error=str(e))

    async def _retry_message(self, message: AgentMessage) -> None:
        """Retry a failed message.
        
        Args:
            message: The message to retry.
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    UPDATE messages 
                    SET status = 'pending', 
                        retry_count = retry_count + 1,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (message.id,))
                await db.commit()
                
            logger.info("Message scheduled for retry", message_id=message.id, retry_count=message.retry_count + 1)
            
        except Exception as e:
            logger.error("Failed to retry message", message_id=message.id, error=str(e))

    async def get_message_history(
        self, 
        agent_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        limit: int = 100
    ) -> List[AgentMessage]:
        """Retrieve message history.
        
        Args:
            agent_id: Filter by agent ID (sender or recipient).
            correlation_id: Filter by correlation ID.
            limit: Maximum number of messages to return.
            
        Returns:
            List of messages matching the criteria.
        """
        try:
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
                
                messages = []
                for row in rows:
                    message = AgentMessage(
                        id=row['id'],
                        sender_agent=row['sender_agent'],
                        recipient_agent=row['recipient_agent'],
                        message_type=row['message_type'],
                        payload=json.loads(row['payload']),
                        timestamp=row['timestamp'],
                        correlation_id=row['correlation_id'],
                        status=row['status'],
                        priority=row['priority'],
                        retry_count=row['retry_count'],
                        max_retries=row['max_retries']
                    )
                    messages.append(message)
                
                return messages
                
        except Exception as e:
            logger.error("Failed to get message history", error=str(e))
            raise DatabaseError(f"Failed to get message history: {e}")

    async def cleanup_old_messages(self, days: int = 30) -> int:
        """Clean up old completed and failed messages.
        
        Args:
            days: Number of days to keep messages.
            
        Returns:
            Number of messages deleted.
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute("""
                    DELETE FROM messages 
                    WHERE status IN ('completed', 'failed') 
                    AND created_at < datetime('now', '-' || ? || ' days')
                """, (days,))
                
                deleted_count = cursor.rowcount
                await db.commit()
                
                logger.info("Cleaned up old messages", deleted_count=deleted_count, days=days)
                return deleted_count
                
        except Exception as e:
            logger.error("Failed to cleanup messages", error=str(e))
            raise DatabaseError(f"Failed to cleanup messages: {e}")

    async def get_stats(self) -> Dict[str, Any]:
        """Get event bus statistics.
        
        Returns:
            Dictionary with statistics.
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                
                # Overall stats
                async with db.execute("""
                    SELECT 
                        COUNT(*) as total_messages,
                        SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending,
                        SUM(CASE WHEN status = 'processing' THEN 1 ELSE 0 END) as processing,
                        SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
                        SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed
                    FROM messages
                """) as cursor:
                    overall = await cursor.fetchone()
                
                # Recent activity (last hour)
                async with db.execute("""
                    SELECT COUNT(*) as recent_messages
                    FROM messages
                    WHERE created_at > datetime('now', '-1 hour')
                """) as cursor:
                    recent = await cursor.fetchone()
                
                # Agent activity
                async with db.execute("""
                    SELECT 
                        recipient_agent,
                        COUNT(*) as message_count,
                        SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending_count
                    FROM messages
                    GROUP BY recipient_agent
                    ORDER BY message_count DESC
                    LIMIT 10
                """) as cursor:
                    agents = await cursor.fetchall()
                
                return {
                    "total_messages": overall["total_messages"],
                    "pending": overall["pending"],
                    "processing": overall["processing"],
                    "completed": overall["completed"],
                    "failed": overall["failed"],
                    "recent_messages_1h": recent["recent_messages"],
                    "active_agents": len(self.subscribers),
                    "polling_tasks": len(self._polling_tasks),
                    "top_agents": [
                        {
                            "agent": row["recipient_agent"],
                            "total_messages": row["message_count"],
                            "pending": row["pending_count"]
                        }
                        for row in agents
                    ]
                }
                
        except Exception as e:
            logger.error("Failed to get stats", error=str(e))
            return {}


# Global event bus instance
_event_bus: Optional[SQLiteEventBus] = None


async def get_event_bus() -> SQLiteEventBus:
    """Get the global event bus instance."""
    global _event_bus
    if _event_bus is None:
        _event_bus = SQLiteEventBus()
        await _event_bus.initialize()
        await _event_bus.start()
    return _event_bus


async def shutdown_event_bus() -> None:
    """Shutdown the global event bus."""
    global _event_bus
    if _event_bus:
        await _event_bus.stop()
        _event_bus = None