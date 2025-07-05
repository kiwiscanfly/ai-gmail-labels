"""SQLite-based state management for the email categorization agent."""

import json
import asyncio
import time
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from pathlib import Path
import structlog

from src.core.config import get_config
from src.core.exceptions import DatabaseError, StateError
from src.core.database_pool import get_database_pool
from src.models.agent import WorkflowCheckpoint, UserInteraction

logger = structlog.get_logger(__name__)


# WorkflowCheckpoint and UserInteraction are now imported from src.models.agent


class SQLiteStateManager:
    """SQLite-based state management system."""

    def __init__(self, db_path: Optional[str] = None):
        """Initialize the state manager.
        
        Args:
            db_path: Path to SQLite database. If None, uses config default.
        """
        config = get_config()
        self.db_path = db_path or config.sqlite.database_path
        self.cleanup_interval = config.state.cleanup_interval
        self.max_history_days = config.state.max_history_days
        self._cleanup_task: Optional[asyncio.Task] = None
        
    async def initialize(self) -> None:
        """Initialize the state management database schema."""
        try:
            # Ensure database directory exists
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            
            pool = await get_database_pool()
            
            # Create schema using transaction
            operations = [
                ("""
                    CREATE TABLE IF NOT EXISTS workflow_state (
                        workflow_id TEXT PRIMARY KEY,
                        workflow_type TEXT NOT NULL,
                        state_data TEXT NOT NULL,
                        checkpoint_time REAL NOT NULL,
                        status TEXT NOT NULL,
                        error_message TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """, None),
                ("""
                    CREATE TABLE IF NOT EXISTS user_preferences (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """, None),
                ("""
                    CREATE TABLE IF NOT EXISTS user_interactions (
                        interaction_id TEXT PRIMARY KEY,
                        email_id TEXT NOT NULL,
                        interaction_type TEXT NOT NULL,
                        data TEXT NOT NULL,
                        status TEXT NOT NULL DEFAULT 'pending',
                        created_at REAL NOT NULL,
                        expires_at REAL,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """, None),
                ("""
                    CREATE TABLE IF NOT EXISTS categorization_feedback (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        email_id TEXT NOT NULL,
                        suggested_label TEXT,
                        correct_label TEXT,
                        confidence_score REAL,
                        feedback_type TEXT,
                        timestamp REAL NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """, None),
                ("""
                    CREATE TABLE IF NOT EXISTS email_queue (
                        email_id TEXT PRIMARY KEY,
                        status TEXT NOT NULL DEFAULT 'pending',
                        priority INTEGER DEFAULT 5,
                        retry_count INTEGER DEFAULT 0,
                        last_error TEXT,
                        metadata TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """, None),
                ("""
                    CREATE TABLE IF NOT EXISTS email_metadata (
                        email_id TEXT PRIMARY KEY,
                        subject TEXT,
                        sender TEXT,
                        content_preview TEXT,
                        labels TEXT,
                        thread_id TEXT,
                        received_at REAL,
                        processed_at REAL,
                        categorized_labels TEXT,
                        confidence_score REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """, None),
                ("""
                    CREATE INDEX IF NOT EXISTS idx_workflow_status 
                    ON workflow_state(status, workflow_type)
                """, None),
                ("""
                    CREATE INDEX IF NOT EXISTS idx_email_queue_status_priority 
                    ON email_queue(status, priority DESC, created_at ASC)
                """, None),
                ("""
                    CREATE INDEX IF NOT EXISTS idx_feedback_email 
                    ON categorization_feedback(email_id)
                """, None),
                ("""
                    CREATE INDEX IF NOT EXISTS idx_interactions_status_expires 
                    ON user_interactions(status, expires_at)
                """, None),
                ("""
                    CREATE INDEX IF NOT EXISTS idx_email_metadata_received 
                    ON email_metadata(received_at DESC)
                """, None)
            ]
            
            await pool.execute_transaction(operations)
                
            # Start cleanup task
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
            
            logger.info("State manager initialized", db_path=self.db_path)
            
        except Exception as e:
            logger.error("Failed to initialize state manager", error=str(e))
            raise DatabaseError(f"Failed to initialize state manager: {e}")

    async def shutdown(self) -> None:
        """Shutdown the state manager."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        logger.info("State manager shutdown")

    # Workflow State Management
    
    async def save_checkpoint(self, checkpoint: WorkflowCheckpoint) -> None:
        """Save workflow checkpoint.
        
        Args:
            checkpoint: The checkpoint to save.
        """
        try:
            pool = await get_database_pool()
            await pool.execute_query("""
                INSERT OR REPLACE INTO workflow_state
                (workflow_id, workflow_type, state_data, checkpoint_time, status, error_message)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                checkpoint.workflow_id,
                checkpoint.workflow_type,
                json.dumps(checkpoint.state_data),
                checkpoint.checkpoint_time,
                checkpoint.status,
                checkpoint.error_message
            ))
                
            logger.debug("Checkpoint saved", workflow_id=checkpoint.workflow_id)
            
        except Exception as e:
            logger.error("Failed to save checkpoint", workflow_id=checkpoint.workflow_id, error=str(e))
            raise StateError(f"Failed to save checkpoint: {e}")

    async def load_checkpoint(self, workflow_id: str) -> Optional[WorkflowCheckpoint]:
        """Load workflow checkpoint.
        
        Args:
            workflow_id: The workflow ID to load.
            
        Returns:
            The checkpoint if found, None otherwise.
        """
        try:
            pool = await get_database_pool()
            row = await pool.execute_query("""
                SELECT workflow_id, workflow_type, state_data, checkpoint_time, status, error_message, created_at, updated_at
                FROM workflow_state WHERE workflow_id = ?
            """, (workflow_id,), fetch_one=True)
            
            if row:
                return WorkflowCheckpoint(
                    workflow_id=row[0],
                    workflow_type=row[1],
                    state_data=json.loads(row[2]),
                    checkpoint_time=row[3],
                    status=row[4],
                    error_message=row[5]
                )
            return None
                
        except Exception as e:
            logger.error("Failed to load checkpoint", workflow_id=workflow_id, error=str(e))
            raise StateError(f"Failed to load checkpoint: {e}")

    async def list_checkpoints(
        self, 
        workflow_type: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100
    ) -> List[WorkflowCheckpoint]:
        """List workflow checkpoints.
        
        Args:
            workflow_type: Filter by workflow type.
            status: Filter by status.
            limit: Maximum number of checkpoints to return.
            
        Returns:
            List of checkpoints.
        """
        try:
            pool = await get_database_pool()
            
            query = "SELECT workflow_id, workflow_type, state_data, checkpoint_time, status, error_message, created_at, updated_at FROM workflow_state WHERE 1=1"
            params = []
            
            if workflow_type:
                query += " AND workflow_type = ?"
                params.append(workflow_type)
            
            if status:
                query += " AND status = ?"
                params.append(status)
            
            query += " ORDER BY checkpoint_time DESC LIMIT ?"
            params.append(limit)
            
            rows = await pool.execute_query(query, params, fetch_all=True)
            
            checkpoints = []
            for row in rows:
                checkpoint = WorkflowCheckpoint(
                    workflow_id=row[0],
                    workflow_type=row[1],
                    state_data=json.loads(row[2]),
                    checkpoint_time=row[3],
                    status=row[4],
                    error_message=row[5]
                )
                checkpoints.append(checkpoint)
            
            return checkpoints
                
        except Exception as e:
            logger.error("Failed to list checkpoints", error=str(e))
            raise StateError(f"Failed to list checkpoints: {e}")

    # User Preferences

    async def set_preference(self, key: str, value: Any) -> None:
        """Set user preference.
        
        Args:
            key: The preference key.
            value: The preference value.
        """
        try:
            pool = await get_database_pool()
            await pool.execute_query("""
                INSERT OR REPLACE INTO user_preferences (key, value)
                VALUES (?, ?)
            """, (key, json.dumps(value)))
                
            logger.debug("Preference set", key=key)
            
        except Exception as e:
            logger.error("Failed to set preference", key=key, error=str(e))
            raise StateError(f"Failed to set preference: {e}")

    async def get_preference(self, key: str, default: Any = None) -> Any:
        """Get user preference.
        
        Args:
            key: The preference key.
            default: Default value if key not found.
            
        Returns:
            The preference value or default.
        """
        try:
            pool = await get_database_pool()
            row = await pool.execute_query("""
                SELECT value FROM user_preferences WHERE key = ?
            """, (key,), fetch_one=True)
            
            if row:
                return json.loads(row[0])
            return default
                
        except Exception as e:
            logger.error("Failed to get preference", key=key, error=str(e))
            return default

    # User Interactions

    async def store_interaction(self, interaction: UserInteraction) -> None:
        """Store user interaction.
        
        Args:
            interaction: The interaction to store.
        """
        try:
            pool = await get_database_pool()
            await pool.execute_query("""
                INSERT OR REPLACE INTO user_interactions 
                (interaction_id, email_id, interaction_type, data, status, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                interaction.interaction_id,
                interaction.email_id,
                interaction.interaction_type,
                json.dumps(interaction.data),
                interaction.status,
                interaction.created_at,
                interaction.expires_at
            ))
                
            logger.debug("Interaction stored", interaction_id=interaction.interaction_id)
            
        except Exception as e:
            logger.error("Failed to store interaction", interaction_id=interaction.interaction_id, error=str(e))
            raise StateError(f"Failed to store interaction: {e}")

    async def get_interaction(self, interaction_id: str) -> Optional[UserInteraction]:
        """Get user interaction.
        
        Args:
            interaction_id: The interaction ID.
            
        Returns:
            The interaction if found, None otherwise.
        """
        try:
            pool = await get_database_pool()
            row = await pool.execute_query("""
                SELECT interaction_id, email_id, interaction_type, data, status, created_at, expires_at, updated_at
                FROM user_interactions WHERE interaction_id = ?
            """, (interaction_id,), fetch_one=True)
            
            if row:
                return UserInteraction(
                    interaction_id=row[0],
                    email_id=row[1],
                    interaction_type=row[2],
                    data=json.loads(row[3]),
                    status=row[4],
                    created_at=row[5],
                    expires_at=row[6]
                )
            return None
                
        except Exception as e:
            logger.error("Failed to get interaction", interaction_id=interaction_id, error=str(e))
            return None

    # Email Queue Management

    async def add_to_queue(
        self, 
        email_id: str, 
        priority: int = 5, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add email to processing queue.
        
        Args:
            email_id: The email ID.
            priority: Processing priority (1-10, lower is higher priority).
            metadata: Optional metadata.
        """
        try:
            pool = await get_database_pool()
            await pool.execute_query("""
                INSERT OR IGNORE INTO email_queue 
                (email_id, priority, metadata)
                VALUES (?, ?, ?)
            """, (email_id, priority, json.dumps(metadata or {})))
                
            logger.debug("Email added to queue", email_id=email_id, priority=priority)
            
        except Exception as e:
            logger.error("Failed to add email to queue", email_id=email_id, error=str(e))
            raise StateError(f"Failed to add email to queue: {e}")

    async def get_next_from_queue(self) -> Optional[str]:
        """Get next email from queue.
        
        Returns:
            The next email ID to process, or None if queue is empty.
        """
        try:
            pool = await get_database_pool()
            
            # Get the next email from queue
            row = await pool.execute_query("""
                SELECT email_id FROM email_queue 
                WHERE status = 'pending'
                ORDER BY priority ASC, created_at ASC
                LIMIT 1
            """, fetch_one=True)
            
            if row:
                email_id = row[0]
                
                # Update status to processing
                await pool.execute_query("""
                    UPDATE email_queue 
                    SET status = 'processing', updated_at = CURRENT_TIMESTAMP
                    WHERE email_id = ?
                """, (email_id,))
                
                return email_id
            
            return None
                
        except Exception as e:
            logger.error("Failed to get next from queue", error=str(e))
            return None

    async def complete_queue_item(self, email_id: str) -> None:
        """Mark queue item as completed.
        
        Args:
            email_id: The email ID to mark as completed.
        """
        try:
            pool = await get_database_pool()
            await pool.execute_query("""
                UPDATE email_queue 
                SET status = 'completed', updated_at = CURRENT_TIMESTAMP
                WHERE email_id = ?
            """, (email_id,))
                
            logger.debug("Queue item completed", email_id=email_id)
            
        except Exception as e:
            logger.error("Failed to complete queue item", email_id=email_id, error=str(e))

    # Feedback Management

    async def record_feedback(
        self,
        email_id: str,
        suggested_label: str,
        correct_label: str,
        confidence_score: float,
        feedback_type: str = "correction"
    ) -> None:
        """Record categorization feedback.
        
        Args:
            email_id: The email ID.
            suggested_label: The label that was suggested.
            correct_label: The correct label.
            confidence_score: The confidence score of the suggestion.
            feedback_type: Type of feedback.
        """
        try:
            pool = await get_database_pool()
            await pool.execute_query("""
                INSERT INTO categorization_feedback 
                (email_id, suggested_label, correct_label, confidence_score, feedback_type, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                email_id,
                suggested_label,
                correct_label,
                confidence_score,
                feedback_type,
                time.time()
            ))
                
            logger.debug("Feedback recorded", email_id=email_id, feedback_type=feedback_type)
            
        except Exception as e:
            logger.error("Failed to record feedback", email_id=email_id, error=str(e))
            raise StateError(f"Failed to record feedback: {e}")

    # Statistics and Monitoring

    async def get_categorization_stats(self) -> Dict[str, Any]:
        """Get categorization statistics.
        
        Returns:
            Dictionary with categorization statistics.
        """
        try:
            pool = await get_database_pool()
            
            # Overall accuracy
            stats = await pool.execute_query("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN suggested_label = correct_label THEN 1 ELSE 0 END) as correct,
                    AVG(confidence_score) as avg_confidence
                FROM categorization_feedback
                WHERE feedback_type = 'correction'
            """, fetch_one=True)
            
            # Label accuracy
            label_stats = await pool.execute_query("""
                SELECT 
                    suggested_label,
                    COUNT(*) as count,
                    SUM(CASE WHEN suggested_label = correct_label THEN 1 ELSE 0 END) as correct
                FROM categorization_feedback
                WHERE feedback_type = 'correction'
                GROUP BY suggested_label
                ORDER BY count DESC
            """, fetch_all=True)
            
            # Queue stats
            queue_stats = await pool.execute_query("""
                SELECT 
                    status,
                    COUNT(*) as count
                FROM email_queue
                GROUP BY status
            """, fetch_all=True)
            
            return {
                "overall_accuracy": stats[1] / stats[0] if stats[0] > 0 else 0,
                "total_processed": stats[0],
                "average_confidence": stats[2] or 0,
                "label_accuracy": [
                    {
                        'label': row[0],
                        'accuracy': row[2] / row[1] if row[1] > 0 else 0,
                        'count': row[1]
                    }
                    for row in label_stats
                ],
                "queue_status": {row[0]: row[1] for row in queue_stats}
            }
                
        except Exception as e:
            logger.error("Failed to get categorization stats", error=str(e))
            return {}

    # Cleanup and Maintenance

    async def _periodic_cleanup(self) -> None:
        """Periodic cleanup task."""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self.cleanup_old_data()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Cleanup task error", error=str(e))

    async def cleanup_old_data(self) -> Dict[str, int]:
        """Clean up old data.
        
        Returns:
            Dictionary with cleanup counts.
        """
        try:
            pool = await get_database_pool()
            
            # Use transaction for all cleanup operations
            operations = [
                ("""
                    DELETE FROM workflow_state 
                    WHERE status IN ('completed', 'failed')
                    AND created_at < datetime('now', '-' || ? || ' days')
                """, (self.max_history_days,)),
                ("""
                    DELETE FROM user_interactions 
                    WHERE expires_at < ? AND expires_at IS NOT NULL
                """, (time.time(),)),
                ("""
                    DELETE FROM email_queue 
                    WHERE status = 'completed'
                    AND updated_at < datetime('now', '-7 days')
                """, None),
                ("""
                    DELETE FROM categorization_feedback 
                    WHERE created_at < datetime('now', '-90 days')
                """, None)
            ]
            
            await pool.execute_transaction(operations)
            
            # Note: We can't get rowcount easily with the connection pool
            # so we'll just log success
            logger.info("Cleanup completed")
            return {"status": "completed"}
                
        except Exception as e:
            logger.error("Failed to cleanup old data", error=str(e))
            return {}


# Global state manager instance
_state_manager: Optional[SQLiteStateManager] = None


async def get_state_manager() -> SQLiteStateManager:
    """Get the global state manager instance."""
    global _state_manager
    if _state_manager is None:
        _state_manager = SQLiteStateManager()
        await _state_manager.initialize()
    return _state_manager


async def shutdown_state_manager() -> None:
    """Shutdown the global state manager."""
    global _state_manager
    if _state_manager:
        await _state_manager.shutdown()
        _state_manager = None