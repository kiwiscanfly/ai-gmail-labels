"""Transaction management for database operations."""

import asyncio
import uuid
from typing import List, Tuple, Any, Optional, Dict, Callable, AsyncContextManager
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
import structlog
import time

from src.core.database_pool import get_database_pool, PooledConnection
from src.core.exceptions import DatabaseError, TransactionError

logger = structlog.get_logger(__name__)


class TransactionIsolation(Enum):
    """Transaction isolation levels."""
    READ_UNCOMMITTED = "READ UNCOMMITTED"
    READ_COMMITTED = "READ COMMITTED"
    REPEATABLE_READ = "REPEATABLE READ"
    SERIALIZABLE = "SERIALIZABLE"


class TransactionState(Enum):
    """Transaction states."""
    STARTED = "started"
    ACTIVE = "active"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


@dataclass
class TransactionOperation:
    """Represents a single database operation within a transaction."""
    query: str
    parameters: Optional[Tuple] = None
    operation_type: str = "unknown"
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class TransactionStats:
    """Statistics for transaction execution."""
    transaction_id: str
    started_at: float
    completed_at: Optional[float] = None
    operation_count: int = 0
    retry_count: int = 0
    state: TransactionState = TransactionState.STARTED
    isolation_level: Optional[TransactionIsolation] = None
    error_message: Optional[str] = None


class TransactionContext:
    """Context manager for database transactions."""
    
    def __init__(
        self,
        connection: PooledConnection,
        isolation_level: Optional[TransactionIsolation] = None,
        timeout_seconds: float = 30.0,
        savepoint_name: Optional[str] = None
    ):
        self.connection = connection
        self.isolation_level = isolation_level
        self.timeout_seconds = timeout_seconds
        self.savepoint_name = savepoint_name
        
        self.transaction_id = str(uuid.uuid4())[:8]
        self.operations: List[TransactionOperation] = []
        self.stats = TransactionStats(
            transaction_id=self.transaction_id,
            started_at=time.time(),
            isolation_level=isolation_level
        )
        self._is_active = False
        self._is_savepoint = savepoint_name is not None
        
    async def __aenter__(self):
        """Start the transaction."""
        try:
            # Set isolation level if specified
            if self.isolation_level:
                await self.connection.execute(
                    f"PRAGMA read_uncommitted = {1 if self.isolation_level == TransactionIsolation.READ_UNCOMMITTED else 0}"
                )
            
            # Start transaction or savepoint
            if self._is_savepoint:
                await self.connection.execute(f"SAVEPOINT {self.savepoint_name}")
                logger.debug(
                    "Savepoint created",
                    transaction_id=self.transaction_id,
                    savepoint=self.savepoint_name
                )
            else:
                await self.connection.execute("BEGIN IMMEDIATE")
                logger.debug(
                    "Transaction started",
                    transaction_id=self.transaction_id,
                    isolation=self.isolation_level.value if self.isolation_level else None
                )
            
            self._is_active = True
            self.stats.state = TransactionState.ACTIVE
            return self
            
        except Exception as e:
            self.stats.state = TransactionState.FAILED
            self.stats.error_message = str(e)
            logger.error(
                "Failed to start transaction",
                transaction_id=self.transaction_id,
                error=str(e)
            )
            raise TransactionError(f"Failed to start transaction: {e}")
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Complete the transaction."""
        try:
            self.stats.completed_at = time.time()
            
            if exc_type is not None:
                # Exception occurred, rollback
                await self.rollback()
                return False  # Don't suppress the exception
            else:
                # No exception, commit
                await self.commit()
                return True
                
        except Exception as e:
            logger.error(
                "Error during transaction completion",
                transaction_id=self.transaction_id,
                error=str(e)
            )
            # Try to rollback if possible
            try:
                await self.rollback()
            except:
                pass
            raise TransactionError(f"Transaction completion failed: {e}")
    
    async def execute(
        self,
        query: str,
        parameters: Optional[Tuple] = None,
        operation_type: str = "unknown"
    ) -> Any:
        """Execute a query within the transaction."""
        if not self._is_active:
            raise TransactionError("Transaction is not active")
        
        operation = TransactionOperation(
            query=query,
            parameters=parameters,
            operation_type=operation_type
        )
        
        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                self.connection.execute(query, parameters),
                timeout=self.timeout_seconds
            )
            
            self.operations.append(operation)
            self.stats.operation_count += 1
            
            logger.debug(
                "Transaction operation executed",
                transaction_id=self.transaction_id,
                operation_type=operation_type,
                query_preview=query[:100]
            )
            
            return result
            
        except asyncio.TimeoutError:
            operation.retry_count += 1
            self.stats.retry_count += 1
            error_msg = f"Query timeout after {self.timeout_seconds}s"
            logger.error(
                "Transaction operation timeout",
                transaction_id=self.transaction_id,
                operation_type=operation_type,
                timeout=self.timeout_seconds
            )
            raise TransactionError(error_msg)
            
        except Exception as e:
            operation.retry_count += 1
            self.stats.retry_count += 1
            logger.error(
                "Transaction operation failed",
                transaction_id=self.transaction_id,
                operation_type=operation_type,
                error=str(e)
            )
            raise TransactionError(f"Operation failed: {e}")
    
    async def executemany(
        self,
        query: str,
        parameters_list: List[Tuple],
        operation_type: str = "batch"
    ) -> Any:
        """Execute multiple queries with the same statement."""
        if not self._is_active:
            raise TransactionError("Transaction is not active")
        
        operation = TransactionOperation(
            query=query,
            parameters=None,  # Multiple parameters
            operation_type=f"{operation_type}_batch_{len(parameters_list)}"
        )
        
        try:
            result = await asyncio.wait_for(
                self.connection.executemany(query, parameters_list),
                timeout=self.timeout_seconds * len(parameters_list)  # Scale timeout
            )
            
            self.operations.append(operation)
            self.stats.operation_count += len(parameters_list)
            
            logger.debug(
                "Transaction batch operation executed",
                transaction_id=self.transaction_id,
                operation_type=operation_type,
                batch_size=len(parameters_list)
            )
            
            return result
            
        except Exception as e:
            operation.retry_count += 1
            self.stats.retry_count += 1
            logger.error(
                "Transaction batch operation failed",
                transaction_id=self.transaction_id,
                operation_type=operation_type,
                batch_size=len(parameters_list),
                error=str(e)
            )
            raise TransactionError(f"Batch operation failed: {e}")
    
    async def commit(self) -> None:
        """Commit the transaction."""
        if not self._is_active:
            raise TransactionError("Transaction is not active")
        
        try:
            if self._is_savepoint:
                await self.connection.execute(f"RELEASE SAVEPOINT {self.savepoint_name}")
                logger.debug(
                    "Savepoint released",
                    transaction_id=self.transaction_id,
                    savepoint=self.savepoint_name
                )
            else:
                await self.connection.commit()
                logger.debug(
                    "Transaction committed",
                    transaction_id=self.transaction_id,
                    operations=len(self.operations)
                )
            
            self._is_active = False
            self.stats.state = TransactionState.COMMITTED
            
        except Exception as e:
            self.stats.state = TransactionState.FAILED
            self.stats.error_message = str(e)
            logger.error(
                "Failed to commit transaction",
                transaction_id=self.transaction_id,
                error=str(e)
            )
            raise TransactionError(f"Failed to commit transaction: {e}")
    
    async def rollback(self) -> None:
        """Rollback the transaction."""
        if not self._is_active:
            return  # Already rolled back or committed
        
        try:
            if self._is_savepoint:
                await self.connection.execute(f"ROLLBACK TO SAVEPOINT {self.savepoint_name}")
                logger.debug(
                    "Savepoint rolled back",
                    transaction_id=self.transaction_id,
                    savepoint=self.savepoint_name
                )
            else:
                await self.connection.rollback()
                logger.info(
                    "Transaction rolled back",
                    transaction_id=self.transaction_id,
                    operations=len(self.operations)
                )
            
            self._is_active = False
            self.stats.state = TransactionState.ROLLED_BACK
            
        except Exception as e:
            self.stats.state = TransactionState.FAILED
            self.stats.error_message = str(e)
            logger.error(
                "Failed to rollback transaction",
                transaction_id=self.transaction_id,
                error=str(e)
            )
            # Don't raise here - rollback should be best effort


class TransactionManager:
    """High-level transaction management."""
    
    def __init__(self):
        self._active_transactions: Dict[str, TransactionContext] = {}
        self._stats_history: List[TransactionStats] = []
        self._max_history = 1000
    
    @asynccontextmanager
    async def transaction(
        self,
        isolation_level: Optional[TransactionIsolation] = None,
        timeout_seconds: float = 30.0
    ) -> AsyncContextManager[TransactionContext]:
        """Create a new transaction context."""
        pool = await get_database_pool()
        
        async with pool.acquire() as connection:
            context = TransactionContext(
                connection=connection,
                isolation_level=isolation_level,
                timeout_seconds=timeout_seconds
            )
            
            self._active_transactions[context.transaction_id] = context
            
            try:
                async with context as tx:
                    yield tx
                    
            finally:
                # Clean up and record stats
                if context.transaction_id in self._active_transactions:
                    del self._active_transactions[context.transaction_id]
                
                self._record_transaction_stats(context.stats)
    
    @asynccontextmanager
    async def savepoint(
        self,
        savepoint_name: str,
        parent_transaction: TransactionContext,
        timeout_seconds: float = 30.0
    ) -> AsyncContextManager[TransactionContext]:
        """Create a savepoint within an existing transaction."""
        context = TransactionContext(
            connection=parent_transaction.connection,
            isolation_level=None,  # Inherit from parent
            timeout_seconds=timeout_seconds,
            savepoint_name=savepoint_name
        )
        
        async with context as sp:
            yield sp
    
    async def execute_atomic(
        self,
        operations: List[Tuple[str, Optional[Tuple], str]],
        isolation_level: Optional[TransactionIsolation] = None,
        timeout_seconds: float = 30.0
    ) -> List[Any]:
        """Execute multiple operations atomically."""
        results = []
        
        async with self.transaction(
            isolation_level=isolation_level,
            timeout_seconds=timeout_seconds
        ) as tx:
            for query, parameters, operation_type in operations:
                result = await tx.execute(query, parameters, operation_type)
                results.append(result)
        
        return results
    
    async def execute_batch_atomic(
        self,
        query: str,
        parameters_list: List[Tuple],
        operation_type: str = "batch",
        batch_size: int = 100,
        isolation_level: Optional[TransactionIsolation] = None
    ) -> None:
        """Execute batch operations atomically with chunking."""
        total_operations = len(parameters_list)
        
        for i in range(0, total_operations, batch_size):
            batch = parameters_list[i:i + batch_size]
            
            async with self.transaction(isolation_level=isolation_level) as tx:
                await tx.executemany(query, batch, f"{operation_type}_chunk_{i}")
                
                logger.debug(
                    "Batch chunk completed",
                    chunk_start=i,
                    chunk_size=len(batch),
                    total=total_operations
                )
    
    def get_active_transactions(self) -> List[str]:
        """Get list of active transaction IDs."""
        return list(self._active_transactions.keys())
    
    def get_transaction_stats(self, transaction_id: str) -> Optional[TransactionStats]:
        """Get stats for a specific transaction."""
        if transaction_id in self._active_transactions:
            return self._active_transactions[transaction_id].stats
        
        # Check history
        for stats in self._stats_history:
            if stats.transaction_id == transaction_id:
                return stats
        
        return None
    
    def get_recent_stats(self, limit: int = 10) -> List[TransactionStats]:
        """Get recent transaction statistics."""
        return self._stats_history[-limit:] if self._stats_history else []
    
    def _record_transaction_stats(self, stats: TransactionStats) -> None:
        """Record transaction statistics."""
        self._stats_history.append(stats)
        
        # Limit history size
        if len(self._stats_history) > self._max_history:
            self._stats_history = self._stats_history[-self._max_history:]
        
        # Log transaction completion
        duration_ms = 0
        if stats.completed_at and stats.started_at:
            duration_ms = (stats.completed_at - stats.started_at) * 1000
        
        logger.info(
            "Transaction completed",
            transaction_id=stats.transaction_id,
            state=stats.state.value,
            duration_ms=duration_ms,
            operations=stats.operation_count,
            retries=stats.retry_count
        )


# Global transaction manager instance
_transaction_manager: Optional[TransactionManager] = None


def get_transaction_manager() -> TransactionManager:
    """Get the global transaction manager."""
    global _transaction_manager
    if _transaction_manager is None:
        _transaction_manager = TransactionManager()
    return _transaction_manager


# Convenience functions
async def atomic_transaction(
    operations: List[Tuple[str, Optional[Tuple], str]],
    isolation_level: Optional[TransactionIsolation] = None
) -> List[Any]:
    """Execute operations in an atomic transaction."""
    manager = get_transaction_manager()
    return await manager.execute_atomic(operations, isolation_level)


async def batch_transaction(
    query: str,
    parameters_list: List[Tuple],
    operation_type: str = "batch",
    batch_size: int = 100
) -> None:
    """Execute batch operations in atomic chunks."""
    manager = get_transaction_manager()
    await manager.execute_batch_atomic(
        query, parameters_list, operation_type, batch_size
    )