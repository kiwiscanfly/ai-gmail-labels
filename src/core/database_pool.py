"""Database connection pooling for SQLite operations."""

import asyncio
import sqlite3
import aiosqlite
from typing import Optional, List, AsyncContextManager
from pathlib import Path
from contextlib import asynccontextmanager
import structlog
import threading
import time
from dataclasses import dataclass

from src.core.config import get_config
from src.core.exceptions import DatabaseError

logger = structlog.get_logger(__name__)


@dataclass
class ConnectionStats:
    """Statistics for connection pool."""
    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    total_requests: int = 0
    failed_requests: int = 0
    avg_wait_time_ms: float = 0.0
    created_at: float = 0.0


class PooledConnection:
    """Wrapper for pooled database connection."""
    
    def __init__(self, connection: aiosqlite.Connection, pool: 'DatabaseConnectionPool'):
        self.connection = connection
        self.pool = pool
        self.created_at = time.time()
        self.last_used = time.time()
        self.is_active = False
        self._lock = asyncio.Lock()
    
    async def execute(self, query: str, parameters=None):
        """Execute a query on this connection."""
        async with self._lock:
            self.last_used = time.time()
            self.is_active = True
            try:
                if parameters:
                    return await self.connection.execute(query, parameters)
                else:
                    return await self.connection.execute(query)
            finally:
                self.is_active = False
    
    async def executemany(self, query: str, parameters):
        """Execute many queries on this connection."""
        async with self._lock:
            self.last_used = time.time()
            self.is_active = True
            try:
                return await self.connection.executemany(query, parameters)
            finally:
                self.is_active = False
    
    async def fetchall(self):
        """Fetch all results from the last query."""
        return await self.connection.fetchall()
    
    async def fetchone(self):
        """Fetch one result from the last query."""
        return await self.connection.fetchone()
    
    async def commit(self):
        """Commit the current transaction."""
        async with self._lock:
            await self.connection.commit()
    
    async def rollback(self):
        """Rollback the current transaction."""
        async with self._lock:
            await self.connection.rollback()
    
    async def close(self):
        """Close the connection."""
        if hasattr(self.connection, 'close'):
            await self.connection.close()


class DatabaseConnectionPool:
    """Connection pool for SQLite database operations."""
    
    def __init__(
        self,
        database_path: str,
        pool_size: int = 5,
        max_wait_time: float = 30.0,
        idle_timeout: float = 300.0  # 5 minutes
    ):
        self.database_path = Path(database_path)
        self.pool_size = pool_size
        self.max_wait_time = max_wait_time
        self.idle_timeout = idle_timeout
        
        self._pool: asyncio.Queue[PooledConnection] = asyncio.Queue(maxsize=pool_size)
        self._all_connections: List[PooledConnection] = []
        self._stats = ConnectionStats(created_at=time.time())
        self._initialized = False
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
    
    async def initialize(self) -> None:
        """Initialize the connection pool."""
        if self._initialized:
            return
        
        async with self._lock:
            if self._initialized:
                return
            
            try:
                # Ensure database directory exists
                self.database_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Create initial connections
                config = get_config()
                
                for i in range(self.pool_size):
                    connection = await aiosqlite.connect(
                        str(self.database_path),
                        timeout=config.sqlite.busy_timeout / 1000.0
                    )
                    
                    # Configure connection
                    await connection.execute(f"PRAGMA journal_mode={config.sqlite.journal_mode}")
                    await connection.execute("PRAGMA foreign_keys=ON")
                    await connection.execute("PRAGMA synchronous=NORMAL")
                    await connection.execute("PRAGMA cache_size=-64000")  # 64MB cache
                    await connection.execute("PRAGMA temp_store=MEMORY")
                    
                    pooled_conn = PooledConnection(connection, self)
                    self._all_connections.append(pooled_conn)
                    await self._pool.put(pooled_conn)
                
                self._stats.total_connections = len(self._all_connections)
                self._stats.idle_connections = len(self._all_connections)
                
                # Start cleanup task
                self._cleanup_task = asyncio.create_task(self._cleanup_idle_connections())
                
                self._initialized = True
                logger.info(
                    "Database connection pool initialized",
                    pool_size=self.pool_size,
                    database_path=str(self.database_path)
                )
                
            except Exception as e:
                logger.error("Failed to initialize connection pool", error=str(e))
                raise DatabaseError(f"Failed to initialize connection pool: {e}")
    
    @asynccontextmanager
    async def acquire(self) -> AsyncContextManager[PooledConnection]:
        """Acquire a connection from the pool."""
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        connection = None
        
        try:
            # Try to get a connection from the pool
            try:
                connection = await asyncio.wait_for(
                    self._pool.get(),
                    timeout=self.max_wait_time
                )
            except asyncio.TimeoutError:
                self._stats.failed_requests += 1
                raise DatabaseError("Timeout waiting for database connection")
            
            # Update stats
            wait_time_ms = (time.time() - start_time) * 1000
            self._stats.total_requests += 1
            self._stats.active_connections += 1
            self._stats.idle_connections -= 1
            
            # Update rolling average wait time
            if self._stats.total_requests == 1:
                self._stats.avg_wait_time_ms = wait_time_ms
            else:
                self._stats.avg_wait_time_ms = (
                    (self._stats.avg_wait_time_ms * (self._stats.total_requests - 1) + wait_time_ms) /
                    self._stats.total_requests
                )
            
            logger.debug(
                "Connection acquired from pool",
                wait_time_ms=wait_time_ms,
                active_connections=self._stats.active_connections
            )
            
            yield connection
            
        finally:
            if connection:
                # Return connection to pool
                self._stats.active_connections -= 1
                self._stats.idle_connections += 1
                
                try:
                    # Reset connection state
                    await connection.rollback()  # Ensure clean state
                    await self._pool.put(connection)
                    
                    logger.debug(
                        "Connection returned to pool",
                        active_connections=self._stats.active_connections
                    )
                    
                except Exception as e:
                    logger.error("Failed to return connection to pool", error=str(e))
                    # Connection is corrupted, create a new one
                    await self._replace_connection(connection)
    
    async def _replace_connection(self, bad_connection: PooledConnection) -> None:
        """Replace a corrupted connection."""
        try:
            # Close the bad connection
            await bad_connection.close()
            
            # Remove from tracking
            if bad_connection in self._all_connections:
                self._all_connections.remove(bad_connection)
            
            # Create new connection
            config = get_config()
            new_connection = await aiosqlite.connect(
                str(self.database_path),
                timeout=config.sqlite.busy_timeout / 1000.0
            )
            
            # Configure new connection
            await new_connection.execute(f"PRAGMA journal_mode={config.sqlite.journal_mode}")
            await new_connection.execute("PRAGMA foreign_keys=ON")
            await new_connection.execute("PRAGMA synchronous=NORMAL")
            await new_connection.execute("PRAGMA cache_size=-64000")
            await new_connection.execute("PRAGMA temp_store=MEMORY")
            
            pooled_conn = PooledConnection(new_connection, self)
            self._all_connections.append(pooled_conn)
            await self._pool.put(pooled_conn)
            
            logger.info("Replaced corrupted database connection")
            
        except Exception as e:
            logger.error("Failed to replace corrupted connection", error=str(e))
            self._stats.total_connections -= 1
    
    async def _cleanup_idle_connections(self) -> None:
        """Cleanup task to handle idle connections."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                now = time.time()
                connections_to_refresh = []
                
                for conn in self._all_connections:
                    if not conn.is_active and (now - conn.last_used) > self.idle_timeout:
                        connections_to_refresh.append(conn)
                
                # Refresh old idle connections
                for conn in connections_to_refresh:
                    if not conn.is_active:  # Double-check it's still idle
                        logger.debug("Refreshing idle connection")
                        await self._replace_connection(conn)
                
            except Exception as e:
                logger.error("Error in connection cleanup task", error=str(e))
                await asyncio.sleep(60)  # Continue despite errors
    
    async def execute_query(
        self,
        query: str,
        parameters=None,
        fetch_one: bool = False,
        fetch_all: bool = False
    ):
        """Execute a query using the connection pool."""
        async with self.acquire() as conn:
            cursor = await conn.execute(query, parameters)
            
            if fetch_one:
                result = await cursor.fetchone()
            elif fetch_all:
                result = await cursor.fetchall()
            else:
                result = cursor
            
            await conn.commit()
            return result
    
    async def execute_transaction(self, operations: List[tuple]) -> None:
        """Execute multiple operations in a single transaction."""
        async with self.acquire() as conn:
            try:
                for query, parameters in operations:
                    await conn.execute(query, parameters or ())
                await conn.commit()
                
                logger.debug("Transaction completed", operations_count=len(operations))
                
            except Exception as e:
                await conn.rollback()
                logger.error("Transaction failed, rolled back", error=str(e))
                raise DatabaseError(f"Transaction failed: {e}")
    
    def get_stats(self) -> ConnectionStats:
        """Get connection pool statistics."""
        return self._stats
    
    async def close(self) -> None:
        """Close all connections in the pool."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Close all connections
        for conn in self._all_connections:
            try:
                await conn.close()
            except Exception as e:
                logger.error("Error closing connection", error=str(e))
        
        self._all_connections.clear()
        
        # Clear the pool queue
        while not self._pool.empty():
            try:
                self._pool.get_nowait()
            except asyncio.QueueEmpty:
                break
        
        self._initialized = False
        logger.info("Database connection pool closed")


# Global connection pool instance
_connection_pool: Optional[DatabaseConnectionPool] = None


async def get_database_pool() -> DatabaseConnectionPool:
    """Get the global database connection pool."""
    global _connection_pool
    if _connection_pool is None:
        config = get_config()
        _connection_pool = DatabaseConnectionPool(
            database_path=config.sqlite.database_path,
            pool_size=config.sqlite.connection_pool_size,
            max_wait_time=30.0,
            idle_timeout=300.0
        )
        await _connection_pool.initialize()
    return _connection_pool


async def shutdown_database_pool() -> None:
    """Shutdown the global database connection pool."""
    global _connection_pool
    if _connection_pool:
        await _connection_pool.close()
        _connection_pool = None