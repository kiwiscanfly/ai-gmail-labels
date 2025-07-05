"""Comprehensive error recovery system for the email categorization agent."""

import asyncio
import time
import traceback
from typing import Dict, List, Optional, Any, Callable, Type, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import structlog
from contextlib import asynccontextmanager

from src.core.config import get_config
from src.core.exceptions import (
    EmailAgentException,
    RetryableError,
    NonRetryableError,
    TemporaryGmailError,
    TemporaryOllamaError,
    TemporaryDatabaseError,
    RateLimitError,
    AuthenticationError,
    ModelNotFoundError
)
from src.core.database_pool import get_database_pool

logger = structlog.get_logger(__name__)


class RecoveryStrategy(Enum):
    """Recovery strategies for different error types."""
    RETRY = "retry"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    CIRCUIT_BREAKER = "circuit_breaker"
    FALLBACK = "fallback"
    ESCALATE = "escalate"
    IGNORE = "ignore"


@dataclass
class ErrorContext:
    """Context information for error recovery."""
    error: Exception
    operation: str
    component: str
    timestamp: float = field(default_factory=time.time)
    retry_count: int = 0
    max_retries: int = 3
    context_data: Dict[str, Any] = field(default_factory=dict)
    recovery_strategy: RecoveryStrategy = RecoveryStrategy.RETRY
    
    def __post_init__(self):
        if self.timestamp == 0:
            self.timestamp = time.time()


@dataclass
class RecoveryAction:
    """Action to take for error recovery."""
    strategy: RecoveryStrategy
    delay_seconds: float = 0.0
    fallback_operation: Optional[Callable] = None
    escalation_target: Optional[str] = None
    max_attempts: int = 3
    
    def should_retry(self, attempt: int) -> bool:
        """Check if we should retry based on current attempt."""
        return attempt < self.max_attempts


class CircuitBreaker:
    """Circuit breaker pattern implementation."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self._lock = asyncio.Lock()
    
    async def call(self, operation: Callable, *args, **kwargs):
        """Execute operation with circuit breaker protection."""
        async with self._lock:
            if self.state == "OPEN":
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = "HALF_OPEN"
                    logger.info("Circuit breaker moving to HALF_OPEN", component=operation.__name__)
                else:
                    raise Exception("Circuit breaker is OPEN")
            
            try:
                result = await operation(*args, **kwargs)
                
                if self.state == "HALF_OPEN":
                    self.state = "CLOSED"
                    self.failure_count = 0
                    logger.info("Circuit breaker recovered to CLOSED", component=operation.__name__)
                
                return result
                
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = "OPEN"
                    logger.warning(
                        "Circuit breaker opened",
                        component=operation.__name__,
                        failure_count=self.failure_count
                    )
                
                raise


class ErrorRecoveryManager:
    """Centralized error recovery and resilience management."""
    
    def __init__(self):
        self.config = get_config()
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.error_history: List[ErrorContext] = []
        self.recovery_handlers: Dict[Type[Exception], RecoveryAction] = {}
        self.fallback_operations: Dict[str, Callable] = {}
        self._setup_default_handlers()
        
    def _setup_default_handlers(self):
        """Set up default error recovery handlers."""
        # Retryable errors
        self.recovery_handlers[TemporaryGmailError] = RecoveryAction(
            strategy=RecoveryStrategy.EXPONENTIAL_BACKOFF,
            max_attempts=3
        )
        
        self.recovery_handlers[TemporaryOllamaError] = RecoveryAction(
            strategy=RecoveryStrategy.EXPONENTIAL_BACKOFF,
            max_attempts=5
        )
        
        self.recovery_handlers[TemporaryDatabaseError] = RecoveryAction(
            strategy=RecoveryStrategy.CIRCUIT_BREAKER,
            max_attempts=10
        )
        
        self.recovery_handlers[RateLimitError] = RecoveryAction(
            strategy=RecoveryStrategy.EXPONENTIAL_BACKOFF,
            delay_seconds=60.0,
            max_attempts=3
        )
        
        # Non-retryable errors
        self.recovery_handlers[AuthenticationError] = RecoveryAction(
            strategy=RecoveryStrategy.ESCALATE,
            escalation_target="authentication_service"
        )
        
        self.recovery_handlers[ModelNotFoundError] = RecoveryAction(
            strategy=RecoveryStrategy.FALLBACK,
            max_attempts=1
        )
        
        # Generic fallback
        self.recovery_handlers[RetryableError] = RecoveryAction(
            strategy=RecoveryStrategy.RETRY,
            max_attempts=3
        )
        
        self.recovery_handlers[NonRetryableError] = RecoveryAction(
            strategy=RecoveryStrategy.ESCALATE,
            escalation_target="error_handler"
        )
    
    def register_fallback_operation(self, operation_name: str, fallback_func: Callable):
        """Register a fallback operation for a specific operation."""
        self.fallback_operations[operation_name] = fallback_func
        logger.debug("Registered fallback operation", operation=operation_name)
    
    def get_circuit_breaker(self, component: str) -> CircuitBreaker:
        """Get or create circuit breaker for component."""
        if component not in self.circuit_breakers:
            self.circuit_breakers[component] = CircuitBreaker()
        return self.circuit_breakers[component]
    
    async def handle_error(self, error_context: ErrorContext) -> Optional[Any]:
        """Handle error using appropriate recovery strategy."""
        error = error_context.error
        error_type = type(error)
        
        # Find matching recovery action
        recovery_action = self._find_recovery_action(error_type)
        if not recovery_action:
            logger.error("No recovery action found", error_type=error_type.__name__)
            return None
        
        # Record error
        self.error_history.append(error_context)
        
        # Execute recovery strategy
        try:
            return await self._execute_recovery_strategy(error_context, recovery_action)
        except Exception as recovery_error:
            logger.error(
                "Recovery strategy failed",
                original_error=str(error),
                recovery_error=str(recovery_error),
                strategy=recovery_action.strategy.value
            )
            return None
    
    def _find_recovery_action(self, error_type: Type[Exception]) -> Optional[RecoveryAction]:
        """Find appropriate recovery action for error type."""
        # Direct match
        if error_type in self.recovery_handlers:
            return self.recovery_handlers[error_type]
        
        # Check parent classes
        for registered_type, action in self.recovery_handlers.items():
            if issubclass(error_type, registered_type):
                return action
        
        # Default action for unknown errors
        return RecoveryAction(strategy=RecoveryStrategy.ESCALATE)
    
    async def _execute_recovery_strategy(
        self, 
        error_context: ErrorContext, 
        recovery_action: RecoveryAction
    ) -> Optional[Any]:
        """Execute the specified recovery strategy."""
        strategy = recovery_action.strategy
        
        if strategy == RecoveryStrategy.RETRY:
            return await self._retry_operation(error_context, recovery_action)
        
        elif strategy == RecoveryStrategy.EXPONENTIAL_BACKOFF:
            return await self._exponential_backoff_retry(error_context, recovery_action)
        
        elif strategy == RecoveryStrategy.CIRCUIT_BREAKER:
            return await self._circuit_breaker_recovery(error_context, recovery_action)
        
        elif strategy == RecoveryStrategy.FALLBACK:
            return await self._fallback_operation(error_context, recovery_action)
        
        elif strategy == RecoveryStrategy.ESCALATE:
            await self._escalate_error(error_context, recovery_action)
            return None
        
        elif strategy == RecoveryStrategy.IGNORE:
            logger.info("Ignoring error as per strategy", error=str(error_context.error))
            return None
        
        else:
            logger.error("Unknown recovery strategy", strategy=strategy.value)
            return None
    
    async def _retry_operation(
        self, 
        error_context: ErrorContext, 
        recovery_action: RecoveryAction
    ) -> Optional[Any]:
        """Simple retry operation."""
        if not recovery_action.should_retry(error_context.retry_count):
            logger.warning("Max retries exceeded", operation=error_context.operation)
            return None
        
        if recovery_action.delay_seconds > 0:
            await asyncio.sleep(recovery_action.delay_seconds)
        
        logger.info(
            "Retrying operation",
            operation=error_context.operation,
            attempt=error_context.retry_count + 1
        )
        
        return "RETRY"  # Signal to retry
    
    async def _exponential_backoff_retry(
        self, 
        error_context: ErrorContext, 
        recovery_action: RecoveryAction
    ) -> Optional[Any]:
        """Retry with exponential backoff."""
        if not recovery_action.should_retry(error_context.retry_count):
            logger.warning("Max retries exceeded", operation=error_context.operation)
            return None
        
        # Calculate exponential backoff delay
        base_delay = recovery_action.delay_seconds or 1.0
        delay = base_delay * (2 ** error_context.retry_count)
        delay = min(delay, 300.0)  # Max 5 minutes
        
        logger.info(
            "Retrying with exponential backoff",
            operation=error_context.operation,
            attempt=error_context.retry_count + 1,
            delay_seconds=delay
        )
        
        await asyncio.sleep(delay)
        return "RETRY"
    
    async def _circuit_breaker_recovery(
        self, 
        error_context: ErrorContext, 
        recovery_action: RecoveryAction
    ) -> Optional[Any]:
        """Handle recovery with circuit breaker pattern."""
        circuit_breaker = self.get_circuit_breaker(error_context.component)
        
        if circuit_breaker.state == "OPEN":
            logger.warning(
                "Circuit breaker is open, cannot retry",
                component=error_context.component
            )
            return None
        
        return "RETRY"
    
    async def _fallback_operation(
        self, 
        error_context: ErrorContext, 
        recovery_action: RecoveryAction
    ) -> Optional[Any]:
        """Execute fallback operation."""
        fallback_func = None
        
        # Check for registered fallback
        if error_context.operation in self.fallback_operations:
            fallback_func = self.fallback_operations[error_context.operation]
        elif recovery_action.fallback_operation:
            fallback_func = recovery_action.fallback_operation
        
        if fallback_func:
            try:
                logger.info("Executing fallback operation", operation=error_context.operation)
                return await fallback_func(error_context)
            except Exception as e:
                logger.error("Fallback operation failed", error=str(e))
                return None
        
        logger.warning("No fallback operation available", operation=error_context.operation)
        return None
    
    async def _escalate_error(
        self, 
        error_context: ErrorContext, 
        recovery_action: RecoveryAction
    ) -> None:
        """Escalate error to appropriate handler."""
        escalation_data = {
            "error": str(error_context.error),
            "error_type": type(error_context.error).__name__,
            "operation": error_context.operation,
            "component": error_context.component,
            "timestamp": error_context.timestamp,
            "retry_count": error_context.retry_count,
            "context": error_context.context_data,
            "traceback": traceback.format_exc()
        }
        
        try:
            # Store escalation in database
            pool = await get_database_pool()
            
            await pool.execute_query("""
                INSERT INTO error_escalations 
                (operation, component, error_type, error_message, escalation_data, created_at)
                VALUES (?, ?, ?, ?, ?, datetime('now'))
            """, (
                error_context.operation,
                error_context.component,
                type(error_context.error).__name__,
                str(error_context.error),
                str(escalation_data)  # JSON string
            ))
            
            logger.critical(
                "Error escalated",
                operation=error_context.operation,
                component=error_context.component,
                target=recovery_action.escalation_target,
                error=str(error_context.error)
            )
            
        except Exception as e:
            logger.error("Failed to escalate error", error=str(e))
    
    @asynccontextmanager
    async def protected_operation(
        self, 
        operation_name: str, 
        component: str = "unknown",
        max_retries: int = 3,
        **context_data
    ):
        """Context manager for protected operation execution."""
        retry_count = 0
        last_exception = None
        
        while retry_count <= max_retries:
            try:
                yield
                return  # Success, exit
                
            except Exception as e:
                last_exception = e
                retry_count += 1
                
                error_context = ErrorContext(
                    error=e,
                    operation=operation_name,
                    component=component,
                    retry_count=retry_count - 1,
                    max_retries=max_retries,
                    context_data=context_data
                )
                
                if retry_count > max_retries:
                    # Final attempt failed
                    await self.handle_error(error_context)
                    raise
                
                # Handle error and decide whether to retry
                result = await self.handle_error(error_context)
                
                if result != "RETRY":
                    # Recovery strategy didn't request retry
                    raise
                
                logger.info(
                    "Retrying protected operation",
                    operation=operation_name,
                    attempt=retry_count + 1
                )
        
        # If we get here, all retries were exhausted
        if last_exception:
            raise last_exception
    
    async def initialize_error_tracking(self) -> None:
        """Initialize error tracking database tables."""
        try:
            pool = await get_database_pool()
            
            await pool.execute_query("""
                CREATE TABLE IF NOT EXISTS error_escalations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    operation TEXT NOT NULL,
                    component TEXT NOT NULL,
                    error_type TEXT NOT NULL,
                    error_message TEXT NOT NULL,
                    escalation_data TEXT,
                    resolved BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    resolved_at TIMESTAMP
                )
            """)
            
            await pool.execute_query("""
                CREATE INDEX IF NOT EXISTS idx_error_escalations_component 
                ON error_escalations(component, created_at DESC)
            """)
            
            await pool.execute_query("""
                CREATE INDEX IF NOT EXISTS idx_error_escalations_unresolved 
                ON error_escalations(resolved, created_at DESC)
            """)
            
            logger.info("Error tracking initialized")
            
        except Exception as e:
            logger.error("Failed to initialize error tracking", error=str(e))
    
    async def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics."""
        try:
            pool = await get_database_pool()
            
            # Recent errors (last 24 hours)
            recent_stats = await pool.execute_query("""
                SELECT 
                    component,
                    error_type,
                    COUNT(*) as error_count
                FROM error_escalations 
                WHERE created_at > datetime('now', '-1 day')
                GROUP BY component, error_type
                ORDER BY error_count DESC
                LIMIT 10
            """, fetch_all=True)
            
            # Unresolved errors
            unresolved_count = await pool.execute_query("""
                SELECT COUNT(*) FROM error_escalations WHERE resolved = FALSE
            """, fetch_one=True)
            
            # Circuit breaker states
            circuit_states = {
                name: {"state": cb.state, "failure_count": cb.failure_count}
                for name, cb in self.circuit_breakers.items()
            }
            
            return {
                "recent_errors": [
                    {
                        "component": row[0],
                        "error_type": row[1],
                        "count": row[2]
                    }
                    for row in recent_stats
                ],
                "unresolved_errors": unresolved_count[0] if unresolved_count else 0,
                "circuit_breakers": circuit_states,
                "error_history_size": len(self.error_history),
                "recovery_handlers": len(self.recovery_handlers)
            }
            
        except Exception as e:
            logger.error("Failed to get error statistics", error=str(e))
            return {}


# Global error recovery manager
_error_recovery_manager: Optional[ErrorRecoveryManager] = None


async def get_error_recovery_manager() -> ErrorRecoveryManager:
    """Get the global error recovery manager."""
    global _error_recovery_manager
    if _error_recovery_manager is None:
        _error_recovery_manager = ErrorRecoveryManager()
        await _error_recovery_manager.initialize_error_tracking()
    return _error_recovery_manager


# Decorator for automatic error recovery
def with_error_recovery(operation_name: str, component: str = "unknown", max_retries: int = 3):
    """Decorator to add error recovery to functions."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            recovery_manager = await get_error_recovery_manager()
            async with recovery_manager.protected_operation(
                operation_name=operation_name,
                component=component,
                max_retries=max_retries
            ):
                return await func(*args, **kwargs)
        return wrapper
    return decorator