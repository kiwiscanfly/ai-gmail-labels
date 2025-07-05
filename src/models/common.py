"""Common enums and types used across models."""

from enum import Enum
from typing import TypeVar, Generic


class Status(str, Enum):
    """Common status values."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Priority(int, Enum):
    """Priority levels (lower number = higher priority)."""
    CRITICAL = 1
    HIGH = 3
    MEDIUM = 5
    LOW = 7
    MINIMAL = 10


class MessageType(str, Enum):
    """Types of messages between agents."""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    ERROR = "error"
    STATUS_UPDATE = "status_update"
    
    
class WorkflowStatus(str, Enum):
    """Workflow execution status."""
    INITIALIZING = "initializing"
    IN_PROGRESS = "in_progress"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ConfidenceLevel(str, Enum):
    """Confidence levels for categorization."""
    VERY_HIGH = "very_high"    # > 0.95
    HIGH = "high"              # 0.85 - 0.95
    MEDIUM = "medium"          # 0.70 - 0.85
    LOW = "low"                # 0.50 - 0.70
    VERY_LOW = "very_low"      # < 0.50


# Type variables for generic models
T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')