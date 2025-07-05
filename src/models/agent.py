"""Agent and workflow-related data models."""

import uuid
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from .common import Status, Priority, MessageType, WorkflowStatus


@dataclass
class AgentMessage:
    """Message passed between agents."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender_agent: str = ""
    recipient_agent: str = ""
    message_type: MessageType = MessageType.REQUEST
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    correlation_id: Optional[str] = None
    status: Status = Status.PENDING
    priority: Priority = Priority.MEDIUM
    retry_count: int = 0
    max_retries: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "sender_agent": self.sender_agent,
            "recipient_agent": self.recipient_agent,
            "message_type": self.message_type.value,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "correlation_id": self.correlation_id,
            "status": self.status.value,
            "priority": self.priority.value,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentMessage":
        """Create message from dictionary."""
        # Convert string values back to enums
        if isinstance(data.get("message_type"), str):
            data["message_type"] = MessageType(data["message_type"])
        if isinstance(data.get("status"), str):
            data["status"] = Status(data["status"])
        if isinstance(data.get("priority"), int):
            data["priority"] = Priority(data["priority"])
        return cls(**data)
    
    def create_response(self, payload: Dict[str, Any]) -> "AgentMessage":
        """Create a response message to this message."""
        return AgentMessage(
            sender_agent=self.recipient_agent,
            recipient_agent=self.sender_agent,
            message_type=MessageType.RESPONSE,
            payload=payload,
            correlation_id=self.id,
            priority=self.priority
        )


@dataclass
class WorkflowCheckpoint:
    """Workflow checkpoint for state persistence."""
    workflow_id: str
    workflow_type: str
    state_data: Dict[str, Any]
    status: WorkflowStatus = WorkflowStatus.IN_PROGRESS
    error_message: Optional[str] = None
    checkpoint_time: float = field(default_factory=time.time)
    parent_workflow_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "workflow_id": self.workflow_id,
            "workflow_type": self.workflow_type,
            "state_data": self.state_data,
            "status": self.status.value,
            "error_message": self.error_message,
            "checkpoint_time": self.checkpoint_time,
            "parent_workflow_id": self.parent_workflow_id,
            "metadata": self.metadata
        }
    
    def update_status(self, status: WorkflowStatus, error_message: Optional[str] = None):
        """Update workflow status."""
        self.status = status
        self.error_message = error_message
        self.checkpoint_time = time.time()


@dataclass
class UserInteraction:
    """User interaction state."""
    interaction_id: str
    email_id: str
    interaction_type: str  # categorization_feedback, label_suggestion, etc.
    data: Dict[str, Any]
    status: Status = Status.PENDING
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    user_response: Optional[Dict[str, Any]] = None
    responded_at: Optional[float] = None
    
    def is_expired(self) -> bool:
        """Check if interaction has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at
    
    def record_response(self, response: Dict[str, Any]):
        """Record user response."""
        self.user_response = response
        self.responded_at = time.time()
        self.status = Status.COMPLETED
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "interaction_id": self.interaction_id,
            "email_id": self.email_id,
            "interaction_type": self.interaction_type,
            "data": self.data,
            "status": self.status.value,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "user_response": self.user_response,
            "responded_at": self.responded_at
        }


@dataclass
class AgentState:
    """State information for an agent."""
    agent_id: str
    agent_type: str
    status: Status = Status.PENDING
    current_task: Optional[str] = None
    last_activity: float = field(default_factory=time.time)
    metrics: Dict[str, Any] = field(default_factory=dict)
    configuration: Dict[str, Any] = field(default_factory=dict)
    
    def update_activity(self, task: Optional[str] = None):
        """Update agent activity."""
        self.last_activity = time.time()
        if task:
            self.current_task = task
    
    def update_metric(self, key: str, value: Any):
        """Update a specific metric."""
        self.metrics[key] = value
        self.last_activity = time.time()
    
    @property
    def is_idle(self) -> bool:
        """Check if agent is idle."""
        return self.status == Status.PENDING and self.current_task is None
    
    @property
    def seconds_since_activity(self) -> float:
        """Get seconds since last activity."""
        return time.time() - self.last_activity