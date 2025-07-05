"""Domain models for the email categorization agent.

This module contains all data models used throughout the application,
providing a centralized location for domain entities.
"""

from .email import EmailMessage, EmailReference, EmailContent
from .gmail import GmailLabel, BatchOperation
from .ollama import ModelInfo, GenerationResult, ChatMessage
from .agent import AgentMessage, WorkflowCheckpoint, UserInteraction
from .common import Priority, Status

__all__ = [
    # Email models
    "EmailMessage",
    "EmailReference", 
    "EmailContent",
    
    # Gmail models
    "GmailLabel",
    "BatchOperation",
    
    # Ollama models
    "ModelInfo",
    "GenerationResult",
    "ChatMessage",
    
    # Agent models
    "AgentMessage",
    "WorkflowCheckpoint",
    "UserInteraction",
    
    # Common enums
    "Priority",
    "Status",
]