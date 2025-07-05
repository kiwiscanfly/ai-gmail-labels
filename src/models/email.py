"""Email-related data models."""

import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

from .common import Status


@dataclass
class EmailMessage:
    """Represents a Gmail message."""
    id: str
    thread_id: str
    label_ids: List[str] = field(default_factory=list)
    snippet: str = ""
    history_id: str = ""
    internal_date: int = 0
    size_estimate: int = 0
    
    # Headers
    subject: str = ""
    sender: str = ""
    recipient: str = ""
    date: str = ""
    
    # Content
    body_text: str = ""
    body_html: str = ""
    attachments: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    received_at: Optional[datetime] = None
    processed_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.internal_date and not self.received_at:
            self.received_at = datetime.fromtimestamp(self.internal_date / 1000)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "thread_id": self.thread_id,
            "label_ids": self.label_ids,
            "subject": self.subject,
            "sender": self.sender,
            "recipient": self.recipient,
            "date": self.date,
            "size_estimate": self.size_estimate,
            "snippet": self.snippet[:100] + "..." if len(self.snippet) > 100 else self.snippet
        }


@dataclass
class EmailReference:
    """Lightweight reference to an email without storing full content."""
    email_id: str
    thread_id: str
    subject: str = ""
    sender: str = ""
    recipient: str = ""
    date: str = ""
    labels: List[str] = field(default_factory=list)
    size_estimate: int = 0
    content_hash: Optional[str] = None
    storage_path: Optional[str] = None
    cached_at: Optional[float] = None
    last_accessed: Optional[float] = None
    
    def __post_init__(self):
        if self.last_accessed is None:
            self.last_accessed = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "email_id": self.email_id,
            "thread_id": self.thread_id,
            "subject": self.subject,
            "sender": self.sender,
            "labels": self.labels,
            "size_estimate": self.size_estimate,
            "last_accessed": self.last_accessed
        }


@dataclass
class EmailContent:
    """Full email content - loaded on demand."""
    email_id: str
    body_text: str = ""
    body_html: str = ""
    attachments: List[Dict[str, Any]] = field(default_factory=list)
    headers: Dict[str, str] = field(default_factory=dict)
    raw_content: Optional[bytes] = None
    compressed: bool = False
    
    def get_size_bytes(self) -> int:
        """Calculate the approximate size in bytes."""
        size = len(self.body_text.encode('utf-8'))
        size += len(self.body_html.encode('utf-8'))
        
        for attachment in self.attachments:
            if 'size' in attachment:
                size += attachment['size']
        
        if self.raw_content:
            size += len(self.raw_content)
            
        return size
    
    def get_text_preview(self, max_length: int = 200) -> str:
        """Get a text preview of the email content."""
        text = self.body_text or ""
        if len(text) > max_length:
            return text[:max_length] + "..."
        return text


@dataclass
class EmailCategory:
    """Email categorization result."""
    email_id: str
    suggested_labels: List[str]
    confidence_scores: Dict[str, float]
    reasoning: str = ""
    requires_user_input: bool = False
    categorized_at: Optional[float] = None
    
    def __post_init__(self):
        if self.categorized_at is None:
            self.categorized_at = time.time()
    
    @property
    def primary_label(self) -> Optional[str]:
        """Get the label with highest confidence."""
        if not self.confidence_scores:
            return None
        return max(self.confidence_scores.items(), key=lambda x: x[1])[0]
    
    @property
    def primary_confidence(self) -> float:
        """Get the highest confidence score."""
        if not self.confidence_scores:
            return 0.0
        return max(self.confidence_scores.values())