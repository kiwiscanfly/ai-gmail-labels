"""Gmail-specific data models."""

import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from .common import Status


@dataclass
class GmailLabel:
    """Represents a Gmail label."""
    id: str
    name: str
    message_list_visibility: str = "show"
    label_list_visibility: str = "labelShow"
    type: str = "user"
    messages_total: int = 0
    messages_unread: int = 0
    threads_total: int = 0
    threads_unread: int = 0
    color: Optional[Dict[str, str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "messages_total": self.messages_total,
            "messages_unread": self.messages_unread,
            "threads_total": self.threads_total,
            "threads_unread": self.threads_unread
        }
    
    @property
    def is_system_label(self) -> bool:
        """Check if this is a system label."""
        return self.type == "system"
    
    @property
    def is_user_label(self) -> bool:
        """Check if this is a user-created label."""
        return self.type == "user"


@dataclass
class BatchOperation:
    """Represents a batch operation."""
    operation_id: str
    operation_type: str  # modify_labels, delete, archive, etc.
    message_ids: List[str]
    parameters: Dict[str, Any] = field(default_factory=dict)
    status: Status = Status.PENDING
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    error: Optional[str] = None
    
    def mark_completed(self, error: Optional[str] = None):
        """Mark operation as completed."""
        self.completed_at = time.time()
        if error:
            self.status = Status.FAILED
            self.error = error
        else:
            self.status = Status.COMPLETED
    
    @property
    def duration(self) -> Optional[float]:
        """Get operation duration in seconds."""
        if self.completed_at:
            return self.completed_at - self.created_at
        return None
    
    @property
    def is_completed(self) -> bool:
        """Check if operation is completed."""
        return self.status in [Status.COMPLETED, Status.FAILED, Status.CANCELLED]


@dataclass
class GmailFilter:
    """Gmail search filter criteria."""
    query: str = ""
    label_ids: List[str] = field(default_factory=list)
    include_spam_trash: bool = False
    max_results: int = 100
    page_token: Optional[str] = None
    
    def to_api_params(self) -> Dict[str, Any]:
        """Convert to Gmail API parameters."""
        params = {
            "includeSpamTrash": self.include_spam_trash,
            "maxResults": min(self.max_results, 500)  # API limit
        }
        
        if self.query:
            params["q"] = self.query
        
        if self.label_ids:
            params["labelIds"] = self.label_ids
            
        if self.page_token:
            params["pageToken"] = self.page_token
            
        return params


@dataclass 
class GmailQuota:
    """Gmail API quota and rate limit tracking."""
    requests_per_second: int = 10
    quota_per_user_per_second: int = 250
    daily_quota_limit: int = 1_000_000_000  # 1 billion quota units
    
    # Current usage tracking
    requests_made: int = 0
    quota_units_used: int = 0
    last_reset: float = field(default_factory=time.time)
    
    def can_make_request(self, quota_cost: int = 1) -> bool:
        """Check if request can be made within quota limits."""
        return self.quota_units_used + quota_cost <= self.quota_per_user_per_second
    
    def record_request(self, quota_cost: int = 1):
        """Record a request and its quota cost."""
        self.requests_made += 1
        self.quota_units_used += quota_cost