"""Ollama LLM-related data models."""

import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ModelInfo:
    """Information about an Ollama model."""
    name: str
    size: int  # in bytes
    digest: str = ""
    modified_at: str = ""
    loaded: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "size": self.size,
            "size_gb": round(self.size / (1024 * 1024 * 1024), 2),
            "digest": self.digest,
            "modified_at": self.modified_at,
            "loaded": self.loaded
        }
    
    @property
    def size_gb(self) -> float:
        """Get model size in GB."""
        return self.size / (1024 * 1024 * 1024)


@dataclass
class GenerationResult:
    """Result from text generation."""
    model: str
    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_duration_ns: int = 0
    load_duration_ns: int = 0
    prompt_eval_duration_ns: int = 0
    eval_duration_ns: int = 0
    done: bool = True
    done_reason: str = "stop"
    
    @property
    def total_tokens(self) -> int:
        """Get total token count."""
        return self.prompt_tokens + self.completion_tokens
    
    @property
    def total_duration_ms(self) -> float:
        """Get total duration in milliseconds."""
        return self.total_duration_ns / 1_000_000
    
    @property
    def tokens_per_second(self) -> float:
        """Calculate tokens per second."""
        if self.eval_duration_ns > 0:
            return self.completion_tokens / (self.eval_duration_ns / 1_000_000_000)
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "model": self.model,
            "content": self.content[:100] + "..." if len(self.content) > 100 else self.content,
            "total_tokens": self.total_tokens,
            "tokens_per_second": round(self.tokens_per_second, 2),
            "duration_ms": round(self.total_duration_ms, 2)
        }


@dataclass
class ChatMessage:
    """Chat message for conversation."""
    role: str  # system, user, assistant
    content: str
    timestamp: Optional[float] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API."""
        return {
            "role": self.role,
            "content": self.content
        }


@dataclass
class ChatSession:
    """Chat session with conversation history."""
    session_id: str
    model: str
    messages: List[ChatMessage] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_message(self, role: str, content: str):
        """Add a message to the session."""
        self.messages.append(ChatMessage(role=role, content=content))
        self.last_activity = time.time()
    
    def get_messages_for_api(self) -> List[Dict[str, str]]:
        """Get messages formatted for Ollama API."""
        return [msg.to_dict() for msg in self.messages]
    
    @property
    def message_count(self) -> int:
        """Get total message count."""
        return len(self.messages)
    
    @property
    def duration(self) -> float:
        """Get session duration in seconds."""
        return self.last_activity - self.created_at


@dataclass
class ModelConfig:
    """Configuration for model behavior."""
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    num_predict: int = 256
    stop: List[str] = field(default_factory=list)
    seed: Optional[int] = None
    num_ctx: int = 2048
    repeat_penalty: float = 1.1
    
    def to_options(self) -> Dict[str, Any]:
        """Convert to Ollama API options."""
        options = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "num_predict": self.num_predict,
            "num_ctx": self.num_ctx,
            "repeat_penalty": self.repeat_penalty
        }
        
        if self.stop:
            options["stop"] = self.stop
            
        if self.seed is not None:
            options["seed"] = self.seed
            
        return options