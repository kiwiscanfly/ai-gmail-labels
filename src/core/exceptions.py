"""Custom exceptions for the email categorization agent."""

from typing import Optional, Dict, Any


class EmailAgentException(Exception):
    """Base exception for email categorization agent."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigurationError(EmailAgentException):
    """Raised when there's a configuration issue."""
    pass


class AuthenticationError(EmailAgentException):
    """Raised when authentication fails."""
    pass


class GmailAPIError(EmailAgentException):
    """Raised when Gmail API operations fail."""
    pass


class OllamaError(EmailAgentException):
    """Raised when Ollama operations fail."""
    pass


class ModelError(OllamaError):
    """Raised when model operations fail."""
    pass


class DatabaseError(EmailAgentException):
    """Raised when database operations fail."""
    pass


class StateError(EmailAgentException):
    """Raised when state management operations fail."""
    pass


class StorageError(EmailAgentException):
    """Raised when storage operations fail."""
    pass


class ServiceError(EmailAgentException):
    """Raised when service operations fail."""
    pass


class AgentError(EmailAgentException):
    """Raised when agent operations fail."""
    pass


class WorkflowError(EmailAgentException):
    """Raised when workflow execution fails."""
    pass


class MCPError(EmailAgentException):
    """Raised when MCP operations fail."""
    pass


class RateLimitError(EmailAgentException):
    """Raised when rate limits are exceeded."""
    pass


class SecurityError(EmailAgentException):
    """Raised when security violations occur."""
    pass


class ValidationError(EmailAgentException):
    """Raised when data validation fails."""
    pass


class TransactionError(DatabaseError):
    """Raised when transaction operations fail."""
    pass


class RetryableError(EmailAgentException):
    """Base class for errors that can be retried."""
    pass


class NonRetryableError(EmailAgentException):
    """Base class for errors that should not be retried."""
    pass


# Specific retryable errors
class TemporaryGmailError(RetryableError, GmailAPIError):
    """Temporary Gmail API error that can be retried."""
    pass


class TemporaryOllamaError(RetryableError, OllamaError):
    """Temporary Ollama error that can be retried."""
    pass


class TemporaryDatabaseError(RetryableError, DatabaseError):
    """Temporary database error that can be retried."""
    pass


# Specific non-retryable errors
class InvalidCredentialsError(NonRetryableError, AuthenticationError):
    """Invalid credentials that won't work on retry."""
    pass


class ModelNotFoundError(NonRetryableError, ModelError):
    """Model not found and won't be available on retry."""
    pass


class InvalidEmailFormatError(NonRetryableError, ValidationError):
    """Invalid email format that won't be fixed on retry."""
    pass