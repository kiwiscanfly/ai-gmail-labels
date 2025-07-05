"""Business logic services for the email categorization agent.

This module contains service classes that orchestrate business logic,
keeping it separate from technical implementation details.
"""

from .email_service import EmailService
from .categorization_service import CategorizationService
from .notification_service import NotificationService

__all__ = [
    "EmailService",
    "CategorizationService", 
    "NotificationService",
]