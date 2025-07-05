"""Notification service for user interactions and alerts."""

import asyncio
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime, timedelta
import structlog

from src.core.event_bus import get_event_bus
from src.core.state_manager import get_state_manager
from src.models.agent import AgentMessage, UserInteraction
from src.models.email import EmailCategory
from src.models.common import Status, MessageType, Priority
from src.core.exceptions import ServiceError

logger = structlog.get_logger(__name__)


class NotificationService:
    """Service for managing notifications and user interactions."""
    
    def __init__(self):
        self._event_bus = None
        self._state_manager = None
        self._initialized = False
        self._notification_handlers: Dict[str, List[Callable]] = {}
        
    async def initialize(self) -> None:
        """Initialize the notification service."""
        try:
            self._event_bus = await get_event_bus()
            self._state_manager = await get_state_manager()
            
            # Subscribe to notification events
            await self._event_bus.subscribe(
                "notification_service",
                self._handle_notification_event
            )
            
            self._initialized = True
            logger.info("Notification service initialized")
            
        except Exception as e:
            logger.error("Failed to initialize notification service", error=str(e))
            raise ServiceError(f"Failed to initialize notification service: {e}")
    
    async def request_user_input(
        self,
        email_id: str,
        category: EmailCategory,
        options: List[str],
        timeout_minutes: int = 30
    ) -> Optional[str]:
        """Request user input for ambiguous categorization.
        
        Args:
            email_id: Email ID
            category: Current categorization
            options: Available label options
            timeout_minutes: Timeout for user response
            
        Returns:
            User-selected label or None if timeout
        """
        if not self._initialized:
            await self.initialize()
            
        try:
            # Create user interaction
            interaction = UserInteraction(
                interaction_id=f"cat_{email_id}_{datetime.now().timestamp()}",
                email_id=email_id,
                interaction_type="label_selection",
                data={
                    "suggested_labels": category.suggested_labels,
                    "confidence_scores": category.confidence_scores,
                    "reasoning": category.reasoning,
                    "options": options
                },
                expires_at=datetime.now().timestamp() + (timeout_minutes * 60)
            )
            
            # Store interaction
            await self._state_manager.store_interaction(interaction)
            
            # Send notification
            await self._send_notification(
                recipient="user_interface",
                notification_type="input_required",
                data={
                    "interaction_id": interaction.interaction_id,
                    "email_id": email_id,
                    "type": "label_selection",
                    "options": options,
                    "timeout": timeout_minutes
                },
                priority=Priority.HIGH
            )
            
            # Wait for response with timeout
            start_time = asyncio.get_event_loop().time()
            timeout = timeout_minutes * 60
            
            while (asyncio.get_event_loop().time() - start_time) < timeout:
                # Check for response
                updated_interaction = await self._state_manager.get_interaction(
                    interaction.interaction_id
                )
                
                if updated_interaction and updated_interaction.user_response:
                    selected_label = updated_interaction.user_response.get("selected_label")
                    logger.info(
                        "User input received",
                        interaction_id=interaction.interaction_id,
                        selected_label=selected_label
                    )
                    return selected_label
                
                # Wait a bit before checking again
                await asyncio.sleep(1)
            
            # Timeout reached
            logger.warning(
                "User input timeout",
                interaction_id=interaction.interaction_id,
                email_id=email_id
            )
            return None
            
        except Exception as e:
            logger.error(
                "Failed to request user input",
                email_id=email_id,
                error=str(e)
            )
            return None
    
    async def notify_categorization_complete(
        self,
        email_id: str,
        category: EmailCategory,
        applied: bool
    ) -> None:
        """Notify that categorization is complete.
        
        Args:
            email_id: Email ID
            category: Categorization result
            applied: Whether label was applied
        """
        if not self._initialized:
            await self.initialize()
            
        await self._send_notification(
            recipient="user_interface",
            notification_type="categorization_complete",
            data={
                "email_id": email_id,
                "label": category.primary_label,
                "confidence": category.primary_confidence,
                "applied": applied,
                "timestamp": datetime.now().isoformat()
            },
            priority=Priority.MEDIUM
        )
    
    async def notify_error(
        self,
        error_type: str,
        error_message: str,
        context: Dict[str, Any]
    ) -> None:
        """Send error notification.
        
        Args:
            error_type: Type of error
            error_message: Error message
            context: Additional context
        """
        if not self._initialized:
            await self.initialize()
            
        await self._send_notification(
            recipient="error_handler",
            notification_type="error",
            data={
                "error_type": error_type,
                "error_message": error_message,
                "context": context,
                "timestamp": datetime.now().isoformat()
            },
            priority=Priority.HIGH
        )
    
    async def notify_batch_progress(
        self,
        batch_id: str,
        total: int,
        processed: int,
        failed: int = 0
    ) -> None:
        """Notify batch processing progress.
        
        Args:
            batch_id: Batch identifier
            total: Total items
            processed: Processed items
            failed: Failed items
        """
        if not self._initialized:
            await self.initialize()
            
        progress_percent = (processed / total * 100) if total > 0 else 0
        
        await self._send_notification(
            recipient="user_interface",
            notification_type="batch_progress",
            data={
                "batch_id": batch_id,
                "total": total,
                "processed": processed,
                "failed": failed,
                "progress_percent": round(progress_percent, 2),
                "timestamp": datetime.now().isoformat()
            },
            priority=Priority.LOW
        )
    
    def register_handler(
        self,
        notification_type: str,
        handler: Callable[[Dict[str, Any]], Any]
    ) -> None:
        """Register a notification handler.
        
        Args:
            notification_type: Type of notification to handle
            handler: Handler function
        """
        if notification_type not in self._notification_handlers:
            self._notification_handlers[notification_type] = []
        
        self._notification_handlers[notification_type].append(handler)
        logger.debug(
            "Notification handler registered",
            notification_type=notification_type
        )
    
    async def _send_notification(
        self,
        recipient: str,
        notification_type: str,
        data: Dict[str, Any],
        priority: Priority = Priority.MEDIUM
    ) -> None:
        """Send a notification via event bus.
        
        Args:
            recipient: Recipient agent/service
            notification_type: Type of notification
            data: Notification data
            priority: Message priority
        """
        try:
            message = AgentMessage(
                sender_agent="notification_service",
                recipient_agent=recipient,
                message_type=MessageType.NOTIFICATION,
                payload={
                    "notification_type": notification_type,
                    "data": data
                },
                priority=priority
            )
            
            await self._event_bus.publish(message)
            
            logger.debug(
                "Notification sent",
                recipient=recipient,
                type=notification_type,
                priority=priority.name
            )
            
        except Exception as e:
            logger.error(
                "Failed to send notification",
                recipient=recipient,
                type=notification_type,
                error=str(e)
            )
    
    async def _handle_notification_event(self, message: AgentMessage) -> None:
        """Handle incoming notification events.
        
        Args:
            message: Incoming message
        """
        try:
            if message.message_type != MessageType.NOTIFICATION:
                return
                
            notification_type = message.payload.get("notification_type")
            data = message.payload.get("data", {})
            
            # Call registered handlers
            handlers = self._notification_handlers.get(notification_type, [])
            for handler in handlers:
                try:
                    await handler(data) if asyncio.iscoroutinefunction(handler) else handler(data)
                except Exception as e:
                    logger.error(
                        "Notification handler error",
                        handler=str(handler),
                        error=str(e)
                    )
                    
        except Exception as e:
            logger.error(
                "Failed to handle notification event",
                message_id=message.id,
                error=str(e)
            )
    
    async def get_pending_interactions(
        self,
        email_id: Optional[str] = None
    ) -> List[UserInteraction]:
        """Get pending user interactions.
        
        Args:
            email_id: Optional email ID filter
            
        Returns:
            List of pending interactions
        """
        if not self._initialized:
            await self.initialize()
            
        try:
            # This would query the state manager for pending interactions
            # For now, return empty list
            return []
            
        except Exception as e:
            logger.error("Failed to get pending interactions", error=str(e))
            return []
    
    async def submit_user_response(
        self,
        interaction_id: str,
        response: Dict[str, Any]
    ) -> bool:
        """Submit user response to an interaction.
        
        Args:
            interaction_id: Interaction ID
            response: User response data
            
        Returns:
            Success status
        """
        if not self._initialized:
            await self.initialize()
            
        try:
            # Get interaction
            interaction = await self._state_manager.get_interaction(interaction_id)
            if not interaction:
                logger.warning("Interaction not found", interaction_id=interaction_id)
                return False
            
            # Check if expired
            if interaction.is_expired():
                logger.warning("Interaction expired", interaction_id=interaction_id)
                return False
            
            # Record response
            interaction.record_response(response)
            
            # Update in state manager
            await self._state_manager.store_interaction(interaction)
            
            logger.info(
                "User response recorded",
                interaction_id=interaction_id,
                response_type=interaction.interaction_type
            )
            
            return True
            
        except Exception as e:
            logger.error(
                "Failed to submit user response",
                interaction_id=interaction_id,
                error=str(e)
            )
            return False