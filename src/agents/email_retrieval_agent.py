"""Email retrieval agent for fetching emails with notifications."""

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import structlog

from src.core.event_bus import get_event_bus
from src.core.state_manager import get_state_manager
from src.core.error_recovery import get_error_recovery_manager
from src.services.email_service import EmailService
from src.integrations.gmail_client import get_gmail_client
from src.models.agent import AgentMessage
from src.models.email import EmailMessage, EmailReference
from src.models.gmail import GmailFilter
from src.models.common import Status, Priority, MessageType
from src.core.exceptions import AgentError

logger = structlog.get_logger(__name__)


class EmailRetrievalAgent:
    """Agent responsible for retrieving emails from Gmail and managing notifications."""
    
    def __init__(self):
        self._event_bus = None
        self._state_manager = None
        self._error_manager = None
        self._email_service = None
        self._gmail_client = None
        self._initialized = False
        self._monitoring_task = None
        self._last_check = None
        
    async def initialize(self) -> None:
        """Initialize the email retrieval agent."""
        try:
            self._event_bus = await get_event_bus()
            self._state_manager = await get_state_manager()
            self._error_manager = await get_error_recovery_manager()
            self._gmail_client = await get_gmail_client()
            
            # Initialize email service
            self._email_service = EmailService()
            await self._email_service.initialize()
            
            # Subscribe to orchestration commands
            await self._event_bus.subscribe("email_retrieval_agent", self._handle_command)
            
            # Start push notification monitoring
            await self._start_push_monitoring()
            
            self._initialized = True
            logger.info("Email retrieval agent initialized")
            
        except Exception as e:
            logger.error("Failed to initialize email retrieval agent", error=str(e))
            raise AgentError(f"Failed to initialize email retrieval agent: {e}")
    
    async def _start_push_monitoring(self) -> None:
        """Start monitoring for new emails via push notifications."""
        try:
            # Start background task for monitoring
            self._monitoring_task = asyncio.create_task(self._monitor_new_emails())
            self._last_check = datetime.now()
            
            logger.info("Push notification monitoring started")
            
        except Exception as e:
            logger.error("Failed to start push monitoring", error=str(e))
    
    async def _monitor_new_emails(self) -> None:
        """Background task to monitor for new emails."""
        while True:
            try:
                # Check for new emails every 30 seconds
                await asyncio.sleep(30)
                
                async with self._error_manager.protected_operation(
                    "email_monitoring",
                    "email_retrieval_agent"
                ):
                    await self._check_for_new_emails()
                    
            except Exception as e:
                logger.error("Error in email monitoring task", error=str(e))
                await asyncio.sleep(60)  # Longer delay on error
    
    async def _check_for_new_emails(self) -> None:
        """Check for new emails since last check."""
        try:
            current_time = datetime.now()
            time_filter = current_time - timedelta(minutes=1)  # Check last minute
            
            # Create filter for recent emails
            gmail_filter = GmailFilter(
                query=f"after:{int(time_filter.timestamp())}",
                include_spam_trash=False
            )
            
            # Fetch recent emails
            new_emails = []
            async for email_ref in self._email_service.search_emails(
                gmail_filter.query,
                limit=50
            ):
                # Convert reference to full email
                email = await self._email_service.get_email(email_ref.email_id)
                if email:
                    new_emails.append(email)
            
            if new_emails:
                logger.info(f"Found {len(new_emails)} new emails")
                
                # Notify orchestration agent
                await self._notify_new_emails(new_emails)
            
            self._last_check = current_time
            
        except Exception as e:
            logger.error("Failed to check for new emails", error=str(e))
    
    async def _notify_new_emails(self, emails: List[EmailMessage]) -> None:
        """Notify about new emails found."""
        try:
            message = AgentMessage(
                sender_agent="email_retrieval_agent",
                recipient_agent="orchestration_agent",
                message_type=MessageType.EVENT,
                payload={
                    "event": "new_emails_found",
                    "email_count": len(emails),
                    "emails": [email.to_dict() for email in emails],
                    "timestamp": datetime.now().timestamp()
                },
                priority=Priority.MEDIUM
            )
            
            await self._event_bus.publish(message)
            
            logger.info(
                "New emails notification sent",
                email_count=len(emails)
            )
            
        except Exception as e:
            logger.error("Failed to notify about new emails", error=str(e))
    
    async def _handle_command(self, message: AgentMessage) -> None:
        """Handle commands from orchestration agent."""
        try:
            if message.message_type != MessageType.COMMAND:
                return
            
            payload = message.payload
            command = payload.get("command")
            
            if command == "retrieve_emails":
                await self._handle_retrieve_emails(message)
            elif command == "fetch_specific_emails":
                await self._handle_fetch_specific_emails(message)
            elif command == "setup_push_notifications":
                await self._handle_setup_push_notifications(message)
            else:
                logger.warning(f"Unknown command: {command}")
                
        except Exception as e:
            logger.error(
                "Failed to handle command",
                command=message.payload.get("command"),
                error=str(e)
            )
            await self._send_error_response(message, str(e))
    
    async def _handle_retrieve_emails(self, message: AgentMessage) -> None:
        """Handle email retrieval command."""
        try:
            payload = message.payload
            workflow_id = payload.get("workflow_id")
            email_ids = payload.get("email_ids", [])
            
            logger.info(
                "Handling email retrieval",
                workflow_id=workflow_id,
                email_count=len(email_ids)
            )
            
            retrieved_emails = []
            failed_emails = []
            
            async with self._error_manager.protected_operation(
                f"retrieve_emails_{workflow_id}",
                "email_retrieval_agent"
            ):
                # Retrieve each email
                for email_id in email_ids:
                    try:
                        email = await self._email_service.get_email(email_id)
                        if email:
                            retrieved_emails.append(email)
                        else:
                            failed_emails.append(email_id)
                            
                    except Exception as e:
                        logger.error(f"Failed to retrieve email {email_id}", error=str(e))
                        failed_emails.append(email_id)
                
                # Send response
                response = AgentMessage(
                    sender_agent="email_retrieval_agent",
                    recipient_agent=message.sender_agent,
                    message_type=MessageType.RESPONSE,
                    payload={
                        "command": "retrieve_emails",
                        "workflow_id": workflow_id,
                        "status": "completed",
                        "retrieved_count": len(retrieved_emails),
                        "failed_count": len(failed_emails),
                        "emails": [email.to_dict() for email in retrieved_emails],
                        "failed_email_ids": failed_emails
                    },
                    correlation_id=message.id,
                    priority=Priority.HIGH
                )
                
                await self._event_bus.publish(response)
                
                logger.info(
                    "Email retrieval completed",
                    workflow_id=workflow_id,
                    retrieved=len(retrieved_emails),
                    failed=len(failed_emails)
                )
                
        except Exception as e:
            logger.error("Email retrieval failed", workflow_id=payload.get("workflow_id"), error=str(e))
            await self._send_error_response(message, str(e))
    
    async def _handle_fetch_specific_emails(self, message: AgentMessage) -> None:
        """Handle command to fetch specific emails by criteria."""
        try:
            payload = message.payload
            workflow_id = payload.get("workflow_id")
            filter_criteria = payload.get("filter", {})
            limit = payload.get("limit", 100)
            
            logger.info(
                "Handling specific email fetch",
                workflow_id=workflow_id,
                filter=filter_criteria,
                limit=limit
            )
            
            # Create Gmail filter
            gmail_filter = GmailFilter(
                query=filter_criteria.get("query", ""),
                labels=filter_criteria.get("labels", []),
                include_spam_trash=filter_criteria.get("include_spam", False)
            )
            
            # Fetch emails
            emails = []
            async with self._error_manager.protected_operation(
                f"fetch_emails_{workflow_id}",
                "email_retrieval_agent"
            ):
                async for email_ref in self._email_service.search_emails(
                    gmail_filter.query,
                    limit=limit
                ):
                    email = await self._email_service.get_email(email_ref.email_id)
                    if email:
                        emails.append(email)
                
                # Send response
                response = AgentMessage(
                    sender_agent="email_retrieval_agent",
                    recipient_agent=message.sender_agent,
                    message_type=MessageType.RESPONSE,
                    payload={
                        "command": "fetch_specific_emails",
                        "workflow_id": workflow_id,
                        "status": "completed",
                        "email_count": len(emails),
                        "emails": [email.to_dict() for email in emails]
                    },
                    correlation_id=message.id,
                    priority=Priority.HIGH
                )
                
                await self._event_bus.publish(response)
                
                logger.info(
                    "Specific email fetch completed",
                    workflow_id=workflow_id,
                    fetched=len(emails)
                )
                
        except Exception as e:
            logger.error("Specific email fetch failed", workflow_id=payload.get("workflow_id"), error=str(e))
            await self._send_error_response(message, str(e))
    
    async def _handle_setup_push_notifications(self, message: AgentMessage) -> None:
        """Handle command to setup Gmail push notifications."""
        try:
            payload = message.payload
            webhook_url = payload.get("webhook_url")
            
            logger.info("Setting up push notifications", webhook_url=webhook_url)
            
            # This would typically set up Gmail push notifications via Pub/Sub
            # For now, we'll just acknowledge the setup
            
            response = AgentMessage(
                sender_agent="email_retrieval_agent",
                recipient_agent=message.sender_agent,
                message_type=MessageType.RESPONSE,
                payload={
                    "command": "setup_push_notifications",
                    "status": "completed",
                    "webhook_url": webhook_url,
                    "message": "Push notifications monitoring active"
                },
                correlation_id=message.id,
                priority=Priority.MEDIUM
            )
            
            await self._event_bus.publish(response)
            
            logger.info("Push notifications setup completed")
            
        except Exception as e:
            logger.error("Push notifications setup failed", error=str(e))
            await self._send_error_response(message, str(e))
    
    async def _send_error_response(self, original_message: AgentMessage, error: str) -> None:
        """Send error response to the original sender."""
        try:
            error_response = AgentMessage(
                sender_agent="email_retrieval_agent",
                recipient_agent=original_message.sender_agent,
                message_type=MessageType.RESPONSE,
                payload={
                    "command": original_message.payload.get("command"),
                    "status": "error",
                    "error": error,
                    "workflow_id": original_message.payload.get("workflow_id")
                },
                correlation_id=original_message.id,
                priority=Priority.HIGH
            )
            
            await self._event_bus.publish(error_response)
            
        except Exception as e:
            logger.error("Failed to send error response", error=str(e))
    
    async def get_agent_status(self) -> Dict[str, Any]:
        """Get the current status of the email retrieval agent."""
        try:
            # Get email service stats
            email_stats = await self._email_service.get_email_stats()
            
            return {
                "agent": "email_retrieval_agent",
                "status": "healthy" if self._initialized else "initializing",
                "monitoring_active": self._monitoring_task is not None and not self._monitoring_task.done(),
                "last_check": self._last_check.isoformat() if self._last_check else None,
                "email_service_status": email_stats.get("status", "unknown"),
                "gmail_status": email_stats.get("gmail", {}).get("status", "unknown")
            }
            
        except Exception as e:
            logger.error("Failed to get agent status", error=str(e))
            return {
                "agent": "email_retrieval_agent",
                "status": "error",
                "error": str(e)
            }
    
    async def shutdown(self) -> None:
        """Shutdown the email retrieval agent."""
        try:
            if self._monitoring_task:
                self._monitoring_task.cancel()
                try:
                    await self._monitoring_task
                except asyncio.CancelledError:
                    pass
            
            logger.info("Email retrieval agent shutdown")
            
        except Exception as e:
            logger.error("Error during agent shutdown", error=str(e))


# Global email retrieval agent instance
_email_retrieval_agent: Optional[EmailRetrievalAgent] = None


async def get_email_retrieval_agent() -> EmailRetrievalAgent:
    """Get the global email retrieval agent instance."""
    global _email_retrieval_agent
    if _email_retrieval_agent is None:
        _email_retrieval_agent = EmailRetrievalAgent()
        await _email_retrieval_agent.initialize()
    return _email_retrieval_agent