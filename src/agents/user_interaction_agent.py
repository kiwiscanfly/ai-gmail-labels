"""User interaction agent for handling ambiguous email categorizations through MCP."""

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import structlog

from src.core.event_bus import get_event_bus
from src.core.state_manager import get_state_manager
from src.core.error_recovery import get_error_recovery_manager
from src.services.user_service import UserService
from src.models.agent import AgentMessage
from src.models.email import EmailMessage, EmailCategory
from src.models.user import UserInteraction, UserFeedback
from src.models.common import Status, Priority, MessageType, ConfidenceLevel
from src.core.exceptions import AgentError

logger = structlog.get_logger(__name__)


class UserInteractionAgent:
    """Agent responsible for handling user interactions for ambiguous email categorizations."""
    
    def __init__(self):
        self._event_bus = None
        self._state_manager = None
        self._error_manager = None
        self._user_service = None
        self._initialized = False
        self._pending_interactions = {}  # workflow_id -> interaction data
        self._interaction_timeout = 300  # 5 minutes
        
    async def initialize(self) -> None:
        """Initialize the user interaction agent."""
        try:
            self._event_bus = await get_event_bus()
            self._state_manager = await get_state_manager()
            self._error_manager = await get_error_recovery_manager()
            
            # Initialize user service
            self._user_service = UserService()
            await self._user_service.initialize()
            
            # Subscribe to orchestration commands
            await self._event_bus.subscribe("user_interaction_agent", self._handle_command)
            
            # Start cleanup task for expired interactions
            asyncio.create_task(self._cleanup_expired_interactions())
            
            self._initialized = True
            logger.info("User interaction agent initialized")
            
        except Exception as e:
            logger.error("Failed to initialize user interaction agent", error=str(e))
            raise AgentError(f"Failed to initialize user interaction agent: {e}")
    
    async def _handle_command(self, message: AgentMessage) -> None:
        """Handle commands from orchestration agent."""
        try:
            if message.message_type != MessageType.COMMAND:
                return
            
            payload = message.payload
            command = payload.get("command")
            
            if command == "handle_interactions":
                await self._handle_user_interactions(message)
            elif command == "submit_feedback":
                await self._handle_submit_feedback(message)
            elif command == "get_pending_interactions":
                await self._handle_get_pending_interactions(message)
            elif command == "resolve_interaction":
                await self._handle_resolve_interaction(message)
            else:
                logger.warning(f"Unknown command: {command}")
                
        except Exception as e:
            logger.error(
                "Failed to handle command",
                command=message.payload.get("command"),
                error=str(e)
            )
            await self._send_error_response(message, str(e))
    
    async def _handle_user_interactions(self, message: AgentMessage) -> None:
        """Handle request for user interactions on ambiguous emails."""
        try:
            payload = message.payload
            workflow_id = payload.get("workflow_id")
            email_ids = payload.get("email_ids", [])
            categorizations = payload.get("categorizations", {})
            
            logger.info(
                "Handling user interactions request",
                workflow_id=workflow_id,
                email_count=len(email_ids)
            )
            
            async with self._error_manager.protected_operation(
                f"user_interactions_{workflow_id}",
                "user_interaction_agent"
            ):
                # Create user interaction records
                interactions = []
                for email_id in email_ids:
                    categorization_data = categorizations.get(email_id, {})
                    
                    interaction = UserInteraction(
                        interaction_id=f"{workflow_id}_{email_id}",
                        workflow_id=workflow_id,
                        email_id=email_id,
                        interaction_type="email_categorization",
                        context_data={
                            "email_id": email_id,
                            "suggested_labels": categorization_data.get("suggested_labels", []),
                            "confidence_scores": categorization_data.get("confidence_scores", {}),
                            "reasoning": categorization_data.get("reasoning", ""),
                            "available_actions": ["approve", "reject", "modify", "skip"]
                        },
                        status=Status.PENDING,
                        created_at=datetime.now().timestamp(),
                        expires_at=(datetime.now() + timedelta(seconds=self._interaction_timeout)).timestamp()
                    )
                    
                    # Save interaction to database
                    await self._user_service.create_interaction(interaction)
                    interactions.append(interaction)
                
                # Store in memory for quick access
                self._pending_interactions[workflow_id] = {
                    "interactions": interactions,
                    "created_at": datetime.now().timestamp(),
                    "email_ids": email_ids
                }
                
                # Send response with interaction details
                response = AgentMessage(
                    sender_agent="user_interaction_agent",
                    recipient_agent=message.sender_agent,
                    message_type=MessageType.RESPONSE,
                    payload={
                        "command": "handle_interactions",
                        "workflow_id": workflow_id,
                        "status": "pending_user_input",
                        "interaction_count": len(interactions),
                        "interactions": [interaction.to_dict() for interaction in interactions],
                        "timeout_seconds": self._interaction_timeout
                    },
                    correlation_id=message.id,
                    priority=Priority.MEDIUM
                )
                
                await self._event_bus.publish(response)
                
                # Notify MCP server about pending interactions
                await self._notify_mcp_server(workflow_id, interactions)
                
                logger.info(
                    "User interactions created",
                    workflow_id=workflow_id,
                    interaction_count=len(interactions)
                )
                
        except Exception as e:
            logger.error("User interactions handling failed", workflow_id=payload.get("workflow_id"), error=str(e))
            await self._send_error_response(message, str(e))
    
    async def _handle_submit_feedback(self, message: AgentMessage) -> None:
        """Handle user feedback submission."""
        try:
            payload = message.payload
            interaction_id = payload.get("interaction_id")
            user_choice = payload.get("user_choice")
            selected_labels = payload.get("selected_labels", [])
            confidence_override = payload.get("confidence_override")
            comments = payload.get("comments", "")
            
            logger.info(
                "Handling feedback submission",
                interaction_id=interaction_id,
                user_choice=user_choice
            )
            
            async with self._error_manager.protected_operation(
                f"submit_feedback_{interaction_id}",
                "user_interaction_agent"
            ):
                # Create feedback record
                feedback = UserFeedback(
                    feedback_id=f"feedback_{interaction_id}_{datetime.now().timestamp()}",
                    interaction_id=interaction_id,
                    user_choice=user_choice,
                    selected_labels=selected_labels,
                    confidence_override=confidence_override,
                    feedback_type="categorization_correction",
                    comments=comments,
                    submitted_at=datetime.now().timestamp()
                )
                
                # Save feedback
                await self._user_service.submit_feedback(feedback)
                
                # Update interaction status
                await self._user_service.update_interaction_status(
                    interaction_id, 
                    Status.COMPLETED
                )
                
                # Remove from pending interactions
                workflow_id = interaction_id.split("_")[0] + "_" + interaction_id.split("_")[1]
                if workflow_id in self._pending_interactions:
                    # Update the specific interaction in memory
                    for interaction in self._pending_interactions[workflow_id]["interactions"]:
                        if interaction.interaction_id == interaction_id:
                            interaction.status = Status.COMPLETED
                            break
                
                # Send response
                response = AgentMessage(
                    sender_agent="user_interaction_agent",
                    recipient_agent=message.sender_agent,
                    message_type=MessageType.RESPONSE,
                    payload={
                        "command": "submit_feedback",
                        "status": "completed",
                        "interaction_id": interaction_id,
                        "feedback_id": feedback.feedback_id
                    },
                    correlation_id=message.id,
                    priority=Priority.MEDIUM
                )
                
                await self._event_bus.publish(response)
                
                # Notify orchestration agent if all interactions for workflow are complete
                await self._check_workflow_completion(workflow_id)
                
                logger.info(
                    "Feedback submitted successfully",
                    interaction_id=interaction_id,
                    feedback_id=feedback.feedback_id
                )
                
        except Exception as e:
            logger.error("Feedback submission failed", interaction_id=payload.get("interaction_id"), error=str(e))
            await self._send_error_response(message, str(e))
    
    async def _handle_get_pending_interactions(self, message: AgentMessage) -> None:
        """Handle request to get pending interactions."""
        try:
            payload = message.payload
            workflow_id = payload.get("workflow_id")
            
            logger.info("Getting pending interactions", workflow_id=workflow_id)
            
            pending_interactions = []
            
            if workflow_id and workflow_id in self._pending_interactions:
                # Get specific workflow interactions
                workflow_data = self._pending_interactions[workflow_id]
                pending_interactions = [
                    interaction.to_dict() 
                    for interaction in workflow_data["interactions"]
                    if interaction.status == Status.PENDING
                ]
            else:
                # Get all pending interactions
                for wf_id, workflow_data in self._pending_interactions.items():
                    pending_interactions.extend([
                        interaction.to_dict()
                        for interaction in workflow_data["interactions"]
                        if interaction.status == Status.PENDING
                    ])
            
            response = AgentMessage(
                sender_agent="user_interaction_agent",
                recipient_agent=message.sender_agent,
                message_type=MessageType.RESPONSE,
                payload={
                    "command": "get_pending_interactions",
                    "status": "completed",
                    "workflow_id": workflow_id,
                    "pending_count": len(pending_interactions),
                    "interactions": pending_interactions
                },
                correlation_id=message.id,
                priority=Priority.LOW
            )
            
            await self._event_bus.publish(response)
            
            logger.info(
                "Pending interactions retrieved",
                workflow_id=workflow_id,
                count=len(pending_interactions)
            )
            
        except Exception as e:
            logger.error("Get pending interactions failed", workflow_id=payload.get("workflow_id"), error=str(e))
            await self._send_error_response(message, str(e))
    
    async def _handle_resolve_interaction(self, message: AgentMessage) -> None:
        """Handle direct resolution of an interaction (e.g., through MCP)."""
        try:
            payload = message.payload
            interaction_id = payload.get("interaction_id")
            resolution = payload.get("resolution", {})
            
            logger.info("Resolving interaction", interaction_id=interaction_id)
            
            async with self._error_manager.protected_operation(
                f"resolve_interaction_{interaction_id}",
                "user_interaction_agent"
            ):
                # Update interaction with resolution
                await self._user_service.update_interaction_status(
                    interaction_id,
                    Status.COMPLETED,
                    resolution_data=resolution
                )
                
                # Update in memory
                workflow_id = "_".join(interaction_id.split("_")[:2])
                if workflow_id in self._pending_interactions:
                    for interaction in self._pending_interactions[workflow_id]["interactions"]:
                        if interaction.interaction_id == interaction_id:
                            interaction.status = Status.COMPLETED
                            break
                
                response = AgentMessage(
                    sender_agent="user_interaction_agent",
                    recipient_agent=message.sender_agent,
                    message_type=MessageType.RESPONSE,
                    payload={
                        "command": "resolve_interaction",
                        "status": "completed",
                        "interaction_id": interaction_id
                    },
                    correlation_id=message.id,
                    priority=Priority.MEDIUM
                )
                
                await self._event_bus.publish(response)
                
                # Check if workflow is complete
                await self._check_workflow_completion(workflow_id)
                
                logger.info("Interaction resolved", interaction_id=interaction_id)
                
        except Exception as e:
            logger.error("Interaction resolution failed", interaction_id=payload.get("interaction_id"), error=str(e))
            await self._send_error_response(message, str(e))
    
    async def _check_workflow_completion(self, workflow_id: str) -> None:
        """Check if all interactions for a workflow are complete and notify orchestration."""
        try:
            if workflow_id not in self._pending_interactions:
                return
            
            workflow_data = self._pending_interactions[workflow_id]
            all_complete = all(
                interaction.status == Status.COMPLETED
                for interaction in workflow_data["interactions"]
            )
            
            if all_complete:
                # All interactions complete, notify orchestration agent
                completion_message = AgentMessage(
                    sender_agent="user_interaction_agent",
                    recipient_agent="orchestration_agent",
                    message_type=MessageType.EVENT,
                    payload={
                        "event": "user_interactions_completed",
                        "workflow_id": workflow_id,
                        "completed_interactions": len(workflow_data["interactions"]),
                        "completion_time": datetime.now().timestamp()
                    },
                    priority=Priority.MEDIUM
                )
                
                await self._event_bus.publish(completion_message)
                
                # Clean up from memory
                del self._pending_interactions[workflow_id]
                
                logger.info(
                    "Workflow interactions completed",
                    workflow_id=workflow_id,
                    interaction_count=len(workflow_data["interactions"])
                )
                
        except Exception as e:
            logger.error("Failed to check workflow completion", workflow_id=workflow_id, error=str(e))
    
    async def _notify_mcp_server(self, workflow_id: str, interactions: List[UserInteraction]) -> None:
        """Notify MCP server about pending interactions."""
        try:
            # This would typically send a notification to the MCP server
            # For now, we'll just log the notification
            logger.info(
                "Notifying MCP server about pending interactions",
                workflow_id=workflow_id,
                interaction_count=len(interactions),
                interaction_ids=[i.interaction_id for i in interactions]
            )
            
            # In a real implementation, this might:
            # 1. Send a webhook to the MCP server
            # 2. Update a shared state that the MCP server monitors
            # 3. Use a message queue to notify the MCP server
            
        except Exception as e:
            logger.error("Failed to notify MCP server", workflow_id=workflow_id, error=str(e))
    
    async def _cleanup_expired_interactions(self) -> None:
        """Background task to clean up expired interactions."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                current_time = datetime.now().timestamp()
                expired_workflows = []
                
                for workflow_id, workflow_data in self._pending_interactions.items():
                    # Check if any interactions have expired
                    expired_interactions = []
                    for interaction in workflow_data["interactions"]:
                        if (interaction.status == Status.PENDING and 
                            interaction.expires_at < current_time):
                            expired_interactions.append(interaction)
                    
                    if expired_interactions:
                        logger.warning(
                            "Interactions expired",
                            workflow_id=workflow_id,
                            expired_count=len(expired_interactions)
                        )
                        
                        # Mark as expired in database
                        for interaction in expired_interactions:
                            await self._user_service.update_interaction_status(
                                interaction.interaction_id,
                                Status.FAILED,
                                error_message="Interaction expired"
                            )
                            interaction.status = Status.FAILED
                        
                        # Check if workflow should be marked as complete
                        all_resolved = all(
                            interaction.status != Status.PENDING
                            for interaction in workflow_data["interactions"]
                        )
                        
                        if all_resolved:
                            expired_workflows.append(workflow_id)
                
                # Clean up expired workflows
                for workflow_id in expired_workflows:
                    del self._pending_interactions[workflow_id]
                    
                    # Notify orchestration agent about expired workflow
                    expiry_message = AgentMessage(
                        sender_agent="user_interaction_agent",
                        recipient_agent="orchestration_agent",
                        message_type=MessageType.EVENT,
                        payload={
                            "event": "user_interactions_expired",
                            "workflow_id": workflow_id,
                            "expiry_time": current_time
                        },
                        priority=Priority.LOW
                    )
                    
                    await self._event_bus.publish(expiry_message)
                    
            except Exception as e:
                logger.error("Error in interaction cleanup task", error=str(e))
                await asyncio.sleep(60)  # Continue after error
    
    async def _send_error_response(self, original_message: AgentMessage, error: str) -> None:
        """Send error response to the original sender."""
        try:
            error_response = AgentMessage(
                sender_agent="user_interaction_agent",
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
        """Get the current status of the user interaction agent."""
        try:
            # Count pending interactions
            total_pending = 0
            active_workflows = 0
            
            for workflow_data in self._pending_interactions.values():
                active_workflows += 1
                for interaction in workflow_data["interactions"]:
                    if interaction.status == Status.PENDING:
                        total_pending += 1
            
            # Get user service stats
            user_stats = await self._user_service.get_interaction_stats()
            
            return {
                "agent": "user_interaction_agent",
                "status": "healthy" if self._initialized else "initializing",
                "active_workflows": active_workflows,
                "pending_interactions": total_pending,
                "interaction_timeout": self._interaction_timeout,
                "total_interactions_today": user_stats.get("today_count", 0),
                "average_response_time": user_stats.get("avg_response_time", 0)
            }
            
        except Exception as e:
            logger.error("Failed to get agent status", error=str(e))
            return {
                "agent": "user_interaction_agent",
                "status": "error",
                "error": str(e)
            }
    
    async def get_interaction_stats(self) -> Dict[str, Any]:
        """Get interaction statistics."""
        try:
            stats = await self._user_service.get_interaction_stats()
            
            # Add current pending stats
            current_pending = 0
            for workflow_data in self._pending_interactions.values():
                for interaction in workflow_data["interactions"]:
                    if interaction.status == Status.PENDING:
                        current_pending += 1
            
            stats["current_pending"] = current_pending
            stats["active_workflows"] = len(self._pending_interactions)
            
            return stats
            
        except Exception as e:
            logger.error("Failed to get interaction stats", error=str(e))
            return {}


# Global user interaction agent instance
_user_interaction_agent: Optional[UserInteractionAgent] = None


async def get_user_interaction_agent() -> UserInteractionAgent:
    """Get the global user interaction agent instance."""
    global _user_interaction_agent
    if _user_interaction_agent is None:
        _user_interaction_agent = UserInteractionAgent()
        await _user_interaction_agent.initialize()
    return _user_interaction_agent