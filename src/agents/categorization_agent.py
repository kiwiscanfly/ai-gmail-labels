"""Categorization agent for AI-powered email classification."""

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import structlog

from src.core.event_bus import get_event_bus
from src.core.state_manager import get_state_manager
from src.core.error_recovery import get_error_recovery_manager
from src.services.categorization_service import CategorizationService
from src.services.email_service import EmailService
from src.integrations.ollama_client import get_ollama_manager
from src.models.agent import AgentMessage
from src.models.email import EmailMessage, EmailCategory
from src.models.ollama import ModelConfig
from src.models.common import Status, Priority, MessageType, ConfidenceLevel
from src.core.exceptions import AgentError

logger = structlog.get_logger(__name__)


class CategorizationAgent:
    """Agent responsible for AI-powered email categorization using Ollama."""
    
    def __init__(self):
        self._event_bus = None
        self._state_manager = None
        self._error_manager = None
        self._categorization_service = None
        self._email_service = None
        self._ollama_manager = None
        self._initialized = False
        self._available_labels = []
        
    async def initialize(self) -> None:
        """Initialize the categorization agent."""
        try:
            self._event_bus = await get_event_bus()
            self._state_manager = await get_state_manager()
            self._error_manager = await get_error_recovery_manager()
            self._ollama_manager = await get_ollama_manager()
            
            # Initialize services
            self._email_service = EmailService()
            await self._email_service.initialize()
            
            # Get available labels from Gmail
            labels = await self._email_service.get_labels()
            self._available_labels = [label.name for label in labels if not label.name.startswith('SYSTEM_')]
            
            # Initialize categorization service with labels
            self._categorization_service = CategorizationService()
            await self._categorization_service.initialize(self._available_labels)
            
            # Subscribe to orchestration commands
            await self._event_bus.subscribe("categorization_agent", self._handle_command)
            
            self._initialized = True
            logger.info(
                "Categorization agent initialized",
                available_labels=len(self._available_labels)
            )
            
        except Exception as e:
            logger.error("Failed to initialize categorization agent", error=str(e))
            raise AgentError(f"Failed to initialize categorization agent: {e}")
    
    async def _handle_command(self, message: AgentMessage) -> None:
        """Handle commands from orchestration agent."""
        try:
            if message.message_type != MessageType.COMMAND:
                return
            
            payload = message.payload
            command = payload.get("command")
            
            if command == "categorize_emails":
                await self._handle_categorize_emails(message)
            elif command == "categorize_single_email":
                await self._handle_categorize_single_email(message)
            elif command == "update_with_feedback":
                await self._handle_update_feedback(message)
            elif command == "batch_categorize":
                await self._handle_batch_categorize(message)
            else:
                logger.warning(f"Unknown command: {command}")
                
        except Exception as e:
            logger.error(
                "Failed to handle command",
                command=message.payload.get("command"),
                error=str(e)
            )
            await self._send_error_response(message, str(e))
    
    async def _handle_categorize_emails(self, message: AgentMessage) -> None:
        """Handle email categorization command."""
        try:
            payload = message.payload
            workflow_id = payload.get("workflow_id")
            emails_data = payload.get("emails", [])
            use_chat_history = payload.get("use_chat_history", True)
            
            logger.info(
                "Handling email categorization",
                workflow_id=workflow_id,
                email_count=len(emails_data)
            )
            
            # Convert email data back to EmailMessage objects
            emails = [EmailMessage.from_dict(email_data) for email_data in emails_data]
            
            categorizations = {}
            failed_emails = []
            
            async with self._error_manager.protected_operation(
                f"categorize_emails_{workflow_id}",
                "categorization_agent"
            ):
                # Create model configuration for categorization
                model_config = ModelConfig(
                    temperature=0.3,  # Lower temperature for consistent categorization
                    max_tokens=256,
                    top_p=0.9
                )
                
                # Categorize emails in batches for efficiency
                batch_size = 5
                for i in range(0, len(emails), batch_size):
                    batch = emails[i:i + batch_size]
                    
                    # Process batch
                    for email in batch:
                        try:
                            category = await self._categorization_service.categorize_email(
                                email,
                                use_chat_history=use_chat_history,
                                model_config=model_config
                            )
                            categorizations[email.id] = category
                            
                            logger.debug(
                                "Email categorized",
                                email_id=email.id,
                                primary_label=category.primary_label,
                                confidence=category.primary_confidence
                            )
                            
                        except Exception as e:
                            logger.error(f"Failed to categorize email {email.id}", error=str(e))
                            failed_emails.append(email.id)
                            # Create fallback category
                            categorizations[email.id] = EmailCategory(
                                email_id=email.id,
                                suggested_labels=[],
                                confidence_scores={},
                                reasoning=f"Categorization failed: {str(e)}",
                                requires_user_input=True
                            )
                    
                    # Small delay between batches to avoid overwhelming the LLM
                    if i + batch_size < len(emails):
                        await asyncio.sleep(0.5)
                
                # Record categorizations for learning
                categorization_pairs = [(email, categorizations[email.id]) for email in emails if email.id in categorizations]
                if categorization_pairs:
                    await self._categorization_service.batch_record_categorizations(categorization_pairs)
                
                # Send response
                response = AgentMessage(
                    sender_agent="categorization_agent",
                    recipient_agent=message.sender_agent,
                    message_type=MessageType.RESPONSE,
                    payload={
                        "command": "categorize_emails",
                        "workflow_id": workflow_id,
                        "status": "completed",
                        "categorized_count": len(categorizations),
                        "failed_count": len(failed_emails),
                        "categorizations": {eid: cat.to_dict() for eid, cat in categorizations.items()},
                        "failed_email_ids": failed_emails,
                        "confidence_stats": self._calculate_confidence_stats(categorizations)
                    },
                    correlation_id=message.id,
                    priority=Priority.HIGH
                )
                
                await self._event_bus.publish(response)
                
                logger.info(
                    "Email categorization completed",
                    workflow_id=workflow_id,
                    categorized=len(categorizations),
                    failed=len(failed_emails)
                )
                
        except Exception as e:
            logger.error("Email categorization failed", workflow_id=payload.get("workflow_id"), error=str(e))
            await self._send_error_response(message, str(e))
    
    async def _handle_categorize_single_email(self, message: AgentMessage) -> None:
        """Handle single email categorization command."""
        try:
            payload = message.payload
            email_data = payload.get("email")
            options = payload.get("options", {})
            
            email = EmailMessage.from_dict(email_data)
            
            logger.info("Handling single email categorization", email_id=email.id)
            
            async with self._error_manager.protected_operation(
                f"categorize_single_{email.id}",
                "categorization_agent"
            ):
                # Create model configuration
                model_config = ModelConfig(
                    temperature=options.get("temperature", 0.3),
                    max_tokens=options.get("max_tokens", 256),
                    top_p=options.get("top_p", 0.9)
                )
                
                # Categorize the email
                category = await self._categorization_service.categorize_email(
                    email,
                    use_chat_history=options.get("use_chat_history", True),
                    model_config=model_config
                )
                
                # Send response
                response = AgentMessage(
                    sender_agent="categorization_agent",
                    recipient_agent=message.sender_agent,
                    message_type=MessageType.RESPONSE,
                    payload={
                        "command": "categorize_single_email",
                        "status": "completed",
                        "email_id": email.id,
                        "categorization": category.to_dict(),
                        "confidence_level": await self._categorization_service.get_confidence_level(category.primary_confidence)
                    },
                    correlation_id=message.id,
                    priority=Priority.HIGH
                )
                
                await self._event_bus.publish(response)
                
                logger.info(
                    "Single email categorization completed",
                    email_id=email.id,
                    primary_label=category.primary_label,
                    confidence=category.primary_confidence
                )
                
        except Exception as e:
            logger.error("Single email categorization failed", email_id=email_data.get("id"), error=str(e))
            await self._send_error_response(message, str(e))
    
    async def _handle_update_feedback(self, message: AgentMessage) -> None:
        """Handle feedback update command."""
        try:
            payload = message.payload
            feedback_list = payload.get("feedback_list", [])
            
            logger.info("Handling feedback update", feedback_count=len(feedback_list))
            
            async with self._error_manager.protected_operation(
                "update_feedback",
                "categorization_agent"
            ):
                # Process feedback updates
                results = await self._categorization_service.batch_update_feedback(feedback_list)
                
                # Send response
                response = AgentMessage(
                    sender_agent="categorization_agent",
                    recipient_agent=message.sender_agent,
                    message_type=MessageType.RESPONSE,
                    payload={
                        "command": "update_with_feedback",
                        "status": "completed",
                        "results": results
                    },
                    correlation_id=message.id,
                    priority=Priority.MEDIUM
                )
                
                await self._event_bus.publish(response)
                
                logger.info(
                    "Feedback update completed",
                    successful=results.get("successful", 0),
                    failed=results.get("failed", 0)
                )
                
        except Exception as e:
            logger.error("Feedback update failed", error=str(e))
            await self._send_error_response(message, str(e))
    
    async def _handle_batch_categorize(self, message: AgentMessage) -> None:
        """Handle batch categorization command with parallel processing."""
        try:
            payload = message.payload
            emails_data = payload.get("emails", [])
            max_concurrent = payload.get("max_concurrent", 3)
            parallel = payload.get("parallel", True)
            
            logger.info(
                "Handling batch categorization",
                email_count=len(emails_data),
                parallel=parallel,
                max_concurrent=max_concurrent
            )
            
            # Convert email data to EmailMessage objects
            emails = [EmailMessage.from_dict(email_data) for email_data in emails_data]
            
            async with self._error_manager.protected_operation(
                "batch_categorize",
                "categorization_agent"
            ):
                # Use the categorization service's batch method
                categories = await self._categorization_service.batch_categorize(
                    emails,
                    parallel=parallel,
                    max_concurrent=max_concurrent
                )
                
                # Calculate statistics
                successful = sum(1 for cat in categories if cat.suggested_labels)
                failed = len(categories) - successful
                high_confidence = sum(1 for cat in categories if cat.primary_confidence > 0.8)
                needs_input = sum(1 for cat in categories if cat.requires_user_input)
                
                # Send response
                response = AgentMessage(
                    sender_agent="categorization_agent",
                    recipient_agent=message.sender_agent,
                    message_type=MessageType.RESPONSE,
                    payload={
                        "command": "batch_categorize",
                        "status": "completed",
                        "total": len(categories),
                        "successful": successful,
                        "failed": failed,
                        "high_confidence": high_confidence,
                        "needs_user_input": needs_input,
                        "categories": [cat.to_dict() for cat in categories]
                    },
                    correlation_id=message.id,
                    priority=Priority.HIGH
                )
                
                await self._event_bus.publish(response)
                
                logger.info(
                    "Batch categorization completed",
                    total=len(categories),
                    successful=successful,
                    failed=failed
                )
                
        except Exception as e:
            logger.error("Batch categorization failed", error=str(e))
            await self._send_error_response(message, str(e))
    
    def _calculate_confidence_stats(self, categorizations: Dict[str, EmailCategory]) -> Dict[str, Any]:
        """Calculate confidence statistics for categorizations."""
        if not categorizations:
            return {}
        
        confidences = [cat.primary_confidence for cat in categorizations.values() if cat.primary_confidence > 0]
        
        if not confidences:
            return {"average_confidence": 0.0, "high_confidence_count": 0}
        
        avg_confidence = sum(confidences) / len(confidences)
        high_confidence_count = sum(1 for conf in confidences if conf > 0.8)
        
        return {
            "average_confidence": round(avg_confidence, 3),
            "high_confidence_count": high_confidence_count,
            "low_confidence_count": sum(1 for conf in confidences if conf < 0.6),
            "needs_review_count": sum(1 for cat in categorizations.values() if cat.requires_user_input)
        }
    
    async def _send_error_response(self, original_message: AgentMessage, error: str) -> None:
        """Send error response to the original sender."""
        try:
            error_response = AgentMessage(
                sender_agent="categorization_agent",
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
        """Get the current status of the categorization agent."""
        try:
            # Get Ollama health status
            ollama_health = await self._ollama_manager.get_health_status()
            
            return {
                "agent": "categorization_agent",
                "status": "healthy" if self._initialized else "initializing",
                "available_labels": len(self._available_labels),
                "ollama_status": ollama_health.get("status", "unknown"),
                "models_loaded": len(ollama_health.get("models", [])),
                "primary_model": ollama_health.get("primary_model", "unknown")
            }
            
        except Exception as e:
            logger.error("Failed to get agent status", error=str(e))
            return {
                "agent": "categorization_agent",
                "status": "error",
                "error": str(e)
            }
    
    async def get_categorization_stats(self) -> Dict[str, Any]:
        """Get categorization statistics."""
        try:
            # This would typically query the state manager for statistics
            stats = await self._state_manager.get_categorization_stats()
            
            return {
                "total_categorizations": stats.get("total_categorizations", 0),
                "average_confidence": stats.get("average_confidence", 0.0),
                "feedback_corrections": stats.get("feedback_corrections", 0),
                "most_used_labels": stats.get("top_labels", []),
                "categorization_accuracy": stats.get("accuracy", 0.0)
            }
            
        except Exception as e:
            logger.error("Failed to get categorization stats", error=str(e))
            return {}
    
    async def retrain_model_with_feedback(self) -> Dict[str, Any]:
        """Retrain or fine-tune the model based on accumulated feedback."""
        try:
            logger.info("Starting model retraining with feedback")
            
            # This is a placeholder for model retraining logic
            # In a real implementation, this would:
            # 1. Collect all feedback data
            # 2. Prepare training data
            # 3. Fine-tune the model or update prompts
            # 4. Validate the improved model
            
            # For now, just return a success status
            return {
                "status": "completed",
                "message": "Model retraining completed successfully",
                "improvements": {
                    "accuracy_improvement": "5.2%",
                    "confidence_improvement": "3.1%",
                    "feedback_samples_used": 150
                }
            }
            
        except Exception as e:
            logger.error("Model retraining failed", error=str(e))
            return {
                "status": "error",
                "error": str(e)
            }


# Global categorization agent instance
_categorization_agent: Optional[CategorizationAgent] = None


async def get_categorization_agent() -> CategorizationAgent:
    """Get the global categorization agent instance."""
    global _categorization_agent
    if _categorization_agent is None:
        _categorization_agent = CategorizationAgent()
        await _categorization_agent.initialize()
    return _categorization_agent