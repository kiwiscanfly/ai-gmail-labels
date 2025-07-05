"""Categorization service for email classification."""

import asyncio
import json
from typing import List, Dict, Any, Optional, Tuple
import structlog

from src.integrations.ollama_client import get_ollama_manager
from src.core.state_manager import get_state_manager
from src.core.transaction_manager import get_transaction_manager
from src.models.email import EmailMessage, EmailCategory
from src.models.ollama import ChatMessage, ModelConfig
from src.models.common import ConfidenceLevel
from src.core.exceptions import ServiceError

logger = structlog.get_logger(__name__)


class CategorizationService:
    """Service for categorizing emails using LLM models."""
    
    # Prompt templates
    SYSTEM_PROMPT = """You are an email categorization assistant. Your task is to analyze emails and suggest appropriate labels/categories.

Available labels: {labels}

Respond with a JSON object containing:
- suggested_labels: array of 1-3 most relevant label names
- confidence_scores: object with label names as keys and confidence scores (0-1) as values
- reasoning: brief explanation of your categorization
- requires_user_input: boolean indicating if the email is ambiguous and needs human review

Be precise and consistent in your categorization."""

    EMAIL_PROMPT = """Categorize this email:

Subject: {subject}
From: {sender}
Content: {content}

Current labels on email: {current_labels}"""
    
    def __init__(self):
        self._ollama_manager = None
        self._state_manager = None
        self._initialized = False
        self._available_labels: List[str] = []
        
    async def initialize(self, labels: List[str]) -> None:
        """Initialize the categorization service.
        
        Args:
            labels: Available email labels
        """
        try:
            self._ollama_manager = await get_ollama_manager()
            self._state_manager = await get_state_manager()
            self._available_labels = labels
            self._initialized = True
            
            logger.info(
                "Categorization service initialized",
                label_count=len(labels)
            )
        except Exception as e:
            logger.error("Failed to initialize categorization service", error=str(e))
            raise ServiceError(f"Failed to initialize categorization service: {e}")
    
    async def categorize_email(
        self,
        email: EmailMessage,
        use_chat_history: bool = True,
        model_config: Optional[ModelConfig] = None
    ) -> EmailCategory:
        """Categorize an email using LLM.
        
        Args:
            email: Email message to categorize
            use_chat_history: Whether to use previous categorizations for context
            model_config: Optional model configuration
            
        Returns:
            Email categorization result
        """
        if not self._initialized:
            raise ServiceError("Categorization service not initialized")
            
        try:
            # Prepare system prompt with available labels
            system_prompt = self.SYSTEM_PROMPT.format(
                labels=", ".join(self._available_labels)
            )
            
            # Prepare email prompt
            content_preview = email.body_text[:500] if email.body_text else email.snippet
            email_prompt = self.EMAIL_PROMPT.format(
                subject=email.subject,
                sender=email.sender,
                content=content_preview,
                current_labels=", ".join(email.label_ids) if email.label_ids else "None"
            )
            
            # Build messages
            messages = [
                ChatMessage(role="system", content=system_prompt),
                ChatMessage(role="user", content=email_prompt)
            ]
            
            # Add chat history if enabled
            if use_chat_history:
                history = await self._get_categorization_history(limit=5)
                for hist in history:
                    messages.insert(1, ChatMessage(
                        role="assistant",
                        content=f"Previous categorization: {hist}"
                    ))
            
            # Get categorization from LLM
            result = await self._ollama_manager.chat(
                messages=[msg.to_dict() for msg in messages],
                model="categorization",
                options=model_config.to_options() if model_config else {
                    "temperature": 0.3,  # Lower temperature for consistency
                    "num_predict": 256
                }
            )
            
            # Parse result
            category = self._parse_categorization_result(
                email.id,
                result.content
            )
            
            # Record feedback for learning
            await self._record_categorization(email, category)
            
            logger.info(
                "Email categorized",
                email_id=email.id,
                primary_label=category.primary_label,
                confidence=category.primary_confidence
            )
            
            return category
            
        except Exception as e:
            logger.error(
                "Failed to categorize email",
                email_id=email.id,
                error=str(e)
            )
            
            # Return low-confidence fallback
            return EmailCategory(
                email_id=email.id,
                suggested_labels=[],
                confidence_scores={},
                reasoning=f"Categorization failed: {str(e)}",
                requires_user_input=True
            )
    
    def _parse_categorization_result(
        self,
        email_id: str,
        llm_response: str
    ) -> EmailCategory:
        """Parse LLM response into EmailCategory.
        
        Args:
            email_id: Email ID
            llm_response: Raw LLM response
            
        Returns:
            Parsed email category
        """
        try:
            # Try to parse JSON response
            data = json.loads(llm_response)
            
            return EmailCategory(
                email_id=email_id,
                suggested_labels=data.get("suggested_labels", []),
                confidence_scores=data.get("confidence_scores", {}),
                reasoning=data.get("reasoning", ""),
                requires_user_input=data.get("requires_user_input", False)
            )
            
        except json.JSONDecodeError:
            # Fallback parsing for non-JSON responses
            logger.warning(
                "Failed to parse JSON response, using fallback",
                email_id=email_id
            )
            
            # Simple heuristic: extract labels mentioned in response
            suggested_labels = []
            for label in self._available_labels:
                if label.lower() in llm_response.lower():
                    suggested_labels.append(label)
            
            # Assign equal confidence to extracted labels
            confidence = 0.7 if suggested_labels else 0.3
            confidence_scores = {
                label: confidence for label in suggested_labels
            }
            
            return EmailCategory(
                email_id=email_id,
                suggested_labels=suggested_labels[:3],  # Limit to 3
                confidence_scores=confidence_scores,
                reasoning=llm_response[:200],
                requires_user_input=len(suggested_labels) == 0
            )
    
    async def batch_categorize(
        self,
        emails: List[EmailMessage],
        parallel: bool = True,
        max_concurrent: int = 5
    ) -> List[EmailCategory]:
        """Categorize multiple emails.
        
        Args:
            emails: List of emails to categorize
            parallel: Whether to process in parallel
            max_concurrent: Maximum concurrent categorizations
            
        Returns:
            List of categorization results
        """
        if not self._initialized:
            raise ServiceError("Categorization service not initialized")
            
        if not emails:
            return []
            
        if parallel:
            # Process in parallel with semaphore
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def categorize_with_limit(email):
                async with semaphore:
                    return await self.categorize_email(email)
            
            tasks = [categorize_with_limit(email) for email in emails]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions
            categories = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(
                        "Batch categorization failed for email",
                        email_id=emails[i].id,
                        error=str(result)
                    )
                    # Create error category
                    categories.append(EmailCategory(
                        email_id=emails[i].id,
                        suggested_labels=[],
                        confidence_scores={},
                        reasoning=f"Error: {str(result)}",
                        requires_user_input=True
                    ))
                else:
                    categories.append(result)
                    
            return categories
        else:
            # Process sequentially
            categories = []
            for email in emails:
                category = await self.categorize_email(email)
                categories.append(category)
            return categories
    
    async def get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Get confidence level enum from score.
        
        Args:
            confidence: Confidence score (0-1)
            
        Returns:
            Confidence level
        """
        if confidence > 0.95:
            return ConfidenceLevel.VERY_HIGH
        elif confidence > 0.85:
            return ConfidenceLevel.HIGH
        elif confidence > 0.70:
            return ConfidenceLevel.MEDIUM
        elif confidence > 0.50:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    async def _get_categorization_history(self, limit: int = 5) -> List[str]:
        """Get recent categorization history for context.
        
        Args:
            limit: Maximum history items
            
        Returns:
            List of categorization summaries
        """
        try:
            # Get recent categorization feedback
            stats = await self._state_manager.get_categorization_stats()
            
            # This is a placeholder - would query actual history
            return []
            
        except Exception as e:
            logger.error("Failed to get categorization history", error=str(e))
            return []
    
    async def _record_categorization(
        self,
        email: EmailMessage,
        category: EmailCategory
    ) -> None:
        """Record categorization for learning.
        
        Args:
            email: Categorized email
            category: Categorization result
        """
        try:
            await self._state_manager.record_feedback(
                email_id=email.id,
                suggested_label=category.primary_label,
                confidence_score=category.primary_confidence
            )
        except Exception as e:
            logger.error(
                "Failed to record categorization",
                email_id=email.id,
                error=str(e)
            )
    
    async def update_with_feedback(
        self,
        email_id: str,
        correct_label: str,
        original_category: EmailCategory
    ) -> None:
        """Update categorization with user feedback.
        
        Args:
            email_id: Email ID
            correct_label: Correct label from user
            original_category: Original categorization
        """
        try:
            await self._state_manager.record_feedback(
                email_id=email_id,
                suggested_label=original_category.primary_label,
                correct_label=correct_label,
                confidence_score=original_category.primary_confidence,
                feedback_type="correction"
            )
            
            logger.info(
                "Categorization feedback recorded",
                email_id=email_id,
                suggested=original_category.primary_label,
                correct=correct_label
            )
            
        except Exception as e:
            logger.error(
                "Failed to record feedback",
                email_id=email_id,
                error=str(e)
            )
    
    async def batch_update_feedback(
        self,
        feedback_list: List[Tuple[str, str, EmailCategory]]
    ) -> Dict[str, Any]:
        """Update multiple categorizations with user feedback atomically.
        
        Args:
            feedback_list: List of (email_id, correct_label, original_category) tuples
            
        Returns:
            Dictionary with success/failure statistics
        """
        if not self._initialized:
            raise ServiceError("Categorization service not initialized")
        
        transaction_manager = get_transaction_manager()
        results = {
            "total": len(feedback_list),
            "successful": 0,
            "failed": 0,
            "errors": []
        }
        
        try:
            # Prepare transaction operations
            operations = []
            current_time = asyncio.get_event_loop().time()
            
            for email_id, correct_label, original_category in feedback_list:
                operations.append((
                    """
                    INSERT INTO categorization_feedback 
                    (email_id, suggested_label, correct_label, confidence_score, feedback_type, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        email_id,
                        original_category.primary_label,
                        correct_label,
                        original_category.primary_confidence,
                        "batch_correction",
                        current_time
                    ),
                    "feedback_insert"
                ))
            
            # Execute all feedback records in a single transaction
            async with transaction_manager.transaction() as tx:
                for query, params, op_type in operations:
                    await tx.execute(query, params, op_type)
                    results["successful"] += 1
            
            logger.info(
                "Batch feedback update completed",
                total=results["total"],
                successful=results["successful"]
            )
            
        except Exception as e:
            results["failed"] = results["total"] - results["successful"]
            results["errors"].append(str(e))
            
            logger.error(
                "Batch feedback update failed",
                total=results["total"],
                successful=results["successful"],
                failed=results["failed"],
                error=str(e)
            )
        
        return results
    
    async def batch_record_categorizations(
        self,
        categorizations: List[Tuple[EmailMessage, EmailCategory]]
    ) -> Dict[str, Any]:
        """Record multiple categorizations atomically.
        
        Args:
            categorizations: List of (email, category) tuples
            
        Returns:
            Dictionary with success/failure statistics
        """
        if not self._initialized:
            raise ServiceError("Categorization service not initialized")
        
        transaction_manager = get_transaction_manager()
        results = {
            "total": len(categorizations),
            "successful": 0,
            "failed": 0,
            "errors": []
        }
        
        try:
            # Prepare transaction operations
            operations = []
            current_time = asyncio.get_event_loop().time()
            
            for email, category in categorizations:
                operations.append((
                    """
                    INSERT INTO categorization_feedback 
                    (email_id, suggested_label, confidence_score, feedback_type, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        email.id,
                        category.primary_label,
                        category.primary_confidence,
                        "automatic",
                        current_time
                    ),
                    "categorization_record"
                ))
            
            # Execute all records in a single transaction
            async with transaction_manager.transaction() as tx:
                for query, params, op_type in operations:
                    await tx.execute(query, params, op_type)
                    results["successful"] += 1
            
            logger.info(
                "Batch categorization recording completed",
                total=results["total"],
                successful=results["successful"]
            )
            
        except Exception as e:
            results["failed"] = results["total"] - results["successful"]
            results["errors"].append(str(e))
            
            logger.error(
                "Batch categorization recording failed",
                total=results["total"],
                successful=results["successful"],
                failed=results["failed"],
                error=str(e)
            )
        
        return results