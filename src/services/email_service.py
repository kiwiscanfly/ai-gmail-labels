"""Email service for managing email operations."""

import asyncio
from typing import List, Optional, Dict, Any, AsyncGenerator, Tuple
import structlog

from src.core.email_storage import get_email_storage
from src.core.transaction_manager import get_transaction_manager
from src.integrations.gmail_client import get_gmail_client
from src.models.email import EmailMessage, EmailReference, EmailContent, EmailCategory
from src.models.gmail import GmailLabel, GmailFilter, BatchOperation
from src.models.common import Status
from src.core.exceptions import ServiceError

logger = structlog.get_logger(__name__)


class EmailService:
    """Service for managing email operations and orchestrating email processing."""
    
    def __init__(self):
        self._gmail_client = None
        self._email_storage = None
        self._initialized = False
        
    async def initialize(self) -> None:
        """Initialize the email service."""
        try:
            self._gmail_client = await get_gmail_client()
            self._email_storage = await get_email_storage()
            self._initialized = True
            logger.info("Email service initialized")
        except Exception as e:
            logger.error("Failed to initialize email service", error=str(e))
            raise ServiceError(f"Failed to initialize email service: {e}")
    
    async def fetch_new_emails(
        self, 
        filter_criteria: Optional[GmailFilter] = None,
        store_content: bool = True
    ) -> List[EmailReference]:
        """Fetch new emails based on filter criteria.
        
        Args:
            filter_criteria: Gmail filter criteria
            store_content: Whether to store email content
            
        Returns:
            List of email references
        """
        if not self._initialized:
            await self.initialize()
            
        try:
            # Default filter if none provided
            if filter_criteria is None:
                filter_criteria = GmailFilter(
                    query="is:unread",
                    max_results=50
                )
            
            references = []
            
            # Search for messages
            async for message in self._gmail_client.search_messages(
                query=filter_criteria.query,
                max_results=filter_criteria.max_results
            ):
                # Create reference
                reference = EmailReference(
                    email_id=message.id,
                    thread_id=message.thread_id,
                    subject=message.subject,
                    sender=message.sender,
                    recipient=message.recipient,
                    date=message.date,
                    labels=message.label_ids,
                    size_estimate=message.size_estimate
                )
                
                # Store reference
                await self._email_storage.store_email_reference(reference)
                references.append(reference)
                
                # Store content if requested and email is large
                if store_content and message.size_estimate > 50000:  # 50KB
                    content = EmailContent(
                        email_id=message.id,
                        body_text=message.body_text,
                        body_html=message.body_html,
                        attachments=message.attachments
                    )
                    storage_path = await self._email_storage.store_email_content(
                        message.id, content
                    )
                    reference.storage_path = storage_path
                    
            logger.info(
                "Fetched new emails",
                count=len(references),
                filter=filter_criteria.query
            )
            
            return references
            
        except Exception as e:
            logger.error("Failed to fetch new emails", error=str(e))
            raise ServiceError(f"Failed to fetch new emails: {e}")
    
    async def get_email_content(self, email_id: str) -> Optional[EmailMessage]:
        """Get full email content, using storage optimization.
        
        Args:
            email_id: Email ID to retrieve
            
        Returns:
            EmailMessage with full content
        """
        if not self._initialized:
            await self.initialize()
            
        try:
            # Try to get from optimized storage first
            reference = await self._email_storage.get_email_reference(email_id)
            if reference and reference.storage_path:
                content = await self._email_storage.load_email_content(email_id)
                if content:
                    # Construct EmailMessage from stored data
                    message = EmailMessage(
                        id=email_id,
                        thread_id=reference.thread_id,
                        label_ids=reference.labels,
                        subject=reference.subject,
                        sender=reference.sender,
                        recipient=reference.recipient,
                        date=reference.date,
                        size_estimate=reference.size_estimate,
                        body_text=content.body_text,
                        body_html=content.body_html,
                        attachments=content.attachments
                    )
                    return message
            
            # Fallback to Gmail API
            return await self._gmail_client.get_message(email_id)
            
        except Exception as e:
            logger.error("Failed to get email content", email_id=email_id, error=str(e))
            return None
    
    async def apply_category(
        self, 
        email_id: str, 
        category: EmailCategory,
        create_labels: bool = True
    ) -> bool:
        """Apply categorization to an email.
        
        Args:
            email_id: Email ID
            category: Categorization result
            create_labels: Whether to create missing labels
            
        Returns:
            Success status
        """
        if not self._initialized:
            await self.initialize()
            
        try:
            # Get primary label to apply
            primary_label = category.primary_label
            if not primary_label:
                logger.warning("No primary label in category", email_id=email_id)
                return False
            
            # Apply label
            await self._gmail_client.apply_label_by_name(
                email_id,
                primary_label,
                create_if_missing=create_labels
            )
            
            # Mark as read if high confidence
            if category.primary_confidence > 0.9:
                await self._gmail_client.modify_labels(
                    email_id,
                    remove_label_ids=["UNREAD"]
                )
            
            logger.info(
                "Applied category to email",
                email_id=email_id,
                label=primary_label,
                confidence=category.primary_confidence
            )
            
            return True
            
        except Exception as e:
            logger.error(
                "Failed to apply category",
                email_id=email_id,
                error=str(e)
            )
            return False
    
    async def get_labels(self, include_stats: bool = False) -> List[GmailLabel]:
        """Get all available Gmail labels.
        
        Args:
            include_stats: Whether to include message count statistics
        
        Returns:
            List of Gmail labels
        """
        if not self._initialized:
            await self.initialize()
            
        return await self._gmail_client.list_labels(include_stats=include_stats)
    
    async def create_label(self, name: str) -> Optional[GmailLabel]:
        """Create a new Gmail label.
        
        Args:
            name: Label name
            
        Returns:
            Created label or None
        """
        if not self._initialized:
            await self.initialize()
            
        try:
            return await self._gmail_client.create_label(name)
        except Exception as e:
            logger.error("Failed to create label", name=name, error=str(e))
            return None
    
    async def search_emails(
        self,
        query: str,
        limit: int = 100
    ) -> AsyncGenerator[EmailReference, None]:
        """Search emails and yield references.
        
        Args:
            query: Search query
            limit: Maximum results
            
        Yields:
            Email references
        """
        if not self._initialized:
            await self.initialize()
            
        count = 0
        # Set a reasonable default limit if none provided
        max_results = limit if limit is not None else 1000
        async for message in self._gmail_client.search_messages(query, max_results=max_results):
            if limit is not None and count >= limit:
                break
                
            reference = EmailReference(
                email_id=message.id,
                thread_id=message.thread_id,
                subject=message.subject,
                sender=message.sender,
                recipient=message.recipient,
                date=message.date,
                labels=message.label_ids,
                size_estimate=message.size_estimate
            )
            
            yield reference
            count += 1
    
    async def get_email_stats(self) -> Dict[str, Any]:
        """Get email statistics.
        
        Returns:
            Statistics dictionary
        """
        if not self._initialized:
            await self.initialize()
            
        try:
            gmail_health = await self._gmail_client.get_health_status()
            storage_stats = await self._email_storage.get_storage_stats()
            
            return {
                "gmail": {
                    "total_messages": gmail_health.get("total_messages", 0),
                    "total_threads": gmail_health.get("total_threads", 0),
                    "labels_count": gmail_health.get("labels_count", 0)
                },
                "storage": storage_stats,
                "status": "healthy" if gmail_health.get("status") == "healthy" else "degraded"
            }
            
        except Exception as e:
            logger.error("Failed to get email stats", error=str(e))
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def batch_apply_labels(
        self,
        email_labels: List[Tuple[str, List[str]]],
        remove_existing: bool = False
    ) -> Dict[str, Any]:
        """Apply labels to multiple emails atomically.
        
        Args:
            email_labels: List of (email_id, label_names) tuples
            remove_existing: Whether to remove existing labels first
            
        Returns:
            Dictionary with success/failure statistics
        """
        if not self._initialized:
            await self.initialize()
        
        transaction_manager = get_transaction_manager()
        results = {
            "total": len(email_labels),
            "successful": 0,
            "failed": 0,
            "errors": []
        }
        
        try:
            # Get all available labels
            all_labels = await self._gmail_client.list_labels()
            label_map = {label.name: label.id for label in all_labels}
            
            # Prepare batch operations
            batch_ops = []
            
            for email_id, label_names in email_labels:
                # Convert label names to IDs
                label_ids = []
                for name in label_names:
                    if name in label_map:
                        label_ids.append(label_map[name])
                    else:
                        logger.warning(f"Label not found: {name}")
                
                if label_ids:
                    batch_ops.append(BatchOperation(
                        operation_type="add_labels",
                        email_ids=[email_id],
                        label_ids=label_ids
                    ))
                    
                    if remove_existing:
                        # Get current labels for removal
                        current_email = await self.get_email(email_id)
                        if current_email and current_email.label_ids:
                            existing_to_remove = [
                                lid for lid in current_email.label_ids 
                                if lid not in label_ids
                            ]
                            if existing_to_remove:
                                batch_ops.append(BatchOperation(
                                    operation_type="remove_labels",
                                    email_ids=[email_id],
                                    label_ids=existing_to_remove
                                ))
            
            # Execute batch operations atomically
            async with transaction_manager.transaction() as tx:
                # Use transaction context for database operations if needed
                # For Gmail API, we rely on Gmail's batch processing
                for batch_op in batch_ops:
                    await self._gmail_client.execute_batch_operation(batch_op)
                    results["successful"] += 1
                
                # Log transaction completion in database
                await tx.execute(
                    "INSERT INTO email_queue (email_id, status, metadata) VALUES (?, ?, ?)",
                    (
                        f"batch_{asyncio.get_event_loop().time()}",
                        "completed",
                        f"Applied labels to {len(email_labels)} emails"
                    ),
                    "batch_label_application"
                )
            
            logger.info(
                "Batch label application completed",
                total=results["total"],
                successful=results["successful"]
            )
            
        except Exception as e:
            results["failed"] = results["total"] - results["successful"]
            results["errors"].append(str(e))
            
            logger.error(
                "Batch label application failed",
                total=results["total"],
                successful=results["successful"],
                failed=results["failed"],
                error=str(e)
            )
        
        return results
    
    async def batch_categorize_and_apply(
        self,
        emails: List[EmailMessage],
        categories: List[EmailCategory]
    ) -> Dict[str, Any]:
        """Categorize and apply labels to multiple emails atomically.
        
        Args:
            emails: List of emails to process
            categories: Corresponding categorization results
            
        Returns:
            Dictionary with success/failure statistics
        """
        if not self._initialized:
            await self.initialize()
        
        if len(emails) != len(categories):
            raise ServiceError("Email and category lists must have same length")
        
        transaction_manager = get_transaction_manager()
        results = {
            "total": len(emails),
            "successful": 0,
            "failed": 0,
            "errors": []
        }
        
        try:
            async with transaction_manager.transaction() as tx:
                for email, category in zip(emails, categories):
                    try:
                        # Apply categorization
                        success = await self.apply_category(email.id, category)
                        
                        if success:
                            # Record successful categorization in database
                            await tx.execute(
                                """
                                INSERT INTO email_metadata 
                                (email_id, subject, sender, categorized_labels, confidence_score, processed_at)
                                VALUES (?, ?, ?, ?, ?, ?)
                                """,
                                (
                                    email.id,
                                    email.subject,
                                    email.sender,
                                    ",".join(category.suggested_labels),
                                    category.primary_confidence,
                                    asyncio.get_event_loop().time()
                                ),
                                "email_categorization"
                            )
                            results["successful"] += 1
                        else:
                            results["failed"] += 1
                            results["errors"].append(f"Failed to apply category to {email.id}")
                            
                    except Exception as e:
                        results["failed"] += 1
                        results["errors"].append(f"Error processing {email.id}: {str(e)}")
            
            logger.info(
                "Batch categorization and application completed",
                total=results["total"],
                successful=results["successful"],
                failed=results["failed"]
            )
            
        except Exception as e:
            results["errors"].append(str(e))
            logger.error(
                "Batch categorization and application failed",
                error=str(e)
            )
        
        return results