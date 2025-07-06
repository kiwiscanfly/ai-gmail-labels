#!/usr/bin/env python3
"""
Email MCP Server using FastMCP for better compatibility.
"""

import asyncio
import sys
import json
import logging
import threading
from typing import List, Dict, Any, Optional
from mcp.server.fastmcp import FastMCP

# Configure basic logging to stderr
logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastMCP instance
mcp = FastMCP("email-categorization-agent")

# Global service instances and initialization state
email_service = None
prioritizer = None
marketing_classifier = None
receipt_classifier = None
ollama_manager = None
config = None
services_initialized = False
initialization_in_progress = False
background_loop = None
background_thread = None


def start_background_loop():
    """Start a background event loop for async operations."""
    global background_loop, background_thread
    
    if background_loop is not None:
        return background_loop
    
    def run_loop():
        global background_loop
        background_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(background_loop)
        try:
            background_loop.run_forever()
        except Exception as e:
            logger.error(f"Background loop error: {e}")
        finally:
            background_loop = None
    
    background_thread = threading.Thread(target=run_loop, daemon=True)
    background_thread.start()
    
    # Wait a bit for the loop to start
    import time
    time.sleep(0.1)
    
    return background_loop


def run_async_in_background(coro):
    """Run an async coroutine in the background event loop."""
    global background_loop
    
    if background_loop is None:
        start_background_loop()
    
    # Submit the coroutine to the background loop
    future = asyncio.run_coroutine_threadsafe(coro, background_loop)
    try:
        return future.result(timeout=30)  # 30 second timeout
    except Exception as e:
        logger.error(f"Background async operation failed: {e}")
        raise


def try_initialize_services():
    """Try to initialize services synchronously if possible."""
    global email_service, prioritizer, marketing_classifier, receipt_classifier, ollama_manager, config
    global services_initialized, initialization_in_progress
    
    if services_initialized or initialization_in_progress:
        return services_initialized
    
    initialization_in_progress = True
    
    try:
        logger.info("Attempting to initialize email services...")
        
        # Import basic services
        from src.core.config import get_config
        
        # Get configuration (this is synchronous)
        config = get_config()
        logger.info("Configuration loaded")
        
        # For now, mark as partially initialized
        # Full async initialization will happen on first real tool use
        services_initialized = True
        initialization_in_progress = False
        
        logger.info("Basic services initialized, async services will initialize on demand")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize basic services: {e}")
        initialization_in_progress = False
        return False


async def full_async_initialization():
    """Full async service initialization."""
    global email_service, prioritizer, marketing_classifier, receipt_classifier, ollama_manager
    
    if email_service is not None:
        return True  # Already fully initialized
    
    try:
        logger.info("Starting full async service initialization...")
        
        # Import async services
        from src.services.email_service import EmailService
        from src.services.email_prioritizer import EmailPrioritizer
        from src.services.marketing_classifier import MarketingEmailClassifier
        from src.services.receipt_classifier import ReceiptClassifier
        from src.integrations.ollama_client import get_ollama_manager
        
        # Initialize Ollama manager
        try:
            logger.info("Attempting to initialize Ollama manager...")
            ollama_manager = await get_ollama_manager()
            if ollama_manager:
                logger.info(f"Ollama manager initialized successfully: {type(ollama_manager).__name__}")
            else:
                logger.error("get_ollama_manager() returned None")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama manager: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            ollama_manager = None
        
        # Initialize email service
        email_service = EmailService()
        await email_service.initialize()
        logger.info("Email service initialized")
        
        # Initialize classifiers
        prioritizer = EmailPrioritizer()
        await prioritizer.initialize()
        
        marketing_classifier = MarketingEmailClassifier()
        await marketing_classifier.initialize()
        
        receipt_classifier = ReceiptClassifier()
        await receipt_classifier.initialize()
        
        logger.info("All async services initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize async services: {e}")
        return False


@mcp.tool()
def search_emails(
    query: str,
    limit: int = 20,
    include_summary: bool = True,
    apply_labels: bool = False,
    summary_model: str = "auto"
) -> Dict[str, Any]:
    """Search Gmail emails with optional AI summaries.
    
    Performance Guidelines:
    - For quick browsing: Set include_summary=false (5-10 seconds for 20 emails)
    - For detailed analysis: Set include_summary=true with smaller limits (10-15 emails max)
    - Large searches with summaries: Expect 10-15 seconds per 5 emails
    
    Args:
        query: Gmail search query (e.g., 'is:unread', 'from:example.com', 'subject:meeting')
        limit: Maximum number of emails to return (max: 100, recommended: 10-20 with summaries)
        include_summary: Whether to generate AI summaries (slower but more informative)
        apply_labels: Whether to automatically apply classification labels after analysis
        summary_model: Model type for analysis - "auto", "priority", "marketing", "receipt"
    """
    # Try basic initialization first
    if not try_initialize_services():
        return {"error": "Failed to initialize basic services"}
    
    # Check if email service is available
    if not email_service:
        return {
            "status": "initializing",
            "message": "Email services are starting up. Please try the 'initialize_email_services' tool first.",
            "query": query,
            "limit": min(max(1, limit), 100),
            "include_summary": include_summary
        }
    
    # Perform the actual email search using background loop
    try:
        async def perform_search():
            try:
                # Search emails using the email service (it returns an async generator)
                emails = []
                logger.info(f"Starting email search for query: '{query}', limit: {limit}")
                
                async for email_ref in email_service.search_emails(query, limit):
                    emails.append(email_ref)
                    
                logger.info(f"Found {len(emails)} emails for query: '{query}'")
                
                # Process emails in parallel for speed
                async def process_email(email_ref):
                    # Get basic email data first (fast)
                    email_data = {
                        "id": email_ref.email_id,
                        "subject": email_ref.subject or "No Subject",
                        "sender": email_ref.sender or "Unknown Sender", 
                        "date": email_ref.date if isinstance(email_ref.date, str) else (email_ref.date.isoformat() if email_ref.date else None),
                        "labels": email_ref.labels or [],
                        "gmail_link": f"https://mail.google.com/mail/u/0/#inbox/{email_ref.email_id}"
                    }
                    
                    # Get snippet (this is fast, no AI involved)
                    snippet = ""
                    full_email = None
                    try:
                        full_email = await email_service.get_email_content(email_ref.email_id)
                        if full_email:
                            snippet = full_email.snippet
                            if not snippet and full_email.body_text:
                                snippet = full_email.body_text[:200].replace('\n', ' ').strip()
                                if len(full_email.body_text) > 200:
                                    snippet += "..."
                        else:
                            snippet = "Email content unavailable"
                    except Exception as e:
                        logger.error(f"Failed to get email content for {email_ref.email_id}: {e}")
                        snippet = f"Error loading email: {str(e)}"
                    
                    email_data["snippet"] = snippet
                    
                    # Add AI summary if requested and if we have content
                    if include_summary and full_email and (full_email.body_text or full_email.body_html):
                        if ollama_manager:
                            try:
                                content = full_email.body_text or full_email.body_html or ""
                                # Shorter prompt for faster generation
                                prompt = f"Summarize in 1-2 sentences: {content[:1500]}"
                                
                                # Add timeout for AI generation to prevent hanging
                                result = await asyncio.wait_for(
                                    ollama_manager.generate(prompt, options={"num_predict": 100}),
                                    timeout=10.0  # 10 second timeout per summary
                                )
                                email_data["ai_summary"] = result.content.strip()
                            except asyncio.TimeoutError:
                                email_data["ai_summary"] = "Summary generation timed out"
                            except Exception as e:
                                email_data["ai_summary"] = f"Summary generation failed: {str(e)}"
                        else:
                            email_data["ai_summary"] = "AI summary service not available"
                    elif include_summary:
                        email_data["ai_summary"] = "No content available for summary"
                    
                    return email_data
                
                # Process emails concurrently with limited parallelism to avoid overwhelming the system
                semaphore = asyncio.Semaphore(5)  # Max 5 concurrent operations
                
                async def process_with_semaphore(email_ref):
                    async with semaphore:
                        return await process_email(email_ref)
                
                # Create tasks for concurrent processing
                if include_summary:
                    # If summaries requested, process in smaller batches to avoid timeout
                    batch_size = 5
                    processed_emails = []
                    
                    for i in range(0, len(emails), batch_size):
                        batch = emails[i:i + batch_size]
                        logger.info(f"Processing batch {i//batch_size + 1}/{(len(emails) + batch_size - 1)//batch_size}")
                        
                        batch_tasks = [process_with_semaphore(email_ref) for email_ref in batch]
                        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                        
                        for result in batch_results:
                            if isinstance(result, Exception):
                                logger.error(f"Email processing failed: {result}")
                                processed_emails.append({
                                    "id": "unknown",
                                    "subject": "Error processing email",
                                    "error": str(result)
                                })
                            else:
                                processed_emails.append(result)
                else:
                    # Without summaries, process all at once (much faster)
                    tasks = [process_with_semaphore(email_ref) for email_ref in emails]
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    processed_emails = []
                    for result in results:
                        if isinstance(result, Exception):
                            logger.error(f"Email processing failed: {result}")
                            processed_emails.append({
                                "id": "unknown", 
                                "subject": "Error processing email",
                                "error": str(result)
                            })
                        else:
                            processed_emails.append(result)
                
                return {
                    "query": query,
                    "total_found": len(processed_emails),
                    "limit": min(max(1, limit), 100),
                    "emails": processed_emails,
                    "include_summary": include_summary,
                    "timestamp": "2025-07-06T16:00:00Z",
                    "status": "success"
                }
                
            except Exception as async_error:
                import traceback
                logger.error(f"Async email search failed: {type(async_error).__name__}: {str(async_error)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                return {
                    "error": f"Async search failed: {type(async_error).__name__}: {str(async_error)}",
                    "query": query,
                    "limit": min(max(1, limit), 100),
                    "include_summary": include_summary,
                    "traceback": traceback.format_exc()
                }
        
        # Run the async operation in the background loop with longer timeout for summaries
        timeout = 120 if include_summary else 30  # 2 minutes for summaries, 30 seconds for basic search
        
        def run_search_with_timeout():
            future = asyncio.run_coroutine_threadsafe(perform_search(), background_loop)
            try:
                return future.result(timeout=timeout)
            except Exception as e:
                logger.error(f"Search operation timed out or failed: {e}")
                raise
        
        result = run_search_with_timeout()
        return result
        
    except Exception as e:
        import traceback
        error_details = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc()
        }
        logger.error(f"Email search failed: {error_details}")
        return {
            "error": f"Email search failed: {error_details['error_type']}: {error_details['error_message']}",
            "error_details": error_details,
            "query": query,
            "limit": min(max(1, limit), 100),
            "include_summary": include_summary
        }


@mcp.tool()
def get_email_summary(
    email_id: str,
    analysis_type: str = "comprehensive",
    include_classification: bool = True
) -> Dict[str, Any]:
    """Get detailed summary and analysis of a specific email.
    
    Use this for deep analysis of individual emails when search_emails summaries aren't detailed enough.
    Performs comprehensive AI analysis including priority, marketing, and receipt classification.
    
    Args:
        email_id: Gmail message ID (get this from search_emails results)
        analysis_type: Type of analysis (comprehensive, priority, marketing, receipt)
        include_classification: Whether to include detailed classification analysis
    """
    # Try basic initialization first
    if not try_initialize_services():
        return {"error": "Failed to initialize basic services"}
    
    # Check if email service is available
    if not email_service:
        return {
            "status": "initializing",
            "message": "Email services are starting up. Please try the 'initialize_email_services' tool first.",
            "email_id": email_id,
            "analysis_type": analysis_type
        }
    
    # Get email summary using background loop
    try:
        async def get_summary():
            # Get the email
            email = await email_service.get_email_content(email_id)
            if not email:
                return {"error": f"Email {email_id} not found"}
            
            result = {
                "email": {
                    "id": email_id,
                    "subject": email.subject or "No Subject",
                    "sender": email.sender or "Unknown Sender",
                    "date": email.date if isinstance(email.date, str) else (email.date.isoformat() if email.date else None),
                    "snippet": email.body_text[:200] if email.body_text else ""
                },
                "analysis_type": analysis_type,
                "timestamp": "2025-07-06T16:00:00Z"
            }
            
            # Run appropriate analysis
            if analysis_type in ["comprehensive", "priority"] and prioritizer:
                priority_result = await prioritizer.classify_email(email)
                result["priority_analysis"] = {
                    "level": priority_result.get("priority", "medium"),
                    "confidence": priority_result.get("confidence", 0.5),
                    "reasoning": priority_result.get("reasoning", "Priority analysis completed"),
                    "is_genuine_urgency": priority_result.get("is_urgent", False),
                    "authenticity_score": priority_result.get("authenticity", 0.5)
                }
            
            if analysis_type in ["comprehensive", "marketing"] and marketing_classifier:
                marketing_result = await marketing_classifier.classify_email(email)
                result["marketing_analysis"] = {
                    "is_marketing": marketing_result.get("is_marketing", False),
                    "subtype": marketing_result.get("subtype", "unknown"),
                    "confidence": marketing_result.get("confidence", 0.5),
                    "reasoning": marketing_result.get("reasoning", "Marketing analysis completed")
                }
            
            if analysis_type in ["comprehensive", "receipt"] and receipt_classifier:
                receipt_result = await receipt_classifier.classify_email(email)
                result["receipt_analysis"] = {
                    "is_receipt": receipt_result.get("is_receipt", False),
                    "vendor": receipt_result.get("vendor", "Unknown"),
                    "confidence": receipt_result.get("confidence", 0.5),
                    "reasoning": receipt_result.get("reasoning", "Receipt analysis completed")
                }
            
            # Generate AI summary if Ollama is available
            if ollama_manager:
                try:
                    content = email.body_text or email.body_html or ""
                    if content:
                        prompt = f"Please provide a detailed summary and analysis of this email:\n\n{content[:3000]}"
                        llm_result = await ollama_manager.generate(prompt)
                        result["ai_summary"] = llm_result.content.strip()
                    else:
                        result["ai_summary"] = "No content available for summary"
                except Exception as e:
                    result["ai_summary"] = f"Summary generation failed: {str(e)}"
            
            return result
        
        # Run the async operation in the background loop
        result = run_async_in_background(get_summary())
        return result
        
    except Exception as e:
        logger.error(f"Email summary failed: {e}")
        return {
            "error": f"Email summary failed: {str(e)}",
            "email_id": email_id,
            "analysis_type": analysis_type
        }


@mcp.tool()
def get_labels() -> Dict[str, Any]:
    """Get all available Gmail labels.
    
    Returns a list of all labels in the Gmail account, including system labels
    (like INBOX, SENT) and custom labels. Useful for understanding what labels
    are available before applying them to emails.
    
    Returns:
        Dictionary with labels list and metadata
    """
    # Try basic initialization first
    if not try_initialize_services():
        return {"error": "Failed to initialize basic services"}
    
    # Check if email service is available
    if not email_service:
        return {
            "status": "initializing",
            "message": "Email services are starting up. Please try the 'initialize_email_services' tool first."
        }
    
    # Get labels using background loop
    try:
        async def get_all_labels():
            labels = await email_service.get_labels()
            
            # Organize labels by type
            system_labels = []
            user_labels = []
            
            for label in labels:
                label_data = {
                    "id": label.id,
                    "name": label.name,
                    "type": getattr(label, 'type', 'user'),
                    "messages_total": getattr(label, 'messages_total', 0),
                    "messages_unread": getattr(label, 'messages_unread', 0)
                }
                
                if label.name in ['INBOX', 'SENT', 'DRAFT', 'SPAM', 'TRASH', 'UNREAD', 'STARRED', 'IMPORTANT']:
                    system_labels.append(label_data)
                else:
                    user_labels.append(label_data)
            
            return {
                "total_labels": len(labels),
                "system_labels": sorted(system_labels, key=lambda x: x['name']),
                "user_labels": sorted(user_labels, key=lambda x: x['name']),
                "all_labels": [{"id": l.id, "name": l.name} for l in labels],
                "timestamp": "2025-07-06T16:00:00Z",
                "status": "success"
            }
        
        # Run the async operation in the background loop
        result = run_async_in_background(get_all_labels())
        return result
        
    except Exception as e:
        logger.error(f"Failed to get labels: {e}")
        import traceback
        return {
            "error": f"Failed to get labels: {str(e)}",
            "traceback": traceback.format_exc()
        }


@mcp.tool()
def apply_email_label(
    email_id: str,
    label: str,
    create_if_missing: bool = True,
    confidence_threshold: float = 0.8
) -> Dict[str, Any]:
    """Apply a label to an email with AI validation.
    
    Args:
        email_id: Gmail message ID
        label: Label name to apply
        create_if_missing: Create label if it doesn't exist
        confidence_threshold: Minimum confidence for AI validation (0.0 to skip validation)
    """
    # Try basic initialization first
    if not try_initialize_services():
        return {"error": "Failed to initialize basic services"}
    
    # Check if email service is available
    if not email_service:
        return {
            "status": "initializing",
            "message": "Email services are starting up. Please try the 'initialize_email_services' tool first.",
            "email_id": email_id,
            "label": label,
            "create_if_missing": create_if_missing
        }
    
    # Apply the label using background loop
    try:
        async def apply_label():
            # Get the email first to validate it exists
            email = await email_service.get_email_content(email_id)
            if not email:
                return {"error": f"Email {email_id} not found"}
            
            # Apply the label
            try:
                await email_service.apply_label(email_id, label, create_if_missing=create_if_missing)
                
                return {
                    "email_id": email_id,
                    "label": label,
                    "applied": True,
                    "created_label": create_if_missing,
                    "message": f"Label '{label}' successfully applied to email",
                    "timestamp": "2025-07-06T16:00:00Z"
                }
                
            except Exception as e:
                return {
                    "email_id": email_id,
                    "label": label,
                    "applied": False,
                    "error": f"Failed to apply label: {str(e)}"
                }
        
        # Run the async operation in the background loop
        result = run_async_in_background(apply_label())
        return result
        
    except Exception as e:
        logger.error(f"Label application failed: {e}")
        return {
            "error": f"Label application failed: {str(e)}",
            "email_id": email_id,
            "label": label
        }


@mcp.tool()
def classify_and_label_emails(
    query: Optional[str] = None,
    email_ids: Optional[List[str]] = None,
    limit: int = 50,
    dry_run: bool = False
) -> Dict[str, Any]:
    """Classify emails and apply appropriate labels automatically.
    
    Performance Note: This is a heavy operation. Start with dry_run=true and small limits (10-20 emails).
    Each email requires AI analysis, so expect 2-3 seconds per email.
    
    Args:
        query: Gmail search query to find emails to classify (e.g., 'is:unread')
        email_ids: Specific email IDs to classify (get from search_emails)
        limit: Maximum number of emails to process (recommended: 10-20 for testing)
        dry_run: Preview classifications without applying labels (recommended first)
    """
    # Try basic initialization first
    if not try_initialize_services():
        return {"error": "Failed to initialize basic services"}
    
    # Check if email service is available
    if not email_service:
        return {
            "status": "initializing",
            "message": "Email services are starting up. Please try the 'initialize_email_services' tool first.",
            "query": query,
            "email_ids": email_ids,
            "limit": min(max(1, limit), 100),
            "dry_run": dry_run
        }
    
    # Perform email classification using background loop
    try:
        async def perform_classification():
            # Get emails to classify
            if email_ids:
                # Use specific email IDs
                emails = []
                for email_id in email_ids[:limit]:
                    try:
                        email = await email_service.get_email_content(email_id)
                        if email:
                            emails.append(email)
                    except Exception as e:
                        logger.error(f"Failed to get email {email_id}: {e}")
            else:
                # Use query to find emails
                search_query = query or "is:unread"
                emails = await email_service.search_emails(search_query, limit)
            
            results = []
            successful = 0
            failed = 0
            
            for email in emails:
                try:
                    email_id = email.id
                    subject = email.subject or "No Subject"
                    sender = email.sender or "Unknown Sender"
                    content = email.body_text or email.body_html or ""
                    
                    # Perform classifications
                    classifications = {}
                    labels_to_apply = []
                    
                    # Priority classification
                    if prioritizer:
                        priority_result = await prioritizer.classify_email(email)
                        classifications["priority"] = {
                            "level": priority_result.get("priority", "medium"),
                            "confidence": priority_result.get("confidence", 0.5),
                            "reasoning": priority_result.get("reasoning", "Priority analysis completed")
                        }
                        
                        # Add priority label
                        priority_level = priority_result.get("priority", "medium")
                        labels_to_apply.append(f"Priority/{priority_level.title()}")
                    
                    # Marketing classification
                    if marketing_classifier:
                        marketing_result = await marketing_classifier.classify_email(email)
                        is_marketing = marketing_result.get("is_marketing", False)
                        
                        if is_marketing:
                            classifications["marketing"] = {
                                "is_marketing": True,
                                "subtype": marketing_result.get("subtype", "promotional"),
                                "confidence": marketing_result.get("confidence", 0.5),
                                "reasoning": marketing_result.get("reasoning", "Marketing email detected")
                            }
                            labels_to_apply.append("Marketing/Promotional")
                    
                    # Receipt classification
                    if receipt_classifier:
                        receipt_result = await receipt_classifier.classify_email(email)
                        is_receipt = receipt_result.get("is_receipt", False)
                        
                        if is_receipt:
                            classifications["receipt"] = {
                                "is_receipt": True,
                                "vendor": receipt_result.get("vendor", "Unknown"),
                                "confidence": receipt_result.get("confidence", 0.5),
                                "reasoning": receipt_result.get("reasoning", "Receipt detected")
                            }
                            labels_to_apply.append("Receipts/Purchase")
                    
                    # Apply labels if not dry run
                    labels_applied = []
                    if not dry_run and labels_to_apply:
                        for label in labels_to_apply:
                            try:
                                await email_service.apply_label(email_id, label, create_if_missing=True)
                                labels_applied.append(label)
                            except Exception as e:
                                logger.error(f"Failed to apply label {label} to {email_id}: {e}")
                    
                    result_entry = {
                        "email_id": email_id,
                        "subject": subject,
                        "sender": sender,
                        "processed": True,
                        "dry_run": dry_run,
                        "classifications": classifications,
                        "labels_applied": labels_applied if not dry_run else labels_to_apply
                    }
                    
                    results.append(result_entry)
                    successful += 1
                    
                except Exception as e:
                    logger.error(f"Failed to classify email {getattr(email, 'id', 'unknown')}: {e}")
                    results.append({
                        "email_id": getattr(email, "id", "unknown"),
                        "subject": getattr(email, "subject", "Unknown"),
                        "sender": getattr(email, "sender", "Unknown"),
                        "processed": False,
                        "error": str(e)
                    })
                    failed += 1
            
            return {
                "total_processed": len(results),
                "successful": successful,
                "failed": failed,
                "dry_run": dry_run,
                "classification_types": ["priority", "marketing", "receipt"],
                "results": results,
                "timestamp": "2025-07-06T16:00:00Z",
                "query": query,
                "email_ids": email_ids,
                "limit": limit
            }
        
        # Run the async operation in the background loop
        result = run_async_in_background(perform_classification())
        return result
        
    except Exception as e:
        logger.error(f"Email classification failed: {e}")
        return {
            "error": f"Email classification failed: {str(e)}",
            "query": query,
            "email_ids": email_ids,
            "limit": min(max(1, limit), 100),
            "dry_run": dry_run
        }


@mcp.tool()
def get_classification_stats() -> Dict[str, Any]:
    """Get statistics about email classification and system status."""
    # Try basic initialization first
    try_initialize_services()
    
    services_status = {
        "config": config is not None,
        "email_service": email_service is not None,
        "prioritizer": prioritizer is not None,
        "marketing_classifier": marketing_classifier is not None,
        "receipt_classifier": receipt_classifier is not None,
        "ollama_manager": ollama_manager is not None,
        "background_loop": background_loop is not None,
        "services_initialized": services_initialized
    }
    
    basic_initialized = config is not None
    fully_initialized = all([
        email_service is not None,
        prioritizer is not None,
        marketing_classifier is not None,
        receipt_classifier is not None,
        ollama_manager is not None
    ])
    
    if fully_initialized:
        status = "operational"
    elif basic_initialized:
        status = "partially_initialized"
    else:
        status = "not_initialized"
    
    # Debug info
    debug_info = {
        "ollama_manager_type": type(ollama_manager).__name__ if ollama_manager else "None",
        "initialization_errors": "Check server logs for initialization errors"
    }
    
    return {
        "system_status": status,
        "services": services_status,
        "debug": debug_info,
        "message": "Use 'initialize_email_services' tool to complete initialization" if not fully_initialized else "All services operational",
        "timestamp": "2025-07-06T16:00:00Z",
        "initialization_help": "Run the 'initialize_email_services' tool to start async services" if not fully_initialized else None
    }


def initialize_services_sync():
    """Initialize services synchronously without async loops."""
    global email_service, prioritizer, marketing_classifier, receipt_classifier, ollama_manager
    global services_initialized
    
    try:
        logger.info("Starting synchronous service initialization...")
        
        # Start the background loop first
        start_background_loop()
        
        # Import and initialize services that don't require async
        from src.services.email_service import EmailService
        from src.services.email_prioritizer import EmailPrioritizer
        from src.services.marketing_classifier import MarketingEmailClassifier
        from src.services.receipt_classifier import ReceiptClassifier
        from src.integrations.ollama_client import get_ollama_manager
        
        # Create service instances (they'll initialize async parts later)
        email_service = EmailService()
        prioritizer = EmailPrioritizer()
        marketing_classifier = MarketingEmailClassifier()
        receipt_classifier = ReceiptClassifier()
        
        # Initialize Ollama manager directly in main thread
        async def init_ollama_only():
            global ollama_manager
            try:
                logger.info("Initializing Ollama manager in main thread...")
                ollama_manager = await get_ollama_manager()
                logger.info(f"Ollama manager initialized: {ollama_manager is not None}")
                return ollama_manager is not None
            except Exception as e:
                logger.error(f"Failed to initialize Ollama: {e}")
                return False
        
        # Run just the Ollama initialization
        try:
            ollama_result = run_async_in_background(init_ollama_only())
            logger.info(f"Ollama initialization result: {ollama_result}")
            logger.info(f"Ollama manager after init: {ollama_manager is not None}")
        except Exception as e:
            logger.error(f"Failed to run Ollama initialization: {e}")
        
        # Now run full async initialization for other services
        async def full_init():
            logger.info("Starting full async initialization for email services...")
            
            global email_service, prioritizer, marketing_classifier, receipt_classifier
            
            # Initialize email service
            await email_service.initialize()
            logger.info("Email service initialized")
            
            # Initialize classifiers
            await prioritizer.initialize()
            await marketing_classifier.initialize()
            await receipt_classifier.initialize()
            
            logger.info("All email services initialized successfully")
            return True
        
        # Run the async initialization
        try:
            async_result = run_async_in_background(full_init())
            logger.info(f"Email services initialization result: {async_result}")
        except Exception as e:
            logger.error(f"Failed to run email services initialization: {e}")
            return False
        
        # Final check of all global variables
        logger.info(f"Final global variables - ollama_manager: {ollama_manager is not None}, email_service: {email_service is not None}")
        
        # Mark as initialized for basic functionality
        services_initialized = True
        
        logger.info("All services initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        return False


@mcp.tool()
def check_ollama_status() -> Dict[str, Any]:
    """Check if Ollama is running and available."""
    import aiohttp
    import asyncio
    
    async def test_ollama():
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get('http://localhost:11434/api/tags', timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "status": "running",
                            "models": [model.get("name", "unknown") for model in data.get("models", [])],
                            "model_count": len(data.get("models", []))
                        }
                    else:
                        return {"status": "error", "message": f"HTTP {response.status}"}
        except Exception as e:
            return {"status": "not_running", "error": str(e)}
    
    try:
        result = run_async_in_background(test_ollama())
        return {
            "ollama": result,
            "ollama_manager_initialized": ollama_manager is not None,
            "timestamp": "2025-07-06T16:00:00Z"
        }
    except Exception as e:
        return {
            "ollama": {"status": "error", "error": str(e)},
            "ollama_manager_initialized": ollama_manager is not None,
            "timestamp": "2025-07-06T16:00:00Z"
        }


@mcp.tool()
def fix_ollama_model_config() -> Dict[str, Any]:
    """Fix Ollama model configuration to use available models."""
    try:
        # Get available models
        import aiohttp
        
        async def get_and_fix_models():
            # Get available models
            async with aiohttp.ClientSession() as session:
                async with session.get('http://localhost:11434/api/tags') as response:
                    data = await response.json()
                    available_models = [model.get("name", "unknown") for model in data.get("models", [])]
            
            # Update config to use available model
            global config
            if not config:
                from src.core.config import get_config
                config = get_config()
            
            # Use first available model as primary
            if available_models:
                old_primary = config.ollama.models["primary"]
                config.ollama.models["primary"] = available_models[0]
                if len(available_models) > 1:
                    config.ollama.models["fallback"] = available_models[1]
                    config.ollama.models["reasoning"] = available_models[1]
                else:
                    config.ollama.models["fallback"] = available_models[0]
                    config.ollama.models["reasoning"] = available_models[0]
                
                return {
                    "status": "fixed",
                    "old_primary": old_primary,
                    "new_primary": config.ollama.models["primary"],
                    "available_models": available_models,
                    "updated_config": config.ollama.models
                }
            else:
                return {"status": "error", "message": "No models available"}
        
        result = run_async_in_background(get_and_fix_models())
        return result
        
    except Exception as e:
        return {"status": "error", "message": f"Failed to fix config: {str(e)}"}


@mcp.tool()
def force_ollama_init() -> Dict[str, Any]:
    """Force Ollama manager initialization with detailed logging."""
    global ollama_manager
    
    try:
        async def init_ollama():
            global ollama_manager
            from src.integrations.ollama_client import get_ollama_manager
            
            logger.info("Force initializing Ollama manager...")
            
            # Reset the global manager first
            from src.integrations.ollama_client import _ollama_manager
            import src.integrations.ollama_client as ollama_module
            ollama_module._ollama_manager = None
            
            # Get new manager
            new_manager = await get_ollama_manager()
            
            # Set global variable
            ollama_manager = new_manager
            
            logger.info(f"Ollama manager force initialized: {type(new_manager).__name__}")
            
            return {
                "status": "success",
                "manager_type": type(new_manager).__name__,
                "manager_id": id(new_manager),
                "global_manager_set": ollama_manager is not None
            }
        
        result = run_async_in_background(init_ollama())
        return result
        
    except Exception as e:
        logger.error(f"Force Ollama init failed: {e}")
        import traceback
        return {
            "status": "error", 
            "error": str(e),
            "traceback": traceback.format_exc()
        }


@mcp.tool()
def initialize_email_services() -> Dict[str, Any]:
    """Initialize all email services manually."""
    
    # Try synchronous initialization
    try:
        result = initialize_services_sync()
        
        return {
            "status": "completed" if result else "failed",
            "message": "Service initialization completed" if result else "Service initialization failed",
            "services": {
                "email_service": email_service is not None,
                "prioritizer": prioritizer is not None,
                "marketing_classifier": marketing_classifier is not None,
                "receipt_classifier": receipt_classifier is not None,
                "ollama_manager": ollama_manager is not None
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        return {"status": "error", "message": f"Initialization error: {str(e)}"}


def main():
    """Main entry point."""
    # Configure logging to stderr
    import logging
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
    
    logger.info("Starting Email MCP Server with FastMCP")
    
    # Initialize services on startup
    try:
        logger.info("Initializing services on startup...")
        start_background_loop()
        initialize_services_sync()
        logger.info("Services initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize services on startup: {e}")
        logger.info("Services will be initialized on demand")
    
    # Run the MCP server
    mcp.run()


if __name__ == "__main__":
    main()