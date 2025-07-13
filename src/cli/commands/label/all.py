"""Combined email classification commands - applies all classifiers."""

import asyncio
from typing import Optional, Dict, Any, List
import typer
import time
from concurrent.futures import ProcessPoolExecutor

from src.cli.base import BaseEmailProcessor, run_async_command
from src.cli.formatters import (
    info, success, warning, error, step, llm_status, 
    classification_result, progress, show_summary_panel, show_results_table,
    verbose_info, user_friendly_status, status_with_spinner, set_verbose_mode
)
from src.services.email_prioritizer import EmailPrioritizer
from src.services.marketing_classifier import MarketingEmailClassifier
from src.services.receipt_classifier import ReceiptClassifier
from src.services.notifications_classifier import NotificationsClassifier
from src.services.custom_classifier import CustomClassifier
from src.models.email import EmailCategory
from src.models.unified_analysis import UnifiedEmailAnalysis
from src.cli.langchain.chains import EmailRouterChain
from src.integrations.ollama_client import get_ollama_manager

app = typer.Typer(help="Combined email classification commands", invoke_without_command=True)


@app.callback()
def main_callback(
    ctx: typer.Context,
    days: int = typer.Option(7, "--days", help="Number of days to look back"),
    priority_confidence: float = typer.Option(0.7, "--priority-confidence", help="Minimum confidence for priority labeling"),
    marketing_confidence: float = typer.Option(0.7, "--marketing-confidence", help="Minimum confidence for marketing labeling"),
    receipt_confidence: float = typer.Option(0.7, "--receipt-confidence", help="Minimum confidence for receipt labeling"),
    notifications_confidence: float = typer.Option(0.7, "--notifications-confidence", help="Minimum confidence for notifications labeling"),
    custom_confidence: float = typer.Option(0.7, "--custom-confidence", help="Minimum confidence for custom category labeling"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview labels without applying them (default: apply labels)"),
    limit: Optional[int] = typer.Option(50, "--limit", "-l", help="Maximum number of emails to process (default: 50)"),
    exclude_personal: bool = typer.Option(True, "--exclude-personal/--include-personal", help="Exclude personal emails from marketing labeling"),
    detailed: bool = typer.Option(False, "--detailed", help="Show detailed analysis results"),
    intelligent_routing: bool = typer.Option(True, "--intelligent-routing/--all-classifiers", help="Use smart routing to determine which classifiers to run (default: intelligent)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed technical information")
):
    """Label recent emails with all classifiers (priority + marketing + receipt + notifications).
    
    This is the default behavior when no subcommand is specified.
    Equivalent to running 'email-agent label all recent'.
    """
    # If no subcommand is invoked, run the recent command as default
    if ctx.invoked_subcommand is None:
        # Set verbose mode based on flag
        set_verbose_mode(verbose)
        
        @run_async_command
        async def run():
            command = AllLabelCommand()
            await command.initialize()
            
            results = await command.execute(
                target=f"recent:{days}days",
                priority_confidence=priority_confidence,
                marketing_confidence=marketing_confidence,
                receipt_confidence=receipt_confidence,
                notifications_confidence=notifications_confidence,
                custom_confidence=custom_confidence,
                dry_run=dry_run,
                limit=limit,
                exclude_personal=exclude_personal,
                detailed=detailed,
                intelligent_routing=intelligent_routing
            )
            
            command.format_output(results)
            return command
        
        run()


class AllLabelCommand(BaseEmailProcessor):
    """Combined labeling command that applies all classifiers."""
    
    def __init__(self):
        super().__init__()
        self.prioritizer = None
        self.marketing_classifier = None
        self.receipt_classifier = None
        self.notifications_classifier = None
        self.custom_classifier = None
        self.router_chain = None
        
        # All label mappings
        self.priority_labels = {
            "critical": "Priority/Critical",
            "high": "Priority/High", 
            "medium": "Priority/Medium",
            "low": "Priority/Low"
        }
        
        self.marketing_labels = {
            "promotional": "Marketing/Promotional",
            "newsletter": "Marketing/Newsletter", 
            "hybrid": "Marketing/Hybrid",
            "general": "Marketing/General"
        }
        
        self.receipt_labels = {
            "purchase": "Receipts/Purchase",
            "service": "Receipts/Service",
            "other": "Receipts/Other"
        }
        
        self.notification_labels = {
            "system": "Notifications/System",
            "update": "Notifications/Update", 
            "alert": "Notifications/Alert",
            "reminder": "Notifications/Reminder",
            "security": "Notifications/Security"
        }
        
        # Non-marketing emails won't get marketing labels
        self.skip_marketing_subtypes = ["personal", "transactional"]
    
    async def process_emails_optimized(
        self, 
        query: str,
        limit: Optional[int] = None,
        dry_run: bool = True
    ) -> List[Any]:
        """Optimized email processing with concurrent fetching."""
        emails = []
        
        with status_with_spinner(f"Searching for emails matching: {query}"):
            # Collect email references first
            email_refs = []
            async for email_ref in self.email_service.search_emails(query, limit):
                email_refs.append(email_ref)
        
        if not email_refs:
            warning(f"No emails found for query: '{query}'")
            return emails
        
        with status_with_spinner(f"Loading {len(email_refs)} emails", f"Ready to analyze {len(email_refs)} emails"):
            # Fetch email content concurrently in batches
            batch_size = 10  # Fetch 10 emails concurrently
            ref_batches = [email_refs[i:i + batch_size] for i in range(0, len(email_refs), batch_size)]
            
            for ref_batch in ref_batches:
                # Create concurrent tasks for email content fetching
                fetch_tasks = [
                    self.email_service.get_email_content(ref.email_id)
                    for ref in ref_batch
                ]
                
                # Execute batch of content fetches concurrently
                batch_emails = await asyncio.gather(*fetch_tasks, return_exceptions=True)
                
                # Process results and filter out failures
                for email in batch_emails:
                    if email and not isinstance(email, Exception):
                        emails.append(email)
                
                # Small delay between batches to avoid overwhelming the API
                await asyncio.sleep(0.05)
        
        return emails
    
    async def initialize(self):
        """Initialize all classification services."""
        await super().initialize()
        
        # Initialize all classifiers
        self.prioritizer = EmailPrioritizer()
        await self.prioritizer.initialize()
        
        self.marketing_classifier = MarketingEmailClassifier()
        await self.marketing_classifier.initialize()
        
        self.receipt_classifier = ReceiptClassifier()
        await self.receipt_classifier.initialize()
        
        self.notifications_classifier = NotificationsClassifier()
        await self.notifications_classifier.initialize()
        
        self.custom_classifier = CustomClassifier()
        await self.custom_classifier.initialize()
        
        # Initialize email router chain for intelligent processing
        ollama_manager = await get_ollama_manager()
        self.router_chain = EmailRouterChain(ollama_manager)
        
        # Ensure all labels exist
        await self._ensure_all_labels_exist()
    
    async def _ensure_all_labels_exist(self) -> None:
        """Ensure all classification labels exist in Gmail with proper nesting."""
        verbose_info("Checking all classification labels")
        
        with status_with_spinner("Setting up email classification", "Email classification setup complete"):
            try:
                # Get existing labels
                existing_labels = await self.email_service.get_labels()
                existing_label_names = {label.name for label in existing_labels}
                
                # Parent labels to create
                parent_labels = ["Priority", "Marketing", "Receipts", "Notifications"]
                
                # Create parent labels if needed
                for parent_label in parent_labels:
                    if parent_label not in existing_label_names:
                        verbose_info(f"   Creating parent label: {parent_label}")
                        await self.email_service.create_label(name=parent_label)
                        existing_label_names.add(parent_label)
                    else:
                        verbose_info(f"   ✓ Parent label exists: {parent_label}")
                
                # Create all sublabels
                all_labels = {**self.priority_labels, **self.marketing_labels, **self.receipt_labels, **self.notification_labels}
                for subtype, label_name in all_labels.items():
                    if label_name not in existing_label_names:
                        verbose_info(f"   Creating nested label: {label_name}")
                        await self.email_service.create_label(name=label_name)
                    else:
                        verbose_info(f"   ✓ Nested label exists: {label_name}")
                        
            except Exception as e:
                error(f"Failed to ensure labels exist: {e}")
                raise
    
    async def classify_email_parallel(self, email, use_intelligent_routing: bool = True) -> Dict[str, Any]:
        """Classify a single email using parallel classifiers with unified analysis.
        
        Args:
            email: Email to classify
            use_intelligent_routing: If True, use EmailRouterChain to determine which classifiers to run
            
        Returns:
            Classification results with labels to apply
        """
        classifications = {}
        labels_to_apply = []
        routing_info = {}
        
        try:
            # Perform unified analysis once for all classifiers
            unified_analysis = UnifiedEmailAnalysis.from_email(email)
            
            # Determine which services to use (enhanced with unified analysis)
            if use_intelligent_routing and self.router_chain:
                recommended_services = await self.router_chain.route_email(email)
                # Override with unified analysis hints for better performance
                if unified_analysis.appears_marketing and "marketing" not in recommended_services:
                    recommended_services.append("marketing")
                if unified_analysis.appears_receipt and "receipt" not in recommended_services:
                    recommended_services.append("receipt")
                if unified_analysis.appears_notification and "notifications" not in recommended_services:
                    recommended_services.append("notifications")
                
                # ALWAYS include priority classification for ALL emails
                if "priority" not in recommended_services:
                    recommended_services.insert(0, "priority")  # Add at beginning
                
                routing_info = {
                    "recommended_services": recommended_services,
                    "routing_method": "intelligent_enhanced",
                    "analysis_hints": {
                        "appears_marketing": unified_analysis.appears_marketing,
                        "appears_receipt": unified_analysis.appears_receipt,
                        "appears_notification": unified_analysis.appears_notification
                    }
                }
            else:
                # Use all services (legacy behavior) - priority is already included
                recommended_services = ["priority", "marketing", "receipt", "notifications"]
                routing_info = {
                    "recommended_services": recommended_services,
                    "routing_method": "all_services"
                }
            
            # Prepare classification tasks for parallel execution with unified analysis
            classification_tasks = []
            
            # Priority classification
            if "priority" in recommended_services and self.prioritizer:
                classification_tasks.append(
                    self._classify_priority_wrapper_unified(email, unified_analysis)
                )
            
            # Marketing classification
            if "marketing" in recommended_services and self.marketing_classifier:
                classification_tasks.append(
                    self._classify_marketing_wrapper_unified(email, unified_analysis)
                )
            
            # Receipt classification
            if "receipt" in recommended_services and self.receipt_classifier:
                classification_tasks.append(
                    self._classify_receipt_wrapper_unified(email, unified_analysis)
                )
            
            # Notifications classification
            if "notifications" in recommended_services and self.notifications_classifier:
                classification_tasks.append(
                    self._classify_notifications_wrapper_unified(email, unified_analysis)
                )
            
            # Execute all classifications in parallel
            if classification_tasks:
                parallel_results = await asyncio.gather(*classification_tasks, return_exceptions=True)
                
                # Process results
                for result in parallel_results:
                    if isinstance(result, Exception):
                        continue  # Skip failed classifications
                    
                    if result and isinstance(result, dict):
                        service_type = result.get("service_type")
                        if service_type and "classification" in result:
                            classifications[service_type] = result["classification"]
                            
                            # Add appropriate label
                            if result.get("suggested_label"):
                                labels_to_apply.append(result["suggested_label"])
            
            # Handle custom classifications separately (lighter weight)
            if self.custom_classifier and "custom" in recommended_services:
                try:
                    custom_results = await self._classify_custom_lightweight(email)
                    if custom_results:
                        classifications["custom"] = custom_results["classifications"]
                        labels_to_apply.extend(custom_results.get("labels", []))
                except Exception as e:
                    # Don't fail the whole process for custom classification errors
                    pass
            
            return {
                "email_id": email.id,
                "subject": email.subject or "No Subject",
                "sender": email.sender or "Unknown Sender",
                "classifications": classifications,
                "suggested_labels": labels_to_apply,
                "routing_info": routing_info,
                "unified_analysis": unified_analysis.to_dict(),
                "processed": True
            }
            
        except Exception as e:
            return {
                "email_id": getattr(email, 'id', 'unknown'),
                "subject": getattr(email, 'subject', 'Unknown'),
                "sender": getattr(email, 'sender', 'Unknown'),
                "error": str(e),
                "processed": False
            }
    
    async def _classify_priority_wrapper_unified(self, email, unified_analysis: UnifiedEmailAnalysis) -> Dict[str, Any]:
        """Wrapper for priority classification using unified analysis."""
        try:
            # Use unified analysis context for priority classification
            priority_context = unified_analysis.get_classifier_context('priority')
            
            # Run the classification (existing method)
            result = await self.prioritizer.analyze_priority(email)
            
            # Determine label from priority
            priority_level = result.level.lower()
            suggested_label = self.priority_labels.get(priority_level)
            
            return {
                "service_type": "priority",
                "classification": {
                    "level": result.level,
                    "confidence": result.confidence,
                    "reasoning": result.reasoning,
                    "context_used": priority_context
                },
                "suggested_label": suggested_label
            }
        except Exception as e:
            return {
                "service_type": "priority",
                "error": str(e)
            }
    
    async def _classify_marketing_wrapper_unified(self, email, unified_analysis: UnifiedEmailAnalysis) -> Dict[str, Any]:
        """Wrapper for marketing classification using unified analysis."""
        try:
            # Use unified analysis context for marketing classification
            marketing_context = unified_analysis.get_classifier_context('marketing')
            
            # Skip marketing classification if unified analysis suggests it's clearly not marketing
            if not unified_analysis.appears_marketing and unified_analysis.appears_personal:
                return {
                    "service_type": "marketing",
                    "classification": {
                        "is_marketing": False,
                        "confidence": 0.95,
                        "subtype": "personal",
                        "reasoning": "Unified analysis determined personal email",
                        "skip_reason": "unified_analysis_optimization"
                    },
                    "suggested_label": None
                }
            
            # Run the classification (existing method)
            result = await self.marketing_classifier.classify_email(email)
            
            # Determine label from marketing classification
            suggested_label = None
            if result.is_marketing and result.subtype not in self.skip_marketing_subtypes:
                suggested_label = self.marketing_labels.get(result.subtype)
            
            return {
                "service_type": "marketing",
                "classification": {
                    "is_marketing": result.is_marketing,
                    "confidence": result.confidence,
                    "subtype": result.subtype,
                    "reasoning": result.reasoning,
                    "context_used": marketing_context
                },
                "suggested_label": suggested_label
            }
        except Exception as e:
            return {
                "service_type": "marketing",
                "error": str(e)
            }
    
    async def _classify_receipt_wrapper_unified(self, email, unified_analysis: UnifiedEmailAnalysis) -> Dict[str, Any]:
        """Wrapper for receipt classification using unified analysis."""
        try:
            # Use unified analysis context for receipt classification
            receipt_context = unified_analysis.get_classifier_context('receipt')
            
            # Skip receipt classification if unified analysis suggests it's clearly not a receipt
            # DISABLED - Let the receipt classifier do the full analysis 
            # if not unified_analysis.appears_receipt and not unified_analysis.monetary_amounts:
            #     return {
            #         "service_type": "receipt",
            #         "classification": {
            #             "is_receipt": False,
            #             "confidence": 0.90,
            #             "reasoning": "Unified analysis found no receipt indicators",
            #             "skip_reason": "unified_analysis_optimization"
            #         },
            #         "suggested_label": None
            #     }
            
            # Run the classification (existing method)
            result = await self.receipt_classifier.classify_receipt(email)
            
            # Determine label from receipt classification
            suggested_label = None
            if result.is_receipt:
                suggested_label = self.receipt_labels.get(result.receipt_type, self.receipt_labels.get("other"))
            
            return {
                "service_type": "receipt",
                "classification": {
                    "is_receipt": result.is_receipt,
                    "confidence": result.confidence,
                    "receipt_type": result.receipt_type,
                    "reasoning": result.reasoning,
                    "context_used": receipt_context
                },
                "suggested_label": suggested_label
            }
        except Exception as e:
            return {
                "service_type": "receipt",
                "error": str(e)
            }
    
    async def _classify_notifications_wrapper_unified(self, email, unified_analysis: UnifiedEmailAnalysis) -> Dict[str, Any]:
        """Wrapper for notifications classification using unified analysis."""
        try:
            # Use unified analysis context for notifications classification
            notifications_context = unified_analysis.get_classifier_context('notifications')
            
            # Skip notifications classification if unified analysis suggests it's clearly personal
            if unified_analysis.appears_personal and not unified_analysis.is_automated_sender:
                return {
                    "service_type": "notifications",
                    "classification": {
                        "is_notification": False,
                        "confidence": 0.90,
                        "reasoning": "Unified analysis determined personal communication",
                        "skip_reason": "unified_analysis_optimization"
                    },
                    "suggested_label": None
                }
            
            # Run the classification (existing method)
            result = await self.notifications_classifier.classify_email(email)
            
            # Determine label from notifications classification
            suggested_label = None
            if result.is_notification:
                suggested_label = self.notification_labels.get(result.notification_type, self.notification_labels.get("system"))
            
            return {
                "service_type": "notifications",
                "classification": {
                    "is_notification": result.is_notification,
                    "confidence": result.confidence,
                    "notification_type": result.notification_type,
                    "reasoning": result.reasoning,
                    "context_used": notifications_context
                },
                "suggested_label": suggested_label
            }
        except Exception as e:
            return {
                "service_type": "notifications",
                "error": str(e)
            }
    
    async def _classify_priority_wrapper(self, email) -> Dict[str, Any]:
        """Legacy wrapper for priority classification (fallback)."""
        try:
            priority_result = await self.prioritizer.analyze_priority(email)
            return {
                "service_type": "priority",
                "classification": {
                    "level": priority_result.level,
                    "confidence": priority_result.confidence,
                    "reasoning": priority_result.reasoning,
                    "is_genuine_urgency": priority_result.is_genuine_urgency,
                    "sender_reputation": priority_result.sender_reputation,
                    "needs_review": priority_result.needs_review,
                    "detected_tactics": priority_result.detected_tactics
                },
                "suggested_label": self.priority_labels.get(priority_result.level, "Priority/Medium")
            }
        except Exception as e:
            return {"service_type": "priority", "error": str(e)}
    
    async def _classify_marketing_wrapper(self, email) -> Dict[str, Any]:
        """Wrapper for marketing classification."""
        try:
            marketing_result = await self.marketing_classifier.classify_email(email)
            suggested_label = None
            if (marketing_result.is_marketing and 
                marketing_result.subtype not in self.skip_marketing_subtypes):
                suggested_label = self.marketing_labels.get(marketing_result.subtype, "Marketing/General")
            
            return {
                "service_type": "marketing",
                "classification": {
                    "is_marketing": marketing_result.is_marketing,
                    "subtype": marketing_result.subtype,
                    "confidence": marketing_result.confidence,
                    "reasoning": marketing_result.reasoning,
                    "marketing_indicators": marketing_result.marketing_indicators,
                    "unsubscribe_detected": marketing_result.unsubscribe_detected,
                    "bulk_sending_indicators": marketing_result.bulk_sending_indicators,
                    "sender_reputation": marketing_result.sender_reputation
                },
                "suggested_label": suggested_label
            }
        except Exception as e:
            return {"service_type": "marketing", "error": str(e)}
    
    async def _classify_receipt_wrapper(self, email) -> Dict[str, Any]:
        """Wrapper for receipt classification."""
        try:
            receipt_result = await self.receipt_classifier.classify_receipt(email)
            suggested_label = None
            if receipt_result.is_receipt:
                suggested_label = self.receipt_labels.get(receipt_result.receipt_type, "Receipts/Other")
            
            return {
                "service_type": "receipt",
                "classification": {
                    "is_receipt": receipt_result.is_receipt,
                    "receipt_type": receipt_result.receipt_type,
                    "confidence": receipt_result.confidence,
                    "reasoning": receipt_result.reasoning,
                    "vendor": receipt_result.vendor,
                    "amount": receipt_result.amount,
                    "currency": receipt_result.currency,
                    "order_number": receipt_result.order_number,
                    "transaction_date": receipt_result.transaction_date,
                    "payment_method": receipt_result.payment_method,
                    "receipt_indicators": receipt_result.receipt_indicators
                },
                "suggested_label": suggested_label
            }
        except Exception as e:
            return {"service_type": "receipt", "error": str(e)}
    
    async def _classify_notifications_wrapper(self, email) -> Dict[str, Any]:
        """Wrapper for notifications classification."""
        try:
            notification_result = await self.notifications_classifier.classify_notification(email)
            suggested_label = None
            if notification_result.is_notification:
                suggested_label = self.notification_labels.get(notification_result.notification_type, "Notifications/Alert")
            
            return {
                "service_type": "notifications",
                "classification": {
                    "is_notification": notification_result.is_notification,
                    "notification_type": notification_result.notification_type,
                    "confidence": notification_result.confidence,
                    "reasoning": notification_result.reasoning,
                    "sender_type": notification_result.sender_type,
                    "urgency_level": notification_result.urgency_level,
                    "action_required": notification_result.action_required,
                    "notification_indicators": notification_result.notification_indicators
                },
                "suggested_label": suggested_label
            }
        except Exception as e:
            return {"service_type": "notifications", "error": str(e)}
    
    async def _classify_custom_lightweight(self, email) -> Optional[Dict[str, Any]]:
        """Lightweight custom classification with limited categories."""
        try:
            # Get available custom categories (limit to 2 for performance)
            custom_categories = await self.custom_classifier.get_categories()
            if not custom_categories:
                return None
            
            custom_results = {}
            custom_labels = []
            
            # Process only the first 2 categories for performance
            for category in custom_categories[:2]:
                try:
                    custom_result = await self.custom_classifier.classify_email(
                        email, category.name, category.search_terms, category.confidence_threshold
                    )
                    custom_results[category.name] = {
                        "is_match": custom_result.is_match,
                        "confidence": custom_result.confidence,
                        "reasoning": custom_result.reasoning,
                        "suggested_label": custom_result.suggested_label
                    }
                    
                    # Add custom label if it matches
                    if custom_result.is_match and custom_result.confidence >= category.confidence_threshold:
                        custom_label = custom_result.suggested_label or category.name
                        custom_labels.append(custom_label)
                        
                except Exception as e:
                    custom_results[category.name] = {"error": str(e)}
            
            return {
                "classifications": custom_results,
                "labels": custom_labels
            }
            
        except Exception as e:
            return None
    
    # Keep the original method for backward compatibility
    async def classify_email(self, email, use_intelligent_routing: bool = True) -> Dict[str, Any]:
        """Legacy method - redirects to optimized parallel version."""
        return await self.classify_email_parallel(email, use_intelligent_routing)
    
    async def execute(
        self,
        target: str = "unread",
        priority_confidence: float = 0.3,
        marketing_confidence: float = 0.5,
        receipt_confidence: float = 0.6,
        notifications_confidence: float = 0.5,
        custom_confidence: float = 0.6,
        dry_run: bool = True,
        limit: Optional[int] = None,
        exclude_personal: bool = True,
        detailed: bool = False,
        intelligent_routing: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute combined classification with optional intelligent routing."""
        
        # Convert target to Gmail query
        query = self._parse_target(target)
        
        # Process emails with optimized concurrent fetching
        emails = await self.process_emails_optimized(query, limit, dry_run)
        
        if not emails:
            return {"processed": 0, "results": [], "dry_run": dry_run}
        
        # Show analysis status (no spinner needed as progress bar will take over)
        if dry_run:
            user_friendly_status(f"Analyzing {len(emails)} emails (preview mode)")
        else:
            user_friendly_status(f"Analyzing and labeling {len(emails)} emails")
        
        if dry_run:
            warning("Preview mode - No labels will be applied")
        
        results = []
        counts = {
            "priority": {"critical": 0, "high": 0, "medium": 0, "low": 0},
            "marketing": {"promotional": 0, "newsletter": 0, "hybrid": 0, "transactional": 0, "personal": 0, "general": 0},
            "receipt": {"purchase": 0, "service": 0, "other": 0, "non_receipt": 0},
            "notifications": {"system": 0, "update": 0, "alert": 0, "reminder": 0, "security": 0, "non_notification": 0},
            "errors": 0
        }
        
        # Process emails with progress bar
        with progress("Processing emails...", total=len(emails)) as progress_tracker:
            
            # Process emails in batches for better performance
            batch_size = min(8, len(emails))  # Process up to 8 emails concurrently
            email_batches = [emails[i:i + batch_size] for i in range(0, len(emails), batch_size)]
            
            for batch_idx, email_batch in enumerate(email_batches):
                try:
                    # Process batch of emails concurrently
                    batch_tasks = [
                        self.classify_email_parallel(email, use_intelligent_routing=intelligent_routing)
                        for email in email_batch
                    ]
                    
                    batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                    
                    # Process batch results
                    for email, result in zip(email_batch, batch_results):
                        try:
                            if isinstance(result, Exception):
                                counts["errors"] += 1
                                results.append({
                                    "email_id": email.id,
                                    "subject": email.subject,
                                    "error": str(result),
                                    "processed": False
                                })
                                continue
                    
                            if not result.get("processed"):
                                results.append(result)
                                counts["errors"] += 1
                                progress_tracker.update()
                                continue
                    
                            # Filter labels based on confidence thresholds
                            labels_to_apply = []
                            classifications = result.get("classifications", {})
                            
                            # Priority label (always apply if meets threshold)
                            priority_info = classifications.get("priority", {})
                            if priority_info.get("confidence", 0) >= priority_confidence:
                                priority_label = self.priority_labels.get(priority_info.get("level"), "Priority/Medium")
                                labels_to_apply.append(priority_label)
                            
                            # Marketing label (only if marketing and meets criteria)
                            marketing_info = classifications.get("marketing", {})
                            if (marketing_info.get("is_marketing") and 
                                marketing_info.get("subtype") not in self.skip_marketing_subtypes and
                                marketing_info.get("confidence", 0) >= marketing_confidence):
                                if not exclude_personal or marketing_info.get("subtype") != "personal":
                                    marketing_label = self.marketing_labels.get(marketing_info.get("subtype"), "Marketing/General")
                                    labels_to_apply.append(marketing_label)
                            
                            # Receipt label (only if receipt and meets threshold)
                            receipt_info = classifications.get("receipt", {})
                            if (receipt_info.get("is_receipt") and 
                                receipt_info.get("confidence", 0) >= receipt_confidence):
                                receipt_label = self.receipt_labels.get(receipt_info.get("receipt_type"), "Receipts/Other")
                                labels_to_apply.append(receipt_label)
                            
                            # Notifications label (only if notification and meets threshold)
                            notifications_info = classifications.get("notifications", {})
                            if (notifications_info.get("is_notification") and 
                                notifications_info.get("confidence", 0) >= notifications_confidence):
                                notification_label = self.notification_labels.get(notifications_info.get("notification_type"), "Notifications/Alert")
                                labels_to_apply.append(notification_label)
                            
                            # Custom labels (only if matches and meets threshold)
                            custom_info = classifications.get("custom", {})
                            for category_name, custom_result in custom_info.items():
                                if (custom_result.get("is_match") and 
                                    custom_result.get("confidence", 0) >= custom_confidence):
                                    custom_label = custom_result.get("suggested_label") or category_name
                                    labels_to_apply.append(custom_label)
                            
                            result["final_labels"] = labels_to_apply
                            results.append(result)
                            
                            # Update counts
                            if priority_info:
                                counts["priority"][priority_info.get("level", "medium")] += 1
                            if marketing_info:
                                counts["marketing"][marketing_info.get("subtype", "general")] += 1
                            if receipt_info:
                                if receipt_info.get("is_receipt"):
                                    counts["receipt"][receipt_info.get("receipt_type", "other")] += 1
                                else:
                                    counts["receipt"]["non_receipt"] += 1
                            if notifications_info:
                                if notifications_info.get("is_notification"):
                                    counts["notifications"][notifications_info.get("notification_type", "alert")] += 1
                                else:
                                    counts["notifications"]["non_notification"] += 1
                            
                        except Exception as e:
                            self.console.print(f"   ❌ Failed to process {email.subject[:40] if email.subject else 'Unknown'}...: {str(e)}")
                            counts["errors"] += 1
                            results.append({
                                "email_id": email.id,
                                "subject": email.subject,
                                "error": str(e),
                                "processed": False
                            })
                        
                        progress_tracker.update()
                    
                    # Batch delay to avoid overwhelming APIs (shorter delay since we're batching)
                    await asyncio.sleep(0.05)
                    
                except Exception as e:
                    # Handle entire batch failure
                    for email in email_batch:
                        counts["errors"] += 1
                        results.append({
                            "email_id": email.id,
                            "subject": email.subject,
                            "error": f"Batch processing error: {str(e)}",
                            "processed": False
                        })
                        progress_tracker.update()
        
        # Store results for batch processing reference
        self._current_results = results
        
        # Now apply labels in batches for better performance
        if not dry_run:
            await self._apply_labels_batch(results, dry_run)
        else:
            # For dry run, show what would be applied
            for result in results:
                if result.get("processed") and result.get("final_labels"):
                    labels = result.get("final_labels", [])
                    subject = result.get("subject", "Unknown")[:40]
                    if labels:
                        result["labels_would_apply"] = labels
                        classification_result(subject, labels, dry_run=True)
                    else:
                        result["labels_skipped"] = "No labels met confidence thresholds"
                        classification_result(subject, [], dry_run=True)
        
        # Print summary
        success("Email classification complete!")
        
        # Display performance metrics
        successful_results = [r for r in results if r.get("processed")]
        if successful_results:
            avg_labels_per_email = sum(len(r.get("final_labels", [])) for r in successful_results) / len(successful_results)
            self.console.print(f"   ⚡ Performance: {len(successful_results)} emails processed with {avg_labels_per_email:.1f} avg labels per email")
        self.console.print(f"   📊 Classification Summary:")
        
        self.console.print(f"      Priority Distribution:")
        for level, count in counts["priority"].items():
            if count > 0:
                self.console.print(f"        {level.title()}: {count}")
        
        self.console.print(f"      Marketing Distribution:")
        for subtype, count in counts["marketing"].items():
            if count > 0:
                self.console.print(f"        {subtype.title()}: {count}")
        
        self.console.print(f"      Receipt Distribution:")
        for receipt_type, count in counts["receipt"].items():
            if count > 0:
                self.console.print(f"        {receipt_type.replace('_', ' ').title()}: {count}")
        
        self.console.print(f"      Notifications Distribution:")
        for notification_type, count in counts["notifications"].items():
            if count > 0:
                self.console.print(f"        {notification_type.replace('_', ' ').title()}: {count}")
        
        if counts["errors"] > 0:
            self.console.print(f"      Errors: {counts['errors']}")
        
        # Show detailed results if requested
        if detailed:
            self._print_detailed_results(results)
        
        return {
            "processed": len(results),
            "counts": counts,
            "results": results,
            "dry_run": dry_run,
            "successful": len([r for r in results if r.get("processed")]),
            "failed": counts["errors"],
            "batch_processing": True,
            "batch_size": min(8, len(emails)) if emails else 0
        }
    
    async def _apply_labels_batch(self, results: List[Dict[str, Any]], dry_run: bool) -> None:
        """Apply labels to emails in batches for optimal performance.
        
        Args:
            results: List of classification results
            dry_run: Whether this is a dry run
        """
        if dry_run:
            return
        
        # Group emails by labels for batch processing
        label_groups = {}
        
        for result in results:
            if not result.get("processed") or not result.get("final_labels"):
                continue
                
            email_id = result.get("email_id")
            labels = result.get("final_labels", [])
            
            if not email_id or not labels:
                continue
            
            # Group by label combination for batch processing
            label_key = tuple(sorted(labels))
            if label_key not in label_groups:
                label_groups[label_key] = []
            label_groups[label_key].append({
                "email_id": email_id,
                "subject": result.get("subject", "Unknown")[:40],
                "labels": labels
            })
        
        # Apply labels in batches
        batch_tasks = []
        for label_combination, email_group in label_groups.items():
            # Process in smaller batches to avoid overwhelming the API
            email_batches = [email_group[i:i + 5] for i in range(0, len(email_group), 5)]
            
            for email_batch in email_batches:
                batch_tasks.append(
                    self._apply_labels_to_batch(email_batch, list(label_combination))
                )
        
        # Execute all batch operations concurrently
        if batch_tasks:
            await asyncio.gather(*batch_tasks, return_exceptions=True)
    
    async def _apply_labels_to_batch(self, email_batch: List[Dict[str, Any]], labels: List[str]) -> None:
        """Apply the same set of labels to a batch of emails.
        
        Args:
            email_batch: List of email info dicts
            labels: Labels to apply to all emails in the batch
        """
        try:
            # Apply labels to each email in the batch
            for email_info in email_batch:
                try:
                    category = EmailCategory(
                        email_id=email_info["email_id"],
                        suggested_labels=labels,
                        confidence_scores={label: 0.8 for label in labels},
                        reasoning="Batch classification result"
                    )
                    await self.email_service.apply_category(
                        email_info["email_id"], category, create_labels=True
                    )
                    
                    # Update result with applied labels
                    if hasattr(self, '_current_results'):
                        for result in self._current_results:
                            if result.get("email_id") == email_info["email_id"]:
                                result["labels_applied"] = labels
                                break
                    
                    self.console.print(f"   🏷️  Applied {len(labels)} labels to: {email_info['subject']}...")
                    
                except Exception as e:
                    self.console.print(f"   ❌ Failed to apply labels to {email_info['subject']}...: {str(e)}")
            
            # Small delay between batches
            await asyncio.sleep(0.1)
            
        except Exception as e:
            for email_info in email_batch:
                self.console.print(f"   ❌ Batch label application failed for {email_info['subject']}...: {str(e)}")
    
    def _print_detailed_results(self, results: List[Dict[str, Any]]) -> None:
        """Print detailed results with combined analysis."""
        self.console.print(f"\n📋 Detailed Combined Analysis:")
        self.console.print("=" * 80)
        
        for result in results:
            if not result.get("processed"):
                continue
                
            self.console.print(f"\nSubject: {result.get('subject', 'Unknown')[:60]}...")
            
            classifications = result.get('classifications', {})
            
            # Priority details
            priority_info = classifications.get('priority', {})
            if priority_info:
                self.console.print(f"Priority: {priority_info.get('level', 'unknown').title()} (confidence: {priority_info.get('confidence', 0):.1%})")
                if priority_info.get('reasoning'):
                    self.console.print(f"  Reasoning: {priority_info.get('reasoning')}")
            
            # Marketing details
            marketing_info = classifications.get('marketing', {})
            if marketing_info:
                marketing_status = "Yes" if marketing_info.get('is_marketing') else "No"
                self.console.print(f"Marketing: {marketing_status} - {marketing_info.get('subtype', 'unknown').title()} (confidence: {marketing_info.get('confidence', 0):.1%})")
                if marketing_info.get('marketing_indicators'):
                    self.console.print(f"  Indicators: {', '.join(marketing_info.get('marketing_indicators', []))}")
            
            # Receipt details
            receipt_info = classifications.get('receipt', {})
            if receipt_info:
                receipt_status = "Yes" if receipt_info.get('is_receipt') else "No"
                self.console.print(f"Receipt: {receipt_status} - {receipt_info.get('receipt_type', 'unknown').title()} (confidence: {receipt_info.get('confidence', 0):.1%})")
                if receipt_info.get('vendor'):
                    self.console.print(f"  Vendor: {receipt_info.get('vendor')}")
                if receipt_info.get('amount'):
                    amount_str = receipt_info.get('amount')
                    if receipt_info.get('currency'):
                        amount_str = f"{receipt_info.get('currency')} {amount_str}"
                    self.console.print(f"  Amount: {amount_str}")
            
            # Notifications details
            notifications_info = classifications.get('notifications', {})
            if notifications_info:
                notification_status = "Yes" if notifications_info.get('is_notification') else "No"
                self.console.print(f"Notification: {notification_status} - {notifications_info.get('notification_type', 'unknown').title()} (confidence: {notifications_info.get('confidence', 0):.1%})")
                if notifications_info.get('sender_type'):
                    self.console.print(f"  Sender Type: {notifications_info.get('sender_type').title()}")
                if notifications_info.get('urgency_level'):
                    self.console.print(f"  Urgency: {notifications_info.get('urgency_level').title()}")
                if notifications_info.get('action_required'):
                    self.console.print("  ⚡ Action Required")
            
            # Final labels
            final_labels = result.get('final_labels', [])
            labels_text = ', '.join(final_labels) if final_labels else 'None'
            self.console.print(f"Final Labels: {labels_text}")
            
            self.console.print("-" * 80)
    
    def format_output(self, results: Dict[str, Any]) -> None:
        """Format and display results."""
        columns = {
            "subject": "Subject",
            "sender": "Sender",
            "final_labels": "Applied Labels"
        }
        self.display_results_table(results, columns)


@app.command()
def unread(
    priority_confidence: float = typer.Option(0.7, "--priority-confidence", help="Minimum confidence for priority labeling"),
    marketing_confidence: float = typer.Option(0.7, "--marketing-confidence", help="Minimum confidence for marketing labeling"),
    receipt_confidence: float = typer.Option(0.7, "--receipt-confidence", help="Minimum confidence for receipt labeling"),
    notifications_confidence: float = typer.Option(0.7, "--notifications-confidence", help="Minimum confidence for notifications labeling"),
    custom_confidence: float = typer.Option(0.7, "--custom-confidence", help="Minimum confidence for custom category labeling"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview labels without applying them (default: apply labels)"),
    limit: Optional[int] = typer.Option(50, "--limit", "-l", help="Maximum number of emails to process (default: 50)"),
    exclude_personal: bool = typer.Option(True, "--exclude-personal/--include-personal", help="Exclude personal emails from marketing labeling"),
    detailed: bool = typer.Option(False, "--detailed", help="Show detailed analysis results"),
    intelligent_routing: bool = typer.Option(True, "--intelligent-routing/--all-classifiers", help="Use smart routing to determine which classifiers to run (default: intelligent)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed technical information")
):
    """Label unread emails with all classifiers (priority + marketing + receipt + notifications)."""
    
    # Set verbose mode based on flag
    set_verbose_mode(verbose)
    
    @run_async_command
    async def run():
        command = AllLabelCommand()
        await command.initialize()
        
        results = await command.execute(
            target="unread",
            priority_confidence=priority_confidence,
            marketing_confidence=marketing_confidence,
            receipt_confidence=receipt_confidence,
            notifications_confidence=notifications_confidence,
            custom_confidence=custom_confidence,
            dry_run=dry_run,
            limit=limit,
            exclude_personal=exclude_personal,
            detailed=detailed,
            intelligent_routing=intelligent_routing
        )
        
        command.format_output(results)
        return command
    
    run()


@app.command()
def recent(
    days: int = typer.Argument(7, help="Number of days to look back"),
    priority_confidence: float = typer.Option(0.7, "--priority-confidence", help="Minimum confidence for priority labeling"),
    marketing_confidence: float = typer.Option(0.7, "--marketing-confidence", help="Minimum confidence for marketing labeling"),
    receipt_confidence: float = typer.Option(0.7, "--receipt-confidence", help="Minimum confidence for receipt labeling"),
    notifications_confidence: float = typer.Option(0.7, "--notifications-confidence", help="Minimum confidence for notifications labeling"),
    custom_confidence: float = typer.Option(0.7, "--custom-confidence", help="Minimum confidence for custom category labeling"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview labels without applying them (default: apply labels)"),
    limit: Optional[int] = typer.Option(50, "--limit", "-l", help="Maximum number of emails to process (default: 50)"),
    exclude_personal: bool = typer.Option(True, "--exclude-personal/--include-personal", help="Exclude personal emails from marketing labeling"),
    detailed: bool = typer.Option(False, "--detailed", help="Show detailed analysis results"),
    intelligent_routing: bool = typer.Option(True, "--intelligent-routing/--all-classifiers", help="Use smart routing to determine which classifiers to run (default: intelligent)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed technical information")
):
    """Label recent emails with all classifiers (priority + marketing + receipt + notifications)."""
    
    # Set verbose mode based on flag
    set_verbose_mode(verbose)
    
    @run_async_command
    async def run():
        command = AllLabelCommand()
        await command.initialize()
        
        results = await command.execute(
            target=f"recent:{days}days",
            priority_confidence=priority_confidence,
            marketing_confidence=marketing_confidence,
            receipt_confidence=receipt_confidence,
            notifications_confidence=notifications_confidence,
            custom_confidence=custom_confidence,
            dry_run=dry_run,
            limit=limit,
            exclude_personal=exclude_personal,
            detailed=detailed,
            intelligent_routing=intelligent_routing
        )
        
        command.format_output(results)
        return command
    
    run()


@app.command()
def query(
    search_query: str = typer.Argument(..., help="Gmail search query"),
    priority_confidence: float = typer.Option(0.7, "--priority-confidence", help="Minimum confidence for priority labeling"),
    marketing_confidence: float = typer.Option(0.7, "--marketing-confidence", help="Minimum confidence for marketing labeling"),
    receipt_confidence: float = typer.Option(0.7, "--receipt-confidence", help="Minimum confidence for receipt labeling"),
    notifications_confidence: float = typer.Option(0.7, "--notifications-confidence", help="Minimum confidence for notifications labeling"),
    custom_confidence: float = typer.Option(0.7, "--custom-confidence", help="Minimum confidence for custom category labeling"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview labels without applying them (default: apply labels)"),
    limit: Optional[int] = typer.Option(50, "--limit", "-l", help="Maximum number of emails to process (default: 50)"),
    exclude_personal: bool = typer.Option(True, "--exclude-personal/--include-personal", help="Exclude personal emails from marketing labeling"),
    detailed: bool = typer.Option(False, "--detailed", help="Show detailed analysis results"),
    intelligent_routing: bool = typer.Option(True, "--intelligent-routing/--all-classifiers", help="Use smart routing to determine which classifiers to run (default: intelligent)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed technical information")
):
    """Label emails matching custom query with all classifiers (priority + marketing + receipt + notifications)."""
    
    # Set verbose mode based on flag
    set_verbose_mode(verbose)
    
    @run_async_command
    async def run():
        command = AllLabelCommand()
        await command.initialize()
        
        results = await command.execute(
            target=f"query:{search_query}",
            priority_confidence=priority_confidence,
            marketing_confidence=marketing_confidence,
            receipt_confidence=receipt_confidence,
            notifications_confidence=notifications_confidence,
            custom_confidence=custom_confidence,
            dry_run=dry_run,
            limit=limit,
            exclude_personal=exclude_personal,
            detailed=detailed,
            intelligent_routing=intelligent_routing
        )
        
        command.format_output(results)
        return command
    
    run()


if __name__ == "__main__":
    app()