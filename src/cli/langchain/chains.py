"""Custom LangChain chains for email classification and analysis."""

import json
import re
from typing import Dict, List, Any, Optional
import structlog

from src.models.email import EmailMessage
from src.integrations.ollama_client import OllamaModelManager
from src.cli.langchain.prompts import (
    SEARCH_TERMS_GENERATION_PROMPT,
    CUSTOM_CLASSIFICATION_PROMPT,
    EMAIL_ROUTING_PROMPT
)

logger = structlog.get_logger(__name__)


class CustomLabelChain:
    """LangChain-style chain for generating search terms and classifying custom labels."""
    
    def __init__(self, ollama_manager: OllamaModelManager):
        """Initialize the custom label chain.
        
        Args:
            ollama_manager: Ollama client manager for LLM interactions
        """
        self.ollama_manager = ollama_manager
        
    async def generate_search_terms(self, category: str, context: str = None) -> List[str]:
        """Generate relevant search terms for a custom category.
        
        Args:
            category: The category name to generate terms for
            context: Additional context about the category (optional)
            
        Returns:
            List of relevant search terms
        """
        try:
            prompt = SEARCH_TERMS_GENERATION_PROMPT.format(
                category=category,
                context=context or "No additional context provided"
            )
            
            response = await self.ollama_manager.chat(
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": 0.3,  # Slightly creative but focused
                    "top_p": 0.9,
                    "num_predict": 150
                }
            )
            
            # Extract terms from response
            content = response.content.strip()
            
            # Try to find the terms after "Terms:" if present
            if "Terms:" in content:
                terms_text = content.split("Terms:")[-1].strip()
            else:
                terms_text = content
            
            # Split by commas and clean up
            terms = [term.strip().strip('"').strip("'") for term in terms_text.split(',')]
            
            # Filter out empty terms and ensure we have reasonable terms
            terms = [term for term in terms if term and len(term) > 1]
            
            # Limit to 10 terms maximum
            terms = terms[:10]
            
            # If we didn't get good terms, fall back to the category itself
            if not terms or len(terms) < 2:
                terms = [category.lower()]
                
            logger.info(
                "Generated search terms",
                category=category,
                terms_count=len(terms),
                terms=terms
            )
            
            return terms
            
        except Exception as e:
            logger.error("Failed to generate search terms", category=category, error=str(e))
            # Fallback to category name
            return [category.lower()]
    
    async def classify_for_category(
        self,
        email: EmailMessage,
        category: str,
        search_terms: List[str]
    ) -> Dict[str, Any]:
        """Classify an email for a specific custom category.
        
        Args:
            email: Email to classify
            category: Category to classify for
            search_terms: Related search terms for the category
            
        Returns:
            Classification result with confidence, reasoning, etc.
        """
        try:
            # Prepare email content for analysis
            content = email.body_text[:1000] if email.body_text else ""
            
            prompt = CUSTOM_CLASSIFICATION_PROMPT.format(
                category=category,
                subject=email.subject or "No subject",
                sender=email.sender or "Unknown sender",
                content=content,
                search_terms=", ".join(search_terms)
            )
            
            response = await self.ollama_manager.chat(
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": 0.1,  # Low temperature for consistent classification
                    "top_p": 0.95,
                    "num_predict": 200
                }
            )
            
            return self._parse_classification_response(response.content, category)
            
        except Exception as e:
            logger.error(
                "Failed to classify email for category",
                email_id=email.id,
                category=category,
                error=str(e)
            )
            return {
                "classification": False,
                "confidence": 0.0,
                "reasoning": f"Error during classification: {str(e)}",
                "suggested_label": None
            }
    
    def _parse_classification_response(self, response: str, category: str) -> Dict[str, Any]:
        """Parse the LLM classification response.
        
        Args:
            response: Raw LLM response
            category: Category being classified
            
        Returns:
            Parsed classification result
        """
        try:
            # Initialize default values
            classification = False
            confidence = 0.0
            reasoning = "Unable to parse response"
            suggested_label = None
            
            # Parse the structured response
            lines = response.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                
                if line.startswith('CLASSIFICATION:'):
                    classification_text = line.split(':', 1)[1].strip().upper()
                    classification = classification_text in ['YES', 'TRUE', '1']
                    
                elif line.startswith('CONFIDENCE:'):
                    confidence_text = line.split(':', 1)[1].strip()
                    try:
                        confidence = float(confidence_text)
                        confidence = max(0.0, min(1.0, confidence))  # Clamp between 0 and 1
                    except ValueError:
                        confidence = 0.5
                        
                elif line.startswith('REASONING:'):
                    reasoning = line.split(':', 1)[1].strip()
                    
                elif line.startswith('SUGGESTED_LABEL:'):
                    label_text = line.split(':', 1)[1].strip()
                    if label_text and label_text.lower() not in ['none', 'null', '']:
                        suggested_label = label_text
            
            # If no suggested label but classification is positive, use category
            if classification and not suggested_label:
                suggested_label = category
                
            result = {
                "classification": classification,
                "confidence": confidence,
                "reasoning": reasoning,
                "suggested_label": suggested_label
            }
            
            logger.debug(
                "Parsed classification response",
                category=category,
                result=result
            )
            
            return result
            
        except Exception as e:
            logger.warning("Failed to parse classification response", response=response, error=str(e))
            return {
                "classification": False,
                "confidence": 0.0,
                "reasoning": "Failed to parse classification response",
                "suggested_label": None
            }


class EmailRouterChain:
    """Chain for intelligently routing emails to appropriate classification services."""
    
    def __init__(self, ollama_manager: OllamaModelManager):
        """Initialize the email router chain.
        
        Args:
            ollama_manager: Ollama client manager for LLM interactions
        """
        self.ollama_manager = ollama_manager
        self._init_heuristic_patterns()
    
    def _init_heuristic_patterns(self):
        """Initialize heuristic patterns for fast routing decisions."""
        # Marketing indicators
        self.marketing_keywords = {
            'unsubscribe', 'newsletter', 'promotion', 'sale', 'discount', 
            'offer', 'deal', 'marketing', 'campaign', 'subscribe', 'promo',
            'exclusive', 'limited time', 'buy now', 'click here', 'free shipping'
        }
        self.marketing_senders = {
            'noreply', 'no-reply', 'marketing', 'newsletter', 'promo', 
            'notifications', 'updates', 'news'
        }
        
        # Receipt indicators
        self.receipt_keywords = {
            'receipt', 'invoice', 'payment', 'order', 'purchase', 'transaction',
            'confirmation', 'billing', 'charged', 'refund', 'shipped', 'delivery',
            'total', 'amount', 'paid', 'subscription', 'renewal'
        }
        self.receipt_senders = {
            'amazon', 'paypal', 'stripe', 'apple', 'google', 'microsoft',
            'billing', 'payments', 'orders', 'receipts'
        }
        
        # Priority indicators
        self.priority_keywords = {
            'urgent', 'asap', 'immediate', 'critical', 'important', 'deadline',
            'emergency', 'action required', 'time sensitive', 'priority',
            'meeting', 'schedule', 'cancel', 'reschedule', 'approval needed'
        }
        self.priority_senders = {
            'boss', 'manager', 'ceo', 'admin', 'support', 'security',
            'alerts', 'urgent', 'critical'
        }
        
        # Notification indicators
        self.notification_keywords = {
            'alert', 'notification', 'reminder', 'update', 'status', 'report',
            'backup', 'security', 'maintenance', 'system', 'automated',
            'service', 'monitor', 'error', 'warning', 'success'
        }
        self.notification_senders = {
            'noreply', 'no-reply', 'system', 'admin', 'alerts', 'notifications',
            'monitoring', 'security', 'backup', 'automated'
        }
    
    async def route_email(self, email: EmailMessage) -> List[str]:
        """Determine which classification services should process an email.
        
        Args:
            email: Email to analyze for routing
            
        Returns:
            List of service names that should process this email
        """
        try:
            # Start with heuristic routing for efficiency
            heuristic_services = self._heuristic_route(email)
            
            # If heuristics give us confident results, use them
            if len(heuristic_services) > 0:
                logger.debug(
                    "Used heuristic routing",
                    email_id=email.id,
                    subject=email.subject[:50] if email.subject else "No subject",
                    services=heuristic_services
                )
                return heuristic_services
            
            # Fall back to LLM routing for ambiguous cases
            llm_services = await self._llm_route(email)
            
            logger.debug(
                "Used LLM routing",
                email_id=email.id, 
                subject=email.subject[:50] if email.subject else "No subject",
                services=llm_services
            )
            
            return llm_services
            
        except Exception as e:
            logger.error("Failed to route email", email_id=email.id, error=str(e))
            # Default routing - apply priority to all emails
            return ["priority"]
    
    def _heuristic_route(self, email: EmailMessage) -> List[str]:
        """Use fast heuristic patterns to route emails.
        
        Args:
            email: Email to analyze
            
        Returns:
            List of services based on heuristic analysis
        """
        services = []
        
        # Prepare text for analysis
        subject_lower = (email.subject or "").lower()
        sender_lower = (email.sender or "").lower()
        body_text = (email.body_text or "")[:500].lower()
        
        combined_text = f"{subject_lower} {body_text}"
        
        # Check for receipt indicators (high confidence)
        if (any(keyword in combined_text for keyword in self.receipt_keywords) or
            any(sender in sender_lower for sender in self.receipt_senders)):
            services.append("receipt")
            
        # Check for marketing indicators (high confidence)
        if (any(keyword in combined_text for keyword in self.marketing_keywords) or
            any(sender in sender_lower for sender in self.marketing_senders) or
            'unsubscribe' in body_text):
            services.append("marketing")
            
        # Check for notification indicators
        if (any(keyword in combined_text for keyword in self.notification_keywords) or
            any(sender in sender_lower for sender in self.notification_senders)):
            services.append("notifications")
            
        # Check for priority indicators
        if (any(keyword in combined_text for keyword in self.priority_keywords) or
            any(sender in sender_lower for sender in self.priority_senders)):
            services.append("priority")
        
        # Always include priority unless it's clearly just a receipt or marketing
        if not services or (len(services) == 1 and services[0] in ["receipt", "marketing"]):
            if "priority" not in services:
                services.append("priority")
        
        return services
    
    def _is_ambiguous(self, email: EmailMessage) -> bool:
        """Determine if an email needs LLM analysis due to ambiguity.
        
        Args:
            email: Email to check
            
        Returns:
            True if email is ambiguous and needs LLM routing
        """
        # Simple heuristics to detect ambiguous cases
        subject = (email.subject or "").lower()
        
        # Very short subjects are often ambiguous
        if len(subject) < 10:
            return True
            
        # Subjects with only common words
        common_words = {'re:', 'fwd:', 'hello', 'hi', 'thanks', 'update', 'info'}
        if any(word in subject for word in common_words) and len(subject) < 30:
            return True
            
        return False
    
    async def _llm_route(self, email: EmailMessage) -> List[str]:
        """Use LLM for intelligent routing of ambiguous emails.
        
        Args:
            email: Email to analyze
            
        Returns:
            List of services from LLM analysis
        """
        # Prepare email snippet for analysis
        snippet = ""
        if email.body_text:
            snippet = email.body_text[:300]
        elif email.body_html:
            # Simple HTML to text conversion
            snippet = re.sub(r'<[^>]+>', ' ', email.body_html[:300])
        
        prompt = EMAIL_ROUTING_PROMPT.format(
            subject=email.subject or "No subject",
            sender=email.sender or "Unknown sender", 
            snippet=snippet
        )
        
        response = await self.ollama_manager.chat(
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": 0.2,
                "top_p": 0.9,
                "num_predict": 100
            }
        )
        
        # Parse the response to extract service names
        return self._parse_routing_response(response.content)
    
    def _parse_routing_response(self, response: str) -> List[str]:
        """Parse the routing response to extract service names.
        
        Args:
            response: Raw LLM response
            
        Returns:
            List of valid service names
        """
        try:
            # Extract service names from response
            content = response.strip().lower()
            
            # Look for "Services:" prefix
            if "services:" in content:
                content = content.split("services:")[-1].strip()
            
            # Split by commas and clean up
            services = [service.strip() for service in content.split(',')]
            
            # Validate service names
            valid_services = {"priority", "marketing", "receipt", "notifications", "custom"}
            filtered_services = [s for s in services if s in valid_services]
            
            # Ensure we always include priority as a fallback
            if not filtered_services:
                filtered_services = ["priority"]
            elif "priority" not in filtered_services:
                filtered_services.insert(0, "priority")
            
            return filtered_services
            
        except Exception as e:
            logger.warning("Failed to parse routing response", response=response, error=str(e))
            return ["priority"]