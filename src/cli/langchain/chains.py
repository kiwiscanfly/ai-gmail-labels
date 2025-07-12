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
    
    async def route_email(self, email: EmailMessage) -> List[str]:
        """Determine which classification services should process an email.
        
        Args:
            email: Email to analyze for routing
            
        Returns:
            List of service names that should process this email
        """
        try:
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
            services = self._parse_routing_response(response.content)
            
            logger.debug(
                "Routed email to services",
                email_id=email.id,
                subject=email.subject[:50] if email.subject else "No subject",
                services=services
            )
            
            return services
            
        except Exception as e:
            logger.error("Failed to route email", email_id=email.id, error=str(e))
            # Default routing - apply priority to all emails
            return ["priority"]
    
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