"""CLI-specific agents for email analysis and insights."""

import re
from typing import Dict, List, Any, Optional
import structlog

from src.models.email import EmailMessage
from src.integrations.ollama_client import OllamaModelManager
from src.cli.langchain.prompts import EMAIL_ANALYSIS_PROMPT

logger = structlog.get_logger(__name__)


class EmailAnalysisAgent:
    """Agent for analyzing email patterns and providing insights."""
    
    def __init__(self, ollama_manager: OllamaModelManager):
        """Initialize the email analysis agent.
        
        Args:
            ollama_manager: Ollama client manager for LLM interactions
        """
        self.ollama_manager = ollama_manager
    
    async def analyze_email_collection(self, emails: List[EmailMessage]) -> Dict[str, Any]:
        """Analyze a collection of emails and provide insights.
        
        Args:
            emails: List of emails to analyze
            
        Returns:
            Analysis results with insights and recommendations
        """
        try:
            if not emails:
                return {
                    "error": "No emails provided for analysis",
                    "email_count": 0
                }
            
            # Prepare email subjects for analysis (limit to first 20 for prompt)
            subjects = [email.subject or "No subject" for email in emails[:20]]
            
            prompt = EMAIL_ANALYSIS_PROMPT.format(
                email_count=len(emails),
                email_subjects="\n".join(f"- {subject}" for subject in subjects)
            )
            
            response = await self.ollama_manager.chat(
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": 0.4,  # Balanced creativity for insights
                    "top_p": 0.9,
                    "num_predict": 800
                }
            )
            
            # Parse and structure the analysis
            analysis = self._parse_analysis_response(response.content, emails)
            
            logger.info(
                "Completed email collection analysis",
                email_count=len(emails),
                suggested_categories=len(analysis.get("suggested_categories", []))
            )
            
            return analysis
            
        except Exception as e:
            logger.error("Failed to analyze email collection", error=str(e), email_count=len(emails))
            return {
                "error": f"Analysis failed: {str(e)}",
                "email_count": len(emails)
            }
    
    def _parse_analysis_response(self, response: str, emails: List[EmailMessage]) -> Dict[str, Any]:
        """Parse the analysis response into structured insights.
        
        Args:
            response: Raw LLM response
            emails: Original emails for additional analysis
            
        Returns:
            Structured analysis results
        """
        try:
            # Add basic statistics
            analysis = {
                "email_count": len(emails),
                "analysis_timestamp": None,
                "raw_analysis": response.strip()
            }
            
            # Extract sender statistics
            sender_stats = self._analyze_senders(emails)
            analysis["sender_statistics"] = sender_stats
            
            # Extract basic patterns
            patterns = self._extract_basic_patterns(emails)
            analysis["basic_patterns"] = patterns
            
            # Try to extract structured insights from LLM response
            sections = self._extract_sections(response)
            analysis.update(sections)
            
            return analysis
            
        except Exception as e:
            logger.warning("Failed to parse analysis response", error=str(e))
            return {
                "email_count": len(emails),
                "raw_analysis": response.strip(),
                "error": f"Failed to parse analysis: {str(e)}"
            }
    
    def _analyze_senders(self, emails: List[EmailMessage]) -> Dict[str, Any]:
        """Analyze sender patterns in the email collection.
        
        Args:
            emails: List of emails to analyze
            
        Returns:
            Sender statistics and patterns
        """
        sender_counts = {}
        domain_counts = {}
        automated_senders = []
        
        for email in emails:
            if not email.sender:
                continue
                
            sender = email.sender
            sender_counts[sender] = sender_counts.get(sender, 0) + 1
            
            # Extract domain
            domain_match = re.search(r'@([\\w.-]+)', sender)
            if domain_match:
                domain = domain_match.group(1)
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
            
            # Detect automated senders
            if any(pattern in sender.lower() for pattern in 
                   ['noreply', 'no-reply', 'donotreply', 'notifications', 'automated']):
                automated_senders.append(sender)
        
        # Get top senders and domains
        top_senders = sorted(sender_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        top_domains = sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "total_unique_senders": len(sender_counts),
            "total_unique_domains": len(domain_counts),
            "top_senders": top_senders,
            "top_domains": top_domains,
            "automated_senders_count": len(set(automated_senders)),
            "automated_senders": list(set(automated_senders))[:5]  # Show first 5
        }
    
    def _extract_basic_patterns(self, emails: List[EmailMessage]) -> Dict[str, Any]:
        """Extract basic patterns from email collection.
        
        Args:
            emails: List of emails to analyze
            
        Returns:
            Basic pattern analysis
        """
        patterns = {
            "total_emails": len(emails),
            "emails_with_subjects": len([e for e in emails if e.subject]),
            "emails_with_body_text": len([e for e in emails if e.body_text]),
            "emails_with_body_html": len([e for e in emails if e.body_html]),
            "average_subject_length": 0,
            "common_subject_words": []
        }
        
        # Calculate average subject length
        subjects_with_length = [len(e.subject) for e in emails if e.subject]
        if subjects_with_length:
            patterns["average_subject_length"] = sum(subjects_with_length) / len(subjects_with_length)
        
        # Find common words in subjects
        all_words = []
        for email in emails:
            if email.subject:
                # Simple word extraction
                words = re.findall(r'\\b\\w+\\b', email.subject.lower())
                all_words.extend(words)
        
        # Count word frequency
        word_counts = {}
        for word in all_words:
            if len(word) > 3:  # Skip short words
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Get most common words
        common_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        patterns["common_subject_words"] = common_words
        
        return patterns
    
    def _extract_sections(self, response: str) -> Dict[str, Any]:
        """Extract structured sections from the LLM analysis response.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Structured sections
        """
        sections = {}
        
        try:
            # Look for numbered sections
            section_patterns = [
                (r'1\\.\\s*COMMON THEMES.*?(?=2\\.|$)', "common_themes"),
                (r'2\\.\\s*SUGGESTED CUSTOM CATEGORIES.*?(?=3\\.|$)', "suggested_categories"),
                (r'3\\.\\s*OPTIMIZATION RECOMMENDATIONS.*?(?=4\\.|$)', "optimization_recommendations"),
                (r'4\\.\\s*SENDER ANALYSIS.*?(?=$)', "sender_analysis")
            ]
            
            for pattern, section_name in section_patterns:
                match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
                if match:
                    content = match.group(0).strip()
                    # Clean up the content
                    content = re.sub(r'^\\d+\\.\\s*[A-Z\\s]+\\n', '', content)
                    sections[section_name] = content.strip()
            
            # Try to extract specific insights
            if "suggested_categories" in sections:
                categories = self._extract_suggested_categories(sections["suggested_categories"])
                if categories:
                    sections["structured_categories"] = categories
                    
        except Exception as e:
            logger.warning("Failed to extract sections from analysis", error=str(e))
        
        return sections
    
    def _extract_suggested_categories(self, categories_text: str) -> List[Dict[str, str]]:
        """Extract suggested categories from the categories section.
        
        Args:
            categories_text: Text containing category suggestions
            
        Returns:
            List of category suggestions with names and descriptions
        """
        categories = []
        
        try:
            # Look for category patterns
            lines = categories_text.split('\\n')
            current_category = None
            current_description = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check if this looks like a category name
                if (line.startswith('-') or line.startswith('*') or 
                    re.match(r'^[A-Za-z]+:', line) or
                    re.match(r'^\\d+\\.', line)):
                    
                    # Save previous category if exists
                    if current_category:
                        categories.append({
                            "name": current_category,
                            "description": " ".join(current_description)
                        })
                    
                    # Extract category name
                    category_name = re.sub(r'^[-*\\d\\.\\s]+', '', line)
                    category_name = re.sub(r':.*$', '', category_name).strip()
                    
                    current_category = category_name
                    current_description = []
                    
                    # Check if description is on same line
                    if ':' in line:
                        desc_part = line.split(':', 1)[1].strip()
                        if desc_part:
                            current_description.append(desc_part)
                else:
                    # This is likely a continuation of description
                    if current_category:
                        current_description.append(line)
            
            # Save last category
            if current_category:
                categories.append({
                    "name": current_category,
                    "description": " ".join(current_description)
                })
                
        except Exception as e:
            logger.warning("Failed to extract suggested categories", error=str(e))
        
        return categories[:5]  # Limit to 5 categories