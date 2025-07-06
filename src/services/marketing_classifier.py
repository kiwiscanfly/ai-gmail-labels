"""Marketing email classification service using LLM-based analysis and structural features."""

import asyncio
import re
import time
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict
import structlog

from src.models.email import EmailMessage
from src.integrations.ollama_client import get_ollama_manager
from src.core.exceptions import ServiceError

logger = structlog.get_logger(__name__)


@dataclass
class MarketingClassificationResult:
    """Marketing email classification result."""
    is_marketing: bool
    confidence: float  # 0.0 to 1.0
    subtype: str  # "promotional", "newsletter", "transactional", "hybrid", "personal"
    reasoning: str
    structural_signals: Dict[str, Any] = field(default_factory=dict)
    sender_reputation: float = 0.0
    marketing_indicators: List[str] = field(default_factory=list)
    unsubscribe_detected: bool = False
    bulk_sending_indicators: bool = False


@dataclass
class SenderMarketingProfile:
    """Sender profile for marketing behavior tracking."""
    email: str
    total_emails: int = 0
    marketing_emails: int = 0
    marketing_rate: float = 0.0
    common_patterns: List[str] = field(default_factory=list)
    sending_schedule: Dict[str, int] = field(default_factory=dict)
    last_updated: float = field(default_factory=time.time)
    domain_type: str = "unknown"  # "business", "platform", "individual"


class MarketingClassificationCache:
    """Caching system for marketing classification results."""
    
    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size
        
    def _generate_cache_key(self, email: EmailMessage) -> str:
        """Generate cache key from email content."""
        content = f"{email.subject}{email.body_text[:500]}{email.sender}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get_cached_result(self, email: EmailMessage) -> Optional[MarketingClassificationResult]:
        """Get cached classification result."""
        cache_key = self._generate_cache_key(email)
        cached = self.cache.get(cache_key)
        
        if cached and time.time() - cached['timestamp'] < 3600:  # 1 hour cache
            return cached['result']
        return None
    
    def cache_result(self, email: EmailMessage, result: MarketingClassificationResult) -> None:
        """Cache classification result."""
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]['timestamp'])
            del self.cache[oldest_key]
        
        cache_key = self._generate_cache_key(email)
        self.cache[cache_key] = {
            'result': result,
            'timestamp': time.time()
        }


class EmailStructureAnalyzer:
    """Analyzes email structure for marketing indicators."""
    
    def __init__(self):
        self.marketing_keywords = [
            "sale", "discount", "offer", "deal", "promotion", "special",
            "limited time", "act now", "free shipping", "order now",
            "subscribe", "newsletter", "unsubscribe", "opt out"
        ]
        
        self.unsubscribe_patterns = [
            r'unsubscribe',
            r'opt.?out',
            r'email preferences',
            r'manage subscription',
            r'remove.*list'
        ]
        
        self.cta_patterns = [
            r'click here',
            r'learn more',
            r'shop now',
            r'buy now',
            r'get started',
            r'sign up',
            r'join now'
        ]
        
        self.promotional_patterns = [
            r'\b\d+% off\b',
            r'\$\d+\.\d+',
            r'save \$\d+',
            r'(free|bonus|gift)\s+(shipping|trial|offer)',
            r'(limited|exclusive)\s+(time|offer)',
            r'ends? (today|soon|tonight)'
        ]
    
    def extract_features(self, email: EmailMessage) -> Dict[str, Any]:
        """Extract structural features from email."""
        text = f"{email.subject} {email.body_text}".lower()
        html = email.body_html or ""
        
        features = {
            'has_unsubscribe': self._detect_unsubscribe(text, html),
            'cta_count': self._count_ctas(text, html),
            'promotional_keywords': self._count_promotional_keywords(text),
            'marketing_patterns': self._count_marketing_patterns(text),
            'html_complexity': self._analyze_html_complexity(html),
            'has_address': self._detect_physical_address(text, html),
            'bulk_indicators': self._detect_bulk_indicators(email),
            'link_count': self._count_links(html),
            'image_count': self._count_images(html),
            'text_to_html_ratio': self._calculate_text_ratio(email)
        }
        
        # Calculate overall marketing score
        features['marketing_score'] = self._calculate_marketing_score(features)
        features['ambiguity_score'] = self._calculate_ambiguity_score(features)
        
        return features
    
    def _detect_unsubscribe(self, text: str, html: str) -> bool:
        """Detect unsubscribe links/text."""
        content = f"{text} {html}".lower()
        return any(re.search(pattern, content, re.IGNORECASE) 
                  for pattern in self.unsubscribe_patterns)
    
    def _count_ctas(self, text: str, html: str) -> int:
        """Count call-to-action phrases."""
        content = f"{text} {html}".lower()
        return sum(len(re.findall(pattern, content, re.IGNORECASE)) 
                  for pattern in self.cta_patterns)
    
    def _count_promotional_keywords(self, text: str) -> int:
        """Count promotional keywords."""
        return sum(1 for keyword in self.marketing_keywords if keyword in text)
    
    def _count_marketing_patterns(self, text: str) -> int:
        """Count promotional patterns like prices, discounts."""
        return sum(len(re.findall(pattern, text, re.IGNORECASE)) 
                  for pattern in self.promotional_patterns)
    
    def _analyze_html_complexity(self, html: str) -> float:
        """Analyze HTML complexity (0.0 to 1.0)."""
        if not html:
            return 0.0
        
        # Count HTML elements
        tag_count = len(re.findall(r'<[^>]+>', html))
        style_blocks = len(re.findall(r'<style[^>]*>.*?</style>', html, re.DOTALL))
        inline_styles = len(re.findall(r'style\s*=', html))
        
        # Normalize complexity score
        complexity = min((tag_count / 100) + (style_blocks / 5) + (inline_styles / 20), 1.0)
        return complexity
    
    def _detect_physical_address(self, text: str, html: str) -> bool:
        """Detect physical address in footer."""
        content = f"{text} {html}".lower()
        address_patterns = [
            r'\d+\s+[a-z\s]+street',
            r'\d+\s+[a-z\s]+(ave|avenue|blvd|boulevard|rd|road)',
            r'[a-z\s]+,\s*[a-z]{2}\s*\d{5}',  # City, State ZIP
            r'\d{3}-\d{3}-\d{4}',  # Phone number
        ]
        return any(re.search(pattern, content, re.IGNORECASE) 
                  for pattern in address_patterns)
    
    def _detect_bulk_indicators(self, email: EmailMessage) -> bool:
        """Detect bulk sending indicators."""
        # Check for bulk sending headers or patterns
        bulk_indicators = [
            'list-unsubscribe' in str(email.body_html).lower(),
            'bulk' in email.sender.lower() if email.sender else False,
            'noreply' in email.sender.lower() if email.sender else False,
            'no-reply' in email.sender.lower() if email.sender else False
        ]
        return any(bulk_indicators)
    
    def _count_links(self, html: str) -> int:
        """Count links in HTML."""
        if not html:
            return 0
        return len(re.findall(r'<a[^>]+href=', html, re.IGNORECASE))
    
    def _count_images(self, html: str) -> int:
        """Count images in HTML."""
        if not html:
            return 0
        return len(re.findall(r'<img[^>]+', html, re.IGNORECASE))
    
    def _calculate_text_ratio(self, email: EmailMessage) -> float:
        """Calculate text to HTML ratio."""
        text_len = len(email.body_text) if email.body_text else 0
        html_len = len(email.body_html) if email.body_html else 0
        
        if html_len == 0:
            return 1.0 if text_len > 0 else 0.0
        
        return text_len / (text_len + html_len)
    
    def _calculate_marketing_score(self, features: Dict[str, Any]) -> float:
        """Calculate overall marketing score based on features."""
        score = 0.0
        
        # Unsubscribe link is strong indicator
        if features['has_unsubscribe']:
            score += 0.3
        
        # Multiple CTAs indicate marketing
        if features['cta_count'] > 2:
            score += 0.2
        elif features['cta_count'] > 0:
            score += 0.1
        
        # Promotional content
        if features['promotional_keywords'] > 3:
            score += 0.2
        elif features['promotional_keywords'] > 0:
            score += 0.1
        
        # Marketing patterns (prices, discounts)
        if features['marketing_patterns'] > 2:
            score += 0.2
        elif features['marketing_patterns'] > 0:
            score += 0.1
        
        # HTML complexity suggests marketing design
        if features['html_complexity'] > 0.7:
            score += 0.15
        elif features['html_complexity'] > 0.3:
            score += 0.05
        
        # Bulk sending indicators
        if features['bulk_indicators']:
            score += 0.15
        
        return min(score, 1.0)
    
    def _calculate_ambiguity_score(self, features: Dict[str, Any]) -> float:
        """Calculate how ambiguous the classification is."""
        marketing_score = features['marketing_score']
        
        # Ambiguity is highest when score is around 0.5
        if 0.3 <= marketing_score <= 0.7:
            return 1.0 - abs(marketing_score - 0.5) * 2
        else:
            return 0.0


class MarketingEmailClassifier:
    """LLM-based marketing email classification service."""
    
    def __init__(self):
        """Initialize the marketing classifier."""
        self.ollama_manager = None
        self.cache = MarketingClassificationCache()
        self.structure_analyzer = EmailStructureAnalyzer()
        self.sender_profiles: Dict[str, SenderMarketingProfile] = {}
        
        # Classification prompt
        self.classification_prompt = """Analyze this email to determine if it's a marketing email. Follow these steps:

1. SENDER ANALYSIS
   - Is the sender a business or individual?
   - Does the sender domain suggest marketing/automation?
   - Are there bulk sending indicators?

2. CONTENT PURPOSE
   - What is the primary intent of this email?
   - Is it informational, transactional, or promotional?
   - Are there commercial elements present?

3. STRUCTURAL MARKERS
   - Unsubscribe links present: {has_unsubscribe}
   - Call-to-action count: {cta_count}
   - Marketing patterns found: {marketing_patterns}
   - HTML complexity: {html_complexity}

4. CLASSIFICATION
   Based on the analysis, classify as one of:
   - MARKETING_PROMOTIONAL (sales, discounts, product launches)
   - MARKETING_NEWSLETTER (educational content with soft promotion)
   - MARKETING_HYBRID (transactional with marketing elements)
   - TRANSACTIONAL (pure transactional: receipts, confirmations, alerts, notifications)
   - PERSONAL (individual non-automated communication)

Email to classify:
From: {sender}
Subject: {subject}
Content: {content}

Respond with exactly one line in this format:
CLASSIFICATION|CONFIDENCE|REASONING

Example: MARKETING_PROMOTIONAL|0.85|Contains discount offers and unsubscribe link
Example: TRANSACTIONAL|0.95|Parking expiration notification with no marketing content

Your response:"""
    
    async def initialize(self) -> None:
        """Initialize the Ollama client."""
        self.ollama_manager = await get_ollama_manager()
        logger.info("Marketing classifier initialized with LLM support")
    
    async def classify_email(self, email: EmailMessage) -> MarketingClassificationResult:
        """Classify email as marketing or not with detailed analysis."""
        try:
            # Check cache first
            cached_result = self.cache.get_cached_result(email)
            if cached_result:
                logger.debug("Using cached marketing classification", email_id=email.id)
                return cached_result
            
            # Extract structural features
            structural_features = self.structure_analyzer.extract_features(email)
            
            # Get sender profile
            sender_profile = self._get_or_create_sender_profile(email.sender)
            
            # Quick classification for obvious cases
            if structural_features['marketing_score'] > 0.8:
                result = MarketingClassificationResult(
                    is_marketing=True,
                    confidence=0.95,
                    subtype="promotional",
                    reasoning="High marketing score from structural analysis",
                    structural_signals=structural_features,
                    sender_reputation=sender_profile.marketing_rate,
                    marketing_indicators=self._extract_marketing_indicators(structural_features),
                    unsubscribe_detected=structural_features['has_unsubscribe'],
                    bulk_sending_indicators=structural_features['bulk_indicators']
                )
            elif structural_features['marketing_score'] < 0.2:
                result = MarketingClassificationResult(
                    is_marketing=False,
                    confidence=0.90,
                    subtype="personal",
                    reasoning="Low marketing score from structural analysis",
                    structural_signals=structural_features,
                    sender_reputation=sender_profile.marketing_rate
                )
            else:
                # Use LLM for ambiguous cases
                result = await self._llm_classify(email, structural_features, sender_profile)
            
            # Update sender profile
            self._update_sender_profile(sender_profile, result)
            
            # Cache result
            self.cache.cache_result(email, result)
            
            return result
            
        except Exception as e:
            logger.error("Failed to classify marketing email", email_id=email.id, error=str(e))
            return MarketingClassificationResult(
                is_marketing=False,
                confidence=0.0,
                subtype="error",
                reasoning=f"Error during classification: {str(e)}"
            )
    
    async def _llm_classify(self, email: EmailMessage, features: Dict[str, Any], 
                           sender_profile: SenderMarketingProfile) -> MarketingClassificationResult:
        """Use LLM to classify ambiguous emails."""
        # Prepare content for analysis
        content = email.body_text[:2000] if email.body_text else ""
        
        prompt = self.classification_prompt.format(
            has_unsubscribe=features['has_unsubscribe'],
            cta_count=features['cta_count'],
            marketing_patterns=features['marketing_patterns'],
            html_complexity=f"{features['html_complexity']:.2f}",
            sender=email.sender or "Unknown",
            subject=email.subject or "No subject",
            content=content
        )
        
        try:
            response = await self.ollama_manager.chat(
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": 0.1,
                    "top_p": 0.95,
                    "num_predict": 150
                }
            )
            
            return self._parse_llm_response(response.content, features, sender_profile)
            
        except Exception as e:
            logger.error("LLM marketing classification failed", error=str(e))
            # Fallback to structural analysis
            return self._fallback_classification(features, sender_profile)
    
    def _parse_llm_response(self, response: str, features: Dict[str, Any], 
                           sender_profile: SenderMarketingProfile) -> MarketingClassificationResult:
        """Parse LLM response into structured result."""
        try:
            # Find the actual classification line
            lines = response.strip().split('\n')
            data_line = None
            
            # Sometimes the LLM includes "REASONING:" prefix, remove it
            cleaned_response = response.replace("REASONING: ", "").strip()
            
            for line in lines:
                if '|' in line and not line.startswith('CLASSIFICATION|CONFIDENCE|REASONING'):
                    parts = line.split('|')
                    if len(parts) >= 3:
                        try:
                            float(parts[1].strip())  # Validate confidence is a number
                            data_line = line
                            break
                        except ValueError:
                            continue
            
            # If no proper format found, check if it's just the reasoning
            if not data_line and cleaned_response and not '|' in cleaned_response:
                # LLM returned only reasoning, need to fallback
                logger.warning("LLM returned only reasoning without classification format", response=response)
                return self._fallback_classification(features, sender_profile)
            
            if data_line:
                parts = data_line.split('|')
                classification = parts[0].strip().upper()
                confidence = float(parts[1].strip())
                reasoning = parts[2].strip()
                
                # Determine if marketing and subtype
                is_marketing = classification.startswith('MARKETING')
                
                if is_marketing:
                    # Extract marketing subtype
                    subtype = classification.lower().replace('marketing_', '')
                elif classification == 'TRANSACTIONAL':
                    is_marketing = False
                    subtype = 'transactional'
                else:
                    # Personal or other non-marketing
                    is_marketing = False
                    subtype = 'personal'
                
                return MarketingClassificationResult(
                    is_marketing=is_marketing,
                    confidence=min(max(confidence, 0.0), 1.0),
                    subtype=subtype,
                    reasoning=reasoning,
                    structural_signals=features,
                    sender_reputation=sender_profile.marketing_rate,
                    marketing_indicators=self._extract_marketing_indicators(features),
                    unsubscribe_detected=features['has_unsubscribe'],
                    bulk_sending_indicators=features['bulk_indicators']
                )
                
        except Exception as e:
            logger.warning("Failed to parse marketing LLM response", response=response, error=str(e))
        
        # Fallback if parsing fails
        return self._fallback_classification(features, sender_profile)
    
    def _fallback_classification(self, features: Dict[str, Any], 
                                sender_profile: SenderMarketingProfile) -> MarketingClassificationResult:
        """Fallback classification based on structural features."""
        marketing_score = features['marketing_score']
        
        # Apply sender reputation adjustment
        if sender_profile.marketing_rate > 0.7:
            marketing_score = min(marketing_score + 0.2, 1.0)
        elif sender_profile.marketing_rate < 0.3:
            marketing_score = max(marketing_score - 0.2, 0.0)
        
        is_marketing = marketing_score > 0.5
        confidence = abs(marketing_score - 0.5) + 0.5  # Convert to confidence
        
        # Determine subtype
        if is_marketing:
            if features['promotional_keywords'] > 2 or features['marketing_patterns'] > 1:
                subtype = "promotional"
            elif features['has_unsubscribe'] and features['cta_count'] <= 2:
                subtype = "newsletter"
            else:
                subtype = "general"
        else:
            subtype = "personal"
        
        return MarketingClassificationResult(
            is_marketing=is_marketing,
            confidence=confidence,
            subtype=subtype,
            reasoning="Rule-based fallback classification",
            structural_signals=features,
            sender_reputation=sender_profile.marketing_rate,
            marketing_indicators=self._extract_marketing_indicators(features),
            unsubscribe_detected=features['has_unsubscribe'],
            bulk_sending_indicators=features['bulk_indicators']
        )
    
    def _extract_marketing_indicators(self, features: Dict[str, Any]) -> List[str]:
        """Extract list of marketing indicators found."""
        indicators = []
        
        if features['has_unsubscribe']:
            indicators.append("unsubscribe_link")
        if features['cta_count'] > 2:
            indicators.append("multiple_ctas")
        if features['promotional_keywords'] > 0:
            indicators.append("promotional_keywords")
        if features['marketing_patterns'] > 0:
            indicators.append("price_discount_patterns")
        if features['html_complexity'] > 0.5:
            indicators.append("complex_html_design")
        if features['bulk_indicators']:
            indicators.append("bulk_sending")
        if features['has_address']:
            indicators.append("physical_address")
        
        return indicators
    
    def _get_or_create_sender_profile(self, sender: str) -> SenderMarketingProfile:
        """Get or create sender marketing profile."""
        if not sender:
            sender = "unknown@unknown.com"
        
        if sender not in self.sender_profiles:
            self.sender_profiles[sender] = SenderMarketingProfile(email=sender)
        
        return self.sender_profiles[sender]
    
    def _update_sender_profile(self, profile: SenderMarketingProfile, 
                              result: MarketingClassificationResult) -> None:
        """Update sender profile based on classification result."""
        profile.total_emails += 1
        profile.last_updated = time.time()
        
        if result.is_marketing:
            profile.marketing_emails += 1
        
        # Update marketing rate
        profile.marketing_rate = profile.marketing_emails / profile.total_emails
        
        # Track sending time patterns
        hour = datetime.now().hour
        profile.sending_schedule[str(hour)] = profile.sending_schedule.get(str(hour), 0) + 1
        
        # Determine domain type
        if not profile.domain_type or profile.domain_type == "unknown":
            if any(platform in profile.email.lower() for platform in 
                   ['mailchimp', 'constantcontact', 'sendgrid', 'mailgun']):
                profile.domain_type = "platform"
            elif profile.marketing_rate > 0.8 and profile.total_emails > 3:
                profile.domain_type = "business"
            elif profile.marketing_rate < 0.2 and profile.total_emails > 3:
                profile.domain_type = "individual"
    
    def get_sender_statistics(self) -> Dict[str, Any]:
        """Get sender marketing statistics."""
        if not self.sender_profiles:
            return {"total_senders": 0}
        
        total_senders = len(self.sender_profiles)
        marketing_senders = sum(1 for p in self.sender_profiles.values() if p.marketing_rate > 0.7)
        individual_senders = sum(1 for p in self.sender_profiles.values() if p.marketing_rate < 0.3)
        
        avg_marketing_rate = sum(p.marketing_rate for p in self.sender_profiles.values()) / total_senders
        
        return {
            "total_senders": total_senders,
            "marketing_senders": marketing_senders,
            "individual_senders": individual_senders,
            "average_marketing_rate": round(avg_marketing_rate, 3)
        }