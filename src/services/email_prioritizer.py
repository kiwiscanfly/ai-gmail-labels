"""Email prioritization service using LLM-based classification and semantic analysis."""

import asyncio
import re
import time
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import structlog

from src.models.email import EmailMessage, EmailReference
from src.integrations.ollama_client import get_ollama_manager
from src.core.exceptions import ServiceError

logger = structlog.get_logger(__name__)


@dataclass
class PriorityScore:
    """Email priority assessment result."""
    level: str  # "critical", "high", "medium", "low"
    confidence: float  # 0.0 to 1.0
    reasoning: str
    is_genuine_urgency: bool
    authenticity_score: float
    detected_tactics: List[str] = field(default_factory=list)
    sender_reputation: float = 0.0
    needs_review: bool = False
    is_marketing: bool = False
    marketing_subtype: str = ""
    marketing_confidence: float = 0.0
    is_receipt: bool = False
    receipt_type: str = ""
    receipt_vendor: str = ""
    receipt_amount: str = ""


@dataclass
class SenderProfile:
    """Sender behavior profile for reputation scoring."""
    email: str
    total_emails: int = 0
    urgency_claims: int = 0
    false_urgency_rate: float = 0.0
    marketing_patterns: int = 0
    last_updated: float = field(default_factory=time.time)
    reputation_score: float = 0.5  # 0.0 = known spam, 1.0 = trusted


class EmailPriorityCache:
    """Caching system for email classification results."""
    
    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size
        
    def _generate_cache_key(self, email: EmailMessage) -> str:
        """Generate cache key from email content."""
        content = f"{email.subject}{email.body_text[:500]}{email.sender}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get_cached_result(self, email: EmailMessage) -> Optional[PriorityScore]:
        """Get cached classification result."""
        cache_key = self._generate_cache_key(email)
        cached = self.cache.get(cache_key)
        
        if cached and time.time() - cached['timestamp'] < 3600:  # 1 hour cache
            return cached['result']
        return None
    
    def cache_result(self, email: EmailMessage, result: PriorityScore) -> None:
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


class SemanticUrgencyDetector:
    """Detects genuine urgency vs marketing manipulation."""
    
    def __init__(self):
        self.marketing_indicators = [
            "act now", "limited time", "expires soon", "don't miss out",
            "final chance", "hurry up", "last call", "ending today",
            "while supplies last", "flash sale", "deal expires",
            "countdown", "only hours left", "going fast"
        ]
        
        self.genuine_indicators = [
            "deadline", "meeting at", "due by", "required by",
            "needs response", "urgent request", "immediate action",
            "time sensitive", "asap", "critical", "emergency", 
            "your order", "your account", "available balance",
        ]
        
        self.marketing_patterns = [
            r'\b\d+% off\b',  # Percentage discounts
            r'\$\d+\.\d+ only',  # Price patterns
            r'save \$\d+',  # Savings claims
            r'(free|bonus|gift)\s+(shipping|trial|offer)',  # Free offers
            r'click (here|now|below)',  # Action buttons
            r'(subscribe|unsubscribe|opt.?out)',  # Marketing controls
        ]
    
    def analyze_urgency_authenticity(self, email: EmailMessage) -> Dict[str, Any]:
        """Analyze if urgency signals are genuine or marketing manipulation."""
        text = f"{email.subject} {email.body_text}".lower()
        
        # Count genuine vs marketing signals
        genuine_signals = sum(1 for indicator in self.genuine_indicators if indicator in text)
        marketing_signals = sum(1 for indicator in self.marketing_indicators if indicator in text)
        
        # Check for marketing patterns
        marketing_patterns = sum(1 for pattern in self.marketing_patterns 
                               if re.search(pattern, text, re.IGNORECASE))
        
        # Analyze temporal consistency
        temporal_validity = self._validate_temporal_claims(text)
        
        # Check semantic coherence
        semantic_coherence = self._check_semantic_coherence(email)
        
        # Calculate authenticity score
        authenticity_score = (
            (genuine_signals / max(genuine_signals + marketing_signals, 1)) * 0.3 +
            (1 - min(marketing_patterns / 5, 1)) * 0.3 +
            temporal_validity * 0.2 +
            semantic_coherence * 0.2
        )
        
        # Identify manipulation tactics
        detected_tactics = []
        if marketing_signals > genuine_signals:
            detected_tactics.append("emotional_manipulation")
        if marketing_patterns > 2:
            detected_tactics.append("sales_pressure")
        if temporal_validity < 0.5:
            detected_tactics.append("false_scarcity")
        if text.count('!') > 3:
            detected_tactics.append("excessive_punctuation")
        
        return {
            'is_genuine': authenticity_score > 0.6,
            'authenticity_score': authenticity_score,
            'detected_tactics': detected_tactics,
            'genuine_signals': genuine_signals,
            'marketing_signals': marketing_signals
        }
    
    def _validate_temporal_claims(self, text: str) -> float:
        """Validate temporal claims for authenticity."""
        # Look for specific vs vague time references
        specific_patterns = [
            r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
            r'\b\d{1,2}:\d{2}\s*(am|pm)\b',
            r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',
            r'\beod\b', r'\bcob\b'  # End of day, close of business
        ]
        
        vague_patterns = [
            r'\bsoon\b', r'\bquickly\b', r'\bimmediately\b',
            r'\bright away\b', r'\bwhile you can\b'
        ]
        
        specific_count = sum(1 for pattern in specific_patterns if re.search(pattern, text))
        vague_count = sum(1 for pattern in vague_patterns if re.search(pattern, text))
        
        # Specific times are more authentic
        total_temporal = specific_count + vague_count
        if total_temporal == 0:
            return 0.5  # Neutral if no temporal claims
        
        return specific_count / total_temporal
    
    def _check_semantic_coherence(self, email: EmailMessage) -> float:
        """Check if urgency signals are coherent with email content."""
        subject_urgent = any(word in email.subject.lower() 
                           for word in self.genuine_indicators + self.marketing_indicators)
        
        body_urgent = any(word in email.body_text.lower() 
                         for word in self.genuine_indicators + self.marketing_indicators)
        
        # Check if subject and body match in urgency
        if subject_urgent and body_urgent:
            return 1.0
        elif not subject_urgent and not body_urgent:
            return 1.0
        else:
            return 0.3  # Mismatch between subject and body urgency


class EmailPrioritizer:
    """LLM-based email prioritization service with semantic analysis."""
    
    def __init__(self):
        """Initialize the email prioritizer."""
        self.ollama_manager = None
        self.cache = EmailPriorityCache()
        self.urgency_detector = SemanticUrgencyDetector()
        self.sender_profiles: Dict[str, SenderProfile] = {}
        self.marketing_classifier = None
        self.receipt_classifier = None
        
        # Classification prompt following spec recommendations
        self.classification_prompt = """Classify this email's priority based on these criteria:

HIGH Priority:
- Specific deadlines within 48 hours
- Direct requests from executives or key clients
- System alerts or security issues
- Meeting changes or cancellations
- Legal or compliance matters
- Financial transactions requiring immediate attention
- Notifications of order status changes or deliveries
- Invoices with payment due within 48 hours
- Something is "expiring soon" or "urgent action required"
- Anything from the landlord or property management company

MEDIUM Priority:
- Project updates with flexible deadlines
- Scheduled meetings (not urgent changes)
- Regular work requests without immediate deadlines
- Important announcements that don't require immediate action
- Domain name renewals
- Receipts or invoices with payment due in more than 48 hours

LOW Priority:
- Newsletters and marketing emails
- Automated notifications
- General announcements
- Social media notifications
- Promotional content

Email to classify:
From: {sender}
Subject: {subject}
Content: {content}

Analyze the email and respond with exactly one line in this format:
PRIORITY|CONFIDENCE|REASONING

Example: MEDIUM|0.8|Regular project update without immediate deadline

Your response:"""
    
    async def initialize(self) -> None:
        """Initialize the Ollama client, marketing classifier, and receipt classifier."""
        self.ollama_manager = await get_ollama_manager()
        
        # Initialize marketing classifier
        from src.services.marketing_classifier import MarketingEmailClassifier
        self.marketing_classifier = MarketingEmailClassifier()
        await self.marketing_classifier.initialize()
        
        # Initialize receipt classifier
        from src.services.receipt_classifier import ReceiptClassifier
        self.receipt_classifier = ReceiptClassifier()
        await self.receipt_classifier.initialize()
        
        logger.info("Email prioritizer initialized with LLM support, marketing and receipt classification")
    
    async def analyze_priority(self, email: EmailMessage, context: Optional[Dict[str, Any]] = None) -> PriorityScore:
        """Analyze email priority using LLM and semantic analysis."""
        try:
            # Check cache first
            cached_result = self.cache.get_cached_result(email)
            if cached_result:
                logger.debug("Using cached priority result", email_id=email.id)
                return cached_result
            
            # Get sender reputation
            sender_profile = self._get_or_create_sender_profile(email.sender)
            
            # Analyze marketing classification first
            marketing_result = await self.marketing_classifier.classify_email(email)
            
            # Analyze receipt classification
            receipt_result = await self.receipt_classifier.classify_receipt(email)
            
            # Analyze urgency authenticity
            urgency_analysis = self.urgency_detector.analyze_urgency_authenticity(email)
            
            # Get LLM classification
            llm_result = await self._classify_with_llm(email)
            
            # Combine signals for final assessment
            final_priority = self._combine_priority_signals(
                llm_result, urgency_analysis, sender_profile, email, marketing_result, receipt_result
            )
            
            # Update sender profile based on results
            self._update_sender_profile(sender_profile, urgency_analysis, final_priority)
            
            # Cache result
            self.cache.cache_result(email, final_priority)
            
            return final_priority
            
        except Exception as e:
            logger.error("Failed to analyze email priority", email_id=email.id, error=str(e))
            return PriorityScore(
                level="medium",
                confidence=0.0,
                reasoning="Error during priority analysis",
                is_genuine_urgency=False,
                authenticity_score=0.0
            )
    
    async def _classify_with_llm(self, email: EmailMessage) -> Dict[str, Any]:
        """Classify email using LLM with optimized prompting."""
        # Truncate content to fit context window
        content = email.body_text[:2000] if email.body_text else ""
        
        prompt = self.classification_prompt.format(
            sender=email.sender or "Unknown",
            subject=email.subject or "No subject",
            content=content
        )
        
        try:
            # Use chat interface for better results
            response = await self.ollama_manager.chat(
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": 0.1,  # Low temperature for consistent classification
                    "top_p": 0.95,
                    "num_predict": 100
                }
            )
            
            return self._parse_llm_response(response.content)
            
        except Exception as e:
            logger.error("LLM classification failed", error=str(e))
            return {
                "priority": "medium",
                "confidence": 0.0,
                "reasoning": "LLM classification unavailable"
            }
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured data."""
        try:
            # Expected format: PRIORITY|CONFIDENCE|REASONING
            lines = response.strip().split('\n')
            
            # Find the line with the actual data (skip template headers)
            data_line = None
            for line in lines:
                line = line.strip()
                if '|' in line and not line.startswith('PRIORITY|CONFIDENCE|REASONING'):
                    # Check if this looks like actual data
                    parts = line.split('|')
                    if len(parts) >= 3:
                        try:
                            # Try to parse the confidence as float to validate it's real data
                            float(parts[1].strip())
                            data_line = line
                            break
                        except ValueError:
                            continue
            
            if data_line:
                parts = data_line.split('|')
                priority = parts[0].strip().lower()
                confidence = float(parts[1].strip())
                reasoning = parts[2].strip()
                
                return {
                    "priority": priority,
                    "confidence": min(max(confidence, 0.0), 1.0),
                    "reasoning": reasoning
                }
        except Exception as e:
            logger.warning("Failed to parse LLM response", response=response, error=str(e))
        
        # Fallback parsing
        response_lower = response.lower()
        if "high" in response_lower:
            priority = "high"
        elif "low" in response_lower:
            priority = "low"
        else:
            priority = "medium"
        
        return {
            "priority": priority,
            "confidence": 0.5,
            "reasoning": "Parsed from unstructured response"
        }
    
    def _get_or_create_sender_profile(self, sender: str) -> SenderProfile:
        """Get or create sender profile for reputation tracking."""
        if not sender:
            sender = "unknown@unknown.com"
        
        if sender not in self.sender_profiles:
            self.sender_profiles[sender] = SenderProfile(email=sender)
        
        return self.sender_profiles[sender]
    
    def _update_sender_profile(self, profile: SenderProfile, urgency_analysis: Dict, priority: PriorityScore) -> None:
        """Update sender profile based on current email analysis."""
        profile.total_emails += 1
        profile.last_updated = time.time()
        
        # Track urgency claims
        if urgency_analysis.get("genuine_signals", 0) > 0 or urgency_analysis.get("marketing_signals", 0) > 0:
            profile.urgency_claims += 1
            
            # Track false urgency if marketing manipulation detected
            if not urgency_analysis.get("is_genuine", True):
                profile.marketing_patterns += 1
        
        # Update false urgency rate
        if profile.urgency_claims > 0:
            profile.false_urgency_rate = profile.marketing_patterns / profile.urgency_claims
        
        # Calculate reputation score
        base_reputation = 0.5
        
        # Penalize high false urgency rates
        urgency_penalty = profile.false_urgency_rate * 0.4
        
        # Reward consistent genuine communications
        if profile.total_emails > 5:
            consistency_bonus = min((profile.total_emails - profile.marketing_patterns) / profile.total_emails, 0.3)
        else:
            consistency_bonus = 0.0
        
        profile.reputation_score = max(0.0, min(1.0, base_reputation - urgency_penalty + consistency_bonus))
    
    def _combine_priority_signals(self, llm_result: Dict, urgency_analysis: Dict, 
                                 sender_profile: SenderProfile, email: EmailMessage, marketing_result=None, receipt_result=None) -> PriorityScore:
        """Combine LLM classification with semantic analysis and sender reputation."""
        llm_priority = llm_result.get("priority", "medium")
        llm_confidence = llm_result.get("confidence", 0.5)
        llm_reasoning = llm_result.get("reasoning", "No reasoning provided")
        
        # Apply confidence-based handling as per spec
        if llm_confidence > 0.8:
            # High confidence - use LLM result directly
            final_priority = llm_priority
            needs_review = False
        elif llm_confidence > 0.6:
            # Medium confidence - use with review flag
            final_priority = llm_priority
            needs_review = True
        else:
            # Low confidence - apply rule-based fallback
            final_priority = self._apply_rule_based_fallback(email, sender_profile)
            needs_review = True
            llm_reasoning += " (Low confidence, rule-based fallback applied)"
        
        # Adjust based on urgency authenticity
        is_genuine = urgency_analysis.get("is_genuine", True)
        authenticity_score = urgency_analysis.get("authenticity_score", 0.5)
        
        # Marketing classification adjustments
        is_marketing = False
        marketing_subtype = ""
        marketing_confidence = 0.0
        
        if marketing_result:
            is_marketing = marketing_result.is_marketing
            marketing_subtype = marketing_result.subtype
            marketing_confidence = marketing_result.confidence
            
            # Downgrade promotional marketing emails
            if is_marketing and marketing_result.subtype == "promotional" and final_priority == "high":
                final_priority = "medium"
                llm_reasoning += " (Downgraded: promotional marketing email)"
            
            # Low priority for general marketing
            elif is_marketing and marketing_result.subtype in ["newsletter", "general"] and final_priority in ["high", "medium"]:
                final_priority = "low"
                llm_reasoning += f" (Marketing email: {marketing_result.subtype})"
        
        # Receipt classification adjustments
        is_receipt = False
        receipt_type = ""
        receipt_vendor = ""
        receipt_amount = ""
        
        if receipt_result:
            is_receipt = receipt_result.is_receipt
            receipt_type = receipt_result.receipt_type
            receipt_vendor = receipt_result.vendor or ""
            receipt_amount = receipt_result.amount or ""
            
            # Receipts are generally low priority unless they need action
            if is_receipt and final_priority == "high":
                if receipt_result.receipt_type != "refund":
                    final_priority = "medium"
                    llm_reasoning += f" (Receipt: {receipt_result.receipt_type})"
        
        # Downgrade marketing emails with false urgency
        if not is_genuine and final_priority == "high":
            final_priority = "medium"
            llm_reasoning += " (Downgraded due to marketing manipulation)"
        
        # Apply sender reputation adjustment
        if sender_profile.reputation_score < 0.3 and final_priority == "high":
            final_priority = "medium"
            llm_reasoning += " (Adjusted for sender reputation)"
        
        # Map to our priority levels
        priority_mapping = {
            "high": "critical",
            "medium": "medium", 
            "low": "low"
        }
        
        return PriorityScore(
            level=priority_mapping.get(final_priority, "medium"),
            confidence=llm_confidence,
            reasoning=llm_reasoning,
            is_genuine_urgency=is_genuine,
            authenticity_score=authenticity_score,
            detected_tactics=urgency_analysis.get("detected_tactics", []),
            sender_reputation=sender_profile.reputation_score,
            needs_review=needs_review,
            is_marketing=is_marketing,
            marketing_subtype=marketing_subtype,
            marketing_confidence=marketing_confidence,
            is_receipt=is_receipt,
            receipt_type=receipt_type,
            receipt_vendor=receipt_vendor,
            receipt_amount=receipt_amount
        )
    
    def _apply_rule_based_fallback(self, email: EmailMessage, sender_profile: SenderProfile) -> str:
        """Apply rule-based classification when LLM confidence is low."""
        text = f"{email.subject} {email.body_text}".lower()
        
        # High priority indicators
        high_keywords = [
            "urgent", "asap", "emergency", "critical", "deadline today",
            "meeting in", "response required", "action needed"
        ]
        
        # Low priority indicators
        low_keywords = [
            "newsletter", "unsubscribe", "marketing", "promotion",
            "sale", "offer", "deal", "free"
        ]
        
        high_matches = sum(1 for keyword in high_keywords if keyword in text)
        low_matches = sum(1 for keyword in low_keywords if keyword in text)
        
        # Apply sender reputation
        if sender_profile.reputation_score > 0.7 and high_matches > 0:
            return "high"
        elif sender_profile.reputation_score < 0.3 or low_matches > 0:
            return "low"
        else:
            return "medium"
    
    def get_sender_statistics(self) -> Dict[str, Any]:
        """Get sender reputation statistics."""
        if not self.sender_profiles:
            return {"total_senders": 0}
        
        total_senders = len(self.sender_profiles)
        trusted_senders = sum(1 for p in self.sender_profiles.values() if p.reputation_score > 0.7)
        suspicious_senders = sum(1 for p in self.sender_profiles.values() if p.reputation_score < 0.3)
        
        avg_reputation = sum(p.reputation_score for p in self.sender_profiles.values()) / total_senders
        
        return {
            "total_senders": total_senders,
            "trusted_senders": trusted_senders,
            "suspicious_senders": suspicious_senders,
            "average_reputation": round(avg_reputation, 3)
        }