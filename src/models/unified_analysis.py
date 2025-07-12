"""Unified email analysis for shared content processing across classifiers."""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Any, Optional
import re
from datetime import datetime

from src.models.email import EmailMessage


@dataclass
class UnifiedEmailAnalysis:
    """Unified analysis of email content shared across all classifiers."""
    
    # Original email
    email: EmailMessage
    
    # Content extraction
    combined_text: str = ""
    subject_lower: str = ""
    sender_lower: str = ""
    body_text_lower: str = ""
    
    # Content features
    word_count: int = 0
    sentence_count: int = 0
    char_count: int = 0
    
    # Pattern detection
    urls: List[str] = field(default_factory=list)
    email_addresses: List[str] = field(default_factory=list)
    phone_numbers: List[str] = field(default_factory=list)
    monetary_amounts: List[str] = field(default_factory=list)
    
    # Keywords and indicators
    keywords: Set[str] = field(default_factory=set)
    marketing_indicators: Set[str] = field(default_factory=set)
    receipt_indicators: Set[str] = field(default_factory=set)
    priority_indicators: Set[str] = field(default_factory=set)
    notification_indicators: Set[str] = field(default_factory=set)
    
    # Structural analysis
    has_unsubscribe_link: bool = False
    has_attachments: bool = False
    is_reply: bool = False
    is_forward: bool = False
    
    # Sender analysis
    sender_domain: str = ""
    is_automated_sender: bool = False
    sender_reputation_signals: Dict[str, Any] = field(default_factory=dict)
    
    # Content classification hints
    appears_marketing: bool = False
    appears_receipt: bool = False
    appears_notification: bool = False
    appears_personal: bool = False
    
    # Performance tracking
    analysis_time: float = 0.0
    cached: bool = False
    
    @classmethod
    def from_email(cls, email: EmailMessage) -> 'UnifiedEmailAnalysis':
        """Create unified analysis from an email message.
        
        Args:
            email: Email message to analyze
            
        Returns:
            UnifiedEmailAnalysis instance
        """
        start_time = datetime.now()
        
        analysis = cls(email=email)
        
        # Basic content extraction
        analysis.subject_lower = (email.subject or "").lower()
        analysis.sender_lower = (email.sender or "").lower()
        analysis.body_text_lower = (email.body_text or "")[:2000].lower()  # Limit for performance
        
        # Combined text for analysis
        analysis.combined_text = f"{analysis.subject_lower} {analysis.body_text_lower}"
        
        # Content metrics
        analysis.word_count = len(analysis.combined_text.split())
        analysis.sentence_count = len(re.findall(r'[.!?]+', analysis.combined_text))
        analysis.char_count = len(analysis.combined_text)
        
        # Extract patterns
        analysis._extract_patterns()
        
        # Analyze sender
        analysis._analyze_sender()
        
        # Detect content indicators
        analysis._detect_content_indicators()
        
        # Structural analysis
        analysis._analyze_structure()
        
        # Classification hints
        analysis._generate_classification_hints()
        
        # Track performance
        end_time = datetime.now()
        analysis.analysis_time = (end_time - start_time).total_seconds()
        
        return analysis
    
    def _extract_patterns(self) -> None:
        """Extract common patterns from email content."""
        text = self.combined_text
        
        # URLs
        url_pattern = r'https?://[^\s<>"{}|\\^`[\]]+|www\.[^\s<>"{}|\\^`[\]]+'
        self.urls = re.findall(url_pattern, text)
        
        # Email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        self.email_addresses = re.findall(email_pattern, text)
        
        # Phone numbers (basic patterns)
        phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b|\(\d{3}\)\s?\d{3}[-.]?\d{4}'
        self.phone_numbers = re.findall(phone_pattern, text)
        
        # Monetary amounts
        money_pattern = r'\$\d+(?:,\d{3})*(?:\.\d{2})?|USD?\s?\d+|\d+\s?(?:dollars?|USD|cents?)'
        self.monetary_amounts = re.findall(money_pattern, text, re.IGNORECASE)
        
        # Extract keywords (important words, excluding common stop words)
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        words = re.findall(r'\b\w{3,}\b', text.lower())
        self.keywords = {word for word in words if word not in stop_words}
    
    def _analyze_sender(self) -> None:
        """Analyze sender characteristics."""
        sender = self.sender_lower
        
        # Extract domain
        if '@' in sender:
            self.sender_domain = sender.split('@')[-1].strip('<>')
        
        # Automated sender detection
        automated_patterns = [
            'noreply', 'no-reply', 'donotreply', 'automated', 'system', 
            'notifications', 'alerts', 'updates', 'mailer', 'daemon'
        ]
        self.is_automated_sender = any(pattern in sender for pattern in automated_patterns)
        
        # Reputation signals
        self.sender_reputation_signals = {
            'is_automated': self.is_automated_sender,
            'domain': self.sender_domain,
            'has_display_name': '<' in self.email.sender if self.email.sender else False,
            'suspicious_chars': bool(re.search(r'[0-9]{3,}', sender))
        }
    
    def _detect_content_indicators(self) -> None:
        """Detect indicators for different email types."""
        text = self.combined_text
        
        # Marketing indicators
        marketing_terms = {
            'unsubscribe', 'newsletter', 'promotion', 'sale', 'discount', 'offer', 
            'deal', 'exclusive', 'limited time', 'buy now', 'click here', 'sign up',
            'subscribe', 'marketing', 'campaign', 'free shipping', 'coupon'
        }
        self.marketing_indicators = {term for term in marketing_terms if term in text}
        
        # Receipt indicators
        receipt_terms = {
            'receipt', 'invoice', 'payment', 'order', 'purchase', 'transaction',
            'confirmation', 'billing', 'charged', 'refund', 'shipped', 'delivery',
            'total', 'amount', 'paid', 'subscription', 'renewal', 'thank you for'
        }
        self.receipt_indicators = {term for term in receipt_terms if term in text}
        
        # Priority indicators
        priority_terms = {
            'urgent', 'asap', 'immediate', 'critical', 'important', 'deadline',
            'emergency', 'action required', 'time sensitive', 'priority',
            'meeting', 'schedule', 'cancel', 'reschedule', 'approval'
        }
        self.priority_indicators = {term for term in priority_terms if term in text}
        
        # Notification indicators
        notification_terms = {
            'alert', 'notification', 'reminder', 'update', 'status', 'report',
            'backup', 'security', 'maintenance', 'system', 'automated',
            'service', 'monitor', 'error', 'warning', 'success'
        }
        self.notification_indicators = {term for term in notification_terms if term in text}
    
    def _analyze_structure(self) -> None:
        """Analyze email structure."""
        text = self.combined_text
        
        # Unsubscribe detection
        unsubscribe_patterns = ['unsubscribe', 'opt out', 'remove me', 'stop emails']
        self.has_unsubscribe_link = any(pattern in text for pattern in unsubscribe_patterns)
        
        # Attachments
        self.has_attachments = bool(getattr(self.email, 'attachments', None))
        
        # Reply/Forward detection
        subject = self.subject_lower
        self.is_reply = subject.startswith('re:') or 'replied to' in text
        self.is_forward = subject.startswith('fwd:') or subject.startswith('fw:') or 'forwarded' in text
    
    def _generate_classification_hints(self) -> None:
        """Generate hints for classification based on analysis."""
        # Marketing hints
        marketing_score = len(self.marketing_indicators)
        if self.has_unsubscribe_link:
            marketing_score += 2
        self.appears_marketing = marketing_score >= 2
        
        # Receipt hints
        receipt_score = len(self.receipt_indicators)
        if self.monetary_amounts:
            receipt_score += 2
        self.appears_receipt = receipt_score >= 2
        
        # Notification hints
        notification_score = len(self.notification_indicators)
        if self.is_automated_sender:
            notification_score += 1
        self.appears_notification = notification_score >= 2
        
        # Personal hints (inverse of automated patterns)
        personal_score = 0
        if not self.is_automated_sender:
            personal_score += 1
        if self.is_reply or self.is_forward:
            personal_score += 1
        if not self.appears_marketing and not self.appears_receipt:
            personal_score += 1
        self.appears_personal = personal_score >= 2
    
    def get_classifier_context(self, classifier_type: str) -> Dict[str, Any]:
        """Get optimized context for a specific classifier type.
        
        Args:
            classifier_type: Type of classifier ('marketing', 'receipt', 'priority', 'notifications')
            
        Returns:
            Relevant context for the classifier
        """
        base_context = {
            'word_count': self.word_count,
            'char_count': self.char_count,
            'has_urls': len(self.urls) > 0,
            'sender_domain': self.sender_domain,
            'is_automated': self.is_automated_sender
        }
        
        if classifier_type == 'marketing':
            return {
                **base_context,
                'marketing_indicators': list(self.marketing_indicators),
                'has_unsubscribe': self.has_unsubscribe_link,
                'appears_marketing': self.appears_marketing,
                'url_count': len(self.urls)
            }
        
        elif classifier_type == 'receipt':
            return {
                **base_context,
                'receipt_indicators': list(self.receipt_indicators),
                'monetary_amounts': self.monetary_amounts,
                'appears_receipt': self.appears_receipt
            }
        
        elif classifier_type == 'priority':
            return {
                **base_context,
                'priority_indicators': list(self.priority_indicators),
                'is_reply': self.is_reply,
                'sender_signals': self.sender_reputation_signals
            }
        
        elif classifier_type == 'notifications':
            return {
                **base_context,
                'notification_indicators': list(self.notification_indicators),
                'appears_notification': self.appears_notification,
                'appears_personal': self.appears_personal
            }
        
        return base_context
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert analysis to dictionary for caching/serialization."""
        return {
            'email_id': self.email.id,
            'combined_text': self.combined_text[:500],  # Truncated for storage
            'word_count': self.word_count,
            'marketing_indicators': list(self.marketing_indicators),
            'receipt_indicators': list(self.receipt_indicators),
            'priority_indicators': list(self.priority_indicators),
            'notification_indicators': list(self.notification_indicators),
            'appears_marketing': self.appears_marketing,
            'appears_receipt': self.appears_receipt,
            'appears_notification': self.appears_notification,
            'appears_personal': self.appears_personal,
            'analysis_time': self.analysis_time,
            'cached': self.cached
        }