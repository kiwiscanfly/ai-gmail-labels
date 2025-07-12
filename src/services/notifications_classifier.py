"""Notification email classification service using LLM-based analysis and pattern recognition."""

import asyncio
import re
import time
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import structlog

from src.models.email import EmailMessage
from src.integrations.ollama_client import get_ollama_manager
from src.core.exceptions import ServiceError

logger = structlog.get_logger(__name__)


@dataclass
class NotificationClassificationResult:
    """Notification email classification result."""
    is_notification: bool
    confidence: float  # 0.0 to 1.0
    notification_type: str  # "system", "update", "alert", "reminder", "security"
    reasoning: str
    sender_type: Optional[str] = None  # "platform", "service", "automated", "security"
    urgency_level: Optional[str] = None  # "low", "medium", "high", "critical"
    action_required: bool = False
    notification_indicators: List[str] = field(default_factory=list)


@dataclass
class NotificationSenderProfile:
    """Sender profile for notification pattern tracking."""
    sender_email: str
    domain: str
    total_emails: int = 0
    notification_emails: int = 0
    notification_rate: float = 0.0
    common_notification_types: List[str] = field(default_factory=list)
    is_automated: bool = False
    last_updated: float = field(default_factory=time.time)


class NotificationPatternAnalyzer:
    """Analyzes email patterns for notification indicators."""
    
    def __init__(self):
        # Notification keywords organized by type
        self.notification_keywords = {
            'system': [
                'system notification', 'service notification', 'maintenance',
                'scheduled maintenance', 'system update', 'service status',
                'platform update', 'system alert', 'service alert'
            ],
            'update': [
                'app update', 'software update', 'new version', 'release notes',
                'update available', 'new feature', 'feature update', 'changelog',
                'version release', 'product update'
            ],
            'alert': [
                'alert', 'warning', 'important notice', 'urgent', 'action required',
                'attention', 'notice', 'notification', 'information'
            ],
            'reminder': [
                'reminder', 'upcoming', 'due', 'expires', 'deadline', 'scheduled',
                'appointment', 'meeting', 'task', 'calendar', 'event'
            ],
            'security': [
                'security alert', 'password', 'login', 'account security',
                'suspicious activity', 'verify', 'confirm', 'authentication',
                'security notification', 'unauthorized', 'protect'
            ]
        }
        
        # Automated sender patterns
        self.automated_sender_patterns = [
            r'no.?reply',
            r'donotreply',
            r'notifications?@',
            r'alerts?@',
            r'system@',
            r'security@',
            r'support@',
            r'service@',
            r'automated@',
            r'bot@',
            r'updates?@'
        ]
        
        # Common notification domains
        self.notification_domains = [
            'notification', 'alerts', 'updates', 'system', 'service',
            'security', 'support', 'automated', 'bot', 'platform'
        ]
        
        # Subject line patterns for each type
        self.subject_patterns = {
            'system': [
                r'system (maintenance|update|alert|notification)',
                r'service (maintenance|update|alert|notification)',
                r'platform (maintenance|update|alert)',
                r'scheduled maintenance',
                r'system status',
                r'service status'
            ],
            'update': [
                r'(app|software|product) update',
                r'new version',
                r'update available',
                r'feature update',
                r'release notes',
                r'changelog',
                r'new feature'
            ],
            'alert': [
                r'^(alert|warning|notice|important)',
                r'action required',
                r'urgent',
                r'attention.*required',
                r'important (notice|information)'
            ],
            'reminder': [
                r'^reminder',
                r'(upcoming|due|expires)',
                r'deadline',
                r'(appointment|meeting|event) reminder',
                r'don\'t forget',
                r'scheduled for'
            ],
            'security': [
                r'security (alert|notification|warning)',
                r'password (reset|change|expir)',
                r'login (attempt|alert)',
                r'account security',
                r'suspicious activity',
                r'verify (your )?account',
                r'authentication',
                r'unauthorized'
            ]
        }
        
        # Content indicators for notifications
        self.content_indicators = {
            'automated_language': [
                'this is an automated', 'automatically generated',
                'do not reply', 'system generated', 'automated notification'
            ],
            'action_required': [
                'action required', 'please', 'click here', 'verify now',
                'confirm', 'update', 'review', 'check'
            ],
            'urgency': [
                'urgent', 'immediate', 'asap', 'critical', 'important',
                'expires soon', 'deadline', 'time sensitive'
            ]
        }
    
    def extract_features(self, email: EmailMessage) -> Dict[str, Any]:
        """Extract notification-related features from email."""
        subject_text = (email.subject or "").lower()
        body_text = (email.body_text or "").lower()
        full_text = f"{subject_text} {body_text}"
        
        features = {
            'notification_keywords': self._count_notification_keywords(full_text),
            'automated_sender': self._check_automated_sender(email.sender or ""),
            'notification_domain': self._check_notification_domain(email.sender or ""),
            'subject_patterns': self._analyze_subject_patterns(subject_text),
            'automated_language': self._detect_automated_language(full_text),
            'action_required': self._detect_action_required(full_text),
            'urgency_indicators': self._detect_urgency(full_text),
            'sender_info': self._extract_sender_info(email)
        }
        
        # Calculate overall notification score
        features['notification_score'] = self._calculate_notification_score(features)
        features['notification_type'] = self._determine_notification_type(subject_text, body_text, features)
        features['urgency_level'] = self._determine_urgency_level(features)
        
        return features
    
    def _count_notification_keywords(self, text: str) -> Dict[str, int]:
        """Count notification keywords by type."""
        counts = {}
        for notification_type, keywords in self.notification_keywords.items():
            count = sum(1 for keyword in keywords if keyword in text)
            counts[notification_type] = count
        return counts
    
    def _check_automated_sender(self, sender: str) -> bool:
        """Check if sender appears to be automated."""
        sender_lower = sender.lower()
        return any(re.search(pattern, sender_lower) for pattern in self.automated_sender_patterns)
    
    def _check_notification_domain(self, sender: str) -> bool:
        """Check if sender domain indicates notifications."""
        sender_lower = sender.lower()
        return any(domain in sender_lower for domain in self.notification_domains)
    
    def _analyze_subject_patterns(self, subject: str) -> Dict[str, bool]:
        """Analyze subject line for notification patterns."""
        patterns_found = {}
        for notification_type, patterns in self.subject_patterns.items():
            found = any(re.search(pattern, subject, re.IGNORECASE) for pattern in patterns)
            patterns_found[notification_type] = found
        return patterns_found
    
    def _detect_automated_language(self, text: str) -> bool:
        """Detect automated language patterns."""
        return any(indicator in text for indicator in self.content_indicators['automated_language'])
    
    def _detect_action_required(self, text: str) -> bool:
        """Detect if action is required from the user."""
        return any(indicator in text for indicator in self.content_indicators['action_required'])
    
    def _detect_urgency(self, text: str) -> int:
        """Count urgency indicators."""
        return sum(1 for indicator in self.content_indicators['urgency'] if indicator in text)
    
    def _extract_sender_info(self, email: EmailMessage) -> Dict[str, str]:
        """Extract sender information."""
        sender_info = {}
        
        if email.sender:
            # Clean sender name
            sender_parts = email.sender.split('<')
            if sender_parts:
                sender_info['sender_name'] = sender_parts[0].strip()
            
            # Extract domain
            domain_match = re.search(r'@([\\w.-]+)', email.sender)
            if domain_match:
                sender_info['domain'] = domain_match.group(1)
        
        return sender_info
    
    def _calculate_notification_score(self, features: Dict[str, Any]) -> float:
        """Calculate overall notification score."""
        score = 0.0
        
        # Automated sender (strong indicator)
        if features['automated_sender']:
            score += 0.4
        
        # Notification domain
        if features['notification_domain']:
            score += 0.2
        
        # Subject patterns (strong indicator)
        if any(features['subject_patterns'].values()):
            score += 0.3
        
        # Automated language
        if features['automated_language']:
            score += 0.2
        
        # Keyword counts
        total_keywords = sum(features['notification_keywords'].values())
        keyword_score = min(total_keywords / 5, 1.0) * 0.3
        score += keyword_score
        
        return min(score, 1.0)
    
    def _determine_notification_type(self, subject: str, body: str, features: Dict[str, Any]) -> str:
        """Determine the primary notification type."""
        # Check subject patterns first (most reliable)
        for notification_type, found in features['subject_patterns'].items():
            if found:
                return notification_type
        
        # Check keyword counts
        keyword_counts = features['notification_keywords']
        if keyword_counts:
            best_type = max(keyword_counts.items(), key=lambda x: x[1])
            if best_type[1] > 0:
                return best_type[0]
        
        # Default classification based on content
        full_text = f"{subject} {body}".lower()
        
        if any(word in full_text for word in ['security', 'password', 'login', 'account']):
            return 'security'
        elif any(word in full_text for word in ['update', 'version', 'release', 'feature']):
            return 'update'
        elif any(word in full_text for word in ['reminder', 'due', 'deadline', 'upcoming']):
            return 'reminder'
        elif any(word in full_text for word in ['system', 'maintenance', 'service']):
            return 'system'
        else:
            return 'alert'
    
    def _determine_urgency_level(self, features: Dict[str, Any]) -> str:
        """Determine urgency level based on features."""
        urgency_count = features['urgency_indicators']
        action_required = features['action_required']
        
        if urgency_count >= 3 or ('security' in features['subject_patterns'] and features['subject_patterns']['security']):
            return 'critical'
        elif urgency_count >= 2 or action_required:
            return 'high'
        elif urgency_count >= 1:
            return 'medium'
        else:
            return 'low'


class NotificationClassificationCache:
    """Caching system for notification classification results."""
    
    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size
        
    def _generate_cache_key(self, email: EmailMessage) -> str:
        """Generate cache key from email content."""
        content = f"{email.subject}{email.body_text[:500]}{email.sender}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get_cached_result(self, email: EmailMessage) -> Optional[NotificationClassificationResult]:
        """Get cached classification result."""
        cache_key = self._generate_cache_key(email)
        cached = self.cache.get(cache_key)
        
        if cached and time.time() - cached['timestamp'] < 3600:  # 1 hour cache
            return cached['result']
        return None
    
    def cache_result(self, email: EmailMessage, result: NotificationClassificationResult) -> None:
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


class NotificationsClassifier:
    """LLM-based notification classification service."""
    
    def __init__(self):
        """Initialize the notifications classifier."""
        self.ollama_manager = None
        self.cache = NotificationClassificationCache()
        self.pattern_analyzer = NotificationPatternAnalyzer()
        self.sender_profiles: Dict[str, NotificationSenderProfile] = {}
        
        # Classification prompt
        self.classification_prompt = """Analyze this email to determine if it's a notification. Follow these steps:

1. SENDER ANALYSIS
   - Is the sender automated (noreply, system, notifications, alerts)?
   - Does the sender appear to be a platform, service, or system?
   - Look for automated sender patterns

2. SUBJECT LINE ANALYSIS
   - Does the subject indicate a notification type (alert, reminder, update)?
   - Look for system/service/platform notifications
   - Check for security, maintenance, or update notifications

3. CONTENT ANALYSIS
   - Is this an automated notification from a service?
   - Does it contain system-generated language?
   - Is it informing about status, updates, or requiring action?

4. NOTIFICATION TYPE ANALYSIS
   If it's a notification, what type:
   - SYSTEM: Platform maintenance, service status, system alerts
   - UPDATE: Software updates, new features, version releases
   - ALERT: Warnings, important notices, general notifications
   - REMINDER: Deadlines, appointments, upcoming events
   - SECURITY: Account security, login alerts, password notifications

IMPORTANT:
- Automated emails from platforms/services are typically notifications
- System-generated content with "do not reply" is usually a notification
- Security alerts, password resets, and account notifications are notifications
- App/software update notifications and maintenance alerts are notifications
- Don't confuse marketing emails with notifications - notifications are informational/functional

Email to analyze:
From: {sender}
Subject: {subject}
Content: {content}

Automated sender detected: {automated_sender}
Notification keywords found: {keyword_count}
Subject patterns: {subject_patterns}

Respond with exactly one line in this format:
CLASSIFICATION|CONFIDENCE|TYPE|URGENCY|SENDER_TYPE|ACTION_REQUIRED|REASONING

Examples:
NOTIFICATION|0.95|SECURITY|HIGH|SECURITY|true|Password reset request from automated security system
NOTIFICATION|0.90|UPDATE|MEDIUM|PLATFORM|false|App update notification from automated system
NOTIFICATION|0.85|REMINDER|MEDIUM|SERVICE|true|Appointment reminder from scheduling service
NOT_NOTIFICATION|0.85|NA|NA|NA|false|Marketing email with promotional content

Your response:"""
    
    async def initialize(self) -> None:
        """Initialize the Ollama client."""
        self.ollama_manager = await get_ollama_manager()
        logger.info("Notifications classifier initialized with LLM support")
    
    async def classify_notification(self, email: EmailMessage) -> NotificationClassificationResult:
        """Classify email as notification or not with detailed analysis."""
        try:
            # Check cache first
            cached_result = self.cache.get_cached_result(email)
            if cached_result:
                logger.debug("Using cached notification classification", email_id=email.id)
                return cached_result
            
            # Extract notification features
            features = self.pattern_analyzer.extract_features(email)
            
            # Get sender profile
            sender_profile = self._get_or_create_sender_profile(email)
            
            # Quick classification for obvious cases
            if features['notification_score'] > 0.8:
                result = await self._create_notification_result(email, features, confidence=0.90)
            elif features['notification_score'] < 0.3:
                result = NotificationClassificationResult(
                    is_notification=False,
                    confidence=0.85,
                    notification_type="none",
                    reasoning="Low notification indicators in pattern analysis"
                )
            else:
                # Use LLM for ambiguous cases
                result = await self._llm_classify(email, features, sender_profile)
            
            # Update sender profile
            self._update_sender_profile(sender_profile, result)
            
            # Cache result
            self.cache.cache_result(email, result)
            
            return result
            
        except Exception as e:
            logger.error("Failed to classify notification", email_id=email.id, error=str(e))
            return NotificationClassificationResult(
                is_notification=False,
                confidence=0.0,
                notification_type="error",
                reasoning=f"Error during classification: {str(e)}"
            )
    
    async def _llm_classify(self, email: EmailMessage, features: Dict[str, Any], 
                           sender_profile: NotificationSenderProfile) -> NotificationClassificationResult:
        """Use LLM to classify ambiguous notifications."""
        # Prepare content for analysis
        content = email.body_text[:2000] if email.body_text else ""
        
        prompt = self.classification_prompt.format(
            sender=email.sender or "Unknown",
            subject=email.subject or "No subject",
            content=content,
            automated_sender=features['automated_sender'],
            keyword_count=sum(features['notification_keywords'].values()),
            subject_patterns=list(k for k, v in features['subject_patterns'].items() if v)
        )
        
        try:
            response = await self.ollama_manager.chat(
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": 0.1,
                    "top_p": 0.95,
                    "num_predict": 200
                }
            )
            
            return self._parse_llm_response(response.content, features, email)
            
        except Exception as e:
            logger.error("LLM notification classification failed", error=str(e))
            # Fallback to pattern-based classification
            return await self._pattern_based_classification(email, features)
    
    def _parse_llm_response(self, response: str, features: Dict[str, Any], 
                           email: EmailMessage) -> NotificationClassificationResult:
        """Parse LLM response into structured result."""
        try:
            # Find the classification line
            lines = response.strip().split('\n')
            data_line = None
            
            for line in lines:
                if '|' in line and not line.startswith('CLASSIFICATION|'):
                    parts = line.split('|')
                    if len(parts) >= 7:
                        data_line = line
                        break
            
            if data_line:
                parts = [p.strip() for p in data_line.split('|')]
                classification = parts[0].upper()
                confidence = float(parts[1])
                notification_type = parts[2].lower() if parts[2] != 'NA' else 'none'
                urgency = parts[3].lower() if parts[3] != 'NA' else 'low'
                sender_type = parts[4].lower() if parts[4] != 'NA' else None
                action_required = parts[5].lower() == 'true' if parts[5] != 'NA' else False
                reasoning = parts[6] if len(parts) > 6 else "No reasoning provided"
                
                is_notification = classification == 'NOTIFICATION'
                
                return NotificationClassificationResult(
                    is_notification=is_notification,
                    confidence=min(max(confidence, 0.0), 1.0),
                    notification_type=notification_type if is_notification else "none",
                    reasoning=reasoning,
                    sender_type=sender_type,
                    urgency_level=urgency if is_notification else None,
                    action_required=action_required,
                    notification_indicators=self._extract_notification_indicators(features)
                )
                
        except Exception as e:
            logger.warning("Failed to parse notification LLM response", response=response, error=str(e))
        
        # Fallback if parsing fails
        return self._create_fallback_result(features, email)
    
    async def _pattern_based_classification(self, email: EmailMessage, 
                                          features: Dict[str, Any]) -> NotificationClassificationResult:
        """Classify based on pattern analysis."""
        notification_score = features['notification_score']
        
        is_notification = notification_score > 0.6
        confidence = min(notification_score + 0.2, 1.0) if is_notification else max(0.8 - notification_score, 0.5)
        
        if is_notification:
            return await self._create_notification_result(email, features, confidence)
        else:
            return NotificationClassificationResult(
                is_notification=False,
                confidence=confidence,
                notification_type="none",
                reasoning="Pattern-based analysis: insufficient notification indicators"
            )
    
    async def _create_notification_result(self, email: EmailMessage, features: Dict[str, Any], 
                                        confidence: float) -> NotificationClassificationResult:
        """Create notification result from features."""
        notification_type = features.get('notification_type', 'alert')
        urgency_level = features.get('urgency_level', 'low')
        sender_type = self._determine_sender_type(email, features)
        action_required = features.get('action_required', False)
        
        return NotificationClassificationResult(
            is_notification=True,
            confidence=confidence,
            notification_type=notification_type,
            reasoning="Strong notification indicators found in pattern analysis",
            sender_type=sender_type,
            urgency_level=urgency_level,
            action_required=action_required,
            notification_indicators=self._extract_notification_indicators(features)
        )
    
    def _create_fallback_result(self, features: Dict[str, Any], 
                               email: EmailMessage) -> NotificationClassificationResult:
        """Create fallback result when parsing fails."""
        is_notification = features['notification_score'] > 0.5
        
        return NotificationClassificationResult(
            is_notification=is_notification,
            confidence=0.5,
            notification_type=features.get('notification_type', 'alert') if is_notification else "none",
            reasoning="Fallback classification based on pattern analysis",
            notification_indicators=self._extract_notification_indicators(features)
        )
    
    def _determine_sender_type(self, email: EmailMessage, features: Dict[str, Any]) -> str:
        """Determine the type of sender."""
        if features['automated_sender']:
            return "automated"
        elif features['notification_domain']:
            return "platform"
        elif 'security' in features['subject_patterns'] and features['subject_patterns']['security']:
            return "security"
        else:
            return "service"
    
    def _extract_notification_indicators(self, features: Dict[str, Any]) -> List[str]:
        """Extract list of notification indicators found."""
        indicators = []
        
        if features['automated_sender']:
            indicators.append("automated_sender")
        
        if features['notification_domain']:
            indicators.append("notification_domain")
        
        if features['automated_language']:
            indicators.append("automated_language")
        
        if features['action_required']:
            indicators.append("action_required")
        
        for notification_type, found in features['subject_patterns'].items():
            if found:
                indicators.append(f"subject_pattern_{notification_type}")
        
        total_keywords = sum(features['notification_keywords'].values())
        if total_keywords > 0:
            indicators.append(f"notification_keywords_{total_keywords}")
        
        return indicators
    
    def _get_or_create_sender_profile(self, email: EmailMessage) -> NotificationSenderProfile:
        """Get or create sender profile."""
        sender_email = email.sender or "unknown"
        
        if sender_email not in self.sender_profiles:
            # Extract domain
            domain = "unknown"
            if email.sender:
                domain_match = re.search(r'@([\\w.-]+)', email.sender)
                if domain_match:
                    domain = domain_match.group(1)
            
            self.sender_profiles[sender_email] = NotificationSenderProfile(
                sender_email=sender_email,
                domain=domain
            )
        
        return self.sender_profiles[sender_email]
    
    def _update_sender_profile(self, profile: NotificationSenderProfile, 
                              result: NotificationClassificationResult) -> None:
        """Update sender profile based on classification result."""
        profile.total_emails += 1
        profile.last_updated = time.time()
        
        if result.is_notification:
            profile.notification_emails += 1
            
            # Track notification types
            if result.notification_type not in profile.common_notification_types:
                profile.common_notification_types.append(result.notification_type)
            
            # Update automated status
            if result.sender_type == "automated":
                profile.is_automated = True
        
        # Update notification rate
        profile.notification_rate = profile.notification_emails / profile.total_emails
    
    def get_sender_statistics(self) -> Dict[str, Any]:
        """Get sender notification statistics."""
        if not self.sender_profiles:
            return {"total_senders": 0}
        
        total_senders = len(self.sender_profiles)
        notification_senders = sum(1 for p in self.sender_profiles.values() if p.notification_rate > 0.7)
        automated_senders = sum(1 for p in self.sender_profiles.values() if p.is_automated)
        
        avg_notification_rate = sum(p.notification_rate for p in self.sender_profiles.values()) / total_senders
        
        # Get top notification senders
        top_senders = sorted(
            [(p.sender_email, p.notification_rate, p.total_emails) for p in self.sender_profiles.values()],
            key=lambda x: x[1] * x[2],  # Sort by notification rate * volume
            reverse=True
        )[:5]
        
        return {
            "total_senders": total_senders,
            "notification_senders": notification_senders,
            "automated_senders": automated_senders,
            "average_notification_rate": round(avg_notification_rate, 3),
            "top_notification_senders": top_senders
        }