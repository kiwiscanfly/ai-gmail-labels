"""Receipt email classification service using LLM-based analysis and pattern recognition."""

import asyncio
import re
import time
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from decimal import Decimal
import structlog

from src.models.email import EmailMessage
from src.integrations.ollama_client import get_ollama_manager
from src.core.exceptions import ServiceError

logger = structlog.get_logger(__name__)


@dataclass
class ReceiptClassificationResult:
    """Receipt email classification result."""
    is_receipt: bool
    confidence: float  # 0.0 to 1.0
    receipt_type: str  # "purchase", "service", "other"
    reasoning: str
    amount: Optional[str] = None
    currency: Optional[str] = None
    vendor: Optional[str] = None
    transaction_date: Optional[str] = None
    order_number: Optional[str] = None
    payment_method: Optional[str] = None
    items_detected: List[str] = field(default_factory=list)
    receipt_indicators: List[str] = field(default_factory=list)


@dataclass
class VendorProfile:
    """Vendor profile for receipt pattern tracking."""
    vendor_name: str
    domain: str
    total_emails: int = 0
    receipt_emails: int = 0
    receipt_rate: float = 0.0
    common_receipt_types: List[str] = field(default_factory=list)
    typical_subject_patterns: List[str] = field(default_factory=list)
    last_updated: float = field(default_factory=time.time)


class ReceiptPatternAnalyzer:
    """Analyzes email patterns for receipt indicators."""
    
    def __init__(self):
        # Receipt keywords and phrases - balanced specificity
        self.receipt_keywords = [
            "receipt", "invoice", "order confirmation", "payment confirmation", 
            "purchase receipt", "transaction receipt", "billing statement",
            "payment received", "thank you for your order", "order summary",
            "your order", "order number", "transaction id", "payment processed",
            "receipt for", "payment to", "transaction", "purchase", "payment"
        ]
        
        # Currency patterns - more specific to receipts
        self.currency_patterns = [
            r'\$[\d,]+\.[\d]{2}\b',  # USD with cents
            r'€[\d,]+\.[\d]{2}\b',   # EUR with cents
            r'£[\d,]+\.[\d]{2}\b',   # GBP with cents
            r'(total|subtotal|amount due|grand total):\s*\$[\d,]+\.[\d]{2}',  # Specific total patterns
            r'order total:\s*\$[\d,]+\.[\d]{2}',  # Order total
            r'payment amount:\s*\$[\d,]+\.[\d]{2}'  # Payment amount
        ]
        
        # Order/Transaction number patterns
        self.order_patterns = [
            r'order\s*#?\s*[\w-]+',
            r'transaction\s*#?\s*[\w-]+',
            r'invoice\s*#?\s*[\w-]+',
            r'reference\s*#?\s*[\w-]+',
            r'confirmation\s*#?\s*[\w-]+',
            r'#[\d]{6,}',  # Generic order numbers
        ]
        
        # Payment method patterns
        self.payment_patterns = [
            r'(visa|mastercard|amex|discover)\s*\*+\s*\d{4}',
            r'card ending in \d{4}',
            r'paypal',
            r'apple pay',
            r'google pay',
            r'bank transfer',
            r'debit card',
            r'credit card'
        ]
        
        # Common receipt domains/senders
        self.receipt_domains = [
            'receipt', 'invoice', 'billing', 'payment', 'order',
            'shop', 'store', 'checkout', 'transaction', 'purchase'
        ]
        
        # Receipt type indicators
        self.type_indicators = {
            'purchase': ['order', 'purchase', 'bought', 'shipped', 'delivery', 'subscription', 'renewal', 'monthly', 'annual', 'recurring'],
            'service': ['service', 'appointment', 'booking', 'reservation', 'hosting', 'utilities', 'parking', 'billing'],
            'other': ['refund', 'return', 'credit', 'reversal', 'reimbursement', 'donation', 'contribution', 'gift', 'charitable']
        }
    
    def extract_features(self, email: EmailMessage) -> Dict[str, Any]:
        """Extract receipt-related features from email."""
        text = f"{email.subject} {email.body_text}".lower()
        html = email.body_html or ""
        
        features = {
            'receipt_keywords': self._count_receipt_keywords(text),
            'currency_mentions': self._extract_currency_mentions(text),
            'order_numbers': self._extract_order_numbers(text),
            'payment_methods': self._extract_payment_methods(text),
            'has_itemized_list': self._detect_itemized_list(text, html),
            'has_totals': self._detect_totals(text),
            'vendor_info': self._extract_vendor_info(email),
            'transaction_date': self._extract_transaction_date(text),
            'receipt_domain_match': self._check_receipt_domain(email.sender or ""),
            'structural_score': self._calculate_structural_score(email, text, html)
        }
        
        # Calculate overall receipt score
        features['receipt_score'] = self._calculate_receipt_score(features)
        features['receipt_type'] = self._determine_receipt_type(text)
        
        return features
    
    def _count_receipt_keywords(self, text: str) -> int:
        """Count receipt-related keywords."""
        return sum(1 for keyword in self.receipt_keywords if keyword in text)
    
    def _extract_currency_mentions(self, text: str) -> List[Dict[str, str]]:
        """Extract currency amounts from text."""
        mentions = []
        for pattern in self.currency_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                mentions.append({
                    'amount': match,
                    'pattern': pattern
                })
        return mentions
    
    def _extract_order_numbers(self, text: str) -> List[str]:
        """Extract potential order/transaction numbers."""
        numbers = []
        for pattern in self.order_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            numbers.extend(matches)
        return numbers
    
    def _extract_payment_methods(self, text: str) -> List[str]:
        """Extract payment method mentions."""
        methods = []
        for pattern in self.payment_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            methods.extend(matches)
        return methods
    
    def _detect_itemized_list(self, text: str, html: str) -> bool:
        """Detect if email contains an itemized list of purchases."""
        # Look for table structures in HTML
        if html:
            has_table = bool(re.search(r'<table.*?</table>', html, re.DOTALL))
            has_rows = len(re.findall(r'<tr.*?</tr>', html, re.DOTALL)) > 2
            if has_table and has_rows:
                return True
        
        # Look for list patterns in text
        list_patterns = [
            r'\d+\s*x\s*\w+',  # Quantity x Item
            r'•\s*\w+.*?\$[\d.]+',  # Bullet points with prices
            r'\n\s*\w+.*?\$[\d.]+',  # Items with prices on new lines
        ]
        
        return any(re.search(pattern, text) for pattern in list_patterns)
    
    def _detect_totals(self, text: str) -> bool:
        """Detect total/subtotal mentions."""
        total_patterns = [
            r'(sub)?total:?\s*\$?[\d,]+\.?\d*',
            r'amount due:?\s*\$?[\d,]+\.?\d*',
            r'grand total:?\s*\$?[\d,]+\.?\d*',
            r'balance:?\s*\$?[\d,]+\.?\d*'
        ]
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in total_patterns)
    
    def _extract_vendor_info(self, email: EmailMessage) -> Dict[str, str]:
        """Extract vendor information from email."""
        vendor_info = {}
        
        # Extract from sender
        if email.sender:
            # Clean sender name
            sender_parts = email.sender.split('<')
            if sender_parts:
                vendor_info['sender_name'] = sender_parts[0].strip()
            
            # Extract domain
            domain_match = re.search(r'@([\w.-]+)', email.sender)
            if domain_match:
                vendor_info['domain'] = domain_match.group(1)
        
        return vendor_info
    
    def _extract_transaction_date(self, text: str) -> Optional[str]:
        """Extract transaction date from email."""
        date_patterns = [
            r'(date|on|dated?):?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'(date|on|dated?):?\s*(\w+\s+\d{1,2},?\s+\d{4})',
            r'transaction date:?\s*([^\n]+)',
            r'order date:?\s*([^\n]+)'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1) if len(match.groups()) == 1 else match.group(2)
        
        return None
    
    def _check_receipt_domain(self, sender: str) -> bool:
        """Check if sender domain indicates receipt."""
        sender_lower = sender.lower()
        return any(domain in sender_lower for domain in self.receipt_domains)
    
    def _calculate_structural_score(self, email: EmailMessage, text: str, html: str) -> float:
        """Calculate structural receipt score."""
        score = 0.0
        
        # Check for typical receipt structure
        if self._detect_totals(text):
            score += 0.3
        
        if len(self._extract_currency_mentions(text)) > 0:
            score += 0.2
        
        if len(self._extract_order_numbers(text)) > 0:
            score += 0.2
        
        if self._detect_itemized_list(text, html):
            score += 0.2
        
        if self._check_receipt_domain(email.sender or ""):
            score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_receipt_score(self, features: Dict[str, Any]) -> float:
        """Calculate overall receipt score based on features - balanced approach."""
        score = 0.0
        
        # Receipt keywords (max 0.4) - adjusted for better sensitivity
        keyword_score = min(features['receipt_keywords'] / 6, 1.0) * 0.4
        score += keyword_score
        
        # Currency mentions (max 0.25)
        if len(features['currency_mentions']) > 0:
            # Give more weight to specific receipt currency patterns
            precise_currency = any('total' in mention['pattern'] or 'amount' in mention['pattern'] 
                                 for mention in features['currency_mentions'])
            if precise_currency:
                score += 0.25
            else:
                score += 0.15  # Increased from 0.1
        
        # Order numbers (max 0.2)
        if len(features['order_numbers']) > 0:
            score += 0.2
        
        # Structural elements (max 0.15)
        score += features['structural_score'] * 0.15
        
        return min(score, 1.0)
    
    def _determine_receipt_type(self, text: str) -> str:
        """Determine the type of receipt."""
        type_scores = {}
        
        for receipt_type, indicators in self.type_indicators.items():
            score = sum(1 for indicator in indicators if indicator in text)
            type_scores[receipt_type] = score
        
        if type_scores:
            best_type = max(type_scores.items(), key=lambda x: x[1])
            if best_type[1] > 0:
                return best_type[0]
        
        return "other"


class ReceiptClassificationCache:
    """Caching system for receipt classification results."""
    
    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size
        
    def _generate_cache_key(self, email: EmailMessage) -> str:
        """Generate cache key from email content."""
        content = f"{email.subject}{email.body_text[:500]}{email.sender}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get_cached_result(self, email: EmailMessage) -> Optional[ReceiptClassificationResult]:
        """Get cached classification result."""
        cache_key = self._generate_cache_key(email)
        cached = self.cache.get(cache_key)
        
        if cached and time.time() - cached['timestamp'] < 3600:  # 1 hour cache
            return cached['result']
        return None
    
    def cache_result(self, email: EmailMessage, result: ReceiptClassificationResult) -> None:
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


class ReceiptClassifier:
    """LLM-based receipt classification service."""
    
    def __init__(self):
        """Initialize the receipt classifier."""
        self.ollama_manager = None
        self.cache = ReceiptClassificationCache()
        self.pattern_analyzer = ReceiptPatternAnalyzer()
        self.vendor_profiles: Dict[str, VendorProfile] = {}
        
        # Rule-based classification patterns - bypass LLM for obvious cases
        # Format: regex pattern -> (receipt_type, confidence, reasoning)
        self.subject_patterns = {
            # Software/App Store receipts → SERVICE
            r"your receipt from apple": ("service", 0.99, "Apple App Store receipt"),
            r"receipt.*apple": ("service", 0.95, "Apple service receipt"),
            r"your receipt from anthropic": ("service", 0.99, "Anthropic AI service receipt"),
            r"anthropic.*receipt": ("service", 0.95, "Anthropic service receipt"),
            
            # Software subscription receipts → SERVICE
            r"your tower receipt": ("service", 0.99, "Tower Git client subscription"),
            r"tower.*receipt": ("service", 0.95, "Tower software receipt"),
            r"your ifttt.*receipt": ("service", 0.99, "IFTTT automation service"),
            r"ifttt.*receipt": ("service", 0.95, "IFTTT service receipt"),
            
            # Payment processors (for software) → SERVICE
            r"receipt for your payment to paddle": ("service", 0.99, "Paddle software payment"),
            r"paddle.*receipt": ("service", 0.95, "Paddle payment processor"),
            
            # Cloud/hosting services → SERVICE
            r"aws.*invoice": ("service", 0.99, "AWS cloud service invoice"),
            r"aws.*billing": ("service", 0.99, "AWS billing statement"),
            r".*billing statement.*": ("service", 0.95, "Service billing statement"),
            r".*invoice available.*": ("service", 0.95, "Service invoice"),
            
            # Parking → SERVICE
            r"parking receipt": ("service", 0.99, "Parking service payment"),
            r"your parking.*receipt": ("service", 0.99, "Parking payment receipt"),
            
            # Physical purchases/food → PURCHASE
            r".*order with uber.*": ("purchase", 0.99, "Uber food delivery"),
            r"uber.*receipt": ("purchase", 0.95, "Uber transaction"),
            r"doordash.*receipt": ("purchase", 0.95, "DoorDash food delivery"),
            
            # PayPal (context dependent) → PURCHASE
            r"receipt for your payment to .*": ("purchase", 0.90, "PayPal payment receipt"),
            r"paypal.*receipt": ("purchase", 0.90, "PayPal transaction"),
            
            # Service renewals → SERVICE
            r"renewal confirmation": ("service", 0.95, "Service renewal"),
            r"subscription.*receipt": ("service", 0.95, "Subscription payment"),
            
            # Generic patterns (lower confidence)
            r"receipt for .*": ("purchase", 0.85, "Generic receipt"),
            r"your.*receipt": ("purchase", 0.80, "Generic receipt format"),
        }
        
        # Classification prompt
        self.classification_prompt = """Analyze this email to determine if it's a receipt. Follow these steps:

1. SUBJECT LINE ANALYSIS
   - Does the subject explicitly mention "receipt", "invoice", "payment confirmation"?
   - Does it say "receipt for your payment" or similar?
   - App store purchases often have "Your receipt" in the subject
   - Look for clear receipt language in the subject

2. TRANSACTION INDICATORS
   - Does it confirm a payment or purchase?
   - Are there order/transaction numbers?
   - Does it show amounts, prices, or totals?
   - Is there a date of transaction?
   - App subscriptions and software purchases count as transactions

3. VENDOR INFORMATION
   - Is the sender a business or service (Apple, Google, Microsoft, SaaS companies)?
   - Does it come from a payment/billing system (PayPal, Stripe, Paddle)?
   - App stores (Apple, Google Play) send legitimate receipts
   - Software companies (IFTTT, Tower) send subscription receipts

4. RECEIPT TYPE ANALYSIS
   If it's a receipt, what type:
   - PURCHASE: Product or goods purchase (including subscriptions, recurring payments)
   - SERVICE: Service payment (parking, appointments, utilities, hosting, etc.)
   - OTHER: Any other receipt type

IMPORTANT: 
- If the subject line explicitly says "receipt" or "receipt for your payment", this is very strong evidence it's a receipt.
- App Store purchases and subscription renewals (Apple, Google Play, Tower, IFTTT, etc.) are always receipts.
- Any email confirming a payment, purchase, or subscription renewal should be classified as a receipt.
- Don't be overly conservative - if there's payment confirmation, it's likely a receipt.

Email to analyze:
From: {sender}
Subject: {subject}
Content: {content}

Currency mentions found: {currency_count}
Order numbers found: {order_count}
Receipt keywords: {keyword_count}

Respond with exactly one line in this format:
CLASSIFICATION|CONFIDENCE|TYPE|AMOUNT|VENDOR|ORDER_NUM|REASONING

Examples:
RECEIPT|0.95|PURCHASE|$49.99|Amazon|123-4567890-1234567|Order confirmation with itemized list and payment details
RECEIPT|0.95|SERVICE|NA|Paddle.com|NA|Subject explicitly states "Receipt for Your Payment to Paddle.com"
RECEIPT|0.95|PURCHASE|$9.99|Apple|NA|App Store purchase receipt or subscription renewal
RECEIPT|0.95|SERVICE|$15.00|IFTTT|NA|Service subscription receipt
NOT_RECEIPT|0.90|NA|NA|NA|NA|Newsletter with no transaction information

Your response:"""
    
    async def initialize(self) -> None:
        """Initialize the Ollama client."""
        self.ollama_manager = await get_ollama_manager()
        logger.info("Receipt classifier initialized with LLM support")
    
    def _check_subject_patterns(self, email: EmailMessage) -> Optional[ReceiptClassificationResult]:
        """Check if email subject matches any rule-based patterns."""
        if not email.subject:
            return None
            
        subject_lower = email.subject.lower().strip()
        
        for pattern, (receipt_type, confidence, reasoning) in self.subject_patterns.items():
            if re.search(pattern, subject_lower):
                logger.info(
                    "Subject pattern match",
                    email_id=email.id,
                    pattern=pattern,
                    receipt_type=receipt_type,
                    confidence=confidence
                )
                
                # Extract vendor from email sender
                vendor = self._extract_vendor_name(email)
                
                return ReceiptClassificationResult(
                    is_receipt=True,
                    confidence=confidence,
                    receipt_type=receipt_type,
                    reasoning=f"Subject pattern match: {reasoning}",
                    vendor=vendor,
                    receipt_indicators=["subject_pattern_match", pattern]
                )
        
        return None
    
    async def classify_receipt(self, email: EmailMessage) -> ReceiptClassificationResult:
        """Classify email as receipt or not with detailed analysis."""
        try:
            # Check cache first
            cached_result = self.cache.get_cached_result(email)
            if cached_result:
                logger.debug("Using cached receipt classification", email_id=email.id)
                return cached_result
            
            # Check rule-based patterns first - bypass LLM for obvious cases
            pattern_result = self._check_subject_patterns(email)
            if pattern_result:
                # Cache the pattern-based result
                self.cache.cache_result(email, pattern_result)
                return pattern_result
            
            # Extract receipt features
            features = self.pattern_analyzer.extract_features(email)
            
            # Get vendor profile
            vendor_profile = self._get_or_create_vendor_profile(email)
            
            # Quick classification for obvious cases - balanced thresholds
            if features['receipt_score'] > 0.85:
                result = await self._create_receipt_result(email, features, confidence=0.95)
            elif features['receipt_score'] < 0.25:
                result = ReceiptClassificationResult(
                    is_receipt=False,
                    confidence=0.90,
                    receipt_type="none",
                    reasoning="Low receipt indicators in structural analysis"
                )
            else:
                # Use LLM for ambiguous cases
                result = await self._llm_classify(email, features, vendor_profile)
            
            # Update vendor profile
            self._update_vendor_profile(vendor_profile, result)
            
            # Cache result
            self.cache.cache_result(email, result)
            
            return result
            
        except Exception as e:
            logger.error("Failed to classify receipt", email_id=email.id, error=str(e))
            return ReceiptClassificationResult(
                is_receipt=False,
                confidence=0.0,
                receipt_type="error",
                reasoning=f"Error during classification: {str(e)}"
            )
    
    async def _llm_classify(self, email: EmailMessage, features: Dict[str, Any], 
                           vendor_profile: VendorProfile) -> ReceiptClassificationResult:
        """Use LLM to classify ambiguous receipts."""
        # Prepare content for analysis
        content = email.body_text[:2000] if email.body_text else ""
        
        prompt = self.classification_prompt.format(
            sender=email.sender or "Unknown",
            subject=email.subject or "No subject",
            content=content,
            currency_count=len(features['currency_mentions']),
            order_count=len(features['order_numbers']),
            keyword_count=features['receipt_keywords']
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
            logger.error("LLM receipt classification failed", error=str(e))
            # Fallback to pattern-based classification
            return await self._pattern_based_classification(email, features)
    
    def _parse_llm_response(self, response: str, features: Dict[str, Any], 
                           email: EmailMessage) -> ReceiptClassificationResult:
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
                receipt_type = parts[2].lower() if parts[2] != 'NA' else 'none'
                amount = parts[3] if parts[3] != 'NA' else None
                vendor = parts[4] if parts[4] != 'NA' else None
                order_num = parts[5] if parts[5] != 'NA' else None
                reasoning = parts[6] if len(parts) > 6 else "No reasoning provided"
                
                is_receipt = classification == 'RECEIPT'
                
                # Extract additional details from features
                currency = None
                if amount and features['currency_mentions']:
                    # Try to extract currency symbol
                    currency_match = re.search(r'([$€£¥]|USD|EUR|GBP)', amount)
                    if currency_match:
                        currency = currency_match.group(1)
                
                return ReceiptClassificationResult(
                    is_receipt=is_receipt,
                    confidence=min(max(confidence, 0.0), 1.0),
                    receipt_type=receipt_type if is_receipt else "none",
                    reasoning=reasoning,
                    amount=amount,
                    currency=currency,
                    vendor=vendor or self._extract_vendor_name(email),
                    transaction_date=features.get('transaction_date'),
                    order_number=order_num,
                    payment_method=features['payment_methods'][0] if features['payment_methods'] else None,
                    items_detected=self._extract_items(email),
                    receipt_indicators=self._extract_receipt_indicators(features)
                )
                
        except Exception as e:
            logger.warning("Failed to parse receipt LLM response", response=response, error=str(e))
        
        # Fallback if parsing fails
        return self._create_fallback_result(features, email)
    
    async def _pattern_based_classification(self, email: EmailMessage, 
                                          features: Dict[str, Any]) -> ReceiptClassificationResult:
        """Classify based on pattern analysis."""
        receipt_score = features['receipt_score']
        
        is_receipt = receipt_score > 0.7  # Higher threshold for pattern-based classification
        confidence = abs(receipt_score - 0.7) + 0.5  # Convert to confidence
        
        if is_receipt:
            return await self._create_receipt_result(email, features, confidence)
        else:
            return ReceiptClassificationResult(
                is_receipt=False,
                confidence=confidence,
                receipt_type="none",
                reasoning="Pattern-based analysis: insufficient receipt indicators"
            )
    
    async def _create_receipt_result(self, email: EmailMessage, features: Dict[str, Any], 
                                   confidence: float) -> ReceiptClassificationResult:
        """Create receipt result from features."""
        # Extract amount
        amount = None
        currency = None
        if features['currency_mentions']:
            first_amount = features['currency_mentions'][0]['amount']
            amount = first_amount
            # Extract currency
            currency_match = re.search(r'([$€£¥]|USD|EUR|GBP)', first_amount)
            if currency_match:
                currency = currency_match.group(1)
        
        # Extract order number
        order_number = features['order_numbers'][0] if features['order_numbers'] else None
        
        # Extract vendor
        vendor = self._extract_vendor_name(email)
        
        # Determine receipt type
        receipt_type = features.get('receipt_type', 'other')
        
        return ReceiptClassificationResult(
            is_receipt=True,
            confidence=confidence,
            receipt_type=receipt_type,
            reasoning="Strong receipt indicators found in pattern analysis",
            amount=amount,
            currency=currency,
            vendor=vendor,
            transaction_date=features.get('transaction_date'),
            order_number=order_number,
            payment_method=features['payment_methods'][0] if features['payment_methods'] else None,
            items_detected=self._extract_items(email),
            receipt_indicators=self._extract_receipt_indicators(features)
        )
    
    def _create_fallback_result(self, features: Dict[str, Any], 
                               email: EmailMessage) -> ReceiptClassificationResult:
        """Create fallback result when parsing fails."""
        is_receipt = features['receipt_score'] > 0.5
        
        return ReceiptClassificationResult(
            is_receipt=is_receipt,
            confidence=0.5,
            receipt_type=features.get('receipt_type', 'other') if is_receipt else "none",
            reasoning="Fallback classification based on pattern analysis",
            receipt_indicators=self._extract_receipt_indicators(features)
        )
    
    def _extract_vendor_name(self, email: EmailMessage) -> Optional[str]:
        """Extract vendor name from email."""
        if not email.sender:
            return None
        
        # Try to extract from sender name
        sender_parts = email.sender.split('<')
        if sender_parts:
            vendor_name = sender_parts[0].strip()
            # Clean up common patterns
            vendor_name = re.sub(r'(receipts?|billing|payment|noreply)[@\s]', '', vendor_name, flags=re.IGNORECASE)
            if vendor_name and vendor_name.lower() not in ['no-reply', 'noreply', 'donotreply']:
                return vendor_name
        
        # Try to extract from domain
        domain_match = re.search(r'@([\w.-]+)', email.sender)
        if domain_match:
            domain = domain_match.group(1)
            # Extract meaningful part from domain
            domain_parts = domain.split('.')
            if domain_parts:
                return domain_parts[0].title()
        
        return None
    
    def _extract_items(self, email: EmailMessage) -> List[str]:
        """Extract purchased items from email."""
        items = []
        
        # This is a simplified implementation
        # In production, would use more sophisticated parsing
        text = email.body_text or ""
        
        # Look for common item patterns
        item_patterns = [
            r'(?:^|\n)\s*\d+\s*x\s*([^$\n]+)',  # "2 x Item Name"
            r'(?:^|\n)\s*•\s*([^$\n]+)',  # Bullet points
        ]
        
        for pattern in item_patterns:
            matches = re.findall(pattern, text, re.MULTILINE)
            items.extend([match.strip() for match in matches if len(match.strip()) > 3])
        
        return items[:10]  # Limit to 10 items
    
    def _extract_receipt_indicators(self, features: Dict[str, Any]) -> List[str]:
        """Extract list of receipt indicators found."""
        indicators = []
        
        if features['receipt_keywords'] > 0:
            indicators.append(f"receipt_keywords_{features['receipt_keywords']}")
        
        if features['currency_mentions']:
            indicators.append("currency_amounts")
        
        if features['order_numbers']:
            indicators.append("order_numbers")
        
        if features['payment_methods']:
            indicators.append("payment_methods")
        
        if features.get('has_itemized_list'):
            indicators.append("itemized_list")
        
        if features.get('has_totals'):
            indicators.append("totals_section")
        
        if features.get('receipt_domain_match'):
            indicators.append("receipt_domain")
        
        return indicators
    
    def _get_or_create_vendor_profile(self, email: EmailMessage) -> VendorProfile:
        """Get or create vendor profile."""
        vendor_info = self.pattern_analyzer._extract_vendor_info(email)
        domain = vendor_info.get('domain', 'unknown')
        
        if domain not in self.vendor_profiles:
            self.vendor_profiles[domain] = VendorProfile(
                vendor_name=vendor_info.get('sender_name', 'Unknown'),
                domain=domain
            )
        
        return self.vendor_profiles[domain]
    
    def _update_vendor_profile(self, profile: VendorProfile, 
                              result: ReceiptClassificationResult) -> None:
        """Update vendor profile based on classification result."""
        profile.total_emails += 1
        profile.last_updated = time.time()
        
        if result.is_receipt:
            profile.receipt_emails += 1
            
            # Track receipt types
            if result.receipt_type not in profile.common_receipt_types:
                profile.common_receipt_types.append(result.receipt_type)
        
        # Update receipt rate
        profile.receipt_rate = profile.receipt_emails / profile.total_emails
    
    def get_vendor_statistics(self) -> Dict[str, Any]:
        """Get vendor receipt statistics."""
        if not self.vendor_profiles:
            return {"total_vendors": 0}
        
        total_vendors = len(self.vendor_profiles)
        receipt_vendors = sum(1 for p in self.vendor_profiles.values() if p.receipt_rate > 0.7)
        
        avg_receipt_rate = sum(p.receipt_rate for p in self.vendor_profiles.values()) / total_vendors
        
        # Get top receipt vendors
        top_vendors = sorted(
            [(p.vendor_name, p.receipt_rate, p.total_emails) for p in self.vendor_profiles.values()],
            key=lambda x: x[1] * x[2],  # Sort by receipt rate * volume
            reverse=True
        )[:5]
        
        return {
            "total_vendors": total_vendors,
            "receipt_vendors": receipt_vendors,
            "average_receipt_rate": round(avg_receipt_rate, 3),
            "top_receipt_vendors": top_vendors
        }