"""Custom email classification service for user-defined categories."""

import asyncio
import json
import time
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import structlog

from src.models.email import EmailMessage
from src.integrations.ollama_client import get_ollama_manager
from src.cli.langchain.chains import CustomLabelChain
from src.core.exceptions import ServiceError

logger = structlog.get_logger(__name__)


@dataclass
class CustomClassificationResult:
    """Custom email classification result."""
    is_match: bool
    confidence: float  # 0.0 to 1.0
    category: str
    reasoning: str
    suggested_label: Optional[str] = None
    search_terms_used: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CustomCategory:
    """Represents a custom email category."""
    name: str
    description: str
    search_terms: List[str] = field(default_factory=list)
    parent_label: Optional[str] = None
    confidence_threshold: float = 0.7
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    usage_count: int = 0
    performance_stats: Dict[str, Any] = field(default_factory=dict)


class CustomCategoryManager:
    """Manages custom email categories and their persistence."""
    
    def __init__(self):
        """Initialize the category manager."""
        self.categories: Dict[str, CustomCategory] = {}
        self.cache_file = "data/custom_categories.json"
        
    async def initialize(self) -> None:
        """Initialize the category manager and load existing categories."""
        try:
            await self._load_categories()
            logger.info("Custom category manager initialized", category_count=len(self.categories))
        except Exception as e:
            logger.error("Failed to initialize category manager", error=str(e))
            raise ServiceError(f"Failed to initialize category manager: {e}")
    
    async def create_category(
        self,
        name: str,
        description: str,
        search_terms: Optional[List[str]] = None,
        parent_label: Optional[str] = None,
        confidence_threshold: float = 0.7
    ) -> CustomCategory:
        """Create a new custom category.
        
        Args:
            name: Category name
            description: Category description
            search_terms: Optional predefined search terms
            parent_label: Optional parent label for hierarchical organization
            confidence_threshold: Minimum confidence for classification
            
        Returns:
            Created custom category
        """
        category = CustomCategory(
            name=name,
            description=description,
            search_terms=search_terms or [],
            parent_label=parent_label,
            confidence_threshold=confidence_threshold
        )
        
        self.categories[name.lower()] = category
        await self._save_categories()
        
        logger.info("Created custom category", name=name, description=description)
        return category
    
    async def get_category(self, name: str) -> Optional[CustomCategory]:
        """Get a category by name.
        
        Args:
            name: Category name
            
        Returns:
            Custom category if found, None otherwise
        """
        return self.categories.get(name.lower())
    
    async def list_categories(self) -> List[CustomCategory]:
        """List all custom categories.
        
        Returns:
            List of all custom categories
        """
        return list(self.categories.values())
    
    async def update_category_usage(self, name: str, performance_data: Dict[str, Any] = None) -> None:
        """Update category usage statistics.
        
        Args:
            name: Category name
            performance_data: Optional performance data to record
        """
        category = self.categories.get(name.lower())
        if category:
            category.last_used = time.time()
            category.usage_count += 1
            
            if performance_data:
                category.performance_stats.update(performance_data)
            
            await self._save_categories()
    
    async def delete_category(self, name: str) -> bool:
        """Delete a custom category.
        
        Args:
            name: Category name to delete
            
        Returns:
            True if deleted, False if not found
        """
        if name.lower() in self.categories:
            del self.categories[name.lower()]
            await self._save_categories()
            logger.info("Deleted custom category", name=name)
            return True
        return False
    
    async def _load_categories(self) -> None:
        """Load categories from persistent storage."""
        try:
            import os
            if not os.path.exists(self.cache_file):
                return
                
            with open(self.cache_file, 'r') as f:
                data = json.load(f)
                
            for name, category_data in data.items():
                category = CustomCategory(
                    name=category_data['name'],
                    description=category_data['description'],
                    search_terms=category_data.get('search_terms', []),
                    parent_label=category_data.get('parent_label'),
                    confidence_threshold=category_data.get('confidence_threshold', 0.7),
                    created_at=category_data.get('created_at', time.time()),
                    last_used=category_data.get('last_used', time.time()),
                    usage_count=category_data.get('usage_count', 0),
                    performance_stats=category_data.get('performance_stats', {})
                )
                self.categories[name] = category
                
        except Exception as e:
            logger.warning("Failed to load categories from cache", error=str(e))
    
    async def _save_categories(self) -> None:
        """Save categories to persistent storage."""
        try:
            import os
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            
            data = {}
            for name, category in self.categories.items():
                data[name] = {
                    'name': category.name,
                    'description': category.description,
                    'search_terms': category.search_terms,
                    'parent_label': category.parent_label,
                    'confidence_threshold': category.confidence_threshold,
                    'created_at': category.created_at,
                    'last_used': category.last_used,
                    'usage_count': category.usage_count,
                    'performance_stats': category.performance_stats
                }
            
            with open(self.cache_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error("Failed to save categories", error=str(e))


class CustomClassificationCache:
    """Caching system for custom classification results."""
    
    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size
        
    def _generate_cache_key(self, email: EmailMessage, category: str) -> str:
        """Generate cache key from email content and category."""
        content = f"{email.subject}{email.body_text[:500]}{email.sender}{category}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get_cached_result(self, email: EmailMessage, category: str) -> Optional[CustomClassificationResult]:
        """Get cached classification result."""
        cache_key = self._generate_cache_key(email, category)
        cached = self.cache.get(cache_key)
        
        if cached and time.time() - cached['timestamp'] < 1800:  # 30 minute cache
            return cached['result']
        return None
    
    def cache_result(self, email: EmailMessage, category: str, result: CustomClassificationResult) -> None:
        """Cache classification result."""
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]['timestamp'])
            del self.cache[oldest_key]
        
        cache_key = self._generate_cache_key(email, category)
        self.cache[cache_key] = {
            'result': result,
            'timestamp': time.time()
        }


class CustomClassifier:
    """Custom email classification service using LLM-based analysis."""
    
    def __init__(self):
        """Initialize the custom classifier."""
        self.ollama_manager = None
        self.label_chain = None
        self.category_manager = CustomCategoryManager()
        self.cache = CustomClassificationCache()
        self.initialized = False
        
    async def initialize(self) -> None:
        """Initialize the custom classifier."""
        try:
            self.ollama_manager = await get_ollama_manager()
            self.label_chain = CustomLabelChain(self.ollama_manager)
            await self.category_manager.initialize()
            self.initialized = True
            logger.info("Custom classifier initialized with LLM support")
        except Exception as e:
            logger.error("Failed to initialize custom classifier", error=str(e))
            raise ServiceError(f"Failed to initialize custom classifier: {e}")
    
    async def classify_email(
        self,
        email: EmailMessage,
        category: str,
        search_terms: Optional[List[str]] = None,
        confidence_threshold: float = 0.7
    ) -> CustomClassificationResult:
        """Classify an email for a specific custom category.
        
        Args:
            email: Email to classify
            category: Category to classify for
            search_terms: Optional predefined search terms (will generate if not provided)
            confidence_threshold: Minimum confidence threshold
            
        Returns:
            Custom classification result
        """
        if not self.initialized:
            await self.initialize()
            
        try:
            # Check cache first
            cached_result = self.cache.get_cached_result(email, category)
            if cached_result:
                logger.debug("Using cached custom classification", email_id=email.id, category=category)
                return cached_result
            
            # Get or generate search terms
            if not search_terms:
                # Check if we have stored terms for this category
                stored_category = await self.category_manager.get_category(category)
                if stored_category and stored_category.search_terms:
                    search_terms = stored_category.search_terms
                else:
                    # Generate search terms using LLM
                    search_terms = await self.label_chain.generate_search_terms(category)
                    
                    # Store the generated terms if we have a stored category
                    if stored_category:
                        stored_category.search_terms = search_terms
                        await self.category_manager._save_categories()
            
            # Classify the email
            classification_result = await self.label_chain.classify_for_category(
                email, category, search_terms
            )
            
            # Create result object
            result = CustomClassificationResult(
                is_match=classification_result.get("classification", False),
                confidence=classification_result.get("confidence", 0.0),
                category=category,
                reasoning=classification_result.get("reasoning", "No reasoning provided"),
                suggested_label=classification_result.get("suggested_label"),
                search_terms_used=search_terms,
                metadata={
                    "classification_time": time.time(),
                    "method": "llm_classification"
                }
            )
            
            # Update category usage stats
            await self.category_manager.update_category_usage(
                category,
                {
                    "last_classification": {
                        "confidence": result.confidence,
                        "is_match": result.is_match,
                        "timestamp": time.time()
                    }
                }
            )
            
            # Cache the result
            self.cache.cache_result(email, category, result)
            
            return result
            
        except Exception as e:
            logger.error("Failed to classify email for custom category", 
                        email_id=email.id, category=category, error=str(e))
            return CustomClassificationResult(
                is_match=False,
                confidence=0.0,
                category=category,
                reasoning=f"Error during classification: {str(e)}",
                metadata={"error": str(e)}
            )
    
    async def create_category(
        self,
        name: str,
        description: str,
        search_terms: Optional[List[str]] = None,
        generate_terms: bool = True,
        parent_label: Optional[str] = None,
        confidence_threshold: float = 0.7
    ) -> CustomCategory:
        """Create a new custom category with optional AI-generated search terms.
        
        Args:
            name: Category name
            description: Category description
            search_terms: Optional predefined search terms
            generate_terms: Whether to generate search terms using AI
            parent_label: Optional parent label for organization
            confidence_threshold: Minimum confidence for classification
            
        Returns:
            Created custom category
        """
        if not self.initialized:
            await self.initialize()
            
        # Generate search terms if requested and not provided
        if generate_terms and not search_terms:
            search_terms = await self.label_chain.generate_search_terms(name, description)
        
        return await self.category_manager.create_category(
            name=name,
            description=description,
            search_terms=search_terms or [],
            parent_label=parent_label,
            confidence_threshold=confidence_threshold
        )
    
    async def get_categories(self) -> List[CustomCategory]:
        """Get all custom categories.
        
        Returns:
            List of all custom categories
        """
        if not self.initialized:
            await self.initialize()
            
        return await self.category_manager.list_categories()
    
    async def get_category_statistics(self) -> Dict[str, Any]:
        """Get statistics about custom categories and their usage.
        
        Returns:
            Category usage statistics
        """
        if not self.initialized:
            await self.initialize()
            
        categories = await self.category_manager.list_categories()
        
        if not categories:
            return {"total_categories": 0}
        
        total_usage = sum(cat.usage_count for cat in categories)
        active_categories = len([cat for cat in categories if cat.usage_count > 0])
        
        # Get most used categories
        most_used = sorted(categories, key=lambda c: c.usage_count, reverse=True)[:5]
        
        # Get recently used categories
        recently_used = sorted(categories, key=lambda c: c.last_used, reverse=True)[:5]
        
        return {
            "total_categories": len(categories),
            "active_categories": active_categories,
            "total_usage": total_usage,
            "average_usage": total_usage / len(categories) if categories else 0,
            "most_used_categories": [
                {"name": cat.name, "usage_count": cat.usage_count} 
                for cat in most_used
            ],
            "recently_used_categories": [
                {"name": cat.name, "last_used": cat.last_used}
                for cat in recently_used
            ]
        }