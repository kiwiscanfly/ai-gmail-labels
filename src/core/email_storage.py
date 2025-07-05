"""Optimized email storage with memory management and lazy loading."""

import asyncio
import json
import time
import gzip
import pickle
from typing import Dict, List, Optional, Any, AsyncGenerator, Union
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import structlog
import hashlib
import weakref
from contextlib import asynccontextmanager

from src.core.config import get_config
from src.core.exceptions import StorageError
from src.core.database_pool import get_database_pool
from src.models.email import EmailReference, EmailContent

logger = structlog.get_logger(__name__)


# EmailReference and EmailContent are now imported from src.models.email


class EmailCache:
    """LRU cache for email content with memory limit."""
    
    def __init__(self, max_size_mb: int = 100, max_items: int = 1000):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.max_items = max_items
        self.cache: Dict[str, EmailContent] = {}
        self.access_times: Dict[str, float] = {}
        self.current_size = 0
        self._lock = asyncio.Lock()
        
    async def get(self, email_id: str) -> Optional[EmailContent]:
        """Get email content from cache."""
        async with self._lock:
            if email_id in self.cache:
                self.access_times[email_id] = time.time()
                logger.debug("Cache hit", email_id=email_id)
                return self.cache[email_id]
            
            logger.debug("Cache miss", email_id=email_id)
            return None
    
    async def put(self, email_id: str, content: EmailContent) -> None:
        """Store email content in cache."""
        async with self._lock:
            content_size = content.get_size_bytes()
            
            # Check if we need to evict items
            while (
                (self.current_size + content_size > self.max_size_bytes) or 
                (len(self.cache) >= self.max_items)
            ) and self.cache:
                await self._evict_lru()
            
            # Store the content
            if email_id in self.cache:
                # Update existing
                old_size = self.cache[email_id].get_size_bytes()
                self.current_size = self.current_size - old_size + content_size
            else:
                # Add new
                self.current_size += content_size
            
            self.cache[email_id] = content
            self.access_times[email_id] = time.time()
            
            logger.debug(
                "Cached email content",
                email_id=email_id,
                size_bytes=content_size,
                cache_size_mb=self.current_size / (1024 * 1024),
                cache_items=len(self.cache)
            )
    
    async def _evict_lru(self) -> None:
        """Evict least recently used item."""
        if not self.access_times:
            return
        
        # Find LRU item
        lru_email_id = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        
        # Remove from cache
        if lru_email_id in self.cache:
            content_size = self.cache[lru_email_id].get_size_bytes()
            self.current_size -= content_size
            del self.cache[lru_email_id]
        
        del self.access_times[lru_email_id]
        
        logger.debug("Evicted from cache", email_id=lru_email_id)
    
    async def clear(self) -> None:
        """Clear the entire cache."""
        async with self._lock:
            self.cache.clear()
            self.access_times.clear()
            self.current_size = 0
            logger.info("Email cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size_mb": self.current_size / (1024 * 1024),
            "max_size_mb": self.max_size_bytes / (1024 * 1024),
            "items": len(self.cache),
            "max_items": self.max_items,
            "utilization": self.current_size / self.max_size_bytes if self.max_size_bytes else 0
        }


class EmailStorage:
    """Optimized email storage with lazy loading and memory management."""
    
    def __init__(self, cache_size_mb: int = 100):
        self.config = get_config()
        self.storage_path = Path("./data/email_storage")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.cache = EmailCache(max_size_mb=cache_size_mb)
        self.references: Dict[str, EmailReference] = {}
        self._weak_refs: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
        
    async def initialize(self) -> None:
        """Initialize the email storage system."""
        try:
            pool = await get_database_pool()
            
            # Create email references table
            await pool.execute_query("""
                CREATE TABLE IF NOT EXISTS email_references (
                    email_id TEXT PRIMARY KEY,
                    thread_id TEXT NOT NULL,
                    subject TEXT,
                    sender TEXT,
                    recipient TEXT,
                    date TEXT,
                    labels TEXT,
                    size_estimate INTEGER DEFAULT 0,
                    content_hash TEXT,
                    storage_path TEXT,
                    cached_at REAL,
                    last_accessed REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes
            await pool.execute_query("""
                CREATE INDEX IF NOT EXISTS idx_email_thread 
                ON email_references(thread_id)
            """)
            
            await pool.execute_query("""
                CREATE INDEX IF NOT EXISTS idx_email_last_accessed 
                ON email_references(last_accessed DESC)
            """)
            
            await pool.execute_query("""
                CREATE INDEX IF NOT EXISTS idx_email_sender 
                ON email_references(sender)
            """)
            
            # Load existing references
            await self._load_references()
            
            logger.info(
                "Email storage initialized",
                storage_path=str(self.storage_path),
                cached_references=len(self.references)
            )
            
        except Exception as e:
            logger.error("Failed to initialize email storage", error=str(e))
            raise StorageError(f"Failed to initialize email storage: {e}")
    
    async def _load_references(self) -> None:
        """Load email references from database."""
        try:
            pool = await get_database_pool()
            
            rows = await pool.execute_query(
                "SELECT * FROM email_references ORDER BY last_accessed DESC LIMIT 10000",
                fetch_all=True
            )
            
            for row in rows:
                ref = EmailReference(
                    email_id=row[0],
                    thread_id=row[1],
                    subject=row[2] or "",
                    sender=row[3] or "",
                    recipient=row[4] or "",
                    date=row[5] or "",
                    labels=json.loads(row[6]) if row[6] else [],
                    size_estimate=row[7] or 0,
                    content_hash=row[8],
                    storage_path=row[9],
                    cached_at=row[10],
                    last_accessed=row[11] or time.time()
                )
                self.references[ref.email_id] = ref
            
            logger.debug("Loaded email references", count=len(self.references))
            
        except Exception as e:
            logger.error("Failed to load email references", error=str(e))
    
    async def store_email_reference(self, reference: EmailReference) -> None:
        """Store an email reference."""
        try:
            pool = await get_database_pool()
            
            reference.last_accessed = time.time()
            
            await pool.execute_query("""
                INSERT OR REPLACE INTO email_references 
                (email_id, thread_id, subject, sender, recipient, date, 
                 labels, size_estimate, content_hash, storage_path, 
                 cached_at, last_accessed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                reference.email_id,
                reference.thread_id,
                reference.subject,
                reference.sender,
                reference.recipient,
                reference.date,
                json.dumps(reference.labels),
                reference.size_estimate,
                reference.content_hash,
                reference.storage_path,
                reference.cached_at,
                reference.last_accessed
            ))
            
            self.references[reference.email_id] = reference
            
            logger.debug("Stored email reference", email_id=reference.email_id)
            
        except Exception as e:
            logger.error("Failed to store email reference", email_id=reference.email_id, error=str(e))
            raise StorageError(f"Failed to store email reference: {e}")
    
    async def get_email_reference(self, email_id: str) -> Optional[EmailReference]:
        """Get an email reference by ID."""
        if email_id in self.references:
            ref = self.references[email_id]
            ref.last_accessed = time.time()
            return ref
        
        # Try loading from database
        try:
            pool = await get_database_pool()
            
            row = await pool.execute_query(
                "SELECT * FROM email_references WHERE email_id = ?",
                (email_id,),
                fetch_one=True
            )
            
            if row:
                ref = EmailReference(
                    email_id=row[0],
                    thread_id=row[1],
                    subject=row[2] or "",
                    sender=row[3] or "",
                    recipient=row[4] or "",
                    date=row[5] or "",
                    labels=json.loads(row[6]) if row[6] else [],
                    size_estimate=row[7] or 0,
                    content_hash=row[8],
                    storage_path=row[9],
                    cached_at=row[10],
                    last_accessed=time.time()
                )
                self.references[email_id] = ref
                return ref
            
            return None
            
        except Exception as e:
            logger.error("Failed to get email reference", email_id=email_id, error=str(e))
            return None
    
    async def store_email_content(self, email_id: str, content: EmailContent) -> str:
        """Store email content and return storage path."""
        try:
            # Calculate content hash
            content_str = json.dumps({
                'body_text': content.body_text,
                'body_html': content.body_html,
                'attachments': content.attachments,
                'headers': content.headers
            }, sort_keys=True)
            
            content_hash = hashlib.sha256(content_str.encode()).hexdigest()
            
            # Create storage path
            storage_dir = self.storage_path / content_hash[:2] / content_hash[2:4]
            storage_dir.mkdir(parents=True, exist_ok=True)
            storage_file = storage_dir / f"{content_hash}.gz"
            
            # Compress and store content
            content_data = {
                'email_id': content.email_id,
                'body_text': content.body_text,
                'body_html': content.body_html,
                'attachments': content.attachments,
                'headers': content.headers,
                'stored_at': time.time()
            }
            
            if content.raw_content:
                content_data['raw_content'] = content.raw_content
            
            with gzip.open(storage_file, 'wb') as f:
                pickle.dump(content_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Store in cache
            content.compressed = True
            await self.cache.put(email_id, content)
            
            logger.debug(
                "Stored email content",
                email_id=email_id,
                storage_path=str(storage_file),
                size_bytes=content.get_size_bytes()
            )
            
            return str(storage_file.relative_to(self.storage_path))
            
        except Exception as e:
            logger.error("Failed to store email content", email_id=email_id, error=str(e))
            raise StorageError(f"Failed to store email content: {e}")
    
    async def load_email_content(self, email_id: str) -> Optional[EmailContent]:
        """Load email content with lazy loading."""
        # Try cache first
        content = await self.cache.get(email_id)
        if content:
            return content
        
        # Get reference
        reference = await self.get_email_reference(email_id)
        if not reference or not reference.storage_path:
            logger.debug("No storage path for email", email_id=email_id)
            return None
        
        try:
            # Load from disk
            storage_file = self.storage_path / reference.storage_path
            
            if not storage_file.exists():
                logger.warning("Storage file not found", email_id=email_id, path=str(storage_file))
                return None
            
            with gzip.open(storage_file, 'rb') as f:
                content_data = pickle.load(f)
            
            # Create EmailContent object
            content = EmailContent(
                email_id=content_data['email_id'],
                body_text=content_data.get('body_text', ''),
                body_html=content_data.get('body_html', ''),
                attachments=content_data.get('attachments', []),
                headers=content_data.get('headers', {}),
                raw_content=content_data.get('raw_content'),
                compressed=True
            )
            
            # Store in cache
            await self.cache.put(email_id, content)
            
            # Update access time
            reference.last_accessed = time.time()
            await self.store_email_reference(reference)
            
            logger.debug("Loaded email content from storage", email_id=email_id)
            return content
            
        except Exception as e:
            logger.error("Failed to load email content", email_id=email_id, error=str(e))
            return None
    
    async def search_references(
        self,
        query: Optional[str] = None,
        sender: Optional[str] = None,
        thread_id: Optional[str] = None,
        labels: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[EmailReference]:
        """Search email references without loading content."""
        try:
            pool = await get_database_pool()
            
            sql = "SELECT * FROM email_references WHERE 1=1"
            params = []
            
            if query:
                sql += " AND (subject LIKE ? OR sender LIKE ?)"
                params.extend([f"%{query}%", f"%{query}%"])
            
            if sender:
                sql += " AND sender LIKE ?"
                params.append(f"%{sender}%")
            
            if thread_id:
                sql += " AND thread_id = ?"
                params.append(thread_id)
            
            if labels:
                # Simple label search - could be improved with proper JSON queries
                for label in labels:
                    sql += " AND labels LIKE ?"
                    params.append(f"%{label}%")
            
            sql += " ORDER BY last_accessed DESC LIMIT ?"
            params.append(limit)
            
            rows = await pool.execute_query(sql, params, fetch_all=True)
            
            references = []
            for row in rows:
                ref = EmailReference(
                    email_id=row[0],
                    thread_id=row[1],
                    subject=row[2] or "",
                    sender=row[3] or "",
                    recipient=row[4] or "",
                    date=row[5] or "",
                    labels=json.loads(row[6]) if row[6] else [],
                    size_estimate=row[7] or 0,
                    content_hash=row[8],
                    storage_path=row[9],
                    cached_at=row[10],
                    last_accessed=row[11] or time.time()
                )
                references.append(ref)
            
            logger.debug("Search completed", query=query, results=len(references))
            return references
            
        except Exception as e:
            logger.error("Failed to search references", error=str(e))
            return []
    
    async def cleanup_old_content(self, days: int = 30) -> int:
        """Clean up old email content files."""
        try:
            cutoff_time = time.time() - (days * 24 * 60 * 60)
            
            pool = await get_database_pool()
            
            # Get old references
            rows = await pool.execute_query("""
                SELECT email_id, storage_path FROM email_references 
                WHERE last_accessed < ? AND storage_path IS NOT NULL
            """, (cutoff_time,), fetch_all=True)
            
            cleaned_count = 0
            for row in rows:
                email_id, storage_path = row[0], row[1]
                
                if storage_path:
                    file_path = self.storage_path / storage_path
                    if file_path.exists():
                        try:
                            file_path.unlink()
                            cleaned_count += 1
                        except Exception as e:
                            logger.error("Failed to delete file", path=str(file_path), error=str(e))
                
                # Update reference to remove storage path
                await pool.execute_query("""
                    UPDATE email_references 
                    SET storage_path = NULL, cached_at = NULL 
                    WHERE email_id = ?
                """, (email_id,))
            
            logger.info("Cleaned up old email content", files_deleted=cleaned_count, days=days)
            return cleaned_count
            
        except Exception as e:
            logger.error("Failed to cleanup old content", error=str(e))
            return 0
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        try:
            pool = await get_database_pool()
            
            # Count references
            ref_stats = await pool.execute_query("""
                SELECT 
                    COUNT(*) as total_references,
                    COUNT(storage_path) as stored_content,
                    SUM(size_estimate) as total_size_estimate
                FROM email_references
            """, fetch_one=True)
            
            # Calculate storage directory size
            storage_size = 0
            file_count = 0
            for file_path in self.storage_path.rglob("*.gz"):
                try:
                    storage_size += file_path.stat().st_size
                    file_count += 1
                except OSError:
                    pass
            
            cache_stats = self.cache.get_stats()
            
            return {
                "references": {
                    "total": ref_stats[0] if ref_stats else 0,
                    "with_content": ref_stats[1] if ref_stats else 0,
                    "size_estimate_mb": (ref_stats[2] or 0) / (1024 * 1024) if ref_stats else 0
                },
                "storage": {
                    "files": file_count,
                    "size_mb": storage_size / (1024 * 1024),
                    "path": str(self.storage_path)
                },
                "cache": cache_stats
            }
            
        except Exception as e:
            logger.error("Failed to get storage stats", error=str(e))
            return {}


# Global email storage instance
_email_storage: Optional[EmailStorage] = None


async def get_email_storage() -> EmailStorage:
    """Get the global email storage instance."""
    global _email_storage
    if _email_storage is None:
        config = get_config()
        cache_size = getattr(config.performance, 'email_cache_size_mb', 100)
        _email_storage = EmailStorage(cache_size_mb=cache_size)
        await _email_storage.initialize()
    return _email_storage


async def shutdown_email_storage() -> None:
    """Shutdown the global email storage."""
    global _email_storage
    if _email_storage:
        await _email_storage.cache.clear()
        _email_storage = None