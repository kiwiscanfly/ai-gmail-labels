"""Gmail API client with OAuth2 authentication and advanced features."""

import asyncio
import json
import time
import base64
import email
from typing import Dict, List, Optional, Any, Union, AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import structlog

# Google API imports
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.auth.exceptions import RefreshError

# Security and rate limiting
import keyring
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.core.config import get_config
from src.core.exceptions import (
    GmailAPIError,
    AuthenticationError,
    RateLimitError,
    TemporaryGmailError,
    InvalidCredentialsError,
    RetryableError
)
from src.core.email_storage import get_email_storage
from src.models.email import EmailMessage, EmailReference, EmailContent
from src.models.gmail import GmailLabel, BatchOperation
from src.models.common import Status

logger = structlog.get_logger(__name__)


# EmailMessage, GmailLabel, and BatchOperation are now imported from src.models


class RateLimiter:
    """Rate limiter for Gmail API requests."""
    
    def __init__(self, requests_per_second: int = 10, quota_per_user_per_second: int = 250):
        self.requests_per_second = requests_per_second
        self.quota_per_user_per_second = quota_per_user_per_second
        self.request_times: List[float] = []
        self.quota_usage: List[float] = []
        self._lock = asyncio.Lock()
    
    async def acquire(self, quota_cost: int = 1):
        """Acquire rate limit permission."""
        async with self._lock:
            now = time.time()
            
            # Clean old entries
            self.request_times = [t for t in self.request_times if now - t < 1.0]
            self.quota_usage = [t for t in self.quota_usage if now - t < 1.0]
            
            # Check rate limits
            if len(self.request_times) >= self.requests_per_second:
                sleep_time = 1.0 - (now - self.request_times[0])
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
            
            if sum(1 for _ in self.quota_usage) + quota_cost > self.quota_per_user_per_second:
                sleep_time = 1.0 - (now - self.quota_usage[0])
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
            
            # Record this request
            self.request_times.append(now)
            for _ in range(quota_cost):
                self.quota_usage.append(now)


class GmailClient:
    """Gmail API client with OAuth2 authentication and advanced features."""
    
    def __init__(self):
        """Initialize the Gmail client."""
        self.config = get_config()
        self.credentials: Optional[Credentials] = None
        self.service = None
        self.rate_limiter = RateLimiter(
            requests_per_second=self.config.gmail.rate_limit["requests_per_second"],
            quota_per_user_per_second=self.config.gmail.rate_limit["quota_per_user_per_second"]
        )
        self.labels_cache: Dict[str, GmailLabel] = {}
        self.watch_response: Optional[Dict[str, Any]] = None
        self._batch_operations: Dict[str, BatchOperation] = {}
        self._email_storage: Optional[Any] = None  # Lazy loaded
        
    async def initialize(self) -> None:
        """Initialize the Gmail client with authentication."""
        try:
            await self._authenticate()
            await self._refresh_labels_cache()
            self._email_storage = await get_email_storage()
            logger.info("Gmail client initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize Gmail client", error=str(e))
            raise GmailAPIError(f"Failed to initialize Gmail client: {e}")
    
    async def _authenticate(self) -> None:
        """Handle OAuth2 authentication with token management."""
        try:
            credentials_path = Path(self.config.gmail.credentials_path)
            token_path = Path(self.config.gmail.token_path)
            
            if not credentials_path.exists():
                raise InvalidCredentialsError(f"Credentials file not found: {credentials_path}")
            
            # Load existing token
            creds = None
            if token_path.exists():
                try:
                    creds = Credentials.from_authorized_user_file(str(token_path), self.config.gmail.scopes)
                except Exception as e:
                    logger.warning("Failed to load existing token", error=str(e))
            
            # Refresh or get new credentials
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    try:
                        creds.refresh(Request())
                        logger.info("Credentials refreshed successfully")
                    except RefreshError as e:
                        logger.error("Failed to refresh credentials", error=str(e))
                        creds = None
                
                if not creds:
                    # Run OAuth flow
                    flow = InstalledAppFlow.from_client_secrets_file(
                        str(credentials_path), 
                        self.config.gmail.scopes
                    )
                    creds = flow.run_local_server(port=0)
                    logger.info("New credentials obtained via OAuth flow")
                
                # Save credentials securely
                await self._save_credentials(creds, token_path)
            
            self.credentials = creds
            self.service = build('gmail', 'v1', credentials=creds)
            logger.debug("Gmail service authenticated")
            
        except Exception as e:
            logger.error("Authentication failed", error=str(e))
            raise AuthenticationError(f"Gmail authentication failed: {e}")
    
    async def _save_credentials(self, creds: Credentials, token_path: Path) -> None:
        """Save credentials securely."""
        try:
            # Save to file
            with open(token_path, 'w') as token_file:
                token_file.write(creds.to_json())
            
            # Also save to keyring for additional security
            if self.config.security.credential_storage == "keyring":
                keyring.set_password("gmail_oauth", "token", creds.to_json())
            
            logger.debug("Credentials saved securely")
            
        except Exception as e:
            logger.error("Failed to save credentials", error=str(e))
            # Don't raise here as authentication might still work
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(RetryableError)
    )
    async def _api_request(self, request_func, quota_cost: int = 1, **kwargs) -> Any:
        """Make an API request with rate limiting and retry logic."""
        await self.rate_limiter.acquire(quota_cost)
        
        try:
            result = request_func(**kwargs).execute()
            return result
        except HttpError as e:
            error_code = e.resp.status
            error_reason = e.error_details[0].get('reason', '') if e.error_details else ''
            
            if error_code == 429 or 'rateLimitExceeded' in error_reason:
                raise RateLimitError(f"Rate limit exceeded: {e}")
            elif error_code in [500, 502, 503, 504]:
                raise TemporaryGmailError(f"Temporary Gmail error: {e}")
            elif error_code == 401:
                # Try to refresh credentials
                await self._authenticate()
                raise TemporaryGmailError(f"Authentication error, retrying: {e}")
            else:
                raise GmailAPIError(f"Gmail API error: {e}")
        except Exception as e:
            logger.error("API request failed", error=str(e))
            raise TemporaryGmailError(f"API request failed: {e}")
    
    # Label Management
    
    async def _refresh_labels_cache(self, include_stats: bool = False) -> None:
        """Refresh the labels cache.
        
        Args:
            include_stats: Whether to fetch detailed message statistics for each label
        """
        try:
            response = await self._api_request(self.service.users().labels().list, userId='me')
            
            self.labels_cache.clear()
            for label_data in response.get('labels', []):
                label = GmailLabel(
                    id=label_data['id'],
                    name=label_data['name'],
                    message_list_visibility=label_data.get('messageListVisibility', 'show'),
                    label_list_visibility=label_data.get('labelListVisibility', 'labelShow'),
                    type=label_data.get('type', 'user'),
                    messages_total=label_data.get('messagesTotal', 0),
                    messages_unread=label_data.get('messagesUnread', 0),
                    threads_total=label_data.get('threadsTotal', 0),
                    threads_unread=label_data.get('threadsUnread', 0),
                    color=label_data.get('color')
                )
                self.labels_cache[label.name] = label
            
            # Fetch detailed statistics if requested
            if include_stats:
                await self._fetch_label_statistics()
            
            logger.debug("Labels cache refreshed", label_count=len(self.labels_cache), include_stats=include_stats)
            
        except Exception as e:
            logger.error("Failed to refresh labels cache", error=str(e))
            raise GmailAPIError(f"Failed to refresh labels cache: {e}")
    
    async def _fetch_label_statistics(self) -> None:
        """Fetch detailed statistics for all labels."""
        try:
            # Fetch detailed label information for each label
            for label_name, label in self.labels_cache.items():
                try:
                    # Get detailed label info from Gmail API
                    label_response = await self._api_request(
                        self.service.users().labels().get,
                        userId='me',
                        id=label.id
                    )
                    
                    # Update the cached label with statistics
                    label.messages_total = label_response.get('messagesTotal', 0)
                    label.messages_unread = label_response.get('messagesUnread', 0)
                    label.threads_total = label_response.get('threadsTotal', 0)
                    label.threads_unread = label_response.get('threadsUnread', 0)
                    
                except Exception as e:
                    logger.warning(f"Could not fetch stats for label {label_name}", error=str(e))
                    continue
                    
                # Rate limiting - small delay between requests
                await asyncio.sleep(0.1)
                
        except Exception as e:
            logger.error("Failed to fetch label statistics", error=str(e))
    
    async def list_labels(self, include_stats: bool = False) -> List[GmailLabel]:
        """List all Gmail labels.
        
        Args:
            include_stats: Whether to include message count statistics
        
        Returns:
            List of GmailLabel objects.
        """
        if not self.labels_cache:
            await self._refresh_labels_cache(include_stats=include_stats)
        elif include_stats:
            # If cache exists but we need stats, fetch them
            await self._fetch_label_statistics()
        return list(self.labels_cache.values())
    
    async def get_label_by_name(self, name: str) -> Optional[GmailLabel]:
        """Get a label by name.
        
        Args:
            name: The label name.
            
        Returns:
            GmailLabel if found, None otherwise.
        """
        if not self.labels_cache:
            await self._refresh_labels_cache()
        return self.labels_cache.get(name)
    
    async def create_label(self, name: str, visibility: str = "labelShow") -> GmailLabel:
        """Create a new Gmail label.
        
        Args:
            name: The label name.
            visibility: Label visibility setting.
            
        Returns:
            The created GmailLabel.
        """
        try:
            # Check if label already exists
            existing = await self.get_label_by_name(name)
            if existing:
                logger.info("Label already exists", name=name)
                return existing
            
            label_object = {
                'name': name,
                'labelListVisibility': visibility,
                'messageListVisibility': 'show'
            }
            
            response = await self._api_request(
                self.service.users().labels().create,
                userId='me',
                body=label_object
            )
            
            # Create label object
            label = GmailLabel(
                id=response['id'],
                name=response['name'],
                message_list_visibility=response.get('messageListVisibility', 'show'),
                label_list_visibility=response.get('labelListVisibility', 'labelShow'),
                type=response.get('type', 'user')
            )
            
            # Update cache
            self.labels_cache[label.name] = label
            
            logger.info("Label created", name=name, id=label.id)
            return label
            
        except Exception as e:
            logger.error("Failed to create label", name=name, error=str(e))
            raise GmailAPIError(f"Failed to create label {name}: {e}")
    
    async def delete_label(self, name: str) -> None:
        """Delete a Gmail label.
        
        Args:
            name: The label name to delete.
        """
        try:
            label = await self.get_label_by_name(name)
            if not label:
                raise GmailAPIError(f"Label not found: {name}")
            
            await self._api_request(
                self.service.users().labels().delete,
                userId='me',
                id=label.id
            )
            
            # Remove from cache
            if name in self.labels_cache:
                del self.labels_cache[name]
            
            logger.info("Label deleted", name=name, id=label.id)
            
        except Exception as e:
            logger.error("Failed to delete label", name=name, error=str(e))
            raise GmailAPIError(f"Failed to delete label {name}: {e}")
    
    # Message Operations
    
    async def list_messages(
        self,
        query: str = "",
        label_ids: Optional[List[str]] = None,
        max_results: int = 100,
        page_token: Optional[str] = None,
        include_spam_trash: bool = False
    ) -> Dict[str, Any]:
        """List Gmail messages with optional filtering.
        
        Args:
            query: Gmail search query.
            label_ids: List of label IDs to filter by.
            max_results: Maximum number of results.
            page_token: Page token for pagination.
            include_spam_trash: Include spam and trash.
            
        Returns:
            Dictionary with messages and pagination info.
        """
        try:
            params = {
                'userId': 'me',
                'maxResults': min(max_results, 500),  # API limit
                'includeSpamTrash': include_spam_trash
            }
            
            if query:
                params['q'] = query
            if label_ids:
                params['labelIds'] = label_ids
            if page_token:
                params['pageToken'] = page_token
            
            response = await self._api_request(
                self.service.users().messages().list,
                quota_cost=5,
                **params
            )
            
            messages = response.get('messages', [])
            next_page_token = response.get('nextPageToken')
            result_size_estimate = response.get('resultSizeEstimate', 0)
            
            logger.debug(
                "Messages listed",
                count=len(messages),
                total_estimate=result_size_estimate,
                has_next_page=bool(next_page_token)
            )
            
            return {
                'messages': messages,
                'nextPageToken': next_page_token,
                'resultSizeEstimate': result_size_estimate
            }
            
        except Exception as e:
            logger.error("Failed to list messages", error=str(e))
            raise GmailAPIError(f"Failed to list messages: {e}")
    
    async def get_message(self, message_id: str, format: str = "full", use_storage: bool = True) -> EmailMessage:
        """Get a Gmail message by ID with optional storage optimization.
        
        Args:
            message_id: The message ID.
            format: Message format (minimal, full, raw, metadata).
            use_storage: Whether to use optimized storage for large emails.
            
        Returns:
            EmailMessage object.
        """
        try:
            # Try to get from storage first if enabled
            if use_storage and self._email_storage and format == "full":
                stored_message = await self._get_message_from_storage(message_id)
                if stored_message:
                    return stored_message
            
            response = await self._api_request(
                self.service.users().messages().get,
                quota_cost=5,
                userId='me',
                id=message_id,
                format=format
            )
            
            # Parse message
            message = self._parse_message(response)
            
            # Store in optimized storage if enabled and message is large
            if (use_storage and self._email_storage and format == "full" and 
                message.size_estimate > 50000):  # 50KB threshold
                await self._store_message_optimized(message)
            
            logger.debug("Message retrieved", message_id=message_id, subject=message.subject[:50])
            return message
            
        except Exception as e:
            logger.error("Failed to get message", message_id=message_id, error=str(e))
            raise GmailAPIError(f"Failed to get message {message_id}: {e}")
    
    def _parse_message(self, msg_data: Dict[str, Any]) -> EmailMessage:
        """Parse Gmail message data into EmailMessage object."""
        message = EmailMessage(
            id=msg_data['id'],
            thread_id=msg_data['threadId'],
            label_ids=msg_data.get('labelIds', []),
            snippet=msg_data.get('snippet', ''),
            history_id=msg_data.get('historyId', ''),
            internal_date=int(msg_data.get('internalDate', 0)),
            size_estimate=msg_data.get('sizeEstimate', 0)
        )
        
        # Parse headers
        payload = msg_data.get('payload', {})
        headers = payload.get('headers', [])
        
        for header in headers:
            name = header['name'].lower()
            value = header['value']
            
            if name == 'subject':
                message.subject = value
            elif name == 'from':
                message.sender = value
            elif name == 'to':
                message.recipient = value
            elif name == 'date':
                message.date = value
        
        # Parse body
        message.body_text, message.body_html = self._extract_body(payload)
        
        # Parse attachments
        message.attachments = self._extract_attachments(payload)
        
        return message
    
    def _extract_body(self, payload: Dict[str, Any]) -> tuple[str, str]:
        """Extract text and HTML body from message payload."""
        text_body = ""
        html_body = ""
        
        def extract_parts(part):
            nonlocal text_body, html_body
            
            mime_type = part.get('mimeType', '')
            
            if mime_type == 'text/plain':
                body = part.get('body', {})
                if 'data' in body:
                    text_body += base64.urlsafe_b64decode(body['data']).decode('utf-8', errors='ignore')
            elif mime_type == 'text/html':
                body = part.get('body', {})
                if 'data' in body:
                    html_body += base64.urlsafe_b64decode(body['data']).decode('utf-8', errors='ignore')
            elif 'parts' in part:
                for subpart in part['parts']:
                    extract_parts(subpart)
        
        extract_parts(payload)
        return text_body, html_body
    
    def _extract_attachments(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract attachment information from message payload."""
        attachments = []
        
        def extract_parts(part):
            filename = ""
            for header in part.get('headers', []):
                if header['name'] == 'Content-Disposition':
                    if 'filename=' in header['value']:
                        filename = header['value'].split('filename=')[1].strip('"')
            
            body = part.get('body', {})
            if body.get('attachmentId'):
                attachments.append({
                    'filename': filename,
                    'mimeType': part.get('mimeType', ''),
                    'size': body.get('size', 0),
                    'attachmentId': body['attachmentId']
                })
            
            if 'parts' in part:
                for subpart in part['parts']:
                    extract_parts(subpart)
        
        extract_parts(payload)
        return attachments
    
    async def modify_labels(
        self,
        message_id: str,
        add_label_ids: Optional[List[str]] = None,
        remove_label_ids: Optional[List[str]] = None
    ) -> EmailMessage:
        """Modify labels on a message.
        
        Args:
            message_id: The message ID.
            add_label_ids: Label IDs to add.
            remove_label_ids: Label IDs to remove.
            
        Returns:
            Updated EmailMessage.
        """
        try:
            body = {}
            if add_label_ids:
                body['addLabelIds'] = add_label_ids
            if remove_label_ids:
                body['removeLabelIds'] = remove_label_ids
            
            response = await self._api_request(
                self.service.users().messages().modify,
                quota_cost=5,
                userId='me',
                id=message_id,
                body=body
            )
            
            # Parse and return updated message
            message = self._parse_message(response)
            
            logger.debug(
                "Labels modified",
                message_id=message_id,
                added=add_label_ids,
                removed=remove_label_ids
            )
            
            return message
            
        except Exception as e:
            logger.error("Failed to modify labels", message_id=message_id, error=str(e))
            raise GmailAPIError(f"Failed to modify labels on message {message_id}: {e}")
    
    async def apply_label_by_name(self, message_id: str, label_name: str, create_if_missing: bool = True) -> EmailMessage:
        """Apply a label to a message by name.
        
        Args:
            message_id: The message ID.
            label_name: The label name.
            create_if_missing: Create label if it doesn't exist.
            
        Returns:
            Updated EmailMessage.
        """
        try:
            label = await self.get_label_by_name(label_name)
            
            if not label:
                if create_if_missing:
                    label = await self.create_label(label_name)
                else:
                    raise GmailAPIError(f"Label not found: {label_name}")
            
            return await self.modify_labels(message_id, add_label_ids=[label.id])
            
        except Exception as e:
            logger.error("Failed to apply label", message_id=message_id, label=label_name, error=str(e))
            raise GmailAPIError(f"Failed to apply label {label_name} to message {message_id}: {e}")
    
    # Batch Operations
    
    async def batch_modify_labels(
        self,
        message_ids: List[str],
        add_label_ids: Optional[List[str]] = None,
        remove_label_ids: Optional[List[str]] = None,
        batch_size: int = 50
    ) -> List[str]:
        """Batch modify labels on multiple messages.
        
        Args:
            message_ids: List of message IDs.
            add_label_ids: Label IDs to add.
            remove_label_ids: Label IDs to remove.
            batch_size: Size of each batch.
            
        Returns:
            List of successfully processed message IDs.
        """
        successful_ids = []
        
        # Process in batches
        for i in range(0, len(message_ids), batch_size):
            batch = message_ids[i:i + batch_size]
            
            try:
                body = {'ids': batch}
                if add_label_ids:
                    body['addLabelIds'] = add_label_ids
                if remove_label_ids:
                    body['removeLabelIds'] = remove_label_ids
                
                await self._api_request(
                    self.service.users().messages().batchModify,
                    quota_cost=len(batch),
                    userId='me',
                    body=body
                )
                
                successful_ids.extend(batch)
                
                logger.debug(
                    "Batch labels modified",
                    batch_size=len(batch),
                    added=add_label_ids,
                    removed=remove_label_ids
                )
                
            except Exception as e:
                logger.error("Batch modify failed", batch_size=len(batch), error=str(e))
                # Continue with next batch
        
        return successful_ids
    
    # Push Notifications
    
    async def setup_push_notifications(self, topic_name: str) -> Dict[str, Any]:
        """Set up Gmail push notifications.
        
        Args:
            topic_name: Cloud Pub/Sub topic name.
            
        Returns:
            Watch response with history ID.
        """
        try:
            request_body = {
                'labelIds': ['INBOX'],
                'topicName': topic_name,
                'labelFilterBehavior': 'INCLUDE'
            }
            
            response = await self._api_request(
                self.service.users().watch,
                userId='me',
                body=request_body
            )
            
            self.watch_response = response
            
            logger.info(
                "Push notifications set up",
                topic=topic_name,
                history_id=response.get('historyId'),
                expiration=response.get('expiration')
            )
            
            return response
            
        except Exception as e:
            logger.error("Failed to set up push notifications", topic=topic_name, error=str(e))
            raise GmailAPIError(f"Failed to set up push notifications: {e}")
    
    async def stop_push_notifications(self) -> None:
        """Stop Gmail push notifications."""
        try:
            await self._api_request(self.service.users().stop, userId='me')
            self.watch_response = None
            logger.info("Push notifications stopped")
            
        except Exception as e:
            logger.error("Failed to stop push notifications", error=str(e))
            raise GmailAPIError(f"Failed to stop push notifications: {e}")
    
    # Utility Methods
    
    async def get_profile(self) -> Dict[str, Any]:
        """Get Gmail profile information.
        
        Returns:
            Dictionary with profile information.
        """
        try:
            response = await self._api_request(
                self.service.users().getProfile,
                userId='me'
            )
            
            logger.debug("Profile retrieved", email=response.get('emailAddress'))
            return response
            
        except Exception as e:
            logger.error("Failed to get profile", error=str(e))
            raise GmailAPIError(f"Failed to get profile: {e}")
    
    async def search_messages(
        self,
        query: str,
        max_results: int = 100
    ) -> AsyncGenerator[EmailMessage, None]:
        """Search messages with pagination support.
        
        Args:
            query: Gmail search query.
            max_results: Maximum total results.
            
        Yields:
            EmailMessage objects.
        """
        page_token = None
        total_retrieved = 0
        
        while total_retrieved < max_results:
            remaining = max_results - total_retrieved
            page_size = min(remaining, 100)  # API limit per page
            
            response = await self.list_messages(
                query=query,
                max_results=page_size,
                page_token=page_token
            )
            
            messages = response.get('messages', [])
            if not messages:
                break
            
            # Fetch full message details
            for msg_info in messages:
                try:
                    message = await self.get_message(msg_info['id'])
                    yield message
                    total_retrieved += 1
                    
                    if total_retrieved >= max_results:
                        break
                        
                except Exception as e:
                    logger.error("Failed to get message details", message_id=msg_info['id'], error=str(e))
                    continue
            
            page_token = response.get('nextPageToken')
            if not page_token:
                break
    
    async def _get_message_from_storage(self, message_id: str) -> Optional[EmailMessage]:
        """Get message from optimized storage."""
        try:
            if not self._email_storage:
                return None
            
            # Get reference
            reference = await self._email_storage.get_email_reference(message_id)
            if not reference:
                return None
            
            # Create lightweight message from reference
            message = EmailMessage(
                id=reference.email_id,
                thread_id=reference.thread_id,
                label_ids=reference.labels,
                subject=reference.subject,
                sender=reference.sender,
                recipient=reference.recipient,
                date=reference.date,
                size_estimate=reference.size_estimate
            )
            
            # Load content on demand (lazy loading)
            content = await self._email_storage.load_email_content(message_id)
            if content:
                message.body_text = content.body_text
                message.body_html = content.body_html
                message.attachments = content.attachments
            
            return message
            
        except Exception as e:
            logger.error("Failed to get message from storage", message_id=message_id, error=str(e))
            return None
    
    async def _store_message_optimized(self, message: EmailMessage) -> None:
        """Store message in optimized storage."""
        try:
            if not self._email_storage:
                return
            
            # Create reference
            reference = EmailReference(
                email_id=message.id,
                thread_id=message.thread_id,
                subject=message.subject,
                sender=message.sender,
                recipient=message.recipient,
                date=message.date,
                labels=message.label_ids,
                size_estimate=message.size_estimate
            )
            
            # Create content
            content = EmailContent(
                email_id=message.id,
                body_text=message.body_text,
                body_html=message.body_html,
                attachments=message.attachments
            )
            
            # Store content and get storage path
            storage_path = await self._email_storage.store_email_content(message.id, content)
            reference.storage_path = storage_path
            
            # Store reference
            await self._email_storage.store_email_reference(reference)
            
            logger.debug("Message stored optimized", message_id=message.id, size=message.size_estimate)
            
        except Exception as e:
            logger.error("Failed to store message optimized", message_id=message.id, error=str(e))
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get Gmail client health status.
        
        Returns:
            Dictionary with health status.
        """
        try:
            # Test authentication
            profile = await self.get_profile()
            
            # Get quota info
            labels = await self.list_labels()
            
            # Get storage stats if available
            storage_stats = {}
            if self._email_storage:
                storage_stats = await self._email_storage.get_storage_stats()
            
            return {
                "status": "healthy",
                "authenticated": True,
                "email_address": profile.get('emailAddress'),
                "total_messages": profile.get('messagesTotal', 0),
                "total_threads": profile.get('threadsTotal', 0),
                "labels_count": len(labels),
                "watch_active": bool(self.watch_response),
                "credentials_valid": self.credentials.valid if self.credentials else False,
                "storage": storage_stats
            }
            
        except Exception as e:
            logger.error("Health check failed", error=str(e))
            return {
                "status": "unhealthy",
                "error": str(e),
                "authenticated": False
            }


# Global Gmail client instance
_gmail_client: Optional[GmailClient] = None


async def get_gmail_client() -> GmailClient:
    """Get the global Gmail client instance."""
    global _gmail_client
    if _gmail_client is None:
        _gmail_client = GmailClient()
        await _gmail_client.initialize()
    return _gmail_client


async def shutdown_gmail_client() -> None:
    """Shutdown the global Gmail client."""
    global _gmail_client
    if _gmail_client:
        if _gmail_client.watch_response:
            await _gmail_client.stop_push_notifications()
        _gmail_client = None