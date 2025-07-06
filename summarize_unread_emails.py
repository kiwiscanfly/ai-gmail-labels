#!/usr/bin/env python3
"""
Email Summarization Script

This script retrieves unread emails from Gmail, summarizes them using Ollama,
and creates markdown files with summaries and key points.
"""

import asyncio
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import structlog

from src.integrations.ollama_client import get_ollama_manager
from src.services.email_service import EmailService
from src.services.email_prioritizer import EmailPrioritizer
from src.models.email import EmailMessage
from src.models.ollama import ModelConfig
from src.core.config import get_config

logger = structlog.get_logger(__name__)


class EmailSummarizer:
    """Email summarization service using Ollama."""
    
    def __init__(self):
        self.email_service = None
        self.ollama_manager = None
        self.prioritizer = EmailPrioritizer()
        self.config = get_config()
        
    async def initialize(self) -> None:
        """Initialize email service and Ollama clients."""
        print("🔄 Initializing email summarizer...")
        
        # Initialize email service
        self.email_service = EmailService()
        await self.email_service.initialize()
        print("✅ Email service connected")
        
        # Initialize Ollama client
        self.ollama_manager = await get_ollama_manager()
        print("✅ Ollama client connected")
        
        # Initialize email prioritizer
        await self.prioritizer.initialize()
        print("✅ Email prioritizer initialized")
        
        # Test Ollama connection
        health = await self.ollama_manager.get_health_status()
        if health.get("status") != "healthy":
            raise Exception("Ollama is not healthy")
        
        print(f"🤖 Using model: {health.get('primary_model', 'unknown')}")
    
    async def get_unread_emails(self, limit: Optional[int] = None) -> List[EmailMessage]:
        """Retrieve unread emails from Gmail."""
        print("📬 Retrieving unread emails...")
        
        unread_emails = []
        
        # Search for unread emails
        async for email_ref in self.email_service.search_emails("is:unread", limit=limit or 100):
            try:
                email = await self.email_service.get_email_content(email_ref.email_id)
                if email:
                    unread_emails.append(email)
                    print(f"   📧 Found: {email.subject[:50]}...")
            except Exception as e:
                logger.error("Failed to retrieve email", email_id=email_ref.email_id, error=str(e))
                continue
        
        print(f"✅ Retrieved {len(unread_emails)} unread emails")
        return unread_emails
    
    async def summarize_email(self, email: EmailMessage) -> Dict[str, Any]:
        """Summarize a single email using Ollama."""
        print(f"🤖 Summarizing: {email.subject[:40]}...")
        
        # Analyze email priority first
        priority_score = await self.prioritizer.analyze_priority(email)
        
        # Extract links from HTML content
        links = []
        if email.body_html:
            links = self._extract_links(email.body_html)
        
        # Extract email content for summarization
        content_parts = []
        
        # Add subject
        content_parts.append(f"Subject: {email.subject}")
        
        # Add sender info
        if email.sender:
            content_parts.append(f"From: {email.sender}")
        
        # Add email body
        if email.body_text:
            # Clean up the text content
            text = email.body_text
            # Remove excessive whitespace and normalize
            text = re.sub(r'\s+', ' ', text).strip()
            # Limit length for summarization
            if len(text) > 4000:
                text = text[:4000] + "..."
            content_parts.append(f"Content: {text}")
        elif email.body_html:
            # If only HTML available, use a portion of it
            html = email.body_html
            # Better HTML to text conversion
            text = self._html_to_text(html)
            text = re.sub(r'\s+', ' ', text).strip()
            if len(text) > 4000:
                text = text[:4000] + "..."
            content_parts.append(f"Content: {text}")
        
        email_text = "\n\n".join(content_parts)
        
        # Create summarization prompt with priority context
        priority_context = f"Priority Assessment: {priority_score.level.title()} ({priority_score.reasoning})"
        
        prompt = f"""Please analyze this email and provide a comprehensive summary. 

Email to analyze:
{email_text}

{priority_context}

Please provide:
1. A concise summary (2-3 sentences) of what this email is about
2. The key points as a bulleted list
3. Any action items or important dates mentioned

Format your response as:

## Summary
[Your 2-3 sentence summary here]

## Key Points
- [Key point 1]
- [Key point 2]
- [Key point 3]
[etc.]

## Action Items
- [Action item 1 if any]
- [Action item 2 if any]
[Or "None identified" if no action items]"""

        try:
            # Configure model for summarization
            model_config = ModelConfig(
                temperature=0.3,  # Lower temperature for more consistent summaries
                num_predict=800,   # Enough for detailed summary
                top_p=0.9
            )
            
            # Get summary from Ollama
            response = await self.ollama_manager.chat(
                messages=[{"role": "user", "content": prompt}],
                options=model_config.to_options()
            )
            
            summary_text = response.content
            
            # Parse the response into structured data
            summary_data = self._parse_summary_response(summary_text)
            
            return {
                "summary": summary_data.get("summary", "Summary not available"),
                "key_points": summary_data.get("key_points", []),
                "action_items": summary_data.get("action_items", []),
                "priority": priority_score.level.title(),
                "priority_score": priority_score.authenticity_score,
                "priority_reasoning": priority_score.reasoning,
                "priority_confidence": priority_score.confidence,
                "priority_factors": {
                    "authenticity": priority_score.authenticity_score,
                    "sender_reputation": priority_score.sender_reputation,
                    "is_genuine": priority_score.is_genuine_urgency
                },
                "links": links,
                "raw_response": summary_text
            }
            
        except Exception as e:
            logger.error("Failed to summarize email", email_id=email.id, error=str(e))
            return {
                "summary": "Failed to generate summary",
                "key_points": ["Error occurred during summarization"],
                "action_items": [],
                "priority": priority_score.level.title(),
                "priority_score": priority_score.authenticity_score,
                "priority_reasoning": priority_score.reasoning,
                "priority_confidence": priority_score.confidence,
                "priority_factors": {
                    "authenticity": priority_score.authenticity_score,
                    "sender_reputation": priority_score.sender_reputation,
                    "is_genuine": priority_score.is_genuine_urgency
                },
                "links": links,
                "raw_response": f"Error: {str(e)}"
            }
    
    def _parse_summary_response(self, response: str) -> Dict[str, Any]:
        """Parse the structured response from Ollama."""
        result = {
            "summary": "",
            "key_points": [],
            "action_items": [],
            "priority": "Medium"
        }
        
        try:
            # Split response into sections
            sections = response.split("##")
            
            for section in sections:
                section = section.strip()
                if not section:
                    continue
                
                if section.lower().startswith("summary"):
                    # Extract summary
                    lines = section.split("\n")
                    summary_lines = [line.strip() for line in lines[1:] if line.strip() and not line.startswith("#")]
                    result["summary"] = " ".join(summary_lines)
                
                elif section.lower().startswith("key points"):
                    # Extract key points
                    lines = section.split("\n")
                    for line in lines[1:]:
                        line = line.strip()
                        if line.startswith("-") or line.startswith("•"):
                            result["key_points"].append(line[1:].strip())
                
                elif section.lower().startswith("action items"):
                    # Extract action items
                    lines = section.split("\n")
                    for line in lines[1:]:
                        line = line.strip()
                        if line.startswith("-") or line.startswith("•"):
                            action = line[1:].strip()
                            if not action.lower().startswith("none"):
                                result["action_items"].append(action)
                
                elif section.lower().startswith("priority"):
                    # Extract priority
                    lines = section.split("\n")
                    if len(lines) > 1:
                        priority_line = lines[1].strip()
                        if any(p in priority_line.lower() for p in ["high", "medium", "low"]):
                            result["priority"] = priority_line
        
        except Exception as e:
            logger.warning("Failed to parse summary response", error=str(e))
        
        return result
    
    def _create_gmail_link(self, email: EmailMessage) -> str:
        """Create a Gmail web link for the email."""
        return f"https://mail.google.com/mail/u/0/#inbox/{email.id}"
    
    def _extract_links(self, html: str) -> List[Dict[str, str]]:
        """Extract links from HTML content with descriptions."""
        links = []
        
        # Find all anchor tags with href attributes
        import re
        link_pattern = r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>(.*?)</a>'
        matches = re.findall(link_pattern, html, re.IGNORECASE | re.DOTALL)
        
        for url, link_text in matches:
            # Clean up the link text
            link_text = re.sub(r'<[^>]+>', '', link_text)  # Remove any HTML tags
            link_text = link_text.strip()
            
            # Skip if it's just whitespace or very short
            if len(link_text) < 2:
                link_text = "Click here"
            
            # Truncate very long link text
            if len(link_text) > 100:
                link_text = link_text[:100] + "..."
            
            # Skip common tracking or empty links
            if any(skip in url.lower() for skip in ['unsubscribe', 'pixel', 'tracking', 'analytics']):
                continue
                
            links.append({
                "url": url,
                "text": link_text,
                "domain": self._get_domain(url)
            })
        
        # Remove duplicates while preserving order
        seen = set()
        unique_links = []
        for link in links:
            link_key = (link["url"], link["text"])
            if link_key not in seen:
                seen.add(link_key)
                unique_links.append(link)
        
        return unique_links[:10]  # Limit to 10 links
    
    def _get_domain(self, url: str) -> str:
        """Extract domain from URL."""
        import re
        domain_match = re.search(r'https?://([^/]+)', url)
        if domain_match:
            domain = domain_match.group(1)
            # Remove www. prefix
            if domain.startswith('www.'):
                domain = domain[4:]
            return domain
        return "unknown"

    def _html_to_text(self, html: str) -> str:
        """Convert HTML to clean text, removing CSS and scripts."""
        # Remove CSS style blocks
        html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove script blocks
        html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove HTML comments
        html = re.sub(r'<!--.*?-->', '', html, flags=re.DOTALL)
        
        # Remove meta, link, and other head elements
        html = re.sub(r'<(meta|link|title|head)[^>]*/?>', '', html, flags=re.IGNORECASE)
        
        # Convert common HTML entities
        html = html.replace('&nbsp;', ' ')
        html = html.replace('&amp;', '&')
        html = html.replace('&lt;', '<')
        html = html.replace('&gt;', '>')
        html = html.replace('&quot;', '"')
        html = html.replace('&#39;', "'")
        
        # Add spaces around block elements to preserve word boundaries
        block_elements = ['div', 'p', 'br', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'tr', 'td', 'th']
        for element in block_elements:
            html = re.sub(f'<{element}[^>]*>', ' ', html, flags=re.IGNORECASE)
            html = re.sub(f'</{element}>', ' ', html, flags=re.IGNORECASE)
        
        # Remove all remaining HTML tags
        html = re.sub(r'<[^>]+>', '', html)
        
        # Clean up whitespace
        html = re.sub(r'\s+', ' ', html)
        html = html.strip()
        
        return html
    
    def _sanitize_filename(self, text: str) -> str:
        """Sanitize text for use as filename."""
        # Remove or replace invalid filename characters
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', text)
        # Limit length
        if len(sanitized) > 100:
            sanitized = sanitized[:100]
        return sanitized
    
    async def create_markdown_file(self, email: EmailMessage, summary_data: Dict[str, Any], output_dir: Path) -> Path:
        """Create a markdown file for the email summary."""
        
        # Create filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_subject = self._sanitize_filename(email.subject)
        filename = f"{timestamp}_{safe_subject}_{email.id[:8]}.md"
        filepath = output_dir / filename
        
        # Format sender information
        sender_info = email.sender or "Unknown Sender"
        
        # Format received date
        received_date = "Unknown"
        if email.received_at:
            try:
                if isinstance(email.received_at, (int, float)):
                    dt = datetime.fromtimestamp(email.received_at)
                else:
                    dt = email.received_at
                received_date = dt.strftime("%B %d, %Y at %I:%M %p")
            except:
                received_date = str(email.received_at)
        
        # Format priority information
        priority_line = f"**{summary_data.get('priority', 'Medium')}**"
        if summary_data.get('priority_reasoning'):
            priority_line += f" - {summary_data.get('priority_reasoning')}"
        
        priority_score = summary_data.get('priority_score', 0)
        confidence = summary_data.get('priority_confidence', 0)
        if priority_score > 0:
            priority_line += f" (Score: {priority_score:.2f}, Confidence: {confidence:.1%})"
        
        # Create markdown content
        markdown_content = f"""# {email.subject}

## Email Details
- **From:** {sender_info}
- **Received:** {received_date}
- **Priority:** {priority_line}
- **Gmail ID:** `{email.id}`
- **Read in Gmail:** [Open Email](https://mail.google.com/mail/u/0/#inbox/{email.id})

---

## Summary
{summary_data.get('summary', 'No summary available')}

## Key Points
"""
        
        # Add key points
        key_points = summary_data.get('key_points', [])
        if key_points:
            for point in key_points:
                markdown_content += f"- {point}\n"
        else:
            markdown_content += "- No key points identified\n"
        
        # Add action items
        markdown_content += "\n## Action Items\n"
        action_items = summary_data.get('action_items', [])
        if action_items:
            for item in action_items:
                markdown_content += f"- [ ] {item}\n"
        else:
            markdown_content += "- No action items identified\n"
        
        # Add links section
        links = summary_data.get('links', [])
        if links:
            markdown_content += "\n## Links and Resources\n"
            for link in links:
                domain = link.get('domain', 'unknown')
                text = link.get('text', 'Link')
                url = link.get('url', '#')
                markdown_content += f"- **[{text}]({url})** - {domain}\n"
        
        # Add labels if any
        if email.label_ids:
            markdown_content += f"\n## Gmail Labels\n"
            for label in email.label_ids:
                markdown_content += f"- {label}\n"
        
        # Add priority analysis section
        priority_factors = summary_data.get('priority_factors', {})
        if priority_factors:
            markdown_content += "\n## Priority Analysis\n"
            for factor, score in priority_factors.items():
                bar_length = int(score * 10)
                bar = "█" * bar_length + "░" * (10 - bar_length)
                markdown_content += f"- **{factor.title()}**: {score:.2f} `{bar}`\n"
        
        # Add metadata
        markdown_content += f"""
---

## Metadata
- **Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Email ID:** {email.id}
- **Thread ID:** {email.thread_id or 'N/A'}
- **Has Attachments:** {'Yes' if email.attachments else 'No'}

"""
        
        # Write the file
        filepath.write_text(markdown_content, encoding='utf-8')
        
        return filepath
    
    async def process_unread_emails(self, output_dir: str = "email_summaries", limit: Optional[int] = None) -> None:
        """Main process to summarize all unread emails."""
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"📁 Output directory: {output_path.absolute()}")
        
        # Get unread emails
        unread_emails = await self.get_unread_emails(limit)
        
        if not unread_emails:
            print("📭 No unread emails found!")
            return
        
        print(f"🔄 Processing {len(unread_emails)} unread emails...")
        
        processed_count = 0
        failed_count = 0
        
        for i, email in enumerate(unread_emails, 1):
            try:
                print(f"\n[{i}/{len(unread_emails)}] Processing: {email.subject[:50]}...")
                
                # Summarize the email
                summary_data = await self.summarize_email(email)
                
                # Create markdown file
                filepath = await self.create_markdown_file(email, summary_data, output_path)
                
                print(f"   ✅ Created: {filepath.name}")
                processed_count += 1
                
                # Small delay to avoid overwhelming the API
                await asyncio.sleep(1)
                
            except Exception as e:
                print(f"   ❌ Failed: {str(e)}")
                logger.error("Failed to process email", email_id=email.id, error=str(e))
                failed_count += 1
                continue
        
        print(f"\n🎉 Processing complete!")
        print(f"   ✅ Successfully processed: {processed_count}")
        print(f"   ❌ Failed: {failed_count}")
        print(f"   📁 Files saved to: {output_path.absolute()}")
    
    async def shutdown(self) -> None:
        """Shutdown the email summarizer and clean up resources."""
        try:
            if self.ollama_manager:
                await self.ollama_manager.shutdown()
            if self.email_service:
                # The email service doesn't have a shutdown method, but we can clean up the storage
                pass
            
            # Clean up database connections
            try:
                from src.core.database_pool import shutdown_database_pool
                await shutdown_database_pool()
                print("🔄 Database connections closed")
            except:
                pass
                
            print("🔄 Cleanup completed")
            
            # Cancel any remaining tasks
            tasks = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
            if tasks:
                print(f"🔄 Cancelling {len(tasks)} remaining tasks")
                for i, task in enumerate(tasks):
                    print(f"   Task {i+1}: {task.get_name() if hasattr(task, 'get_name') else 'unnamed'}")
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                print("🔄 All tasks cancelled")
                
        except Exception as e:
            logger.error("Error during shutdown", error=str(e))


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Summarize unread Gmail emails")
    parser.add_argument("--limit", type=int, help="Limit number of emails to process")
    parser.add_argument("--output", default="email_summaries", help="Output directory for markdown files")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print(" 📧 Email Summarization Tool")
    print("=" * 60)
    
    summarizer = None
    try:
        summarizer = EmailSummarizer()
        await summarizer.initialize()
        
        await summarizer.process_unread_emails(
            output_dir=args.output,
            limit=args.limit
        )
        
    except KeyboardInterrupt:
        print("\n⏹️  Process interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        logger.error("Script failed", error=str(e))
    finally:
        if summarizer:
            await summarizer.shutdown()


if __name__ == "__main__":
    asyncio.run(main())