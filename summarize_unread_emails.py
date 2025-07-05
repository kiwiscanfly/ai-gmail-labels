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
from src.models.email import EmailMessage
from src.models.ollama import ModelConfig
from src.core.config import get_config

logger = structlog.get_logger(__name__)


class EmailSummarizer:
    """Email summarization service using Ollama."""
    
    def __init__(self):
        self.email_service = None
        self.ollama_manager = None
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
            # Simple HTML to text conversion
            text = re.sub(r'<[^>]+>', ' ', html)
            text = re.sub(r'\s+', ' ', text).strip()
            if len(text) > 4000:
                text = text[:4000] + "..."
            content_parts.append(f"Content: {text}")
        
        email_text = "\n\n".join(content_parts)
        
        # Create summarization prompt
        prompt = f"""Please analyze this email and provide a comprehensive summary. 

Email to analyze:
{email_text}

Please provide:
1. A concise summary (2-3 sentences) of what this email is about
2. The key points as a bulleted list
3. Any action items or important dates mentioned
4. The overall tone/urgency level (low/medium/high)

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
[Or "None identified" if no action items]

## Priority Level
[Low/Medium/High] - [Brief reason]"""

        try:
            # Configure model for summarization
            model_config = ModelConfig(
                temperature=0.3,  # Lower temperature for more consistent summaries
                max_tokens=800,   # Enough for detailed summary
                top_p=0.9
            )
            
            # Get summary from Ollama
            response = await self.ollama_manager.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model_config=model_config
            )
            
            summary_text = response.get("message", {}).get("content", "")
            
            # Parse the response into structured data
            summary_data = self._parse_summary_response(summary_text)
            
            return {
                "summary": summary_data.get("summary", "Summary not available"),
                "key_points": summary_data.get("key_points", []),
                "action_items": summary_data.get("action_items", []),
                "priority": summary_data.get("priority", "Medium"),
                "raw_response": summary_text
            }
            
        except Exception as e:
            logger.error("Failed to summarize email", email_id=email.id, error=str(e))
            return {
                "summary": "Failed to generate summary",
                "key_points": ["Error occurred during summarization"],
                "action_items": [],
                "priority": "Unknown",
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
        
        # Create markdown content
        markdown_content = f"""# {email.subject}

## Email Details
- **From:** {sender_info}
- **Received:** {received_date}
- **Priority:** {summary_data.get('priority', 'Medium')}
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
        
        # Add labels if any
        if email.label_ids:
            markdown_content += f"\n## Gmail Labels\n"
            for label in email.label_ids:
                markdown_content += f"- {label}\n"
        
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


if __name__ == "__main__":
    asyncio.run(main())