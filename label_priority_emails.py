#!/usr/bin/env python3
"""
Email Priority Labeling Script

This script analyzes unread emails using the LLM-based EmailPrioritizer
and applies Gmail labels based on priority levels using nested labels:
- Priority/Critical
- Priority/High  
- Priority/Medium
- Priority/Low
"""

import asyncio
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import structlog

from src.integrations.ollama_client import get_ollama_manager
from src.services.email_service import EmailService
from src.services.email_prioritizer import EmailPrioritizer
from src.models.email import EmailMessage
from src.core.config import get_config

logger = structlog.get_logger(__name__)


class EmailPriorityLabeler:
    """Service for labeling emails with priority-based Gmail labels."""
    
    def __init__(self):
        self.email_service = None
        self.prioritizer = EmailPrioritizer()
        self.config = get_config()
        
        # Priority label mapping
        self.priority_labels = {
            "critical": "Priority/Critical",
            "high": "Priority/High", 
            "medium": "Priority/Medium",
            "low": "Priority/Low"
        }
        
    async def initialize(self) -> None:
        """Initialize email service and prioritizer."""
        print("🔄 Initializing priority labeler...")
        
        # Initialize email service
        self.email_service = EmailService()
        await self.email_service.initialize()
        print("✅ Email service connected")
        
        # Initialize email prioritizer
        await self.prioritizer.initialize()
        print("✅ Email prioritizer initialized")
        
        # Ensure priority labels exist
        await self._ensure_priority_labels_exist()
        print("✅ Priority labels verified")
    
    async def _ensure_priority_labels_exist(self) -> None:
        """Ensure all priority labels exist in Gmail with proper nesting."""
        print("🏷️  Checking priority labels...")
        
        try:
            # Get existing labels
            existing_labels = await self.email_service.get_labels()
            existing_label_names = {label.name for label in existing_labels}
            
            # First ensure the parent "Priority" label exists
            parent_label = "Priority"
            if parent_label not in existing_label_names:
                print(f"   Creating parent label: {parent_label}")
                await self.email_service.create_label(name=parent_label)
                existing_label_names.add(parent_label)
            else:
                print(f"   ✓ Parent label exists: {parent_label}")
            
            # Create missing priority sublabels
            for priority_level, label_name in self.priority_labels.items():
                if label_name not in existing_label_names:
                    print(f"   Creating nested label: {label_name}")
                    await self.email_service.create_label(name=label_name)
                else:
                    print(f"   ✓ Nested label exists: {label_name}")
                    
        except Exception as e:
            logger.error("Failed to ensure priority labels exist", error=str(e))
            raise
    
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
    
    async def analyze_and_label_email(self, email: EmailMessage) -> Dict[str, Any]:
        """Analyze email priority and apply appropriate label."""
        try:
            # Analyze email priority
            priority_score = await self.prioritizer.analyze_priority(email)
            
            # Get the corresponding label name
            label_name = self.priority_labels.get(priority_score.level, "Priority/Medium")
            
            # Apply the label to the email using apply_category
            from src.models.email import EmailCategory
            category = EmailCategory(
                email_id=email.id,
                suggested_labels=[label_name],
                confidence_scores={label_name: priority_score.confidence},
                reasoning=priority_score.reasoning
            )
            await self.email_service.apply_category(email.id, category, create_labels=True)
            
            print(f"   🏷️  Applied {label_name} to: {email.subject[:40]}...")
            
            return {
                "email_id": email.id,
                "subject": email.subject,
                "priority_level": priority_score.level,
                "confidence": priority_score.confidence,
                "reasoning": priority_score.reasoning,
                "label_applied": label_name,
                "is_genuine_urgency": priority_score.is_genuine_urgency,
                "sender_reputation": priority_score.sender_reputation,
                "needs_review": priority_score.needs_review,
                "detected_tactics": priority_score.detected_tactics
            }
            
        except Exception as e:
            logger.error("Failed to analyze and label email", email_id=email.id, error=str(e))
            return {
                "email_id": email.id,
                "subject": email.subject,
                "priority_level": "error",
                "confidence": 0.0,
                "reasoning": f"Error: {str(e)}",
                "label_applied": None,
                "error": str(e)
            }
    
    async def process_unread_emails(self, limit: Optional[int] = None, dry_run: bool = False) -> Dict[str, Any]:
        """Process all unread emails and apply priority labels."""
        
        # Get unread emails
        unread_emails = await self.get_unread_emails(limit)
        
        if not unread_emails:
            print("📭 No unread emails found!")
            return {"processed": 0, "results": []}
        
        print(f"🔄 Processing {len(unread_emails)} unread emails...")
        if dry_run:
            print("🔍 DRY RUN MODE - No labels will be applied")
        
        results = []
        priority_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0, "error": 0}
        
        for i, email in enumerate(unread_emails, 1):
            try:
                print(f"\n[{i}/{len(unread_emails)}] Processing: {email.subject[:50]}...")
                
                if dry_run:
                    # In dry run mode, only analyze without applying labels
                    priority_score = await self.prioritizer.analyze_priority(email)
                    label_name = self.priority_labels.get(priority_score.level, "Priority/Medium")
                    
                    result = {
                        "email_id": email.id,
                        "subject": email.subject,
                        "priority_level": priority_score.level,
                        "confidence": priority_score.confidence,
                        "reasoning": priority_score.reasoning,
                        "label_would_apply": label_name,
                        "is_genuine_urgency": priority_score.is_genuine_urgency,
                        "sender_reputation": priority_score.sender_reputation,
                        "needs_review": priority_score.needs_review,
                        "detected_tactics": priority_score.detected_tactics
                    }
                    print(f"   🔍 Would apply {label_name} (confidence: {priority_score.confidence:.1%})")
                else:
                    # Normal mode - analyze and apply labels
                    result = await self.analyze_and_label_email(email)
                
                results.append(result)
                priority_counts[result.get("priority_level", "error")] += 1
                
                # Small delay to avoid overwhelming the API
                await asyncio.sleep(0.5)
                
            except Exception as e:
                print(f"   ❌ Failed: {str(e)}")
                logger.error("Failed to process email", email_id=email.id, error=str(e))
                priority_counts["error"] += 1
                continue
        
        # Print summary
        print(f"\n🎉 Processing complete!")
        print(f"   📊 Priority Distribution:")
        for level, count in priority_counts.items():
            if count > 0:
                print(f"      {level.title()}: {count}")
        
        return {
            "processed": len(results),
            "priority_counts": priority_counts,
            "results": results
        }
    
    def print_detailed_results(self, results: Dict[str, Any]) -> None:
        """Print detailed results with priority analysis."""
        print(f"\n📋 Detailed Results:")
        print("=" * 80)
        
        for result in results.get("results", []):
            print(f"\nSubject: {result.get('subject', 'Unknown')[:60]}...")
            print(f"Priority: {result.get('priority_level', 'unknown').title()}")
            print(f"Confidence: {result.get('confidence', 0):.1%}")
            print(f"Label: {result.get('label_applied') or result.get('label_would_apply', 'None')}")
            
            if result.get('reasoning'):
                print(f"Reasoning: {result.get('reasoning')}")
            
            if result.get('detected_tactics'):
                print(f"Marketing Tactics: {', '.join(result.get('detected_tactics', []))}")
            
            if result.get('needs_review'):
                print("⚠️  Needs Review: Low confidence classification")
            
            if not result.get('is_genuine_urgency', True):
                print("🚨 Marketing Manipulation Detected")
            
            print("-" * 80)
    
    async def shutdown(self) -> None:
        """Shutdown the labeler and clean up resources."""
        try:
            print("🔄 Shutting down priority labeler...")
            
            # Clean up database connections
            try:
                from src.core.database_pool import shutdown_database_pool
                await shutdown_database_pool()
                print("🔄 Database connections closed")
            except:
                pass
                
            print("🔄 Cleanup completed")
            
        except Exception as e:
            logger.error("Error during shutdown", error=str(e))


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Label unread Gmail emails with priority")
    parser.add_argument("--limit", type=int, help="Limit number of emails to process")
    parser.add_argument("--dry-run", action="store_true", help="Analyze emails without applying labels")
    parser.add_argument("--detailed", action="store_true", help="Show detailed results")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print(" 🏷️  Email Priority Labeling Tool")
    print("=" * 60)
    
    labeler = None
    try:
        labeler = EmailPriorityLabeler()
        await labeler.initialize()
        
        # Process emails
        results = await labeler.process_unread_emails(
            limit=args.limit,
            dry_run=args.dry_run
        )
        
        # Show detailed results if requested
        if args.detailed:
            labeler.print_detailed_results(results)
        
    except KeyboardInterrupt:
        print("\n⏹️  Process interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        logger.error("Script failed", error=str(e))
    finally:
        if labeler:
            await labeler.shutdown()


if __name__ == "__main__":
    asyncio.run(main())