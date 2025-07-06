#!/usr/bin/env python3
"""
Marketing Email Labeling Script

This script analyzes unread emails using the LLM-based MarketingEmailClassifier
and applies Gmail labels based on marketing classification:
- Marketing/Promotional
- Marketing/Newsletter
- Marketing/Hybrid
- Marketing/Transactional
- Marketing/Personal (for non-marketing emails)
"""

import asyncio
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import structlog

from src.integrations.ollama_client import get_ollama_manager
from src.services.email_service import EmailService
from src.services.marketing_classifier import MarketingEmailClassifier
from src.models.email import EmailMessage, EmailCategory
from src.core.config import get_config

logger = structlog.get_logger(__name__)


class MarketingEmailLabeler:
    """Service for labeling emails with marketing-based Gmail labels."""
    
    def __init__(self):
        self.email_service = None
        self.marketing_classifier = MarketingEmailClassifier()
        self.config = get_config()
        
        # Marketing label mapping - only for actual marketing emails
        self.marketing_labels = {
            "promotional": "Marketing/Promotional",
            "newsletter": "Marketing/Newsletter", 
            "hybrid": "Marketing/Hybrid",
            "general": "Marketing/General"
        }
        
        # Non-marketing emails won't get marketing labels
        self.skip_label_subtypes = ["personal", "transactional"]
        
    async def initialize(self) -> None:
        """Initialize email service and marketing classifier."""
        print("🔄 Initializing marketing email labeler...")
        
        # Initialize email service
        self.email_service = EmailService()
        await self.email_service.initialize()
        print("✅ Email service connected")
        
        # Initialize marketing classifier
        await self.marketing_classifier.initialize()
        print("✅ Marketing classifier initialized")
        
        # Ensure marketing labels exist
        await self._ensure_marketing_labels_exist()
        print("✅ Marketing labels verified")
    
    async def _ensure_marketing_labels_exist(self) -> None:
        """Ensure all marketing labels exist in Gmail with proper nesting."""
        print("🏷️  Checking marketing labels...")
        
        try:
            # Get existing labels
            existing_labels = await self.email_service.get_labels()
            existing_label_names = {label.name for label in existing_labels}
            
            # First ensure the parent "Marketing" label exists
            parent_label = "Marketing"
            if parent_label not in existing_label_names:
                print(f"   Creating parent label: {parent_label}")
                await self.email_service.create_label(name=parent_label)
                existing_label_names.add(parent_label)
            else:
                print(f"   ✓ Parent label exists: {parent_label}")
            
            # Create missing marketing sublabels (only for marketing types)
            for subtype, label_name in self.marketing_labels.items():
                if label_name not in existing_label_names:
                    print(f"   Creating nested label: {label_name}")
                    await self.email_service.create_label(name=label_name)
                else:
                    print(f"   ✓ Nested label exists: {label_name}")
                    
        except Exception as e:
            logger.error("Failed to ensure marketing labels exist", error=str(e))
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
        """Analyze email marketing classification and apply appropriate label."""
        try:
            # Analyze marketing classification
            marketing_result = await self.marketing_classifier.classify_email(email)
            
            # Only apply marketing labels to actual marketing emails
            if marketing_result.is_marketing and marketing_result.subtype not in self.skip_label_subtypes:
                # Get the corresponding label name
                label_name = self.marketing_labels.get(marketing_result.subtype, "Marketing/General")
                
                # Apply the label to the email
                category = EmailCategory(
                    email_id=email.id,
                    suggested_labels=[label_name],
                    confidence_scores={label_name: marketing_result.confidence},
                    reasoning=marketing_result.reasoning
                )
                await self.email_service.apply_category(email.id, category, create_labels=True)
                
                print(f"   🏷️  Applied {label_name} to: {email.subject[:40]}...")
            else:
                # Non-marketing email - no label applied
                label_name = None
                print(f"   ✉️  Non-marketing email: {email.subject[:40]}... (no label applied)")
            
            return {
                "email_id": email.id,
                "subject": email.subject,
                "is_marketing": marketing_result.is_marketing,
                "marketing_subtype": marketing_result.subtype,
                "confidence": marketing_result.confidence,
                "reasoning": marketing_result.reasoning,
                "label_applied": label_name,
                "marketing_indicators": marketing_result.marketing_indicators,
                "unsubscribe_detected": marketing_result.unsubscribe_detected,
                "bulk_sending_indicators": marketing_result.bulk_sending_indicators,
                "sender_reputation": marketing_result.sender_reputation
            }
            
        except Exception as e:
            logger.error("Failed to analyze and label email", email_id=email.id, error=str(e))
            return {
                "email_id": email.id,
                "subject": email.subject,
                "is_marketing": False,
                "marketing_subtype": "error",
                "confidence": 0.0,
                "reasoning": f"Error: {str(e)}",
                "label_applied": None,
                "error": str(e)
            }
    
    async def process_unread_emails(self, limit: Optional[int] = None, dry_run: bool = False) -> Dict[str, Any]:
        """Process all unread emails and apply marketing labels."""
        
        # Get unread emails
        unread_emails = await self.get_unread_emails(limit)
        
        if not unread_emails:
            print("📭 No unread emails found!")
            return {"processed": 0, "results": []}
        
        print(f"🔄 Processing {len(unread_emails)} unread emails...")
        if dry_run:
            print("🔍 DRY RUN MODE - No labels will be applied")
        
        results = []
        marketing_counts = {
            "promotional": 0, "newsletter": 0, "hybrid": 0, 
            "transactional": 0, "personal": 0, "general": 0, "error": 0
        }
        
        for i, email in enumerate(unread_emails, 1):
            try:
                print(f"\n[{i}/{len(unread_emails)}] Processing: {email.subject[:50]}...")
                
                if dry_run:
                    # In dry run mode, only analyze without applying labels
                    marketing_result = await self.marketing_classifier.classify_email(email)
                    
                    # Determine what label would be applied
                    if marketing_result.is_marketing and marketing_result.subtype not in self.skip_label_subtypes:
                        label_name = self.marketing_labels.get(marketing_result.subtype, "Marketing/General")
                        print(f"   🔍 Would apply {label_name} (confidence: {marketing_result.confidence:.1%})")
                    else:
                        label_name = None
                        print(f"   🔍 Non-marketing email - no label would be applied (confidence: {marketing_result.confidence:.1%})")
                    
                    result = {
                        "email_id": email.id,
                        "subject": email.subject,
                        "is_marketing": marketing_result.is_marketing,
                        "marketing_subtype": marketing_result.subtype,
                        "confidence": marketing_result.confidence,
                        "reasoning": marketing_result.reasoning,
                        "label_would_apply": label_name,
                        "marketing_indicators": marketing_result.marketing_indicators,
                        "unsubscribe_detected": marketing_result.unsubscribe_detected,
                        "bulk_sending_indicators": marketing_result.bulk_sending_indicators,
                        "sender_reputation": marketing_result.sender_reputation
                    }
                    
                    if marketing_result.is_marketing and marketing_result.marketing_indicators:
                        print(f"       Marketing indicators: {', '.join(marketing_result.marketing_indicators)}")
                else:
                    # Normal mode - analyze and apply labels
                    result = await self.analyze_and_label_email(email)
                
                results.append(result)
                marketing_counts[result.get("marketing_subtype", "error")] += 1
                
                # Small delay to avoid overwhelming the API
                await asyncio.sleep(0.5)
                
            except Exception as e:
                print(f"   ❌ Failed: {str(e)}")
                logger.error("Failed to process email", email_id=email.id, error=str(e))
                marketing_counts["error"] += 1
                continue
        
        # Print summary
        print(f"\n🎉 Processing complete!")
        print(f"   📊 Marketing Classification Distribution:")
        for subtype, count in marketing_counts.items():
            if count > 0:
                print(f"      {subtype.title()}: {count}")
        
        return {
            "processed": len(results),
            "marketing_counts": marketing_counts,
            "results": results
        }
    
    def print_detailed_results(self, results: Dict[str, Any]) -> None:
        """Print detailed results with marketing analysis."""
        print(f"\n📋 Detailed Marketing Analysis:")
        print("=" * 80)
        
        for result in results.get("results", []):
            print(f"\nSubject: {result.get('subject', 'Unknown')[:60]}...")
            print(f"Marketing Type: {result.get('marketing_subtype', 'unknown').title()}")
            print(f"Is Marketing: {'Yes' if result.get('is_marketing') else 'No'}")
            print(f"Confidence: {result.get('confidence', 0):.1%}")
            print(f"Label: {result.get('label_applied') or result.get('label_would_apply', 'None')}")
            
            if result.get('reasoning'):
                print(f"Reasoning: {result.get('reasoning')}")
            
            if result.get('marketing_indicators'):
                print(f"Marketing Indicators: {', '.join(result.get('marketing_indicators', []))}")
            
            if result.get('unsubscribe_detected'):
                print("📧 Unsubscribe Link Detected")
            
            if result.get('bulk_sending_indicators'):
                print("📤 Bulk Sending Indicators Found")
            
            sender_rep = result.get('sender_reputation', 0)
            if sender_rep > 0:
                print(f"📊 Sender Reputation: {sender_rep:.1%}")
            
            print("-" * 80)
    
    def print_sender_statistics(self) -> None:
        """Print sender marketing statistics."""
        stats = self.marketing_classifier.get_sender_statistics()
        
        print(f"\n📈 Sender Statistics:")
        print(f"   Total Senders: {stats.get('total_senders', 0)}")
        print(f"   Marketing Senders: {stats.get('marketing_senders', 0)}")
        print(f"   Individual Senders: {stats.get('individual_senders', 0)}")
        print(f"   Average Marketing Rate: {stats.get('average_marketing_rate', 0):.1%}")
    
    async def shutdown(self) -> None:
        """Shutdown the labeler and clean up resources."""
        try:
            print("🔄 Shutting down marketing email labeler...")
            
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
    parser = argparse.ArgumentParser(description="Label unread Gmail emails with marketing classification")
    parser.add_argument("--limit", type=int, help="Limit number of emails to process")
    parser.add_argument("--dry-run", action="store_true", help="Analyze emails without applying labels")
    parser.add_argument("--detailed", action="store_true", help="Show detailed results")
    parser.add_argument("--sender-stats", action="store_true", help="Show sender statistics")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print(" 🏷️  Marketing Email Labeling Tool")
    print("=" * 60)
    
    labeler = None
    try:
        labeler = MarketingEmailLabeler()
        await labeler.initialize()
        
        # Process emails
        results = await labeler.process_unread_emails(
            limit=args.limit,
            dry_run=args.dry_run
        )
        
        # Show detailed results if requested
        if args.detailed:
            labeler.print_detailed_results(results)
        
        # Show sender statistics if requested
        if args.sender_stats:
            labeler.print_sender_statistics()
        
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