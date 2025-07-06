#!/usr/bin/env python3
"""
Receipt Email Labeling Script

This script analyzes unread emails using the LLM-based ReceiptClassifier
and applies Gmail labels based on receipt classification:
- Receipts/Purchase
- Receipts/Subscription
- Receipts/Service
- Receipts/Refund
- Receipts/Donation
- Receipts/Other
"""

import asyncio
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import structlog

from src.integrations.ollama_client import get_ollama_manager
from src.services.email_service import EmailService
from src.services.receipt_classifier import ReceiptClassifier
from src.models.email import EmailMessage, EmailCategory
from src.core.config import get_config

logger = structlog.get_logger(__name__)


class ReceiptEmailLabeler:
    """Service for labeling emails with receipt-based Gmail labels."""
    
    def __init__(self):
        self.email_service = None
        self.receipt_classifier = ReceiptClassifier()
        self.config = get_config()
        
        # Receipt label mapping - only for actual receipts
        self.receipt_labels = {
            "purchase": "Receipts/Purchase",
            "subscription": "Receipts/Subscription",
            "service": "Receipts/Service",
            "refund": "Receipts/Refund",
            "donation": "Receipts/Donation",
            "other": "Receipts/Other"
        }
        
    async def initialize(self) -> None:
        """Initialize email service and receipt classifier."""
        print("🔄 Initializing receipt email labeler...")
        
        # Initialize email service
        self.email_service = EmailService()
        await self.email_service.initialize()
        print("✅ Email service connected")
        
        # Initialize receipt classifier
        await self.receipt_classifier.initialize()
        print("✅ Receipt classifier initialized")
        
        # Ensure receipt labels exist
        await self._ensure_receipt_labels_exist()
        print("✅ Receipt labels verified")
    
    async def _ensure_receipt_labels_exist(self) -> None:
        """Ensure all receipt labels exist in Gmail with proper nesting."""
        print("🏷️  Checking receipt labels...")
        
        try:
            # Get existing labels
            existing_labels = await self.email_service.get_labels()
            existing_label_names = {label.name for label in existing_labels}
            
            # First ensure the parent "Receipts" label exists
            parent_label = "Receipts"
            if parent_label not in existing_label_names:
                print(f"   Creating parent label: {parent_label}")
                await self.email_service.create_label(name=parent_label)
                existing_label_names.add(parent_label)
            else:
                print(f"   ✓ Parent label exists: {parent_label}")
            
            # Create missing receipt sublabels
            for receipt_type, label_name in self.receipt_labels.items():
                if label_name not in existing_label_names:
                    print(f"   Creating nested label: {label_name}")
                    await self.email_service.create_label(name=label_name)
                else:
                    print(f"   ✓ Nested label exists: {label_name}")
                    
        except Exception as e:
            logger.error("Failed to ensure receipt labels exist", error=str(e))
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
        """Analyze email receipt classification and apply appropriate label."""
        try:
            # Analyze receipt classification
            receipt_result = await self.receipt_classifier.classify_receipt(email)
            
            # Only apply receipt labels to actual receipts
            if receipt_result.is_receipt:
                # Get the corresponding label name
                label_name = self.receipt_labels.get(receipt_result.receipt_type, "Receipts/Other")
                
                # Apply the label to the email
                category = EmailCategory(
                    email_id=email.id,
                    suggested_labels=[label_name],
                    confidence_scores={label_name: receipt_result.confidence},
                    reasoning=receipt_result.reasoning
                )
                await self.email_service.apply_category(email.id, category, create_labels=True)
                
                print(f"   🧾 Applied {label_name} to: {email.subject[:40]}...")
                
                # Print receipt details if available
                if receipt_result.vendor:
                    print(f"      Vendor: {receipt_result.vendor}")
                if receipt_result.amount:
                    print(f"      Amount: {receipt_result.amount}")
                if receipt_result.order_number:
                    print(f"      Order #: {receipt_result.order_number}")
            else:
                # Non-receipt email - no label applied
                label_name = None
                print(f"   📧 Non-receipt email: {email.subject[:40]}... (no label applied)")
            
            return {
                "email_id": email.id,
                "subject": email.subject,
                "is_receipt": receipt_result.is_receipt,
                "receipt_type": receipt_result.receipt_type,
                "confidence": receipt_result.confidence,
                "reasoning": receipt_result.reasoning,
                "label_applied": label_name,
                "vendor": receipt_result.vendor,
                "amount": receipt_result.amount,
                "currency": receipt_result.currency,
                "order_number": receipt_result.order_number,
                "transaction_date": receipt_result.transaction_date,
                "payment_method": receipt_result.payment_method,
                "receipt_indicators": receipt_result.receipt_indicators
            }
            
        except Exception as e:
            logger.error("Failed to analyze and label email", email_id=email.id, error=str(e))
            return {
                "email_id": email.id,
                "subject": email.subject,
                "is_receipt": False,
                "receipt_type": "error",
                "confidence": 0.0,
                "reasoning": f"Error: {str(e)}",
                "label_applied": None,
                "error": str(e)
            }
    
    async def process_unread_emails(self, limit: Optional[int] = None, dry_run: bool = False) -> Dict[str, Any]:
        """Process all unread emails and apply receipt labels."""
        
        # Get unread emails
        unread_emails = await self.get_unread_emails(limit)
        
        if not unread_emails:
            print("📭 No unread emails found!")
            return {"processed": 0, "results": []}
        
        print(f"🔄 Processing {len(unread_emails)} unread emails...")
        if dry_run:
            print("🔍 DRY RUN MODE - No labels will be applied")
        
        results = []
        receipt_counts = {
            "purchase": 0, "subscription": 0, "service": 0, 
            "refund": 0, "donation": 0, "other": 0, "non_receipt": 0, "error": 0
        }
        
        for i, email in enumerate(unread_emails, 1):
            try:
                print(f"\n[{i}/{len(unread_emails)}] Processing: {email.subject[:50]}...")
                
                if dry_run:
                    # In dry run mode, only analyze without applying labels
                    receipt_result = await self.receipt_classifier.classify_receipt(email)
                    
                    # Determine what label would be applied
                    if receipt_result.is_receipt:
                        label_name = self.receipt_labels.get(receipt_result.receipt_type, "Receipts/Other")
                        print(f"   🔍 Would apply {label_name} (confidence: {receipt_result.confidence:.1%})")
                        
                        # Print receipt details
                        if receipt_result.vendor:
                            print(f"      Vendor: {receipt_result.vendor}")
                        if receipt_result.amount:
                            print(f"      Amount: {receipt_result.amount}")
                        if receipt_result.order_number:
                            print(f"      Order #: {receipt_result.order_number}")
                    else:
                        label_name = None
                        print(f"   🔍 Non-receipt email - no label would be applied (confidence: {receipt_result.confidence:.1%})")
                    
                    result = {
                        "email_id": email.id,
                        "subject": email.subject,
                        "is_receipt": receipt_result.is_receipt,
                        "receipt_type": receipt_result.receipt_type,
                        "confidence": receipt_result.confidence,
                        "reasoning": receipt_result.reasoning,
                        "label_would_apply": label_name,
                        "vendor": receipt_result.vendor,
                        "amount": receipt_result.amount,
                        "currency": receipt_result.currency,
                        "order_number": receipt_result.order_number,
                        "transaction_date": receipt_result.transaction_date,
                        "payment_method": receipt_result.payment_method,
                        "receipt_indicators": receipt_result.receipt_indicators
                    }
                    
                    if receipt_result.is_receipt and receipt_result.receipt_indicators:
                        print(f"       Receipt indicators: {', '.join(receipt_result.receipt_indicators)}")
                else:
                    # Normal mode - analyze and apply labels
                    result = await self.analyze_and_label_email(email)
                
                results.append(result)
                
                # Update counts
                if result.get("is_receipt"):
                    receipt_counts[result.get("receipt_type", "other")] += 1
                else:
                    receipt_counts["non_receipt"] += 1
                
                # Small delay to avoid overwhelming the API
                await asyncio.sleep(0.5)
                
            except Exception as e:
                print(f"   ❌ Failed: {str(e)}")
                logger.error("Failed to process email", email_id=email.id, error=str(e))
                receipt_counts["error"] += 1
                continue
        
        # Print summary
        print(f"\n🎉 Processing complete!")
        print(f"   📊 Receipt Classification Distribution:")
        for receipt_type, count in receipt_counts.items():
            if count > 0:
                print(f"      {receipt_type.replace('_', ' ').title()}: {count}")
        
        return {
            "processed": len(results),
            "receipt_counts": receipt_counts,
            "results": results
        }
    
    def print_detailed_results(self, results: Dict[str, Any]) -> None:
        """Print detailed results with receipt analysis."""
        print(f"\n📋 Detailed Receipt Analysis:")
        print("=" * 80)
        
        for result in results.get("results", []):
            print(f"\nSubject: {result.get('subject', 'Unknown')[:60]}...")
            print(f"Receipt Type: {result.get('receipt_type', 'unknown').replace('_', ' ').title()}")
            print(f"Is Receipt: {'Yes' if result.get('is_receipt') else 'No'}")
            print(f"Confidence: {result.get('confidence', 0):.1%}")
            print(f"Label: {result.get('label_applied') or result.get('label_would_apply', 'None')}")
            
            if result.get('reasoning'):
                print(f"Reasoning: {result.get('reasoning')}")
            
            # Receipt details
            if result.get('is_receipt'):
                if result.get('vendor'):
                    print(f"Vendor: {result.get('vendor')}")
                if result.get('amount'):
                    amount_str = result.get('amount')
                    if result.get('currency'):
                        amount_str = f"{result.get('currency')} {amount_str}"
                    print(f"Amount: {amount_str}")
                if result.get('order_number'):
                    print(f"Order Number: {result.get('order_number')}")
                if result.get('transaction_date'):
                    print(f"Transaction Date: {result.get('transaction_date')}")
                if result.get('payment_method'):
                    print(f"Payment Method: {result.get('payment_method')}")
            
            if result.get('receipt_indicators'):
                print(f"Receipt Indicators: {', '.join(result.get('receipt_indicators', []))}")
            
            print("-" * 80)
    
    def print_vendor_statistics(self) -> None:
        """Print vendor receipt statistics."""
        stats = self.receipt_classifier.get_vendor_statistics()
        
        print(f"\n📈 Vendor Receipt Statistics:")
        print(f"   Total Vendors: {stats.get('total_vendors', 0)}")
        print(f"   Receipt Vendors: {stats.get('receipt_vendors', 0)}")
        print(f"   Average Receipt Rate: {stats.get('average_receipt_rate', 0):.1%}")
        
        top_vendors = stats.get('top_receipt_vendors', [])
        if top_vendors:
            print(f"\n   Top Receipt Vendors:")
            for vendor_name, receipt_rate, total_emails in top_vendors:
                print(f"      {vendor_name}: {receipt_rate:.1%} ({total_emails} emails)")
    
    async def shutdown(self) -> None:
        """Shutdown the labeler and clean up resources."""
        try:
            print("🔄 Shutting down receipt email labeler...")
            
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
    parser = argparse.ArgumentParser(description="Label unread Gmail emails with receipt classification")
    parser.add_argument("--limit", type=int, help="Limit number of emails to process")
    parser.add_argument("--dry-run", action="store_true", help="Analyze emails without applying labels")
    parser.add_argument("--detailed", action="store_true", help="Show detailed results")
    parser.add_argument("--vendor-stats", action="store_true", help="Show vendor statistics")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print(" 🧾 Receipt Email Labeling Tool")
    print("=" * 60)
    
    labeler = None
    try:
        labeler = ReceiptEmailLabeler()
        await labeler.initialize()
        
        # Process emails
        results = await labeler.process_unread_emails(
            limit=args.limit,
            dry_run=args.dry_run
        )
        
        # Show detailed results if requested
        if args.detailed:
            labeler.print_detailed_results(results)
        
        # Show vendor statistics if requested
        if args.vendor_stats:
            labeler.print_vendor_statistics()
        
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