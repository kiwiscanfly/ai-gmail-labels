# MCP Tools for Email Categorization

This document describes the MCP (Model Context Protocol) tools available for email search, analysis, and categorization.

**✅ Status: WORKING** - MCP server successfully implemented using FastMCP framework

## Overview

The email categorization MCP server provides powerful tools for:
- **Email Search**: Find emails using Gmail queries with AI-powered summaries
- **Email Analysis**: Get detailed classification and priority analysis
- **Label Management**: Apply labels automatically or manually with AI validation
- **Batch Processing**: Classify and label multiple emails efficiently
- **System Monitoring**: Track classification performance and system health

## Available Tools

### 1. `search_emails`

Search Gmail emails and optionally generate summaries and apply labels.

**Parameters:**
- `query` (string, required): Gmail search query (e.g., "is:unread", "from:example.com", "subject:important")
- `limit` (int, default: 20): Maximum number of emails to return (max: 100)
- `include_summary` (bool, default: true): Whether to generate AI summaries
- `apply_labels` (bool, default: false): Whether to automatically apply classification labels
- `summary_model` (string, default: "auto"): Model for summarization - "auto", "priority", "marketing", "receipt"

**Example Usage:**
```
Search for unread emails with summaries:
search_emails(query="is:unread", limit=10, include_summary=true)

Search for emails from a specific domain and apply labels:
search_emails(query="from:amazon.com", limit=20, apply_labels=true)

Search for urgent emails with priority analysis only:
search_emails(query="subject:urgent", summary_model="priority")
```

**Response:**
```json
{
  "query": "is:unread",
  "total_found": 10,
  "limit": 10,
  "emails": [
    {
      "id": "email_id_123",
      "subject": "Important Meeting Update",
      "sender": "boss@company.com",
      "date": "2024-01-15",
      "snippet": "The meeting has been moved...",
      "labels": ["INBOX", "UNREAD"],
      "priority_analysis": {
        "level": "high",
        "confidence": 0.85,
        "reasoning": "Meeting change from executive requires immediate attention",
        "is_genuine_urgency": true,
        "authenticity_score": 0.9
      },
      "ai_summary": "Meeting rescheduled by executive, requires calendar update and team notification."
    }
  ],
  "timestamp": "2024-01-15T10:30:00Z",
  "include_summary": true,
  "apply_labels": false
}
```

### 2. `get_email_summary`

Get detailed summary and analysis of a specific email.

**Parameters:**
- `email_id` (string, required): Gmail message ID
- `analysis_type` (string, default: "comprehensive"): Type of analysis - "comprehensive", "priority", "marketing", "receipt", "quick"
- `include_classification` (bool, default: true): Whether to include classification analysis

**Example Usage:**
```
Get comprehensive analysis:
get_email_summary(email_id="abc123", analysis_type="comprehensive")

Get only priority analysis:
get_email_summary(email_id="abc123", analysis_type="priority")

Get marketing-focused analysis:
get_email_summary(email_id="abc123", analysis_type="marketing")
```

**Response:**
```json
{
  "email": {
    "id": "abc123",
    "subject": "50% Off Sale - Limited Time!",
    "sender": "sales@retailer.com",
    "date": "2024-01-15",
    "snippet": "Don't miss our biggest sale..."
  },
  "analysis_type": "comprehensive",
  "priority_analysis": {
    "level": "low",
    "confidence": 0.92,
    "reasoning": "Marketing email with promotional content, downgraded from medium priority",
    "is_genuine_urgency": false,
    "authenticity_score": 0.3,
    "detected_tactics": ["emotional_manipulation", "sales_pressure"]
  },
  "marketing_analysis": {
    "is_marketing": true,
    "subtype": "promotional",
    "confidence": 0.95,
    "reasoning": "Contains discount offers, unsubscribe link, and promotional language",
    "marketing_indicators": ["unsubscribe_link", "price_discount_patterns", "promotional_keywords"],
    "unsubscribe_detected": true
  },
  "ai_summary": "Promotional marketing email advertising a 50% off sale with typical marketing pressure tactics.",
  "timestamp": "2024-01-15T10:35:00Z"
}
```

### 3. `apply_email_label`

Apply a specific label to an email with optional AI validation.

**Parameters:**
- `email_id` (string, required): Gmail message ID
- `label` (string, required): Label name to apply
- `create_if_missing` (bool, default: true): Create label if it doesn't exist
- `confidence_threshold` (float, default: 0.8): Minimum confidence for AI validation (0.0 to skip)

**Example Usage:**
```
Apply priority label with validation:
apply_email_label(email_id="abc123", label="Priority/High", confidence_threshold=0.8)

Apply custom label without validation:
apply_email_label(email_id="abc123", label="Projects/Alpha", confidence_threshold=0.0)

Apply receipt label with validation:
apply_email_label(email_id="abc123", label="Receipts/Purchase")
```

**Response:**
```json
{
  "email_id": "abc123",
  "label": "Priority/High",
  "applied": true,
  "validation": {
    "recommended": true,
    "confidence": 0.85,
    "reasoning": "AI analysis supports high priority classification"
  },
  "created_label": false,
  "message": "Label 'Priority/High' successfully applied to email"
}
```

### 4. `classify_and_label_emails`

Classify emails and apply appropriate labels automatically.

**Parameters:**
- `email_ids` (list[string], optional): Specific email IDs to classify (overrides query)
- `query` (string, optional): Gmail search query to find emails to classify
- `limit` (int, default: 50): Maximum number of emails to process (max: 100)
- `dry_run` (bool, default: false): Preview classifications without applying labels
- `classification_types` (list[string], default: ["priority", "marketing", "receipt"]): Types of classification to apply

**Example Usage:**
```
Classify unread emails (dry run):
classify_and_label_emails(query="is:unread", limit=20, dry_run=true)

Classify specific emails with all types:
classify_and_label_emails(email_ids=["abc123", "def456"], classification_types=["priority", "marketing", "receipt"])

Classify and label recent emails with priority only:
classify_and_label_emails(query="newer_than:1d", classification_types=["priority"], dry_run=false)
```

**Response:**
```json
{
  "total_processed": 15,
  "successful": 14,
  "failed": 1,
  "dry_run": false,
  "classification_types": ["priority", "marketing", "receipt"],
  "results": [
    {
      "email_id": "abc123",
      "subject": "Meeting Reminder - Project Alpha",
      "sender": "team@company.com",
      "processed": true,
      "dry_run": false,
      "classifications": {
        "priority": {
          "level": "medium",
          "confidence": 0.78,
          "reasoning": "Project meeting reminder with flexible deadline"
        }
      },
      "labels_applied": ["Priority/Medium"]
    }
  ],
  "timestamp": "2024-01-15T10:40:00Z"
}
```

### 5. `get_classification_stats`

Get statistics about email classification and system status.

**Parameters:** None

**Example Usage:**
```
get_classification_stats()
```

**Response:**
```json
{
  "timestamp": "2024-01-15T10:45:00Z",
  "system_status": "operational",
  "services": {
    "email_service": true,
    "prioritizer": true,
    "marketing_classifier": true,
    "receipt_classifier": true,
    "ollama_manager": true
  },
  "sender_statistics": {
    "total_senders": 150,
    "trusted_senders": 45,
    "suspicious_senders": 8,
    "average_reputation": 0.721
  },
  "marketing_statistics": {
    "total_senders": 89,
    "marketing_senders": 34,
    "individual_senders": 55,
    "average_marketing_rate": 0.382
  },
  "receipt_statistics": {
    "total_vendors": 23,
    "receipt_vendors": 18,
    "average_receipt_rate": 0.891,
    "top_receipt_vendors": [
      ["Amazon", 0.95, 47],
      ["Apple", 0.98, 23]
    ]
  }
}
```

## Gmail Search Query Examples

The `query` parameter supports Gmail's search syntax:

### Basic Searches
- `is:unread` - Unread emails
- `is:read` - Read emails
- `is:starred` - Starred emails
- `has:attachment` - Emails with attachments

### Sender/Recipient Searches
- `from:example@domain.com` - From specific sender
- `to:me` - Emails to you
- `cc:person@domain.com` - Emails where person is CC'd

### Date Searches
- `newer_than:1d` - Newer than 1 day
- `older_than:1w` - Older than 1 week
- `after:2024/1/1` - After specific date
- `before:2024/1/15` - Before specific date

### Subject/Content Searches
- `subject:meeting` - Subject contains "meeting"
- `"exact phrase"` - Exact phrase in email
- `filename:pdf` - Emails with PDF attachments

### Label Searches
- `label:important` - Emails with "important" label
- `-label:spam` - Exclude spam
- `has:nouserlabels` - No user-applied labels

### Advanced Examples
- `from:amazon.com has:attachment newer_than:7d` - Recent Amazon emails with attachments
- `is:unread -label:spam subject:(urgent OR important)` - Unread urgent emails, excluding spam
- `from:bank.com OR from:paypal.com` - Financial emails from multiple sources

## Usage Examples

### Example 1: Daily Email Triage
```
# Find all unread emails with AI analysis
search_emails(
  query="is:unread", 
  limit=50, 
  include_summary=true, 
  summary_model="auto"
)

# Classify and label high-priority emails
classify_and_label_emails(
  query="is:unread subject:(urgent OR important OR deadline)", 
  classification_types=["priority"],
  dry_run=false
)
```

### Example 2: Marketing Email Cleanup
```
# Find potential marketing emails
search_emails(
  query="has:nouserlabels newer_than:3d", 
  limit=30, 
  summary_model="marketing"
)

# Classify and label marketing emails
classify_and_label_emails(
  query="has:nouserlabels newer_than:3d",
  classification_types=["marketing"],
  dry_run=false
)
```

### Example 3: Receipt Organization
```
# Find potential receipts
search_emails(
  query="subject:(receipt OR invoice OR order OR payment) newer_than:30d",
  summary_model="receipt"
)

# Classify and organize receipts
classify_and_label_emails(
  query="subject:(receipt OR invoice OR order OR payment) newer_than:30d",
  classification_types=["receipt"],
  dry_run=false
)
```

### Example 4: Specific Email Analysis
```
# Get detailed analysis of a specific email
get_email_summary(
  email_id="abc123",
  analysis_type="comprehensive"
)

# Apply custom label with AI validation
apply_email_label(
  email_id="abc123",
  label="Projects/WebsiteRedesign",
  confidence_threshold=0.7
)
```

## Error Handling

All tools return structured error responses when issues occur:

```json
{
  "error": true,
  "code": "INTERNAL_ERROR",
  "message": "Failed to search emails: Gmail API quota exceeded",
  "timestamp": "2024-01-15T10:50:00Z"
}
```

Common error types:
- `INVALID_PARAMS`: Invalid parameters provided
- `INTERNAL_ERROR`: System or service errors
- `QUOTA_EXCEEDED`: API rate limits reached
- `AUTH_ERROR`: Authentication issues

## Configuration

The MCP server is now working with Claude Desktop! Use this configuration in your MCP settings:

```json
{
  "mcpServers": {
    "email-categorization-agent": {
      "command": "uv",
      "args": [
        "run", 
        "python", 
        "src/email_mcp_server.py"
      ],
      "cwd": "/Users/rebecca/repos/email-agents",
      "env": {
        "PYTHONPATH": "/Users/rebecca/repos/email-agents"
      }
    }
  }
}
```

**Implementation Details:**
- **Framework**: Uses FastMCP for better compatibility with Claude Desktop
- **Logging**: All output properly directed to stderr to avoid MCP protocol interference
- **Services**: Gmail API, Ollama LLM, and classification services integration
- **Tools**: 5 fully functional email management tools

## Security and Privacy

- **Local Processing**: All AI analysis happens locally using Ollama
- **No Data Storage**: Email content is not permanently stored
- **OAuth2 Security**: Secure Gmail API authentication
- **Rate Limiting**: Respects Gmail API quotas and limits
- **Audit Trail**: Classification decisions are logged for transparency

## Performance Notes

- **Caching**: Classification results are cached for 1 hour
- **Batch Processing**: Efficient handling of multiple emails
- **Quick Classification**: Fast pattern-based decisions for obvious cases
- **Fallback**: Graceful degradation when services are unavailable
- **Memory Management**: Efficient handling of large email sets