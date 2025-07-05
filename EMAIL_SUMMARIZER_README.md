# 📧 Email Summarization Tool

This tool retrieves your unread Gmail messages, summarizes them using AI, and creates organized markdown files for easy review.

## Features

- ✅ **Retrieves unread emails** from your Gmail account
- 🤖 **AI-powered summaries** using local Ollama models
- 📝 **Markdown output** with structured summaries
- 🔗 **Gmail links** to read full emails
- 📋 **Key points extraction** in bullet format
- ✅ **Action items identification** with checkboxes
- 🎯 **Priority assessment** (Low/Medium/High)

## Quick Start

### 1. Prerequisites

Ensure Ollama is running:
```bash
ollama serve
```

Make sure you have a model available:
```bash
ollama pull gemma2:3b
```

### 2. Run Email Summarization

**Option 1: Using the wrapper script (easiest)**
```bash
./summarize_emails.sh
```

**Option 2: Direct Python execution**
```bash
uv run python summarize_unread_emails.py
```

**Option 3: With options**
```bash
# Limit to 5 emails
./summarize_emails.sh --limit 5

# Custom output directory
./summarize_emails.sh --output my_summaries

# Both options
./summarize_emails.sh --limit 10 --output urgent_emails
```

## Output Format

Each email generates a markdown file with:

### File Structure
```
email_summaries/
├── 20240706_141230_Meeting_Reminder_a1b2c3d4.md
├── 20240706_141245_Invoice_Due_e5f6g7h8.md
└── 20240706_141300_Newsletter_i9j0k1l2.md
```

### File Content Example
```markdown
# Meeting Reminder: Q4 Planning Session

## Email Details
- **From:** John Smith <john@company.com>
- **Received:** July 6, 2024 at 2:30 PM
- **Priority:** High
- **Gmail ID:** `a1b2c3d4e5f6g7h8`
- **Read in Gmail:** [Open Email](https://mail.google.com/mail/u/0/#inbox/a1b2c3d4e5f6g7h8)

---

## Summary
This email is a reminder about the Q4 planning session scheduled for next Tuesday. The meeting will cover budget planning, goal setting, and resource allocation for the next quarter.

## Key Points
- Q4 planning meeting scheduled for Tuesday, July 9th at 10:00 AM
- Meeting location: Conference Room B or via Zoom link provided
- Agenda includes budget review, goal setting, and resource planning
- Pre-meeting materials shared in company drive folder
- Expected duration: 2 hours with optional lunch discussion

## Action Items
- [ ] Review pre-meeting materials in shared drive
- [ ] Prepare Q3 performance summary
- [ ] Submit budget requests by July 8th
- [ ] Confirm attendance by responding to this email

## Gmail Labels
- INBOX
- Important

---

## Metadata
- **Generated:** 2024-07-06 14:12:30
- **Email ID:** a1b2c3d4e5f6g7h8
- **Thread ID:** i9j0k1l2m3n4o5p6
- **Has Attachments:** Yes
```

## Command Line Options

| Option | Description | Example |
|--------|-------------|---------|
| `--limit` | Limit number of emails to process | `--limit 5` |
| `--output` | Custom output directory | `--output summaries` |
| `--help` | Show help message | `--help` |

## How It Works

1. **Authentication**: Uses your existing Gmail OAuth2 credentials
2. **Email Retrieval**: Searches for emails with `is:unread` filter
3. **Content Extraction**: Extracts subject, sender, and body content
4. **AI Summarization**: Sends content to Ollama for analysis
5. **Markdown Generation**: Creates structured markdown files
6. **File Organization**: Saves files with timestamps and sanitized names

## AI Summary Features

The AI analysis provides:

- **Concise Summary**: 2-3 sentence overview of the email
- **Key Points**: Bulleted list of important information
- **Action Items**: Specific tasks or deadlines mentioned
- **Priority Level**: Assessment of urgency (Low/Medium/High)

## Troubleshooting

### Common Issues

**Error: Ollama not responding**
```bash
# Start Ollama
ollama serve

# Check if model is available
ollama list
```

**Error: Gmail authentication failed**
```bash
# Re-run Gmail setup
uv run python setup_gmail_auth.py
```

**Error: No unread emails found**
- Check if you actually have unread emails in Gmail
- Verify Gmail authentication is working: `uv run python -m src.cli test-gmail`

**Error: Permission denied**
```bash
# Make scripts executable
chmod +x summarize_emails.sh
chmod +x summarize_unread_emails.py
```

### Performance Notes

- **Processing Speed**: ~2-3 seconds per email (depends on content length)
- **Rate Limiting**: Includes 1-second delays between emails
- **Memory Usage**: Optimized for batch processing
- **Model Choice**: Uses your configured Ollama model (default: gemma2:3b)

## Example Usage Scenarios

### Daily Email Review
```bash
# Process all unread emails
./summarize_emails.sh --output daily_review
```

### Quick Triage (5 most recent)
```bash
# Process only 5 emails for quick review
./summarize_emails.sh --limit 5 --output triage
```

### Specific Project Emails
```bash
# Custom output for project-related emails
./summarize_emails.sh --output project_updates
```

## Integration Tips

### With Your Workflow
1. **Morning Routine**: Run daily email summarization
2. **Review Process**: Read markdown files for quick overview
3. **Action Items**: Use checkboxes to track completed tasks
4. **Follow-up**: Click Gmail links for full email context

### Automation Possibilities
- Schedule with cron for daily processing
- Integrate with note-taking apps
- Export to project management tools
- Generate daily digest reports

## Files Created

- `email_summaries/` - Default output directory
- `*.md` - Individual email summary files
- Format: `YYYYMMDD_HHMMSS_Subject_EmailID.md`

## Support

If you encounter issues:
1. Check Ollama is running: `ollama list`
2. Verify Gmail auth: `uv run python -m src.cli test-gmail`
3. Check dependencies: `uv sync`
4. Review logs for detailed error messages