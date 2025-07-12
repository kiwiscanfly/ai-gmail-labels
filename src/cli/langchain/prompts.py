"""Specialized prompts for custom labeling and email analysis."""

# Search term generation prompt
SEARCH_TERMS_GENERATION_PROMPT = """Generate 5-10 relevant search terms for finding emails related to "{category}".

Category: {category}
Context: {context}

Consider:
- Synonyms and related terms
- Common email patterns for this category
- Professional and casual language variations
- Industry-specific terminology
- Acronyms and abbreviations
- Related activities and concepts

Examples for reference:
- "programming" → "code, development, software, debugging, repository, commit, pull request, IDE, framework"
- "finance" → "budget, investment, banking, payment, invoice, expense, tax, financial, money, transaction"
- "travel" → "flight, hotel, booking, reservation, itinerary, visa, passport, vacation, trip, destination"

Return 5-10 terms separated by commas, ordered by relevance (most relevant first).
Focus on terms that would commonly appear in email subjects and content.

Terms:"""

# Custom category classification prompt  
CUSTOM_CLASSIFICATION_PROMPT = """Analyze this email to determine if it belongs to the "{category}" category.

Email Subject: {subject}
Email Sender: {sender}
Email Content: {content}

Category: {category}
Related Search Terms: {search_terms}

Classification Criteria:
1. Does the email content relate to the specified category?
2. Do any of the search terms appear in the subject or content?
3. Is the sender type consistent with this category?
4. Does the overall context match the category theme?

Provide your analysis in this exact format:
CLASSIFICATION: [YES/NO]
CONFIDENCE: [0.0-1.0]
REASONING: [Brief explanation of your decision]
SUGGESTED_LABEL: [Specific label to apply, e.g., "Projects/Programming" or just "{category}"]

Examples:
CLASSIFICATION: YES
CONFIDENCE: 0.85
REASONING: Email discusses code review and contains programming-related terms like "repository" and "pull request"
SUGGESTED_LABEL: Projects/Programming

CLASSIFICATION: NO
CONFIDENCE: 0.90
REASONING: Email is about vacation planning which doesn't relate to the programming category
SUGGESTED_LABEL: None

Your analysis:"""

# Email routing prompt for intelligent classification
EMAIL_ROUTING_PROMPT = """Analyze this email and determine which classification services should process it.

Email Subject: {subject}
Email Sender: {sender}
Email Snippet: {snippet}

Available classification services:
- priority: Assigns priority levels (critical/high/medium/low) based on urgency and importance
- marketing: Detects marketing emails (promotional/newsletter/hybrid/transactional)
- receipt: Identifies receipts and transactions (purchase/service/other)
- notifications: Classifies system and service notifications (system/update/alert/reminder/security)
- custom: Applies custom user-defined categories

Guidelines:
- An email can be processed by multiple services
- Only suggest services that are genuinely relevant
- Consider the email's primary purpose and content
- Priority should be applied to most emails
- Marketing is for commercial/promotional content
- Receipts are for transaction confirmations
- Notifications are for automated system messages
- Custom categories are for user-specific organizational needs

Return only the service names that should process this email, separated by commas.
Example: "priority, receipt" or "priority, marketing" or "priority, notifications"

Services:"""

# Email analysis prompt for insights and suggestions
EMAIL_ANALYSIS_PROMPT = """Analyze this collection of {email_count} emails and provide insights for email organization.

Email Subjects (sample):
{email_subjects}

Provide analysis on:

1. COMMON THEMES AND PATTERNS
   - What are the main themes/topics in these emails?
   - Are there recurring sender patterns?
   - What types of content are most common?

2. SUGGESTED CUSTOM CATEGORIES
   - What custom categories would be useful for organizing these emails?
   - Consider work, personal, project, or domain-specific categories
   - Suggest 3-5 practical categories with brief descriptions

3. OPTIMIZATION RECOMMENDATIONS
   - How could email organization be improved?
   - Are there any obvious patterns that could be automated?
   - What Gmail search queries would be most useful?

4. SENDER ANALYSIS
   - Which senders appear most frequently?
   - Are there senders that could benefit from specific filtering?
   - Any patterns in automated vs. personal emails?

Provide actionable insights that would help improve email organization and productivity.

Analysis:"""