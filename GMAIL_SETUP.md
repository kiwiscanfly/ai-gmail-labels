# Gmail Authentication Setup Guide

This guide will help you set up Gmail OAuth2 authentication for the email categorization system.

## Quick Start

Run the automated setup script:

```bash
uv run python setup_gmail_auth.py
```

## Manual Setup Instructions

### 1. Prerequisites

Ensure you have installed the project dependencies:

```bash
uv sync
```

### 2. Google Cloud Console Setup

#### A. Create a Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Click "Select a project" → "New Project"
3. Enter project name: `email-categorization-agent`
4. Click "Create"

#### B. Enable Gmail API

1. In the Cloud Console, go to "APIs & Services" → "Library"
2. Search for "Gmail API"
3. Click on it and press "Enable"

#### C. Configure OAuth Consent Screen

1. Go to "APIs & Services" → "OAuth consent screen"
2. Choose user type:
   - **External**: For personal Gmail accounts
   - **Internal**: For G Suite/Workspace organizations
3. Fill in required information:
   - **App name**: `Email Categorization Agent`
   - **User support email**: Your email address
   - **Developer contact information**: Your email address
4. Add scopes (optional step, will be configured later)
5. Add test users:
   - Add your Gmail address under "Test users"
6. Review and submit

#### D. Create OAuth 2.0 Credentials

1. Go to "APIs & Services" → "Credentials"
2. Click "Create Credentials" → "OAuth 2.0 Client IDs"
3. Select application type: **Desktop application**
4. Name: `Email Categorization Agent`
5. Click "Create"
6. **Download the JSON file** and save it as `credentials.json` in your project root

### 3. Project Configuration

#### A. Environment Setup

Create a `.env` file from the template:

```bash
cp .env.example .env
```

Verify your `.env` file contains:

```env
# Gmail Configuration  
GMAIL_CREDENTIALS_PATH=./credentials.json
GMAIL_TOKEN_PATH=./token.json
```

#### B. File Structure

Your project should have:

```
email-agents/
├── credentials.json          # OAuth 2.0 credentials (downloaded from Google)
├── token.json               # Access token (created after first auth)
├── .env                     # Environment configuration
└── src/                     # Project source code
```

### 4. Authentication Flow

#### A. First-Time Authentication

When you run the system for the first time:

1. **Browser opens automatically** to Google OAuth page
2. **Sign in** to your Gmail account
3. **Review permissions** the app is requesting:
   - Read your email messages and settings
   - Manage your email labels
   - Modify your email (apply labels)
4. **Click "Allow"** to grant permissions
5. **Token saved** automatically to `token.json`

#### B. Subsequent Use

- The system uses the saved token in `token.json`
- Tokens are automatically refreshed when expired
- Re-authentication only needed if tokens are revoked

### 5. Gmail API Scopes

The system requests these permissions:

| Scope | Purpose | Required For |
|-------|---------|--------------|
| `gmail.readonly` | Read emails and labels | Email fetching and analysis |
| `gmail.modify` | Apply labels to emails | Categorization actions |
| `gmail.labels` | Manage custom labels | Creating categorization labels |

### 6. Testing Authentication

#### A. Using the Setup Script

```bash
uv run python setup_gmail_auth.py
```

#### B. Manual Testing

```bash
# Test with CLI
uv run python -m src.cli status

# Test Gmail connection specifically
uv run python -m src.cli test-gmail
```

### 7. Troubleshooting

#### A. Common Issues

**Error: `credentials.json not found`**
- Download OAuth 2.0 credentials from Google Cloud Console
- Save as `credentials.json` in project root

**Error: `Invalid credentials format`**
- Ensure you downloaded "OAuth 2.0 Client ID" not "Service Account"
- File should contain `"installed"` or `"web"` key

**Error: `OAuth consent screen not configured`**
- Complete OAuth consent screen setup in Google Cloud Console
- Add your email as a test user if using External user type

**Error: `Gmail API not enabled`**
- Enable Gmail API in Google Cloud Console Library

**Error: `Insufficient permissions`**
- Check OAuth scopes in your consent screen configuration
- Revoke and re-authorize if scopes were changed

#### B. Reset Authentication

To reset authentication (clears saved tokens):

```bash
# Remove saved tokens
rm token.json

# Re-run authentication
uv run python setup_gmail_auth.py
```

#### C. Security Considerations

**Token Storage**:
- `token.json` contains access tokens
- Keep this file secure and private
- Add `token.json` to `.gitignore` (already included)

**Credential Security**:
- `credentials.json` contains OAuth client ID/secret
- These are not as sensitive as access tokens
- Can be shared in team environments
- Still recommended to keep private

### 8. Production Deployment

#### A. OAuth Consent Screen

For production use:
1. Submit OAuth consent screen for verification
2. Move from "Testing" to "Published" status
3. Remove user limits

#### B. Environment Variables

Instead of file paths, use environment variables:

```env
GMAIL_CREDENTIALS_JSON={"installed":{"client_id":"...","client_secret":"..."}}
GMAIL_TOKEN_JSON={"token":"...","refresh_token":"..."}
```

#### C. Secret Management

Use secure secret management:
- Google Secret Manager
- AWS Secrets Manager
- Azure Key Vault
- HashiCorp Vault

### 9. API Quotas and Limits

Gmail API has usage quotas:
- **250 quota units per user per second**
- **1 billion quota units per day**

Typical operations:
- Read email: 5 units
- Modify email: 5 units
- List messages: 5 units

The system includes automatic rate limiting to stay within quotas.

### 10. Next Steps

After authentication setup:

1. **Test email retrieval**:
   ```bash
   uv run python -m src.cli test-gmail
   ```

2. **Run categorization**:
   ```bash
   uv run python -m src.cli categorize --limit 10
   ```

3. **Monitor system**:
   ```bash
   uv run python -m src.cli status
   ```

## Support

If you encounter issues:

1. Check the [troubleshooting section](#7-troubleshooting)
2. Run `uv run python setup_gmail_auth.py` for guided setup
3. Review Google Cloud Console configuration
4. Check project logs for detailed error messages