#!/usr/bin/env python3
"""
Gmail Authentication Setup Script

This script helps you set up Gmail OAuth2 authentication for the email categorization system.
Run this script to configure Gmail API access step by step.
"""

import os
import json
import webbrowser
from pathlib import Path
from typing import Dict, Any
import requests

def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def print_step(step: int, description: str):
    """Print a formatted step."""
    print(f"\n📋 Step {step}: {description}")
    print("-" * 50)

def create_env_file():
    """Create .env file from .env.example if it doesn't exist."""
    env_path = Path(".env")
    env_example_path = Path(".env.example")
    
    if not env_path.exists() and env_example_path.exists():
        env_path.write_text(env_example_path.read_text())
        print(f"✅ Created .env file from .env.example")
    elif env_path.exists():
        print(f"✅ .env file already exists")
    else:
        print(f"❌ Neither .env nor .env.example found")
        return False
    return True

def check_credentials_file():
    """Check if credentials.json exists."""
    creds_path = Path("credentials.json")
    if creds_path.exists():
        print(f"✅ credentials.json found")
        try:
            with open(creds_path) as f:
                creds = json.load(f)
                if "installed" in creds or "web" in creds:
                    return True
                else:
                    print(f"❌ credentials.json has invalid format")
                    return False
        except json.JSONDecodeError:
            print(f"❌ credentials.json is not valid JSON")
            return False
    else:
        print(f"❌ credentials.json not found")
        return False

def setup_google_cloud_project():
    """Guide user through Google Cloud Console setup."""
    print_step(1, "Set up Google Cloud Project and Gmail API")
    
    print("""
🔧 You need to create a Google Cloud Project and enable the Gmail API:

1. Go to the Google Cloud Console:
   https://console.cloud.google.com/

2. Create a new project or select an existing one:
   - Click "Select a project" → "New Project"
   - Enter project name (e.g., "email-categorization-agent")
   - Click "Create"

3. Enable the Gmail API:
   - Go to "APIs & Services" → "Library"
   - Search for "Gmail API"
   - Click on it and press "Enable"

4. Create OAuth 2.0 credentials:
   - Go to "APIs & Services" → "Credentials"
   - Click "Create Credentials" → "OAuth 2.0 Client IDs"
   - Configure OAuth consent screen if prompted:
     * User Type: External (for personal use) or Internal (for organization)
     * Fill required fields (App name, User support email, etc.)
     * Add your email to test users
   - For Application type, choose "Desktop application"
   - Give it a name (e.g., "Email Categorization Agent")
   - Click "Create"

5. Download the credentials:
   - Click the download button (⬇️) next to your OAuth 2.0 Client ID
   - Save the file as "credentials.json" in this project directory
    """)
    
    input("\n Press Enter when you have downloaded credentials.json...")
    
    if check_credentials_file():
        print("✅ credentials.json setup complete!")
        return True
    else:
        print("❌ Please ensure credentials.json is in the project directory")
        return False

def test_authentication():
    """Test the authentication setup."""
    print_step(2, "Test Authentication")
    
    print("""
🧪 Now let's test the authentication setup.

The system will:
1. Open your web browser for OAuth authorization
2. You'll need to sign in to your Google account
3. Grant permission to access Gmail
4. The system will save the token for future use
    """)
    
    try:
        # Import and test the Gmail client
        import asyncio
        import sys
        sys.path.append(str(Path.cwd()))
        
        from src.integrations.gmail_client import get_gmail_client
        
        async def test_gmail():
            try:
                print("🔄 Initializing Gmail client...")
                client = await get_gmail_client()
                
                print("🔄 Testing API access...")
                profile = await client.get_profile()
                
                print(f"✅ Authentication successful!")
                print(f"   Email: {profile.get('emailAddress')}")
                print(f"   Messages Total: {profile.get('messagesTotal')}")
                print(f"   History ID: {profile.get('historyId')}")
                
                return True
                
            except Exception as e:
                print(f"❌ Authentication failed: {e}")
                return False
        
        # Run the test
        success = asyncio.run(test_gmail())
        return success
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you've installed dependencies: uv sync")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def check_dependencies():
    """Check if required dependencies are installed."""
    print_step(0, "Check Dependencies")
    
    # Map package names to their import names
    required_packages = {
        "google-auth": "google.auth",
        "google-auth-oauthlib": "google_auth_oauthlib", 
        "google-api-python-client": "googleapiclient",
        "google-auth-httplib2": "google_auth_httplib2"
    }
    
    missing = []
    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"✅ {package_name}")
        except ImportError:
            print(f"❌ {package_name}")
            missing.append(package_name)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("Install with: uv sync")
        return False
    else:
        print("✅ All dependencies installed")
        return True

def setup_gmail_scopes():
    """Explain Gmail scopes."""
    print_step(3, "Understanding Gmail Scopes")
    
    print("""
🔐 The system uses these Gmail API scopes:

1. https://www.googleapis.com/auth/gmail.readonly
   - Read emails and labels
   - Required for email categorization

2. https://www.googleapis.com/auth/gmail.modify  
   - Apply labels to emails
   - Required for categorization actions

3. https://www.googleapis.com/auth/gmail.labels
   - Manage custom labels
   - Required for creating categorization labels

These are the minimum permissions needed for the email categorization system.
    """)

def main():
    """Main setup process."""
    print_header("Gmail Authentication Setup")
    print("This script will help you set up Gmail OAuth2 authentication")
    
    # Step 0: Check dependencies
    if not check_dependencies():
        return
    
    # Create .env file
    if not create_env_file():
        return
    
    # Step 1: Google Cloud setup
    if not setup_google_cloud_project():
        return
    
    # Step 2: Explain scopes
    setup_gmail_scopes()
    
    # Step 3: Test authentication
    if test_authentication():
        print_header("Setup Complete! 🎉")
        print("""
✅ Gmail authentication is now configured!

Next steps:
1. Your credentials are saved in 'token.json'
2. The system can now access your Gmail account
3. You can run the email categorization agents

To start categorizing emails:
   python -m src.cli.main categorize

To check system status:
   python -m src.cli.main status
        """)
    else:
        print_header("Setup Failed ❌")
        print("""
Setup was not completed successfully. Common issues:

1. credentials.json file format is incorrect
   - Re-download from Google Cloud Console
   - Ensure it's a valid JSON file

2. OAuth consent screen not configured
   - Complete the consent screen setup in Google Cloud Console
   - Add your email as a test user

3. Gmail API not enabled
   - Enable Gmail API in Google Cloud Console

Run this script again after fixing the issues.
        """)

if __name__ == "__main__":
    main()