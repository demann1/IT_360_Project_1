import os
import pickle
from pathlib import Path
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

# Authenticate and build the Gmail API service
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

# Path to the credentials.json file
CREDENTIALS_PATH = Path(__file__).resolve().parent.parent / "config" / "credentials.json"

def authenticate_gmail():
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    return build('gmail', 'v1', credentials=creds)

def fetch_emails():
    """Fetch the latest emails from the Gmail inbox."""
    service = authenticate_gmail()
    results = service.users().messages().list(userId='me', maxResults=10).execute()
    messages = results.get('messages', [])
    email_texts = []

    for message in messages:
        msg = service.users().messages().get(userId='me', id=message['id']).execute()
        payload = msg.get('payload', {})
        headers = payload.get('headers', [])
        
        # Extract subject and snippet
        subject = next((header['value'] for header in headers if header['name'] == 'Subject'), "No Subject")
        snippet = msg.get('snippet', "No Snippet Available")
        
        # Format the email output
        email_texts.append(f"Subject: {subject}\nSnippet: {snippet}\n{'-' * 40}")
    
    return email_texts