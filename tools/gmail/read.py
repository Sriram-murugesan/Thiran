"""
tools/gmail/read.py
===================
Implements the gmail.read intent.

Fetches recent emails via the Gmail API, using OAuth 2.0.
Summarizes the snippets of the emails via the Groq LLM to yield a concise,
spoken summary of the inbox. 
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import aiohttp
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from core.models import ToolResult, IntentName
from tools.base import BaseTool, SlotField

# If modifying these scopes, delete the file token.json.
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


class GmailReadTool(BaseTool):
    """
    Reads the user's latest emails and summarizes them using the LLM.
    """

    @property
    def name(self) -> str:
        return IntentName.GMAIL_READ.value

    @property
    def description(self) -> str:
        return "Reads and summarizes the user's latest unread or recent emails from Gmail."

    @property
    def slot_fields(self) -> list[SlotField]:
        return [
            SlotField(
                name="count",
                type_hint="integer",
                required=False,
                default=5,
                description="The number of recent emails to read/summarize. Default is 5. Maximum is 10.",
            )
        ]

    def _get_credentials(self) -> Credentials | None:
        """
        Retrieves or generates OAuth 2.0 credentials for the Gmail API.
        Will trigger a blocking browser login if credentials are not valid.
        """
        creds = None
        base_dir = Path(__file__).parent.parent.parent
        token_path = base_dir / "token.json"
        credentials_path = base_dir / "credentials.json"

        # The file token.json stores the user's access and refresh tokens.
        if token_path.exists():
            creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                except Exception:
                    # If refresh fails, fall through to re-authentication.
                    creds = None

            if not creds:
                if not credentials_path.exists():
                    return None
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(credentials_path), SCOPES
                )
                creds = flow.run_local_server(port=0)

            # Save the credentials for the next run
            token_path.write_text(creds.to_json())

        return creds

    async def _fetch_emails(self, creds: Credentials, max_results: int, log: Any) -> list[dict] | None:
        """Fetch the latest emails from the Gmail API."""
        import asyncio
        loop = asyncio.get_running_loop()

        def _blocking_fetch():
            service = build("gmail", "v1", credentials=creds)
            # Fetch message list (primary inbox)
            results = service.users().messages().list(
                userId="me", labelIds=["INBOX"], maxResults=max_results
            ).execute()
            messages = results.get("messages", [])
            
            email_data = []
            for msg in messages:
                # Fetch message details
                msg_detail = service.users().messages().get(
                    userId="me", id=msg["id"], format="metadata", metadataHeaders=["Subject", "From"]
                ).execute()
                
                snippet = msg_detail.get("snippet", "")
                headers = msg_detail.get("payload", {}).get("headers", [])
                
                subject = "No Subject"
                sender = "Unknown Sender"
                
                for header in headers:
                    if header["name"] == "Subject":
                        subject = header["value"]
                    if header["name"] == "From":
                        sender = header["value"]
                        
                email_data.append({
                    "from": sender,
                    "subject": subject,
                    "snippet": snippet
                })
                
            return email_data

        try:
            return await loop.run_in_executor(None, _blocking_fetch)
        except Exception as exc:
            log.error("gmail.fetch_failed", error=str(exc))
            return None

    async def _summarize_with_groq(self, emails: list[dict], log: Any) -> str | None:
        """Use Groq to turn the raw email data into a concise, spoken summary."""
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            log.error("gmail.summarize_failed", error="GROQ_API_KEY not set")
            return None

        # Build prompt
        prompt = (
            "You are an AI voice assistant. Summarize the following list of recent emails for the user. "
            "Keep the summary very concise (3 to 4 sentences). Group similar items if possible. "
            "Highlight anything that seems urgent, like meetings, invoices, or important alerts. "
            "Speak naturally, as this text will be read aloud by TTS. Do not use markdown or lists.\n\n"
        )
        for i, em in enumerate(emails, 1):
            prompt += f"Email {i}:\nFrom: {em['from']}\nSubject: {em['subject']}\nSnippet: {em['snippet']}\n\n"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": "llama-3.1-8b-instant",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a concise voice assistant summarizing emails. Answer in 3-4 natural sentences."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.3,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=10.0
                ) as resp:
                    if resp.status != 200:
                        log.error("gmail.summarize_api_error", status=resp.status)
                        return None
                        
                    data = await resp.json()
                    return data["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            log.error("gmail.summarize_failed", error=str(exc))
            return None

    async def _execute_impl(
        self, slots: dict[str, Any], request_id: str
    ) -> ToolResult:
        log = self._log(request_id)
        count = slots.get("count", 5)

        log.info("gmail.read.start", count=count)

        # 1. Get credentials (may trigger browser auth if first time)
        try:
            import asyncio
            loop = asyncio.get_running_loop()
            creds = await loop.run_in_executor(None, self._get_credentials)
        except Exception as exc:
            log.error("gmail.auth_error", error=str(exc))
            return ToolResult(
                request_id=request_id,
                tool_name=self.name,
                success=False,
                message="I had trouble authorizing with Gmail. Please check the console logs.",
                error_code="GMAIL_AUTH_ERROR",
                duration_ms=0,
            )

        if not creds:
            return ToolResult(
                request_id=request_id,
                tool_name=self.name,
                success=False,
                message="Gmail credentials are not configured. Please add the credentials JSON file.",
                error_code="GMAIL_CREDS_MISSING",
                duration_ms=0,
            )

        # 2. Fetch emails
        emails = await self._fetch_emails(creds, max_results=count, log=log)
        if emails is None:
            return ToolResult(
                request_id=request_id,
                tool_name=self.name,
                success=False,
                message="I couldn't fetch your emails from Gmail. Please try again later.",
                error_code="GMAIL_FETCH_FAILED",
                duration_ms=0,
            )

        if not emails:
            return ToolResult(
                request_id=request_id,
                tool_name=self.name,
                success=True,
                message="You have no new emails in your inbox.",
                duration_ms=0,
            )

        # 3. Summarize with Groq
        summary = await self._summarize_with_groq(emails, log)
        if not summary:
            return ToolResult(
                request_id=request_id,
                tool_name=self.name,
                success=False,
                message=f"You have {len(emails)} new emails, but I had trouble summarizing them.",
                error_code="SUMMARIZE_FAILED",
                duration_ms=0,
            )

        return ToolResult(
            request_id=request_id,
            tool_name=self.name,
            success=True,
            message=summary,
            data={"fetched_count": len(emails)},
            duration_ms=0,
        )
