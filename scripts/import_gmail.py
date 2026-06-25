#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from email_system.gmail import GMAIL_READONLY_SCOPE, GmailAPIClient, build_gmail_service
from email_system.io import write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import recent Gmail messages to EmailSystem JSONL.")
    parser.add_argument("--credentials", default="secrets/gmail_credentials.json")
    parser.add_argument("--token", default="data/auth/gmail_token.json")
    parser.add_argument("--query", default="in:inbox newer_than:30d")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--output", default="data/eval_sets/gmail_inbox.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    service = build_gmail_service(credentials_path=args.credentials, token_path=args.token, scopes=[GMAIL_READONLY_SCOPE])
    emails = GmailAPIClient(service).list_emails(query=args.query, limit=args.limit)
    write_jsonl(args.output, [email_to_dict(email) for email in emails])
    print(f"imported={len(emails)} output={args.output}")


def email_to_dict(email) -> dict:
    return {
        "email_id": email.email_id,
        "thread_id": email.thread_id,
        "subject": email.subject,
        "from": email.sender,
        "to": email.to,
        "cc": email.cc,
        "timestamp": email.timestamp,
        "body_text": email.body_text,
        "attachments": email.attachments,
        "labels": email.labels.to_dict(),
    }


if __name__ == "__main__":
    main()
