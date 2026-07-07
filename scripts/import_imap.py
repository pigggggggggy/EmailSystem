#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from email_system.env import load_local_env
from email_system.imap_mail import IMAPConfig, IMAPEmailClient, emails_to_dicts
from email_system.io import write_jsonl

load_local_env(ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import emails from an IMAP mailbox to EmailSystem JSONL.")
    parser.add_argument("--host", default="imap.gmail.com")
    parser.add_argument("--port", type=int, default=993)
    parser.add_argument("--mailbox", default="INBOX")
    parser.add_argument("--user", default=os.environ.get("EMAILSYSTEM_IMAP_USER"))
    parser.add_argument("--password-env", default="EMAILSYSTEM_IMAP_PASSWORD")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--search", default="ALL")
    parser.add_argument("--output", default="data/eval_sets/imap_inbox.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    password = os.environ.get(args.password_env)
    if not args.user:
        raise SystemExit("Missing --user or EMAILSYSTEM_IMAP_USER")
    if not password:
        raise SystemExit(f"Missing password environment variable: {args.password_env}")
    client = IMAPEmailClient(
        user=args.user,
        password=password,
        config=IMAPConfig(host=args.host, port=args.port, mailbox=args.mailbox),
    )
    emails = client.fetch_recent(limit=args.limit, search=args.search)
    write_jsonl(args.output, emails_to_dicts(emails))
    print(f"imported={len(emails)} output={args.output}")


if __name__ == "__main__":
    main()
