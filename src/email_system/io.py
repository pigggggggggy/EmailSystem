from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from email_system.schemas import Email


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_emails(path: str | Path) -> list[Email]:
    return [Email.from_dict(row) for row in read_jsonl(path)]
