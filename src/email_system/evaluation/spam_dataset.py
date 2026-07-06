from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from email import policy
from email.parser import BytesParser
from pathlib import Path
from typing import Iterable, Iterator


def iter_enron_rows(path: str | Path) -> Iterator[dict]:
    csv.field_size_limit(sys.maxsize)
    source = Path(path)
    with source.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader((line.replace("\0", "") for line in handle))
        for row in reader:
            label = _label(row.get("Spam/Ham", ""))
            subject = _clean(row.get("Subject", ""))
            body = _clean(row.get("Message", ""))
            source_id = str(row.get("Message ID", ""))
            yield _record(
                source="enron_spam",
                source_id=source_id,
                subject=subject,
                body=body,
                label=label,
                timestamp=_clean(row.get("Date", "")) or None,
            )


def iter_trec_rows(root: str | Path, index_path: str | Path | None = None) -> Iterator[dict]:
    root = Path(root)
    index = Path(index_path) if index_path else root / "full" / "index"
    for raw_line in index.read_text(encoding="utf-8", errors="replace").splitlines():
        if not raw_line.strip():
            continue
        raw_label, relative_path = raw_line.split(maxsplit=1)
        label = _label(raw_label)
        message_path = (index.parent / relative_path).resolve()
        message = BytesParser(policy=policy.default).parsebytes(message_path.read_bytes())
        subject = _clean(str(message.get("Subject", "")))
        body = _message_body(message)
        yield _record(
            source="trec06c",
            source_id=relative_path,
            subject=subject,
            body=body,
            label=label,
            timestamp=str(message.get("Date", "")) or None,
        )


def split_records(
    records: Iterable[dict],
    *,
    seed: int = 20260629,
    train_ratio: float = 0.7,
    validation_ratio: float = 0.1,
) -> tuple[dict[str, list[dict]], dict]:
    if not 0 <= train_ratio <= 1 or not 0 <= validation_ratio <= 1 or train_ratio + validation_ratio > 1:
        raise ValueError("invalid split ratios")

    unique: dict[str, dict] = {}
    duplicate_count = 0
    conflict_count = 0
    source_counts: Counter = Counter()
    for record in records:
        source_counts[record.get("source", "unknown")] += 1
        fingerprint = content_fingerprint(record)
        existing = unique.get(fingerprint)
        if existing is not None:
            duplicate_count += 1
            if existing["labels"]["spam_label"] != record["labels"]["spam_label"]:
                conflict_count += 1
            continue
        item = dict(record)
        item["content_fingerprint"] = fingerprint
        unique[fingerprint] = item

    by_label: dict[str, list[dict]] = defaultdict(list)
    for item in unique.values():
        by_label[item["labels"]["spam_label"]].append(item)

    splits = {"train": [], "validation": [], "test": []}
    for label, items in sorted(by_label.items()):
        ordered = sorted(items, key=lambda item: _split_key(seed, item["content_fingerprint"]))
        train_end = int(len(ordered) * train_ratio)
        validation_end = train_end + int(len(ordered) * validation_ratio)
        splits["train"].extend(ordered[:train_end])
        splits["validation"].extend(ordered[train_end:validation_end])
        splits["test"].extend(ordered[validation_end:])

    for name in splits:
        splits[name].sort(key=lambda item: (item["source"], item["email_id"]))

    manifest = {
        "schema_version": 1,
        "task": "spam_detection",
        "seed": seed,
        "ratios": {
            "train": train_ratio,
            "validation": validation_ratio,
            "test": round(1.0 - train_ratio - validation_ratio, 10),
        },
        "input_records": len(unique) + duplicate_count,
        "unique_records": len(unique),
        "duplicates_removed": duplicate_count,
        "conflicting_duplicates": conflict_count,
        "source_records": dict(sorted(source_counts.items())),
        "splits": {
            name: {
                "records": len(items),
                "labels": dict(sorted(Counter(item["labels"]["spam_label"] for item in items).items())),
            }
            for name, items in splits.items()
        },
    }
    return splits, manifest


def write_dataset(output_dir: str | Path, splits: dict[str, list[dict]], manifest: dict) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        with (output / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def content_fingerprint(record: dict) -> str:
    normalized = "\n".join(
        [
            _normalize_for_hash(record.get("subject", "")),
            _normalize_for_hash(record.get("body_text", "")),
        ]
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _record(*, source: str, source_id: str, subject: str, body: str, label: str, timestamp: str | None) -> dict:
    stable_id = hashlib.sha256(f"{source}:{source_id}".encode("utf-8")).hexdigest()[:24]
    return {
        "email_id": f"{source}:{stable_id}",
        "thread_id": None,
        "subject": subject,
        "from": "",
        "to": [],
        "cc": [],
        "timestamp": timestamp,
        "body_text": body,
        "attachments": [],
        "labels": {
            **({"category": "spam"} if label == "spam" else {}),
            "spam": label == "spam",
            "spam_label": label,
        },
        "source": source,
        "source_id": source_id,
    }


def _message_body(message) -> str:
    try:
        body = message.get_body(preferencelist=("plain", "html"))
        if body is not None:
            return _clean(body.get_content())
    except Exception:
        pass

    parts = []
    try:
        candidates = message.walk()
        for part in candidates:
            if not hasattr(part, "get_content_type") or part.is_multipart():
                continue
            if part.get_content_type() not in {"text/plain", "text/html"}:
                continue
            payload = part.get_payload(decode=True)
            if isinstance(payload, bytes):
                parts.append(_decode_bytes(payload, part.get_content_charset()))
            else:
                raw = part.get_payload()
                if isinstance(raw, str):
                    parts.append(raw)
    except Exception:
        pass
    if parts:
        return _clean("\n".join(parts))

    payload = message.get_payload(decode=True)
    if isinstance(payload, bytes):
        return _clean(_decode_bytes(payload, message.get_content_charset()))
    return _clean(payload if isinstance(payload, str) else "")


def _decode_bytes(payload: bytes, charset: str | None) -> str:
    try:
        return payload.decode(charset or "utf-8", errors="replace")
    except (LookupError, TypeError):
        return payload.decode("utf-8", errors="replace")


def _label(value: str) -> str:
    label = value.strip().lower()
    if label not in {"spam", "ham"}:
        raise ValueError(f"unsupported spam label: {value!r}")
    return label


def _clean(value: object) -> str:
    return str(value or "").replace("\0", "").strip()


def _normalize_for_hash(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().lower()


def _split_key(seed: int, fingerprint: str) -> str:
    return hashlib.sha256(f"{seed}:{fingerprint}".encode("ascii")).hexdigest()
