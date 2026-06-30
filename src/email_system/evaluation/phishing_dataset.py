from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Iterator


SOURCES = (
    "CEAS_08.csv",
    "Enron.csv",
    "Ling.csv",
    "Nazario.csv",
    "Nigerian_Fraud.csv",
    "SpamAssasin.csv",
    "phishing_email.csv",
)


def iter_phishing_rows(root: str | Path) -> Iterator[dict]:
    root = Path(root)
    csv.field_size_limit(sys.maxsize)
    for file_name in SOURCES:
        path = root / file_name
        if not path.exists():
            continue
        yield from _iter_file(path)


def split_records(
    records: Iterable[dict],
    *,
    seed: int = 20260630,
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
            if existing["labels"].get("phishing_label") != record["labels"].get("phishing_label"):
                conflict_count += 1
            candidate = _with_fingerprint(record, fingerprint)
            if _canonical_key(candidate) < _canonical_key(existing):
                unique[fingerprint] = candidate
            continue
        unique[fingerprint] = _with_fingerprint(record, fingerprint)

    by_label: dict[str, list[dict]] = defaultdict(list)
    for item in unique.values():
        by_label[item["labels"]["phishing_label"]].append(item)

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
        "task": "phishing_detection",
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
                "labels": dict(sorted(Counter(item["labels"]["phishing_label"] for item in items).items())),
            }
            for name, items in splits.items()
        },
    }
    return splits, manifest


def _with_fingerprint(record: dict, fingerprint: str) -> dict:
    item = dict(record)
    item["content_fingerprint"] = fingerprint
    return item


def _canonical_key(record: dict) -> tuple[str, str]:
    return (str(record.get("source", "")), str(record.get("email_id", "")))


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


def _iter_file(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader((line.replace("\0", "") for line in handle))
        for row_number, row in enumerate(reader, start=1):
            label = _label(row.get("label", ""))
            if path.name == "phishing_email.csv":
                subject = ""
                body = _clean(row.get("text_combined", ""))
                sender = ""
                to = []
                cc = []
                timestamp = None
            else:
                subject = _clean(row.get("subject", ""))
                body = _clean(row.get("body", ""))
                sender = _clean(row.get("sender", ""))
                to = _split_recipients(row.get("receiver", ""))
                cc = []
                timestamp = _clean(row.get("date", "")) or None
            if not subject and not body:
                continue
            yield _record(
                source=path.stem,
                source_id=str(row_number),
                subject=subject,
                body=body,
                label=label,
                sender=sender,
                to=to,
                cc=cc,
                timestamp=timestamp,
                urls=_int_or_none(row.get("urls")),
            )


def _record(
    *,
    source: str,
    source_id: str,
    subject: str,
    body: str,
    label: str,
    sender: str,
    to: list[str],
    cc: list[str],
    timestamp: str | None,
    urls: int | None,
) -> dict:
    stable_id = hashlib.sha256(f"{source}:{source_id}".encode("utf-8")).hexdigest()[:24]
    is_phishing = label == "phishing"
    metadata = {"urls": urls} if urls is not None else {}
    return {
        "email_id": f"{source}:{stable_id}",
        "thread_id": None,
        "subject": subject,
        "from": sender,
        "to": to,
        "cc": cc,
        "timestamp": timestamp,
        "body_text": body,
        "attachments": [],
        "labels": {
            "category": "spam" if is_phishing else "other",
            "spam": is_phishing,
            "spam_label": "spam" if is_phishing else "ham",
            "phishing": is_phishing,
            "phishing_label": label,
        },
        "metadata": metadata,
        "source": source,
        "source_id": source_id,
    }


def _label(value: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"1", "true", "phishing", "spam"}:
        return "phishing"
    if normalized in {"0", "false", "legitimate", "ham", "safe"}:
        return "legitimate"
    raise ValueError(f"unsupported phishing label: {value!r}")


def _clean(value: object) -> str:
    text = str(value or "").replace("\0", "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    compacted = []
    previous_blank = False
    for line in lines:
        if not line:
            if not previous_blank:
                compacted.append("")
            previous_blank = True
            continue
        compacted.append(line)
        previous_blank = False
    return "\n".join(compacted).strip()


def _normalize_for_hash(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().lower()


def _split_key(seed: int, fingerprint: str) -> str:
    return hashlib.sha256(f"{seed}:{fingerprint}".encode("ascii")).hexdigest()


def _split_recipients(value: object) -> list[str]:
    text = _clean(value)
    if not text:
        return []
    return [item.strip() for item in re.split(r"[,;]", text) if item.strip()]


def _int_or_none(value: object) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return int(float(str(value).strip()))
    except ValueError:
        return None
