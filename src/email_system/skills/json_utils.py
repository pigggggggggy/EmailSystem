from __future__ import annotations

import json
import re
from typing import Any


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = _strip_fence(text.strip())
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        value = json.loads(_first_json_object(cleaned))
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object from model output")
    return value


def _strip_fence(text: str) -> str:
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    return match.group(1).strip() if match else text


def _first_json_object(text: str) -> str:
    start = text.find("{")
    if start < 0:
        raise json.JSONDecodeError("No JSON object found", text, 0)
    depth = 0
    in_string = False
    escape = False
    for index, char in enumerate(text[start:], start=start):
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    raise json.JSONDecodeError("Unclosed JSON object", text, start)
