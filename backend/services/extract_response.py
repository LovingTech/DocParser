"""Recover and normalize a JSON object from raw VLM text against a schema.

The VLM is asked to return JSON, but real models wrap it in prose, code
fences, trailing commas, single quotes, or return values of the wrong type.
This module turns that messy output into the requested field structure.

Design
------
* ``repair_json`` re-parses the *same* bytes more cleverly (fences, first
  balanced object, trailing commas, JS-isms). No retry -- it only re-parses.
* ``normalize_response`` validates the top level is an object and coerces each
  requested field's value to the type the schema declares (string <-> number
  <-> bool, array-of-objects wrapping, etc.).
* ``parse_extracted`` ties them together and raises ``ExtractResponseError``
  (a ``ValueError``) only when the response cannot be recovered at all. The
  service layer retries on that raised error; lighter recoveries happen inline.
"""

from __future__ import annotations

import json
import re
from typing import Any

from backend.models import FieldSchema, FieldType, Schema

ParsedContent = dict[str, Any]


class ExtractResponseError(ValueError):
    """Raised when a model response cannot be recovered into a JSON object.

    Subclasses ``ValueError`` so the extraction route maps it to HTTP 400
    ("the request/response was bad") rather than a server 500.
    """


# --- Recovery (re-parse the same bytes, no retry) -------------------------- #
_FENCE_RE = re.compile(r"```(?:json)\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_UNDEF_RE = re.compile(r"\b(?:undefined|NaN|Infinity)\b")
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")


def repair_json(content: str | None) -> str | None:
    """Turn raw LLM text into a JSON string, or ``None`` if impossible.

    Applies a ladder of increasingly aggressive strategies and returns the
    first candidate that actually parses.
    """
    for text in _candidate_texts(content):
        normalized = _normalize_json_text(text)
        if normalized is None:
            continue
        try:
            json.loads(normalized)
        except json.JSONDecodeError:
            continue
        return normalized
    return None


def _candidate_texts(content: str | None) -> list[str]:
    if not content:
        return []
    text = content.strip()
    candidates: list[str] = [text]

    fence_match = _FENCE_RE.search(text)
    if fence_match and fence_match.group(1).strip().startswith("{"):
        candidates.append(fence_match.group(1).strip())

    obj = _first_object(text)
    if obj is not None and obj != text:
        candidates.append(obj)

    return candidates


def _first_object(text: str) -> str | None:
    """Return the first balanced ``{...}`` block, skipping over string bodies."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    escape = False
    quote = ""
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote:
                in_str = False
            continue
        if ch in ('"', "'"):
            in_str = True
            quote = ch
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _normalize_json_text(text: str) -> str | None:
    text = text.strip()
    if not (text.startswith("{") and text.endswith("}")):
        return None
    text = _UNDEF_RE.sub("null", text)
    text = _TRAILING_COMMA_RE.sub(r"\1", text)
    if '"' not in text and "'" in text:
        text = text.replace("'", '"')
    return text


# --- Validation + coercion ------------------------------------------------- #
def parse_extracted(content: str, schema: Schema) -> ParsedContent:
    """Recover a JSON object from ``content`` and normalize it to ``schema``."""
    repaired = repair_json(content)
    if repaired is None:
        raise ExtractResponseError(
            "could not recover a JSON object from the model response"
        )
    data = json.loads(repaired)
    if not isinstance(data, dict):
        raise ExtractResponseError(
            "the model response was not a JSON object; expected named fields"
        )
    return normalize_response(data, schema)


def normalize_response(data: dict[str, Any], schema: Schema) -> ParsedContent:
    """Return only the requested fields, coerced to their declared types."""
    field_map = {field.name: field for field in schema.fields}
    result: dict[str, Any] = {}
    for name, field in field_map.items():
        if name in data:
            result[name] = coerce_value(data[name], field)
    return result


def coerce_value(value: Any, field: FieldSchema) -> Any:
    if field.type == FieldType.ANY:
        return value
    if field.type == FieldType.ARRAY:
        return coerce_array(value, field.items)
    if field.type == FieldType.OBJECT:
        return coerce_object(value, field.items)
    return coerce_scalar(value, field.type)


def coerce_scalar(value: Any, ftype: FieldType) -> Any:
    if value is None:
        return None
    if ftype == FieldType.STRING:
        return _to_string(value)
    if ftype == FieldType.INTEGER:
        return _to_int(value)
    if ftype == FieldType.NUMBER:
        return _to_number(value)
    if ftype == FieldType.BOOLEAN:
        return _to_bool(value)
    return value  # DATE / unknown types are passed through best-effort


def coerce_array(value: Any, items: list[FieldSchema]) -> Any:
    if value is None:
        return []
    if not isinstance(value, list):
        value = [value]
    if not items:
        return value
    coerced: list[Any] = []
    for entry in value:
        if isinstance(entry, dict):
            sub: dict[str, Any] = {}
            for item in items:
                if item.name in entry:
                    sub[item.name] = coerce_value(entry[item.name], item)
            coerced.append(sub)
        else:
            coerced.append(entry)
    return coerced


def coerce_object(value: Any, items: list[FieldSchema]) -> Any:
    if not isinstance(value, dict):
        return value
    result: dict[str, Any] = {}
    for item in items:
        if item.name in value:
            result[item.name] = coerce_value(value[item.name], item)
    return result


_STRING_TRUE = {"true", "t", "1", "yes", "y", "on"}
_STRING_FALSE = {"false", "f", "0", "no", "n", "off"}
_INT_RE = re.compile(r"^[+-]?\d+$")
_NUMBER_RE = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$")


def _to_string(value: Any) -> Any:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value
    return str(value)


def _to_int(value: Any) -> Any:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "").replace(" ", "")
        if _INT_RE.match(cleaned):
            return int(cleaned)
    return value


def _to_number(value: Any) -> Any:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = (
            value.strip().replace(",", "").replace(" ", "").replace("$", "").replace("+", "")
        )
        if _NUMBER_RE.match(cleaned):
            return float(cleaned)
    return value


def _to_bool(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        token = value.strip().lower()
        if token in _STRING_TRUE:
            return True
        if token in _STRING_FALSE:
            return False
    return value
