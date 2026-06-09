"""Lenient JSON extraction for LLM responses.

LLMs frequently return JSON that is *almost* valid: wrapped in ```json fences,
preceded/followed by prose, truncated at a token cap, or — the common one for this
domain — with thousands-separator commas inside large numbers (``309,389,563``),
which is invalid JSON. This module recovers a usable object from all of those.
"""

from __future__ import annotations

import json
import re
from typing import Any


def _strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    return text


def _strip_inter_digit_commas(text: str) -> str:
    """``1,234,567`` -> ``1234567``. Only commas BETWEEN two digits are removed, so
    commas in strings ("Company, Inc") and JSON separators are left intact."""
    return re.sub(r"(?<=\d),(?=\d)", "", text)


def _first_balanced(text: str, open_ch: str, close_ch: str) -> str | None:
    """Return the first balanced ``open_ch..close_ch`` span, respecting strings."""
    start = text.find(open_ch)
    if start == -1:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def salvage_objects(text: str) -> list[dict[str, Any]]:
    """Recover every complete ``{...}`` object inside the first array (for truncated
    or comma-laden responses). Used as a last resort by array-shaped callers."""
    text = _strip_inter_digit_commas(text)
    idx = text.find("[")
    if idx == -1:
        return []
    objs: list[dict[str, Any]] = []
    depth, start, in_str, esc = 0, None, False, False
    for i in range(idx, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    objs.append(json.loads(text[start : i + 1]))
                except json.JSONDecodeError:
                    pass
                start = None
    return objs


def loads_lenient(text: str, *, array_key: str | None = None) -> dict[str, Any]:
    """Parse an LLM response into a dict, tolerating fences/prose/commas/truncation.

    If ``array_key`` is given and full parsing fails, salvaged objects are returned
    as ``{array_key: [...]}`` so array-shaped callers still get partial results.
    """
    cleaned = _strip_inter_digit_commas(_strip_fences(text))
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    span = _first_balanced(cleaned, "{", "}")
    if span:
        try:
            return json.loads(span)
        except json.JSONDecodeError:
            pass
    if array_key:
        objs = salvage_objects(cleaned)
        if objs:
            return {array_key: objs}
    raise json.JSONDecodeError("no parseable JSON object", text, 0)
