"""Grounding / anti-hallucination guard for narrated insights.

The narrator may only restate numbers that exist in the source Findings. This
module extracts every dollar amount, percentage, and count from a piece of
narration and checks each against the pool of numbers present in the Findings'
metrics/impact. Anything unmatched is flagged as a potential fabrication.

The check is deliberately lenient about *formatting* (``$8.3M`` == ``8253782`` to
two sig-figs, ``~$1,500`` == ``1500``) but strict about *existence*: a number that
matches nothing in the findings is reported.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from .findings import Finding

# $1,234.56  |  $8.3M  |  $5k  |  1,200  |  21%  |  4
# The k/m/b suffix only counts when directly attached to the digits AND not the
# first letter of a following word (so "$49.94 benchmark" is not read as billions).
_NUM_RE = re.compile(
    r"""
    (?P<dollar>\$\s?\d[\d,]*\.?\d*(?P<suf>[kKmMbB])?(?![A-Za-z]))   # money, optional k/m/b
    | (?P<pct>\d+\.?\d*)\s?%                                        # percentage
    | (?P<bare>\b\d[\d,]*\.?\d*\b)                                  # bare number
    """,
    re.VERBOSE,
)

_SUFFIX = {"k": 1e3, "m": 1e6, "b": 1e9}


@dataclass
class GroundingResult:
    ok: bool
    total_numbers: int
    matched: int
    unmatched: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "total_numbers": self.total_numbers,
            "matched": self.matched,
            "unmatched": self.unmatched,
        }


def _collect_numbers(obj: Any, out: list[float]) -> None:
    """Recursively gather every numeric value from a findings structure."""
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect_numbers(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _collect_numbers(v, out)
    elif isinstance(obj, str):
        # numbers embedded in one_line strings are themselves derived from metrics
        for m in re.finditer(r"-?\d[\d,]*\.?\d*", obj):
            try:
                out.append(float(m.group().replace(",", "")))
            except ValueError:
                pass


def allowed_value_pool(findings: Iterable[Finding]) -> set[float]:
    """Build the set of numbers a narration is allowed to cite.

    Includes raw metric values and impacts, their integer rounding, and the
    percentage form (``*100``) of any ratio in [0, 1] so ``0.21`` justifies ``21%``.
    """
    raw: list[float] = []
    for f in findings:
        _collect_numbers(f.metrics, raw)
        _collect_numbers(f.entities, raw)
        _collect_numbers(f.evidence, raw)  # row-ids and proof values are part of the finding
        if f.est_impact_usd is not None:
            raw.append(float(f.est_impact_usd))
        raw.append(float(f.severity))
    pool: set[float] = set()
    for v in raw:
        if not math.isfinite(v):  # skip NaN/inf (data can carry NaN metrics)
            continue
        pool.add(v)
        pool.add(round(v))
        pool.add(round(v, 2))
        # A ratio justifies its percentage form. Handle negative ratios too: a
        # -0.64 YoY change is narrated as "64%" (sign carried in words).
        if abs(v) <= 1.0:
            for pct in (v * 100, abs(v) * 100):
                pool.add(round(pct, 2))
                pool.add(round(pct))
    return pool


def _parse_token(m: re.Match) -> tuple[float, str] | None:
    if m.group("dollar"):
        txt = m.group("dollar").replace("$", "").replace(",", "").strip()
        suf = m.group("suf")
        if suf:
            txt = txt[: -1].strip()
            try:
                return float(txt) * _SUFFIX[suf.lower()], "dollar"
            except ValueError:
                return None
        try:
            return float(txt), "dollar"
        except ValueError:
            return None
    if m.group("pct") is not None:
        try:
            return float(m.group("pct")), "pct"
        except ValueError:
            return None
    if m.group("bare"):
        try:
            return float(m.group("bare").replace(",", "")), "bare"
        except ValueError:
            return None
    return None


def _matches(value: float, pool: set[float], rel_tol: float, abs_tol: float) -> bool:
    for allowed in pool:
        if abs(value - allowed) <= abs_tol:
            return True
        denom = max(abs(allowed), 1e-9)
        if abs(value - allowed) / denom <= rel_tol:
            return True
    return False


def check_grounding(
    narration: str,
    findings: Iterable[Finding],
    *,
    rel_tol: float = 0.02,
    abs_tol: float = 1.0,
    ignore_below: float = 0.0,
) -> GroundingResult:
    """Verify every number in ``narration`` traces to a Finding.

    ``rel_tol`` allows for rounding (2% by default). ``ignore_below`` skips tiny
    integers (e.g. list markers ``1.``, ``2.``) that aren't real claims — set to 0
    to check everything. Returns a :class:`GroundingResult`; ``ok`` is False if any
    number is unmatched.
    """
    findings = list(findings)
    pool = allowed_value_pool(findings)
    total = 0
    matched = 0
    unmatched: list[dict[str, Any]] = []
    for m in _NUM_RE.finditer(narration):
        parsed = _parse_token(m)
        if parsed is None:
            continue
        value, kind = parsed
        token = m.group().strip()
        # Skip identifier-like tokens (leading-zero numbers, e.g. PO row-ids "00115").
        if kind == "bare" and re.match(r"0\d", token.replace(",", "")):
            continue
        if kind == "bare" and value <= ignore_below:
            continue
        total += 1
        if _matches(value, pool, rel_tol, abs_tol):
            matched += 1
        else:
            unmatched.append({"text": m.group().strip(), "value": value, "kind": kind})
    return GroundingResult(ok=len(unmatched) == 0, total_numbers=total, matched=matched, unmatched=unmatched)
