"""Ranking, merging, and top-N selection of findings (Tier-2 pre-step).

The narrator used to pass through every detector finding, so it added no quality
over raw detectors and surfaced many low-value items. This module fixes that: it
**merges** findings about the same subject (same supplier+item across detector
types), **ranks** the merged groups by a configurable business-impact score, and
**selects** only the top-N. The narrator then writes prose for those N groups only.

Pure functions, no LLM, no I/O — so the ranking/selection is unit-tested
deterministically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .findings import Finding


def _merge_key(f: Finding) -> str:
    """Group findings that describe the same subject.

    Primary key is (supplier, item). Findings missing one of those fall back to
    whichever entity they do carry, so e.g. two findings on the same supplier+item
    from different detectors merge, while unrelated findings stay separate.
    """
    ents = f.entities or {}
    supplier = ents.get("supplier") or ents.get("worst_supplier") or ents.get("best_supplier")
    item = ents.get("item") or ents.get("category")
    if supplier and item:
        return f"sup={supplier}|item={item}"
    if supplier:
        return f"sup={supplier}|sig={ents.get('signal', '')}"
    if item:
        return f"item={item}"
    # No entities (e.g. tail_spend) — keep each such finding distinct by type+oneline.
    return f"type={f.type}|{f.one_line[:40]}"


def score_finding(f: Finding, *, formula: str = "severity_x_impact", null_impact_usd: float = 20000.0) -> float:
    """Business-impact score for ranking. Higher = surface sooner.

    Strategies:
      * ``severity_x_impact`` (default): ``severity * impact`` where impact falls
        back to ``null_impact_usd`` when ``est_impact_usd`` is None.
      * ``impact``: the dollar impact alone (null -> ``null_impact_usd``).
      * ``severity``: the 0-1 severity alone.
    """
    impact = f.est_impact_usd if f.est_impact_usd is not None else null_impact_usd
    impact = max(0.0, float(impact))
    sev = max(0.0, float(f.severity))
    if formula == "impact":
        return impact
    if formula == "severity":
        return sev
    # default
    return sev * impact


@dataclass
class InsightGroup:
    """A merged set of findings about one subject, with an aggregate score."""

    key: str
    members: list[Finding] = field(default_factory=list)
    score: float = 0.0
    entities: dict[str, Any] = field(default_factory=dict)
    est_impact_usd: float | None = None
    source_types: list[str] = field(default_factory=list)

    @property
    def representative(self) -> Finding:
        """Highest-severity member — used for the headline entities/oneline."""
        return max(self.members, key=lambda m: m.severity)


def merge_findings(findings: list[Finding]) -> list[InsightGroup]:
    """Collapse findings into :class:`InsightGroup`s by subject (merge key)."""
    groups: dict[str, InsightGroup] = {}
    for f in findings:
        k = _merge_key(f)
        g = groups.setdefault(k, InsightGroup(key=k))
        g.members.append(f)
    for g in groups.values():
        rep = g.representative
        g.entities = dict(rep.entities)
        g.source_types = sorted({m.type for m in g.members})
        impacts = [m.est_impact_usd for m in g.members if m.est_impact_usd is not None]
        # Aggregate impact = max member impact (avoid double-counting overlapping savings).
        g.est_impact_usd = max(impacts) if impacts else None
    return list(groups.values())


def select_findings(
    findings: list[Finding],
    *,
    top_n: int = 8,
    formula: str = "severity_x_impact",
    null_impact_usd: float = 20000.0,
    max_per_type: int | None = None,
) -> list[InsightGroup]:
    """Merge, score, and return the top-``top_n`` groups, highest score first.

    Each group's score is the **max** score over its members (the strongest signal
    drives the ranking), so merging never dilutes an important finding.

    ``max_per_type`` caps how many groups of any one detector type appear, so the
    report stays diverse when one type dominates the high-impact tail (e.g. many
    billion-dollar single-source contracts). If the cap leaves fewer than ``top_n``,
    the remainder is back-filled by score regardless of type.
    """
    groups = merge_findings(findings)
    for g in groups:
        g.score = max(
            (score_finding(m, formula=formula, null_impact_usd=null_impact_usd) for m in g.members),
            default=0.0,
        )
    groups.sort(key=lambda g: g.score, reverse=True)
    top_n = max(0, top_n)
    if not max_per_type:
        return groups[:top_n]

    chosen: list[InsightGroup] = []
    counts: dict[str, int] = {}
    for g in groups:
        t = g.representative.type
        if counts.get(t, 0) < max_per_type:
            chosen.append(g)
            counts[t] = counts.get(t, 0) + 1
        if len(chosen) >= top_n:
            return chosen
    # Back-fill if the per-type cap left us short of top_n.
    if len(chosen) < top_n:
        picked = set(id(g) for g in chosen)
        for g in groups:
            if id(g) not in picked:
                chosen.append(g)
                if len(chosen) >= top_n:
                    break
    return chosen[:top_n]


def selected_findings(groups: list[InsightGroup]) -> list[Finding]:
    """Flatten selected groups back to a finding list (for scoring/grounding)."""
    out: list[Finding] = []
    for g in groups:
        out.extend(g.members)
    return out
