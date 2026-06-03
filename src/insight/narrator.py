"""Tier 2 — thin LLM narrator + prioritizer.

Input is the **list of Findings only** (never the raw dataset). The LLM ranks by
business impact, merges duplicates, and writes a short recommendation per top
insight. A hard prompt rule forbids inventing numbers; after generation the
grounding guard (``insight.grounding``) verifies every figure traces to a Finding.

If no API key is available the narrator falls back to a deterministic ranking of
the detector ``one_line``s — the pipeline still produces a report, just without
LLM prose. This keeps every command runnable offline and is reported honestly.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .findings import Finding, findings_to_json
from .grounding import GroundingResult, check_grounding
from .llm import LLMClient

SYSTEM = (
    "You are a procurement analyst. You will be given a JSON list of FINDINGS, each "
    "already containing exact numbers (metrics, est_impact_usd) computed deterministically "
    "from the data. Your job: (1) rank insights by business impact, (2) merge findings that "
    "describe the same underlying issue, (3) for the top insights write a short, plain-language "
    "recommendation.\n\n"
    "HARD RULE: You may ONLY use numbers that appear in the FINDINGS JSON. Never invent, "
    "extrapolate, or compute new figures. If you cite a dollar amount, count, or percentage, "
    "it MUST be copied from a finding's metrics/est_impact_usd. When unsure, restate the "
    "finding's own numbers verbatim.\n\n"
    "Return ONLY valid JSON (no markdown fence) of the form:\n"
    "{\"insights\": [{\"title\": str, \"what\": str, \"why\": str, \"est_impact_usd\": number|null, "
    "\"action\": str, \"source_types\": [str], \"source_indices\": [int]}]}\n"
    "Order the insights array from highest to lowest business impact. Include at most 12 insights."
)

USER_TEMPLATE = (
    "Here are {n} findings as JSON. Produce the ranked, merged insights per the rules.\n\n"
    "FINDINGS:\n{findings_json}"
)


@dataclass
class Insight:
    title: str
    what: str
    why: str
    action: str
    est_impact_usd: float | None = None
    source_types: list[str] = field(default_factory=list)
    source_indices: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "what": self.what,
            "why": self.why,
            "action": self.action,
            "est_impact_usd": self.est_impact_usd,
            "source_types": self.source_types,
            "source_indices": self.source_indices,
        }


@dataclass
class NarrationResult:
    insights: list[Insight]
    grounding: GroundingResult
    used_llm: bool
    markdown: str
    raw_response: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "used_llm": self.used_llm,
            "grounding": self.grounding.to_dict(),
            "insights": [i.to_dict() for i in self.insights],
        }


def _salvage_insights(text: str) -> list[dict[str, Any]]:
    """Recover complete insight objects from a truncated ``"insights": [...]`` array.

    The LLM occasionally hits the output-token cap mid-array; rather than lose the
    whole response we bracket-scan and keep every fully-closed object.
    """
    idx = text.find("[")
    if idx == -1:
        return []
    objs: list[dict[str, Any]] = []
    depth = 0
    start = None
    in_str = False
    esc = False
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


def _extract_json(text: str) -> dict[str, Any]:
    """Pull a JSON object out of an LLM response, tolerating stray prose/fences.

    Falls back to salvaging complete objects from a truncated array.
    """
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
        salvaged = _salvage_insights(text)
        if salvaged:
            return {"insights": salvaged}
        raise


def _deterministic_insights(findings: list[Finding], top: int) -> list[Insight]:
    """Fallback: rank findings by severity and template their own one_line."""
    out: list[Insight] = []
    for i, f in enumerate(sorted(findings, key=lambda x: x.severity, reverse=True)[:top]):
        out.append(
            Insight(
                title=f.type.replace("_", " ").title(),
                what=f.one_line,
                why=f"Severity {f.severity:.2f}; "
                + (f"estimated impact ${f.est_impact_usd:,.0f}." if f.est_impact_usd else "qualitative risk."),
                action="Review the evidence rows and address the flagged pattern.",
                est_impact_usd=f.est_impact_usd,
                source_types=[f.type],
                source_indices=[i],
            )
        )
    return out


def _render_markdown(insights: list[Insight], used_llm: bool, grounding: GroundingResult) -> str:
    lines = ["# Procurement Insights Report", ""]
    badge = "LLM narrator (Claude Haiku)" if used_llm else "deterministic fallback (no API key)"
    lines.append(f"_Generated via {badge}._")
    g = grounding
    lines.append(
        f"_Grounding check: {g.matched}/{g.total_numbers} cited numbers traced to findings "
        + ("✓ all grounded._" if g.ok else f"— ⚠ {len(g.unmatched)} ungrounded._")
    )
    lines.append("")
    total_impact = sum(i.est_impact_usd or 0 for i in insights)
    if total_impact:
        lines.append(f"**Total flagged impact across insights: ~${total_impact:,.0f}**")
        lines.append("")
    for rank, ins in enumerate(insights, 1):
        impact = f"~${ins.est_impact_usd:,.0f}" if ins.est_impact_usd else "n/a"
        lines.append(f"## {rank}. {ins.title}  (est. impact {impact})")
        lines.append(f"- **What:** {ins.what}")
        lines.append(f"- **Why it matters:** {ins.why}")
        lines.append(f"- **Suggested action:** {ins.action}")
        if ins.source_types:
            lines.append(f"- _Sources: {', '.join(sorted(set(ins.source_types)))}_")
        lines.append("")
    if not grounding.ok:
        lines.append("---")
        lines.append("> ⚠ **Ungrounded figures flagged by the anti-hallucination guard:**")
        for u in grounding.unmatched:
            lines.append(f"> - `{u['text']}` (parsed {u['value']}) — not found in any finding's metrics.")
    return "\n".join(lines)


def narrate(
    findings: list[Finding],
    client: LLMClient,
    *,
    top: int = 12,
    rel_tol: float = 0.02,
) -> NarrationResult:
    """Run the narrator over ``findings`` and return a grounded report.

    Uses the LLM if a key is available; otherwise a deterministic fallback. The
    grounding guard always runs over the produced narration.
    """
    if not findings:
        gr = GroundingResult(ok=True, total_numbers=0, matched=0)
        return NarrationResult([], gr, used_llm=False, markdown="# Procurement Insights Report\n\n_No findings._")

    used_llm = client.available
    insights: list[Insight] = []
    raw = ""
    if used_llm:
        prompt = USER_TEMPLATE.format(n=len(findings), findings_json=findings_to_json(findings))
        try:
            raw = client.complete(prompt, system=SYSTEM, model=client.cfg.get("narrator_model"))
            data = _extract_json(raw)
            for item in data.get("insights", [])[:top]:
                insights.append(
                    Insight(
                        title=str(item.get("title", "")).strip(),
                        what=str(item.get("what", "")).strip(),
                        why=str(item.get("why", "")).strip(),
                        action=str(item.get("action", "")).strip(),
                        est_impact_usd=item.get("est_impact_usd"),
                        source_types=list(item.get("source_types", []) or []),
                        source_indices=list(item.get("source_indices", []) or []),
                    )
                )
        except Exception:
            # Any LLM/parse failure degrades to the deterministic path, honestly labeled.
            used_llm = False
            insights = []

    if not insights:
        used_llm = False
        insights = _deterministic_insights(findings, top)

    # Grounding: concatenate the narration text and check against the finding pool.
    narration_text = "\n".join(
        f"{i.title} {i.what} {i.why} {i.action} {i.est_impact_usd or ''}" for i in insights
    )
    grounding = check_grounding(narration_text, findings, rel_tol=rel_tol, ignore_below=12)

    markdown = _render_markdown(insights, used_llm, grounding)
    return NarrationResult(insights, grounding, used_llm, markdown, raw)
