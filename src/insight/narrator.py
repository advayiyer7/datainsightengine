"""Tier 2 — thin LLM narrator + prioritizer.

Pipeline:
  1. **Select** (no LLM): merge findings by subject, rank by business-impact score,
     keep the top-N (``insight.selection``). The narrator only ever sees the
     selected set — this is where SYS earns its quality over raw detectors.
  2. **Narrate** (LLM): write one plain-language insight per selected group, using
     ONLY numbers that appear verbatim in the findings.
  3. **Repair** (grounding guard): any insight citing an untraceable number is
     regenerated once under a stricter reminder; if it still fails, the offending
     sentence is stripped. The post-fix spurious rate is reported.

Input is the **list of Findings only** — never the raw dataset. Without an API key
the narrator falls back to a deterministic write-up of the same selected set, so
every command runs offline.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .findings import Finding
from .grounding import GroundingResult, check_grounding
from .llm import LLMClient
from .selection import InsightGroup, select_findings, selected_findings

SYSTEM = (
    "You are a procurement analyst. You are given a list of CANDIDATE insight groups. Each group "
    "bundles one or more deterministic findings about the same subject, with exact numbers already "
    "computed from the data.\n\n"
    "Write exactly ONE insight per candidate group (same count, same `id`s — do not drop, add, split, "
    "or merge groups). For each: a short title, what the issue is, why it matters, and a concrete "
    "suggested action.\n\n"
    "ABSOLUTE NUMBER RULE: Every dollar amount, count, and percentage you write MUST appear VERBATIM "
    "in that group's findings (in a metric, est_impact_usd, or one_line). Do NOT round, rescale, sum, "
    "average, or invent any figure. Do NOT introduce illustrative numbers, targets, or thresholds that "
    "are not already in the findings. If you have no exact number for a point, describe it in words "
    "with no number.\n\n"
    "Return ONLY valid JSON (no markdown fence):\n"
    "{\"insights\":[{\"id\":int,\"title\":str,\"what\":str,\"why\":str,\"action\":str}]}"
)


@dataclass
class Insight:
    title: str
    what: str
    why: str
    action: str
    est_impact_usd: float | None = None
    source_types: list[str] = field(default_factory=list)
    group_id: int = -1

    def text(self) -> str:
        return " ".join([self.title, self.what, self.why, self.action])

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "what": self.what,
            "why": self.why,
            "action": self.action,
            "est_impact_usd": self.est_impact_usd,
            "source_types": self.source_types,
        }


@dataclass
class NarrationResult:
    insights: list[Insight]
    grounding: GroundingResult
    used_llm: bool
    markdown: str
    selected: list[Finding] = field(default_factory=list)
    representatives: list[Finding] = field(default_factory=list)  # one headline finding per insight
    spurious_before: int = 0
    spurious_after: int = 0
    total_numbers: int = 0
    raw_response: str = ""

    @property
    def spurious_rate_before(self) -> float:
        return self.spurious_before / self.total_numbers if self.total_numbers else 0.0

    @property
    def spurious_rate_after(self) -> float:
        return self.spurious_after / self.total_numbers if self.total_numbers else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "used_llm": self.used_llm,
            "n_selected_findings": len(self.selected),
            "grounding": self.grounding.to_dict(),
            "spurious_before": self.spurious_before,
            "spurious_after": self.spurious_after,
            "total_numbers": self.total_numbers,
            "spurious_rate_after": round(self.spurious_rate_after, 4),
            "insights": [i.to_dict() for i in self.insights],
        }


# ---------------------------------------------------------------------------
# JSON extraction (tolerant of fences / truncation)
# ---------------------------------------------------------------------------
def _salvage_insights(text: str) -> list[dict[str, Any]]:
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


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        s, e = text.find("{"), text.rfind("}")
        if s != -1 and e != -1 and e > s:
            try:
                return json.loads(text[s : e + 1])
            except json.JSONDecodeError:
                pass
        salvaged = _salvage_insights(text)
        if salvaged:
            return {"insights": salvaged}
        raise


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------
def _group_payload(group: InsightGroup, gid: int) -> dict[str, Any]:
    return {
        "id": gid,
        "entities": group.entities,
        "source_types": group.source_types,
        "est_impact_usd": group.est_impact_usd,
        "findings": [
            {"type": m.type, "one_line": m.one_line, "metrics": m.metrics, "est_impact_usd": m.est_impact_usd}
            for m in group.members
        ],
    }


def _allowed_numbers(findings: list[Finding]) -> list[float]:
    """The exact numbers an insight is allowed to cite (for the strict reminder)."""
    vals: list[float] = []
    def grab(o: Any) -> None:
        if isinstance(o, bool):
            return
        if isinstance(o, (int, float)):
            vals.append(float(o))
        elif isinstance(o, dict):
            for v in o.values():
                grab(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                grab(v)
    for f in findings:
        grab(f.metrics)
        if f.est_impact_usd is not None:
            vals.append(float(f.est_impact_usd))
    seen, uniq = set(), []
    for v in vals:
        r = round(v, 2)
        if r not in seen:
            seen.add(r)
            uniq.append(r)
    return uniq


# ---------------------------------------------------------------------------
# Grounding repair (Fix 3)
# ---------------------------------------------------------------------------
def _split_sentences(s: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", s.strip())
    return [p for p in parts if p]


def _strip_ungrounded_sentences(field_text: str, bad_tokens: list[str]) -> str:
    """Drop sentences that contain any flagged (untraceable) number token."""
    if not bad_tokens:
        return field_text
    kept = []
    for sent in _split_sentences(field_text):
        if any(tok and tok in sent for tok in bad_tokens):
            continue
        kept.append(sent)
    return " ".join(kept).strip()


def _regenerate_insight(client: LLMClient, group: InsightGroup, gid: int, model: str) -> Insight | None:
    """Re-ask for a single insight under a stricter, number-restricted reminder."""
    allowed = _allowed_numbers(group.members)
    allowed_str = ", ".join(f"{a:g}" for a in allowed) or "(none)"
    sys = (
        SYSTEM
        + "\n\nSTRICT RETRY: A previous draft cited a number not in the findings. The ONLY numbers you "
        "may write for this insight are exactly these: [" + allowed_str + "]. Use a subset of them or "
        "no numbers at all. Any other digit is forbidden."
    )
    user = "Single candidate group:\n" + json.dumps(_group_payload(group, gid), default=str)
    try:
        raw = client.complete(user, system=sys, model=model, max_tokens=700)
        data = _extract_json(raw)
        items = data.get("insights", [])
        item = items[0] if items else data
        return Insight(
            title=str(item.get("title", "")).strip(),
            what=str(item.get("what", "")).strip(),
            why=str(item.get("why", "")).strip(),
            action=str(item.get("action", "")).strip(),
            est_impact_usd=group.est_impact_usd,
            source_types=group.source_types,
            group_id=gid,
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _render_markdown(insights: list[Insight], used_llm: bool, res: "NarrationResult") -> str:
    lines = ["# Procurement Insights Report", ""]
    badge = "LLM narrator (Claude Haiku)" if used_llm else "deterministic fallback (no API key)"
    lines.append(f"_Generated via {badge}. Showing the top {len(insights)} selected insights "
                 f"(merged & ranked from the full detector set)._")
    g = res.grounding
    lines.append(
        f"_Grounding: {g.matched}/{g.total_numbers} cited numbers traced to findings"
        + (" ✓ all grounded._" if g.ok else
           f" — {len(g.unmatched)} still ungrounded after repair (was {res.spurious_before})._"))
    lines.append("")
    total_impact = sum(i.est_impact_usd or 0 for i in insights)
    if total_impact:
        lines.append(f"**Total flagged impact across selected insights: ~${total_impact:,.0f}**")
        lines.append("")
    for rank, ins in enumerate(insights, 1):
        impact = f"~${ins.est_impact_usd:,.0f}" if ins.est_impact_usd else "n/a"
        lines.append(f"## {rank}. {ins.title}  (est. impact {impact})")
        if ins.what:
            lines.append(f"- **What:** {ins.what}")
        if ins.why:
            lines.append(f"- **Why it matters:** {ins.why}")
        if ins.action:
            lines.append(f"- **Suggested action:** {ins.action}")
        if ins.source_types:
            lines.append(f"- _Sources: {', '.join(sorted(set(ins.source_types)))}_")
        lines.append("")
    if not g.ok:
        lines.append("---")
        lines.append("> ⚠ **Figures the grounding guard could not trace (left flagged, not trusted):**")
        for u in g.unmatched:
            lines.append(f"> - `{u['text']}`")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Deterministic fallback (no key)
# ---------------------------------------------------------------------------
def _deterministic_insights(groups: list[InsightGroup]) -> list[Insight]:
    out: list[Insight] = []
    for gid, g in enumerate(groups):
        rep = g.representative
        out.append(
            Insight(
                title=rep.type.replace("_", " ").title(),
                what=rep.one_line,
                why=(f"Estimated impact ${g.est_impact_usd:,.0f}." if g.est_impact_usd else "Qualitative risk.")
                + (f" Related signals: {', '.join(g.source_types)}." if len(g.source_types) > 1 else ""),
                action="Review the evidence rows and address the flagged pattern.",
                est_impact_usd=g.est_impact_usd,
                source_types=g.source_types,
                group_id=gid,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------
def narrate(
    findings: list[Finding],
    client: LLMClient,
    *,
    top_n: int = 8,
    rank_formula: str = "severity_x_impact",
    null_impact_usd: float = 20000.0,
    rel_tol: float = 0.02,
    repair: bool = True,
) -> NarrationResult:
    """Select the top-``top_n`` insight groups and narrate them with grounding repair."""
    if not findings:
        gr = GroundingResult(ok=True, total_numbers=0, matched=0)
        return NarrationResult([], gr, used_llm=False,
                               markdown="# Procurement Insights Report\n\n_No findings._")

    groups = select_findings(findings, top_n=top_n, formula=rank_formula, null_impact_usd=null_impact_usd)
    sel = selected_findings(groups)
    model = client.cfg.get("narrator_model")

    used_llm = client.available
    insights: list[Insight] = []
    raw = ""
    if used_llm:
        payload = [_group_payload(g, gid) for gid, g in enumerate(groups)]
        user = ("Candidate insight groups (write one insight each, preserving id):\n"
                + json.dumps(payload, default=str))
        try:
            raw = client.complete(user, system=SYSTEM, model=model)
            data = _extract_json(raw)
            by_id = {int(it.get("id", i)): it for i, it in enumerate(data.get("insights", []))}
            for gid, g in enumerate(groups):
                it = by_id.get(gid, {})
                insights.append(
                    Insight(
                        title=str(it.get("title", "")).strip() or g.representative.type.replace("_", " ").title(),
                        what=str(it.get("what", "")).strip() or g.representative.one_line,
                        why=str(it.get("why", "")).strip(),
                        action=str(it.get("action", "")).strip(),
                        est_impact_usd=g.est_impact_usd,
                        source_types=g.source_types,
                        group_id=gid,
                    )
                )
        except Exception:
            used_llm = False
            insights = []

    if not insights:
        used_llm = False
        insights = _deterministic_insights(groups)

    # --- spurious-number measurement + repair (Fix 3) ---
    spurious_before = 0
    if used_llm:
        for ins in insights:
            grp = groups[ins.group_id]
            gres = check_grounding(ins.text(), grp.members, rel_tol=rel_tol, ignore_below=12)
            spurious_before += len(gres.unmatched)
            if repair and not gres.ok and client.available:
                regen = _regenerate_insight(client, grp, ins.group_id, model)
                if regen is not None:
                    rres = check_grounding(regen.text(), grp.members, rel_tol=rel_tol, ignore_below=12)
                    if len(rres.unmatched) < len(gres.unmatched):
                        ins.title, ins.what, ins.why, ins.action = regen.title, regen.what, regen.why, regen.action
                        gres = rres
                # final guard: strip any sentence still carrying a flagged number
                if not gres.ok:
                    bad = [u["text"] for u in gres.unmatched]
                    ins.what = _strip_ungrounded_sentences(ins.what, bad)
                    ins.why = _strip_ungrounded_sentences(ins.why, bad)
                    ins.action = _strip_ungrounded_sentences(ins.action, bad)

    # --- final overall grounding over the (repaired) insights ---
    narration_text = "\n".join(i.text() for i in insights)
    grounding = check_grounding(narration_text, sel, rel_tol=rel_tol, ignore_below=12)

    representatives = [groups[i.group_id].representative for i in insights]

    res = NarrationResult(
        insights=insights,
        grounding=grounding,
        used_llm=used_llm,
        markdown="",
        selected=sel,
        representatives=representatives,
        spurious_before=spurious_before,
        spurious_after=len(grounding.unmatched),
        total_numbers=grounding.total_numbers,
        raw_response=raw,
    )
    res.markdown = _render_markdown(insights, used_llm, res)
    return res
