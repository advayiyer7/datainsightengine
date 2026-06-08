"""LLM-as-judge precision pass.

Scores each surfaced finding 0 / 1 / 2 (wrong / trivial / valuable) against its own
evidence and numbers. Used by the evaluator to estimate *validity (precision)* per
approach. Falls back to the answer-key's analyst ``validity`` labels when no LLM is
available, and to a neutral score when neither exists.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .findings import Finding
from .llm import LLMClient

SYSTEM = (
    "You are a strict procurement-insight reviewer. For each finding, judge how valuable and "
    "correct it is GIVEN ONLY its own numbers and evidence. Score:\n"
    "  0 = wrong, misleading, or not a real issue\n"
    "  1 = technically true but trivial / not actionable\n"
    "  2 = a real, non-trivial, actionable insight\n"
    "Return ONLY JSON: {\"scores\":[{\"index\":int,\"score\":0|1|2,\"why\":str}]} with one entry per finding."
)


@dataclass
class JudgeResult:
    scores: list[int] = field(default_factory=list)
    rationales: list[str] = field(default_factory=list)
    used_llm: bool = False
    by_key: dict[str, int] = field(default_factory=dict)  # finding.key() -> score

    @property
    def mean_score(self) -> float:
        return sum(self.scores) / len(self.scores) if self.scores else 0.0

    @property
    def valuable_fraction(self) -> float:
        """Share scored 2 (the precision headline)."""
        return (sum(1 for s in self.scores if s >= 2) / len(self.scores)) if self.scores else 0.0


def _extract_json(text: str) -> dict[str, Any]:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        s, e = text.find("{"), text.rfind("}")
        if s != -1 and e > s:
            return json.loads(text[s : e + 1])
        raise


def judge_findings(
    findings: list[Finding],
    client: LLMClient,
    *,
    max_judged: int = 15,
    fallback_scores: dict[str, int] | None = None,
) -> JudgeResult:
    """Score up to ``max_judged`` findings (highest severity first).

    ``fallback_scores`` maps a finding key -> analyst validity, used when the LLM is
    unavailable. Findings with neither LLM nor fallback score get a neutral 1.
    """
    if not findings:
        return JudgeResult(scores=[], used_llm=False)

    ranked = sorted(findings, key=lambda f: f.severity, reverse=True)[:max_judged]

    if client.available:
        payload = [
            {"index": i, "type": f.type, "entities": f.entities, "metrics": f.metrics,
             "est_impact_usd": f.est_impact_usd, "one_line": f.one_line}
            for i, f in enumerate(ranked)
        ]
        prompt = "Findings to score:\n" + json.dumps(payload, default=str)
        try:
            # Scale output budget to the number of findings (one score object each).
            jt = min(4096, 400 + 80 * len(ranked))
            raw = client.complete(prompt, system=SYSTEM, model=client.cfg.get("narrator_model"),
                                  max_tokens=jt)
            data = _extract_json(raw)
            by_idx = {int(s["index"]): s for s in data.get("scores", [])}
            scores, whys = [], []
            for i in range(len(ranked)):
                s = by_idx.get(i, {})
                scores.append(int(max(0, min(2, s.get("score", 1)))))
                whys.append(str(s.get("why", "")))
            by_key = {ranked[i].key(): scores[i] for i in range(len(ranked))}
            return JudgeResult(scores=scores, rationales=whys, used_llm=True, by_key=by_key)
        except Exception:
            pass  # fall through to non-LLM scoring

    fallback_scores = fallback_scores or {}
    scores = [int(fallback_scores.get(f.key(), 1)) for f in ranked]
    by_key = {ranked[i].key(): scores[i] for i in range(len(ranked))}
    return JudgeResult(scores=scores, rationales=["(no LLM; fallback label)"] * len(scores),
                       used_llm=False, by_key=by_key)
