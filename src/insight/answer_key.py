"""Ground-truth insight set ("answer key") for the evaluation harness.

Built in two passes:
  1. Run the detectors at STRICT thresholds → high-confidence candidate insights.
  2. Run the LLM-judge over the candidates and keep ONLY those scored *valuable*
     (2/2). This stops recall from being near-circular — without it, the answer key
     is just raw detector output and detector-based approaches trivially score 1.0.

The result is written with ``"curated": false`` and a header telling you to review,
edit, and — importantly — ADD insights the detectors cannot catch. Recall against a
detector-only key is upper-bounded by construction until a human does that.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .detectors import run_all
from .findings import Finding
from .judge import judge_findings
from .llm import LLMClient

# Stricter overrides layered on top of the user's detector config: fewer, surer hits.
STRICT_OVERRIDES: dict[str, dict[str, Any]] = {
    "supplier_concentration": {},                       # always a single, solid finding
    "tail_spend": {},
    "yoy_spend_movers": {"min_abs_change": 100000.0, "top_k": 5},
    "negative_or_anomalous_spend": {"outlier_k": 4.0, "top_outliers": 5},
    "new_and_churned_suppliers": {"material": 50000.0},
}


def _strict_cfg(detector_cfg: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(detector_cfg or {})
    for name, ov in STRICT_OVERRIDES.items():
        out.setdefault(name, {})
        out[name].update(ov)
    return out


def build_answer_key(
    df: pd.DataFrame,
    detector_cfg: dict[str, Any],
    client: LLMClient | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return (entries, meta) for the answer key.

    Candidates come from strict-threshold detectors. If an LLM ``client`` is
    available, the judge scores them and only *valuable* (>=2) candidates are kept;
    otherwise all candidates are kept unjudged (validity 2) and ``meta`` says so.
    """
    candidates = run_all(df, _strict_cfg(detector_cfg))
    # On large datasets the strict detectors can still emit thousands of candidates;
    # cap the ground-truth set to the highest-severity ones so the judge call is bounded.
    MAX_CANDIDATES = 60
    capped = len(candidates) > MAX_CANDIDATES
    candidates = sorted(candidates, key=lambda f: f.severity, reverse=True)[:MAX_CANDIDATES]
    judged = False
    scores: dict[str, int] = {}
    if client is not None and client.available and candidates:
        jr = judge_findings(candidates, client, max_judged=len(candidates))
        judged = jr.used_llm
        scores = jr.by_key

    entries: list[dict[str, Any]] = []
    n_dropped = 0
    for f in candidates:
        score = scores.get(f.key(), 2)
        if judged and score < 2:
            n_dropped += 1
            continue  # keep only judge-valuable insights
        entries.append(
            {
                "key": f.key(),
                "type": f.type,
                "entities": f.entities,
                "metrics": f.metrics,
                "est_impact_usd": f.est_impact_usd,
                "one_line": f.one_line,
                "include": True,        # set False to drop from ground truth
                "validity": int(score), # 0 wrong / 1 trivial / 2 valuable (judge or analyst)
                "source": "detector",   # human-added entries should use source:"manual"
            }
        )
    meta = {
        "curated": False,
        "judged": judged,
        "candidates": len(candidates),
        "kept_valuable": len(entries),
        "dropped_trivial_or_wrong": n_dropped,
        "candidates_capped_at": MAX_CANDIDATES if capped else None,
    }
    return entries, meta


def write_answer_key(entries: list[dict[str, Any]], path: str | Path, meta: dict[str, Any] | None = None) -> None:
    meta = meta or {}
    payload = {
        "_README": (
            "REVIEW REQUIRED. This key was auto-generated from detectors and filtered by the LLM-judge "
            "to keep only insights it scored valuable (2/2). It is NOT yet curated. To make recall "
            "meaningful you should: (1) review each entry and set validity 0/1/2 or include=false; "
            "(2) ADD insights the fixed detectors cannot catch (set source:\"manual\") — otherwise "
            "detector-based approaches are upper-bounded to ~1.0 recall by construction. "
            "Then set curated:true."
        ),
        "curated": bool(meta.get("curated", False)),
        "meta": meta,
        "entries": entries,
    }
    Path(path).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def load_answer_key(path: str | Path) -> tuple[list[dict[str, Any]], bool]:
    """Return (included entries, curated_flag)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = [e for e in data.get("entries", []) if e.get("include", True)]
    curated = bool(data.get("curated", False))
    return entries, curated
