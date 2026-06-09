"""B0 — the "dump everything into the LLM" baseline.

Feeds as much raw data as the context budget allows directly to the LLM and asks
for insights. This is the expensive ceiling the thesis argues against: cost grows
with row count while the detector approach stays roughly flat. If the data exceeds
``baseline_max_rows`` we sample and say so.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .findings import Finding
from .jsonx import loads_lenient
from .llm import LLMClient

SYSTEM = (
    "You are a procurement spend analyst. You are given RAW supplier-spend rows as CSV (one row per "
    "supplier-year: supplier, spend, prev_spend, ...). Find the most valuable, actionable insights "
    "(spend concentration, tail spend, big YoY moves, negative/anomalous spend, new/churned suppliers). "
    "Any provided yoyChange/flag columns are noisy (tiny prior-spend denominators) — recompute YoY "
    "yourself. For each insight return a finding object. Return ONLY JSON: "
    "{\"findings\":[{\"type\":str,\"severity\":0-1,\"entities\":{...},\"metrics\":{...numbers...},"
    "\"evidence\":{...},\"est_impact_usd\":number|null,\"one_line\":str}]}.\n"
    "Use finding 'type' values from this vocabulary when applicable so results are comparable: "
    "supplier_concentration, tail_spend, yoy_spend_movers, negative_or_anomalous_spend, "
    "new_and_churned_suppliers. Base every number on the rows you were given."
)


@dataclass
class BaselineResult:
    findings: list[Finding] = field(default_factory=list)
    rows_fed: int = 0
    sampled: bool = False
    note: str = ""
    raw_response: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows_fed": self.rows_fed,
            "sampled": self.sampled,
            "note": self.note,
            "n_findings": len(self.findings),
            "findings": [f.to_dict() for f in self.findings],
        }


def _extract_json(text: str) -> dict[str, Any]:
    return loads_lenient(text, array_key="findings")


def _columns_for_baseline(df: pd.DataFrame) -> list[str]:
    # Spend-grain columns first, then transaction-grain fallbacks.
    prefer = ["supplier", "spend", "prev_spend", "year",
              "item", "category", "quantity", "effective_unit_price",
              "total", "order_date", "lead_time_days", "order_status"]
    return [c for c in prefer if c in df.columns]


def dump_to_llm(df: pd.DataFrame, client: LLMClient, cfg: dict[str, Any]) -> BaselineResult:
    """Run the B0 baseline. ``cfg`` is the top-level config (uses ``evaluate`` + ``llm``)."""
    max_rows = int(cfg.get("evaluate", {}).get("baseline_max_rows", 400))
    res = BaselineResult()
    if not client.available:
        res.note = "skipped: no ANTHROPIC_API_KEY (B0 requires the LLM)"
        return res

    cols = _columns_for_baseline(df)
    feed = df[cols]
    if len(feed) > max_rows:
        feed = feed.sample(max_rows, random_state=0).sort_index()
        res.sampled = True
        res.note = f"sampled {max_rows} of {len(df)} rows to fit context budget"
    res.rows_fed = len(feed)

    csv_text = feed.to_csv(index=False)
    prompt = (
        f"There are {len(df)} total rows"
        + (f" (showing a random sample of {len(feed)})" if res.sampled else "")
        + f".\n\nRAW ROWS (CSV):\n{csv_text}"
    )
    model = client.cfg.get("baseline_model", client.cfg.get("narrator_model"))
    try:
        raw = client.complete(prompt, system=SYSTEM, model=model,
                              max_tokens=int(client.cfg.get("max_output_tokens", 2048)))
        res.raw_response = raw
        data = _extract_json(raw)
        for fd in data.get("findings", []) or []:
            try:
                res.findings.append(Finding.from_dict({**fd, "source": "baseline_llm"}))
            except Exception:
                continue
    except Exception as exc:
        res.note = f"B0 failed: {exc}"
    return res
