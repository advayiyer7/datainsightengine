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
from .llm import LLMClient

SYSTEM = (
    "You are a procurement analyst. You are given RAW purchase-order rows as CSV. Find the most "
    "valuable, actionable insights (avoidable cost, risk, anomalies). For each, return a finding "
    "object. Return ONLY JSON: {\"findings\":[{\"type\":str,\"severity\":0-1,\"entities\":{...},"
    "\"metrics\":{...numbers...},\"evidence\":{...},\"est_impact_usd\":number|null,\"one_line\":str}]}.\n"
    "Use finding 'type' values from this vocabulary when applicable so results are comparable: "
    "fragmented_orders, supplier_concentration, maverick_price_variance, tail_spend, "
    "single_source_risk, timing_anomaly, duplicate_order. Base every number on the rows you were given."
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
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        s, e = text.find("{"), text.rfind("}")
        if s != -1 and e > s:
            return json.loads(text[s : e + 1])
        raise


def _columns_for_baseline(df: pd.DataFrame) -> list[str]:
    prefer = ["row_id", "supplier", "item", "category", "quantity", "effective_unit_price",
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
