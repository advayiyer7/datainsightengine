"""Tier 1 — deterministic detector library.

Each detector is ``detect(df, params) -> list[Finding]`` operating on the cleaned,
canonical DataFrame from Tier 0. No detector touches the LLM. The :data:`REGISTRY`
maps a detector name to its function; :func:`run_all` executes every detector and
returns a flat, severity-sorted list of findings.
"""

from __future__ import annotations

from typing import Any, Callable

import pandas as pd

from ..findings import Finding
from .library import (
    detect_negative_or_anomalous_spend,
    detect_new_and_churned_suppliers,
    detect_supplier_concentration,
    detect_tail_spend,
    detect_yoy_spend_movers,
)

# Spend-grain detector library (one row per supplier-year). The transaction-grain
# detectors (fragmented_orders, maverick_price_variance, single_source_risk,
# timing_anomaly, duplicate_order) were removed in the retarget — they require
# items/quantities/dates this data does not have.
REGISTRY: dict[str, Callable[[pd.DataFrame, dict[str, Any]], list[Finding]]] = {
    "supplier_concentration": detect_supplier_concentration,
    "tail_spend": detect_tail_spend,
    "yoy_spend_movers": detect_yoy_spend_movers,
    "negative_or_anomalous_spend": detect_negative_or_anomalous_spend,
    "new_and_churned_suppliers": detect_new_and_churned_suppliers,
}


def run_all(df: pd.DataFrame, detector_cfg: dict[str, Any] | None = None) -> list[Finding]:
    """Run every registered detector and return findings sorted by severity desc.

    ``detector_cfg`` is the ``detectors:`` block of the config; each detector reads
    its own sub-dict. A detector that raises is skipped (the engine never crashes
    on one bad detector), and the rest still run.
    """
    detector_cfg = detector_cfg or {}
    findings: list[Finding] = []
    for name, fn in REGISTRY.items():
        params = detector_cfg.get(name, {})
        try:
            findings.extend(fn(df, params))
        except Exception as exc:  # defensive: one detector must not kill the run
            findings.append(
                Finding(
                    type=f"{name}_error",
                    severity=0.0,
                    metrics={"error": str(exc)},
                    one_line=f"detector {name} failed: {exc}",
                )
            )
    # Drop 'skipped' markers (a detector whose required columns are absent) from the
    # ranked output — they are informational, not insights.
    findings = [f for f in findings if not f.type.endswith("_skipped")]
    findings.sort(key=lambda f: f.severity, reverse=True)
    return findings


__all__ = ["REGISTRY", "run_all", "Finding"]
