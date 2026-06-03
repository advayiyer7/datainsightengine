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
    detect_duplicate_order,
    detect_fragmented_orders,
    detect_maverick_price_variance,
    detect_single_source_risk,
    detect_supplier_concentration,
    detect_tail_spend,
    detect_timing_anomaly,
)

REGISTRY: dict[str, Callable[[pd.DataFrame, dict[str, Any]], list[Finding]]] = {
    "fragmented_orders": detect_fragmented_orders,
    "supplier_concentration": detect_supplier_concentration,
    "maverick_price_variance": detect_maverick_price_variance,
    "tail_spend": detect_tail_spend,
    "single_source_risk": detect_single_source_risk,
    "timing_anomaly": detect_timing_anomaly,
    "duplicate_order": detect_duplicate_order,
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
    findings.sort(key=lambda f: f.severity, reverse=True)
    return findings


__all__ = ["REGISTRY", "run_all", "Finding"]
