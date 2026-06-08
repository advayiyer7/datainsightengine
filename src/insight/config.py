"""Configuration loading with sensible defaults.

Defaults live here so the tool runs even with no ``config.yaml``. Any keys present
in the YAML are deep-merged over these defaults.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "column_map": {},
    "llm": {
        "narrator_model": "claude-haiku-4-5",
        "agentic_model": "claude-haiku-4-5",
        "baseline_model": "claude-haiku-4-5",
        "price_per_mtok_input": 1.0,
        "price_per_mtok_output": 5.0,
        "max_output_tokens": 4096,
    },
    "narrator": {
        "top_n": 8,                      # surface only the top-N selected insights
        # Ranking score for selection. Strategies:
        #   "severity_x_impact" (default), "impact", "severity".
        "rank_formula": "severity_x_impact",
        # Dollar value assumed for findings with no est_impact_usd, so qualitative
        # risks (concentration, single-source) still rank instead of scoring 0.
        "null_impact_usd": 20000.0,
        # Cap groups of any one detector type in the top-N so the report stays
        # diverse when one type dominates the high-impact tail. null = no cap.
        "max_per_type": 3,
    },
    "detectors": {
        "fragmented_orders": {
            "min_orders": 3,
            "window_days": 30,
            "shipping_cost_per_order": 250.0,
        },
        "supplier_concentration": {
            "share_threshold": 0.5,
            "hhi_threshold": 0.30,
        },
        "maverick_price_variance": {
            "min_orders": 3,
            "variance_pct": 0.15,
        },
        "tail_spend": {
            "spend_quantile": 0.20,
            "max_share": 0.05,
            "processing_cost_per_order": 100.0,
        },
        "single_source_risk": {
            "min_spend": 10000.0,
        },
        "timing_anomaly": {
            "lead_time_z": 2.0,
            "end_of_period_share": 0.40,
        },
        "duplicate_order": {
            "window_days": 5,
            "amount_tol_pct": 0.01,
        },
    },
    "agentic": {
        "max_iterations": 8,
        "max_wall_clock_seconds": 120,
        "max_tokens": 60000,
        "goal": (
            "Find avoidable procurement cost or anomalies worth more than $5000 "
            "that the fixed detectors might miss."
        ),
    },
    "evaluate": {
        "slices": [0.05, 0.1, 0.25, 0.5, 1.0],
        "baseline_max_rows": 1500,
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` onto a copy of ``base``."""
    out = copy.deepcopy(base)
    for key, val in (override or {}).items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def load_config(path: str | Path | None) -> dict[str, Any]:
    """Load config from ``path``, merged over the built-in defaults.

    A missing or ``None`` path returns the defaults unchanged.
    """
    if path is None:
        return copy.deepcopy(DEFAULTS)
    p = Path(path)
    if not p.exists():
        return copy.deepcopy(DEFAULTS)
    with p.open("r", encoding="utf-8") as fh:
        user_cfg = yaml.safe_load(fh) or {}
    return _deep_merge(DEFAULTS, user_cfg)
