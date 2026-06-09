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
    # Spend-grain detectors (one row per supplier-year). Thresholds tunable here.
    "detectors": {
        "supplier_concentration": {
            "top_n": 10,
        },
        "tail_spend": {
            "tail_threshold_usd": 10000.0,
            "admin_cost_per_supplier": 500.0,
        },
        "yoy_spend_movers": {
            "base_floor": 1000.0,       # ignore movers whose prior spend is below this
            "min_abs_change": 10000.0,  # ...or whose dollar change is below this
            "top_k": 10,                # surface top-k risers and top-k fallers
        },
        "negative_or_anomalous_spend": {
            "outlier_k": 3.0,           # spend > mean + k*std is a high outlier
            "top_negatives": 8,
            "top_outliers": 8,
        },
        "new_and_churned_suppliers": {
            "near_zero": 1.0,           # |spend| below this counts as "absent"
            "material": 10000.0,        # current/prior spend above this counts as "material"
        },
    },
    "agentic": {
        "max_iterations": 8,
        "max_wall_clock_seconds": 120,
        "max_tokens": 60000,
        "goal": (
            "Find spend patterns or anomalies in this supplier-year data (concentration, "
            "tail spend, robust YoY movement, negative/outlier spend, new/churned suppliers) "
            "worth more than $50000 that the fixed detectors might miss."
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
