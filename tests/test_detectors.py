"""Unit tests for the SPEND-GRAIN detectors (one row per supplier-year).

Covers 5 detectors with small hand-built frames, INCLUDING the two data-quality
edge cases the real dataset has: negative totalSpend, and tiny prevYearSpend that
makes the naive YoY ratio explode (must be excluded by the robust YoY logic).
"""

from __future__ import annotations

import pandas as pd
import pytest

from insight.detectors.library import (
    detect_negative_or_anomalous_spend,
    detect_new_and_churned_suppliers,
    detect_supplier_concentration,
    detect_tail_spend,
    detect_yoy_spend_movers,
)


def make_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if "row_id" not in df:
        df["row_id"] = ["r%d" % i for i in range(len(df))]
    return df


# --- supplier_concentration -------------------------------------------------
def test_supplier_concentration_share_and_hhi():
    # One supplier with 80% of positive spend.
    df = make_df([
        {"supplier": "Big", "spend": 800.0, "prev_spend": 700.0},
        {"supplier": "B", "spend": 100.0, "prev_spend": 100.0},
        {"supplier": "C", "spend": 100.0, "prev_spend": 100.0},
    ])
    out = detect_supplier_concentration(df, {"top_n": 1})
    assert len(out) == 1
    f = out[0]
    assert f.metrics["top_n_share"] == pytest.approx(0.8)
    assert f.metrics["total_positive_spend"] == pytest.approx(1000.0)
    # HHI = 0.8^2 + 0.1^2 + 0.1^2 = 0.66
    assert f.metrics["hhi"] == pytest.approx(0.66)
    assert f.entities["top_supplier"] == "Big"


def test_concentration_ignores_negative_spend_in_total():
    # Negative spend must not be counted in the positive-spend denominator.
    df = make_df([
        {"supplier": "A", "spend": 900.0, "prev_spend": 0.0},
        {"supplier": "B", "spend": 100.0, "prev_spend": 0.0},
        {"supplier": "Refund", "spend": -500.0, "prev_spend": 0.0},
    ])
    out = detect_supplier_concentration(df, {"top_n": 1})
    assert out[0].metrics["total_positive_spend"] == pytest.approx(1000.0)


# --- tail_spend -------------------------------------------------------------
def test_tail_spend_counts_and_admin_cost():
    rows = [{"supplier": f"small{i}", "spend": 100.0, "prev_spend": 0.0} for i in range(4)]
    rows.append({"supplier": "Big", "spend": 1_000_000.0, "prev_spend": 0.0})
    df = make_df(rows)
    out = detect_tail_spend(df, {"tail_threshold_usd": 10000.0, "admin_cost_per_supplier": 500.0})
    assert len(out) == 1
    f = out[0]
    assert f.metrics["tail_suppliers"] == 4
    assert f.est_impact_usd == pytest.approx(2000.0)  # 4 * 500


# --- yoy_spend_movers (robust) ---------------------------------------------
def test_yoy_real_mover_surfaced_with_correct_numbers():
    df = make_df([
        {"supplier": "Riser", "spend": 60000.0, "prev_spend": 20000.0},
    ])
    out = detect_yoy_spend_movers(df, {"base_floor": 1000.0, "min_abs_change": 10000.0, "top_k": 5})
    assert len(out) == 1
    f = out[0]
    assert f.metrics["dollar_change"] == pytest.approx(40000.0)
    assert f.metrics["pct_change"] == pytest.approx(2.0)  # 40000 / 20000
    assert f.entities["direction"] == "rose"


def test_yoy_tiny_prevspend_noise_is_excluded():
    # prevYearSpend in cents -> naive ratio explodes, but robust logic must DROP it:
    # prev_spend < base_floor AND abs change < min_abs_change.
    df = make_df([
        {"supplier": "NoiseCo", "spend": 3459.02, "prev_spend": 0.0287},
    ])
    out = detect_yoy_spend_movers(df, {"base_floor": 1000.0, "min_abs_change": 10000.0, "top_k": 5})
    assert out == []


def test_yoy_sign_flip_flagged():
    df = make_df([
        {"supplier": "Flipper", "spend": -5000.0, "prev_spend": 50000.0},
    ])
    out = detect_yoy_spend_movers(df, {"base_floor": 1000.0, "min_abs_change": 10000.0, "top_k": 5})
    assert len(out) == 1
    assert out[0].metrics["sign_flip"] is True
    assert out[0].entities["direction"] == "fell"


# --- negative_or_anomalous_spend (edge case: negatives) ---------------------
def test_negative_spend_surfaced():
    df = make_df([
        {"supplier": "Normal", "spend": 1000.0, "prev_spend": 1000.0},
        {"supplier": "CreditA", "spend": -5000.0, "prev_spend": 0.0},
        {"supplier": "CreditB", "spend": -20000.0, "prev_spend": 0.0},
    ])
    out = detect_negative_or_anomalous_spend(df, {"outlier_k": 3.0})
    neg = [f for f in out if f.entities.get("signal") == "negative_spend"]
    assert len(neg) == 1
    assert neg[0].metrics["negative_suppliers"] == 2
    assert neg[0].metrics["total_negative_spend"] == pytest.approx(-25000.0)
    assert neg[0].metrics["most_negative_value"] == pytest.approx(-20000.0)


def test_high_outlier_surfaced():
    rows = [{"supplier": f"s{i}", "spend": 1000.0, "prev_spend": 0.0} for i in range(20)]
    rows.append({"supplier": "Giant", "spend": 1_000_000.0, "prev_spend": 0.0})
    df = make_df(rows)
    out = detect_negative_or_anomalous_spend(df, {"outlier_k": 3.0})
    outliers = [f for f in out if f.entities.get("signal") == "high_outlier"]
    assert any(f.entities.get("supplier") == "Giant" for f in outliers)


# --- new_and_churned_suppliers ----------------------------------------------
def test_new_and_churned_detection():
    df = make_df([
        {"supplier": "NewCo", "spend": 50000.0, "prev_spend": 0.0},      # new
        {"supplier": "GoneCo", "spend": 0.0, "prev_spend": 80000.0},     # churned
        {"supplier": "Steady", "spend": 30000.0, "prev_spend": 30000.0}, # neither
    ])
    out = detect_new_and_churned_suppliers(df, {"near_zero": 1.0, "material": 10000.0})
    signals = {(f.entities["supplier"], f.entities["signal"]) for f in out}
    assert ("NewCo", "new") in signals
    assert ("GoneCo", "churned") in signals
    assert len(out) == 2


def test_detectors_skip_when_columns_absent():
    # Wrong-grain frame (no spend column) -> detectors emit a 'skipped' marker, not a crash.
    df = make_df([{"supplier": "A", "item": "x", "quantity": 1}])
    out = detect_supplier_concentration(df, {})
    assert out and out[0].type.endswith("_skipped")
    out2 = detect_yoy_spend_movers(df, {})
    assert out2 and out2[0].type.endswith("_skipped")
