"""Unit tests for Tier-1 detectors using small hand-built frames with known answers.

Covers 5 of the 7 detectors (spec requires >= 3). Each test constructs a tiny
DataFrame where the correct finding is obvious, then asserts the detector's numbers.
"""

from __future__ import annotations

import pandas as pd
import pytest

from insight.detectors.library import (
    detect_duplicate_order,
    detect_fragmented_orders,
    detect_maverick_price_variance,
    detect_single_source_risk,
    detect_supplier_concentration,
)


def make_df(rows: list[dict]) -> pd.DataFrame:
    """Build a cleaned canonical frame from row dicts, filling derived columns."""
    df = pd.DataFrame(rows)
    if "order_date" in df:
        df["order_date"] = pd.to_datetime(df["order_date"])
    if "effective_unit_price" not in df and "unit_price" in df:
        df["effective_unit_price"] = df["unit_price"]
    if "total" not in df and {"quantity", "effective_unit_price"} <= set(df.columns):
        df["total"] = df["quantity"] * df["effective_unit_price"]
    if "category" not in df and "item" in df:
        df["category"] = df["item"]
    if "row_id" not in df:
        df["row_id"] = ["r%d" % i for i in range(len(df))]
    return df


def test_fragmented_orders_detects_dense_window():
    # 4 orders to same supplier+item within 10 days -> consolidation candidate.
    df = make_df(
        [
            {"supplier": "A", "item": "widget", "quantity": 1, "effective_unit_price": 10, "order_date": "2023-01-01"},
            {"supplier": "A", "item": "widget", "quantity": 1, "effective_unit_price": 10, "order_date": "2023-01-04"},
            {"supplier": "A", "item": "widget", "quantity": 1, "effective_unit_price": 10, "order_date": "2023-01-07"},
            {"supplier": "A", "item": "widget", "quantity": 1, "effective_unit_price": 10, "order_date": "2023-01-10"},
            # a far-away order should not be folded into the window
            {"supplier": "A", "item": "widget", "quantity": 1, "effective_unit_price": 10, "order_date": "2023-06-01"},
        ]
    )
    out = detect_fragmented_orders(df, {"min_orders": 3, "window_days": 30, "shipping_cost_per_order": 250})
    assert len(out) == 1
    f = out[0]
    assert f.metrics["orders"] == 4
    assert f.metrics["redundant_orders"] == 3
    assert f.est_impact_usd == pytest.approx(750.0)  # 3 * 250


def test_fragmented_orders_below_threshold_silent():
    df = make_df(
        [
            {"supplier": "A", "item": "w", "quantity": 1, "effective_unit_price": 10, "order_date": "2023-01-01"},
            {"supplier": "A", "item": "w", "quantity": 1, "effective_unit_price": 10, "order_date": "2023-01-02"},
        ]
    )
    assert detect_fragmented_orders(df, {"min_orders": 3, "window_days": 30}) == []


def test_maverick_price_variance_benchmarks_cheapest_supplier():
    # Supplier B averages 10/unit, supplier A averages 20/unit for the same item.
    rows = []
    for _ in range(3):
        rows.append({"supplier": "A", "item": "x", "quantity": 100, "effective_unit_price": 20, "order_date": "2023-01-01"})
        rows.append({"supplier": "B", "item": "x", "quantity": 100, "effective_unit_price": 10, "order_date": "2023-01-01"})
    df = make_df(rows)
    out = detect_maverick_price_variance(df, {"min_orders": 3, "variance_pct": 0.15})
    assert len(out) == 1
    f = out[0]
    assert f.metrics["benchmark_unit_price"] == pytest.approx(10.0)
    assert f.entities["best_supplier"] == "B"
    assert f.entities["worst_supplier"] == "A"
    # Overpayment: A's 3*100 units at 20 vs benchmark 10 = 3*100*10 = 3000; B at benchmark = 0.
    assert f.est_impact_usd == pytest.approx(3000.0)


def test_maverick_ignores_tiny_supplier_sample():
    # B has only 1 order -> cannot be the benchmark; with only A qualifying, no finding.
    rows = [{"supplier": "A", "item": "x", "quantity": 10, "effective_unit_price": 20, "order_date": "2023-01-01"} for _ in range(3)]
    rows.append({"supplier": "B", "item": "x", "quantity": 10, "effective_unit_price": 5, "order_date": "2023-01-01"})
    df = make_df(rows)
    assert detect_maverick_price_variance(df, {"min_orders": 3, "variance_pct": 0.15}) == []


def test_supplier_concentration_flags_dominant_supplier():
    # One supplier owns 90% of a category's spend.
    df = make_df(
        [
            {"supplier": "Dom", "item": "cat1", "quantity": 90, "effective_unit_price": 100, "order_date": "2023-01-01"},
            {"supplier": "Small", "item": "cat1", "quantity": 10, "effective_unit_price": 100, "order_date": "2023-01-02"},
        ]
    )
    out = detect_supplier_concentration(df, {"share_threshold": 0.5, "hhi_threshold": 0.3})
    assert len(out) == 1
    f = out[0]
    assert f.entities["supplier"] == "Dom"
    assert f.metrics["top_supplier_share"] == pytest.approx(0.9)
    # HHI = 0.9^2 + 0.1^2 = 0.82
    assert f.metrics["hhi"] == pytest.approx(0.82)


def test_supplier_concentration_diverse_market_silent():
    df = make_df(
        [
            {"supplier": s, "item": "cat1", "quantity": 25, "effective_unit_price": 100, "order_date": "2023-01-01"}
            for s in ["A", "B", "C", "D"]
        ]
    )
    assert detect_supplier_concentration(df, {"share_threshold": 0.5, "hhi_threshold": 0.3}) == []


def test_single_source_risk_flags_sole_supplier():
    df = make_df(
        [
            {"supplier": "Only", "item": "rare", "quantity": 1000, "effective_unit_price": 50, "order_date": "2023-01-01"},
            {"supplier": "Only", "item": "rare", "quantity": 1000, "effective_unit_price": 50, "order_date": "2023-02-01"},
        ]
    )
    out = detect_single_source_risk(df, {"min_spend": 10000})
    assert len(out) == 1
    assert out[0].entities["supplier"] == "Only"
    assert out[0].metrics["spend_at_risk"] == pytest.approx(100000.0)


def test_single_source_risk_multi_supplier_silent():
    df = make_df(
        [
            {"supplier": "A", "item": "common", "quantity": 1000, "effective_unit_price": 50, "order_date": "2023-01-01"},
            {"supplier": "B", "item": "common", "quantity": 1000, "effective_unit_price": 50, "order_date": "2023-02-01"},
        ]
    )
    assert detect_single_source_risk(df, {"min_spend": 10000}) == []


def test_duplicate_order_flags_near_identical():
    df = make_df(
        [
            {"supplier": "A", "item": "x", "quantity": 100, "effective_unit_price": 10, "order_date": "2023-01-01"},
            {"supplier": "A", "item": "x", "quantity": 100, "effective_unit_price": 10, "order_date": "2023-01-03"},
        ]
    )
    out = detect_duplicate_order(df, {"window_days": 5, "amount_tol_pct": 0.01})
    assert len(out) == 1
    assert out[0].metrics["days_apart"] == 2
    assert out[0].metrics["amount"] == pytest.approx(1000.0)


def test_duplicate_order_outside_window_silent():
    df = make_df(
        [
            {"supplier": "A", "item": "x", "quantity": 100, "effective_unit_price": 10, "order_date": "2023-01-01"},
            {"supplier": "A", "item": "x", "quantity": 100, "effective_unit_price": 10, "order_date": "2023-02-01"},
        ]
    )
    assert detect_duplicate_order(df, {"window_days": 5, "amount_tol_pct": 0.01}) == []
