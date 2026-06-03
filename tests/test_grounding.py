"""Tests for the anti-hallucination grounding guard."""

from __future__ import annotations

from insight.findings import Finding
from insight.grounding import check_grounding


def _finding() -> Finding:
    return Finding(
        type="fragmented_orders",
        severity=0.5,
        entities={"supplier": "Gamma_Co", "item": "MRO"},
        metrics={"orders": 4, "window_days": 18, "est_extra_shipping": 1240.0},
        est_impact_usd=1240.0,
        one_line="4 orders to Gamma_Co for MRO within 18 days; ~$1,240 avoidable.",
    )


def test_grounded_narration_passes():
    text = "Gamma_Co placed 4 orders for MRO over 18 days, costing about $1,240 extra."
    res = check_grounding(text, [_finding()], ignore_below=12)
    assert res.ok
    assert res.matched == res.total_numbers


def test_invented_dollar_amount_is_flagged():
    # $9,999 appears nowhere in the finding -> must be caught.
    text = "Consolidating these 4 orders would save $9,999 in shipping."
    res = check_grounding(text, [_finding()], ignore_below=12)
    assert not res.ok
    assert any(abs(u["value"] - 9999) < 1 for u in res.unmatched)


def test_percentage_form_of_ratio_is_grounded():
    f = Finding(type="supplier_concentration", severity=0.6,
                entities={"category": "MRO"}, metrics={"top_supplier_share": 0.62, "hhi": 0.41})
    text = "One supplier holds 62% of the category."
    res = check_grounding(text, [f], ignore_below=12)
    assert res.ok


def test_rounding_tolerance():
    f = Finding(type="x", severity=0.5, metrics={"est": 1240.0}, est_impact_usd=1240.0)
    # $1.2k should match 1240 within tolerance after suffix expansion + rel_tol.
    res = check_grounding("about $1,238 saved", [f], ignore_below=12, rel_tol=0.02)
    assert res.ok
