"""Tests for the ranking / merging / top-N selection logic (Tier-2 pre-step)."""

from __future__ import annotations

from insight.findings import Finding
from insight.selection import merge_findings, score_finding, select_findings


def mk(type_, sev, impact, entities, one_line="x"):
    return Finding(type=type_, severity=sev, entities=entities, est_impact_usd=impact, one_line=one_line)


def test_score_formulas_and_null_impact():
    f = mk("a", 0.5, 1000.0, {})
    assert score_finding(f, formula="severity_x_impact") == 500.0
    assert score_finding(f, formula="impact") == 1000.0
    assert score_finding(f, formula="severity") == 0.5
    # null impact falls back to the configured value, not zero
    fn = mk("b", 0.5, None, {})
    assert score_finding(fn, formula="severity_x_impact", null_impact_usd=20000.0) == 10000.0


def test_merge_same_supplier_item_across_detectors():
    a = mk("fragmented_orders", 0.4, 1000.0, {"supplier": "Gamma", "item": "MRO"})
    b = mk("duplicate_order", 0.9, 5000.0, {"supplier": "Gamma", "item": "MRO"})
    c = mk("tail_spend", 0.3, 500.0, {})  # different subject, stays separate
    groups = merge_findings([a, b, c])
    # a and b merge into one group; c is its own
    sizes = sorted(len(g.members) for g in groups)
    assert sizes == [1, 2]
    merged = max(groups, key=lambda g: len(g.members))
    assert {m.type for m in merged.members} == {"fragmented_orders", "duplicate_order"}
    # group impact = max member impact, source_types lists both
    assert merged.est_impact_usd == 5000.0
    assert merged.source_types == ["duplicate_order", "fragmented_orders"]


def test_select_top_n_ranks_by_business_impact_and_truncates():
    findings = [
        mk("t1", 0.9, 100000.0, {"supplier": "A", "item": "x"}),   # score 90000
        mk("t2", 0.8, 50000.0, {"supplier": "B", "item": "y"}),    # score 40000
        mk("t3", 0.2, 1000.0, {"supplier": "C", "item": "z"}),     # score 200
        mk("t4", 0.5, 2000.0, {"supplier": "D", "item": "w"}),     # score 1000
    ]
    top = select_findings(findings, top_n=2)
    assert len(top) == 2
    # ordered by score desc
    assert top[0].entities["supplier"] == "A"
    assert top[1].entities["supplier"] == "B"
    assert top[0].score > top[1].score


def test_selection_keeps_qualitative_findings_via_null_impact():
    # A high-severity but impact-less finding should still rank above a trivial one.
    risky = mk("supplier_concentration", 0.95, None, {"category": "MRO", "supplier": "A"})
    trivial = mk("fragmented_orders", 0.1, 500.0, {"supplier": "B", "item": "y"})
    top = select_findings([trivial, risky], top_n=1, null_impact_usd=20000.0)
    assert top[0].source_types == ["supplier_concentration"]
