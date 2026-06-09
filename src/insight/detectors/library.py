"""Spend-grain detectors (one row per supplier-year).

This engine was originally built for transaction-grain purchase-order data. The real
dataset is a DIFFERENT grain — one row per supplier with ``spend``, ``prev_spend``,
and a (noisy) provided YoY flag — so the transaction detectors (fragmented_orders,
maverick_price_variance, single_source_risk, timing_anomaly, duplicate_order) were
removed: they require items / quantities / dates that don't exist here. Each detector
below guards on its required columns and returns ``[]`` (logged as a skip) rather than
crashing if run against the wrong grain.

Conventions (unchanged):
  * Input ``df`` is the cleaned canonical frame from Tier 0. Columns relied upon:
    ``supplier``, ``spend``, ``prev_spend`` (``year`` optional). ``row_id`` for evidence.
  * Every Finding carries real numbers in ``metrics`` and proof (supplier names) in
    ``evidence``. ``one_line`` is templated from the numbers — never LLM-written.
  * The provided ``yoy_change`` and ``spend_flag`` columns are NOT trusted; YoY is
    recomputed robustly in :func:`detect_yoy_spend_movers`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..findings import Finding


def _has(df: pd.DataFrame, *cols: str) -> bool:
    return all(c in df.columns for c in cols)


def _skip(name: str, *cols: str) -> list[Finding]:
    """A no-op 'skipped: required columns absent' marker (kept out of rankings)."""
    return [
        Finding(
            type=f"{name}_skipped",
            severity=0.0,
            metrics={"required_columns": list(cols)},
            one_line=f"{name} skipped: required columns absent ({', '.join(cols)}).",
        )
    ]


def _impact_severity(impact: float, *, soft: float = 1_000_000.0, floor: float = 0.1) -> float:
    """Map a dollar magnitude to a 0-1 severity via a saturating curve."""
    impact = abs(impact)
    if impact <= 0:
        return floor
    return float(min(1.0, floor + (1 - floor) * impact / (impact + soft)))


# ---------------------------------------------------------------------------
# 1. supplier_concentration
# ---------------------------------------------------------------------------
def detect_supplier_concentration(df: pd.DataFrame, params: dict[str, Any]) -> list[Finding]:
    """Top-N suppliers by spend and their cumulative share of total POSITIVE spend,
    plus a Herfindahl-Hirschman concentration index over positive-spend shares."""
    if not _has(df, "supplier", "spend"):
        return _skip("supplier_concentration", "supplier", "spend")
    top_n = int(params.get("top_n", 10))

    pos = df[df["spend"] > 0]
    total = float(pos["spend"].sum())
    if total <= 0 or len(pos) == 0:
        return []
    by_sup = pos.groupby("supplier")["spend"].sum().sort_values(ascending=False)
    shares = by_sup / total
    hhi = float((shares ** 2).sum())
    topn = by_sup.head(top_n)
    top_spend = float(topn.sum())
    top_share = top_spend / total

    sev = float(min(1.0, 0.5 * top_share + 0.5 * min(1.0, hhi / 0.25)))
    return [
        Finding(
            type="supplier_concentration",
            severity=sev,
            entities={"top_supplier": str(by_sup.index[0])},
            metrics={
                "top_n": int(top_n),
                "top_n_share": round(top_share, 4),
                "top_n_spend": round(top_spend, 2),
                "total_positive_spend": round(total, 2),
                "hhi": round(hhi, 4),
                "n_suppliers": int(by_sup.shape[0]),
                "largest_supplier_spend": round(float(by_sup.iloc[0]), 2),
            },
            evidence={"top_suppliers": {str(k): round(float(v), 2) for k, v in topn.items()}},
            est_impact_usd=round(top_spend, 2),
            one_line=(
                f"Top {top_n} suppliers account for {top_share:.0%} of "
                f"${total:,.0f} total positive spend (HHI {hhi:.3f}, {int(by_sup.shape[0])} suppliers); "
                f"largest is {by_sup.index[0]} at ${float(by_sup.iloc[0]):,.0f}."
            ),
        )
    ]


# ---------------------------------------------------------------------------
# 2. tail_spend
# ---------------------------------------------------------------------------
def detect_tail_spend(df: pd.DataFrame, params: dict[str, Any]) -> list[Finding]:
    """Many small-spend suppliers that inflate administrative overhead. Counts
    suppliers below ``tail_threshold_usd`` and their combined share; the opportunity
    is ``admin_cost_per_supplier`` * count if consolidated away."""
    if not _has(df, "supplier", "spend"):
        return _skip("tail_spend", "supplier", "spend")
    threshold = float(params.get("tail_threshold_usd", 10000.0))
    admin = float(params.get("admin_cost_per_supplier", 500.0))

    by_sup = df.groupby("supplier")["spend"].sum()
    total_pos = float(by_sup[by_sup > 0].sum())
    tail = by_sup[(by_sup >= 0) & (by_sup < threshold)]
    n_tail = int(len(tail))
    if n_tail == 0 or total_pos <= 0:
        return []
    tail_spend = float(tail.sum())
    share = tail_spend / total_pos
    impact = n_tail * admin
    return [
        Finding(
            type="tail_spend",
            severity=_impact_severity(impact, soft=100_000.0),
            entities={},
            metrics={
                "tail_suppliers": n_tail,
                "spend_threshold": round(threshold, 2),
                "tail_spend": round(tail_spend, 2),
                "tail_share_of_spend": round(share, 4),
                "admin_cost_per_supplier": round(admin, 2),
                "est_admin_overhead": round(impact, 2),
            },
            evidence={"example_tail_suppliers":
                      {str(k): round(float(v), 2) for k, v in tail.sort_values().head(15).items()}},
            est_impact_usd=round(impact, 2),
            one_line=(
                f"{n_tail} suppliers each below ${threshold:,.0f} make up only {share:.1%} of "
                f"${total_pos:,.0f} spend but carry ~${impact:,.0f} in admin overhead "
                f"(~${admin:,.0f}/supplier) — consolidation candidate."
            ),
        )
    ]


# ---------------------------------------------------------------------------
# 3. yoy_spend_movers
# ---------------------------------------------------------------------------
def detect_yoy_spend_movers(df: pd.DataFrame, params: dict[str, Any]) -> list[Finding]:
    """Biggest YoY increases and decreases, computed ROBUSTLY.

    The provided ``yoyChange`` ratio and ``flagGreaterThan50PercentChange`` are noise
    (tiny ``prevYearSpend`` denominators make ~69% of rows flag TRUE), so they are
    ignored. Here ``pct_change = (spend - prev_spend) / max(prev_spend, base_floor)``
    and a mover is surfaced only if BOTH ``abs(spend - prev_spend) >= min_abs_change``
    AND ``prev_spend >= base_floor`` — excluding tiny-denominator noise.
    """
    if not _has(df, "supplier", "spend", "prev_spend"):
        return _skip("yoy_spend_movers", "supplier", "spend", "prev_spend")
    base_floor = float(params.get("base_floor", 1000.0))
    min_abs = float(params.get("min_abs_change", 10000.0))
    top_k = int(params.get("top_k", 10))

    d = df.dropna(subset=["spend", "prev_spend"]).copy()
    d["abs_change"] = (d["spend"] - d["prev_spend"]).abs()
    d["dollar_change"] = d["spend"] - d["prev_spend"]
    # Robust denominator: floor the prior spend so cents don't explode the ratio.
    d["pct_change"] = d["dollar_change"] / d["prev_spend"].clip(lower=base_floor)
    qualified = d[(d["abs_change"] >= min_abs) & (d["prev_spend"] >= base_floor)]
    if len(qualified) == 0:
        return []

    findings: list[Finding] = []
    risers = qualified[qualified["dollar_change"] > 0].nlargest(top_k, "abs_change")
    fallers = qualified[qualified["dollar_change"] < 0].nlargest(top_k, "abs_change")
    for _, row in pd.concat([risers, fallers]).iterrows():
        change = float(row["dollar_change"])
        pct = float(row["pct_change"])
        prev = float(row["prev_spend"])
        cur = float(row["spend"])
        sign_flip = bool(prev > 0 and cur < 0)
        direction = "rose" if change > 0 else "fell"
        flip_txt = " (flipped positive→negative)" if sign_flip else ""
        findings.append(
            Finding(
                type="yoy_spend_movers",
                severity=_impact_severity(change, soft=2_000_000.0),
                entities={"supplier": str(row["supplier"]), "direction": direction},
                metrics={
                    "spend": round(cur, 2),
                    "prev_spend": round(prev, 2),
                    "dollar_change": round(change, 2),
                    "pct_change": round(pct, 4),
                    "sign_flip": sign_flip,
                },
                evidence={"row_id": str(row.get("row_id", ""))},
                est_impact_usd=round(abs(change), 2),
                one_line=(
                    f"{row['supplier']} spend {direction} ${abs(change):,.0f} ({pct:+.0%}) YoY "
                    f"(${prev:,.0f} → ${cur:,.0f}){flip_txt}."
                ),
            )
        )
    return findings


# ---------------------------------------------------------------------------
# 4. negative_or_anomalous_spend
# ---------------------------------------------------------------------------
def detect_negative_or_anomalous_spend(df: pd.DataFrame, params: dict[str, Any]) -> list[Finding]:
    """Negative totalSpend (refunds/credits/adjustments or errors) and statistical
    high outliers (spend beyond mean + ``outlier_k``*std). Data-quality + anomaly."""
    if not _has(df, "supplier", "spend"):
        return _skip("negative_or_anomalous_spend", "supplier", "spend")
    k = float(params.get("outlier_k", 3.0))
    top_neg = int(params.get("top_negatives", 8))
    top_out = int(params.get("top_outliers", 8))

    findings: list[Finding] = []
    by_sup = df.groupby("supplier")["spend"].sum()

    neg = by_sup[by_sup < 0].sort_values()
    if len(neg) > 0:
        total_neg = float(neg.sum())
        findings.append(
            Finding(
                type="negative_or_anomalous_spend",
                severity=_impact_severity(total_neg, soft=200_000.0, floor=0.3),
                entities={"signal": "negative_spend", "largest": str(neg.index[0])},
                metrics={
                    "negative_suppliers": int(len(neg)),
                    "total_negative_spend": round(total_neg, 2),
                    "most_negative_value": round(float(neg.iloc[0]), 2),
                },
                evidence={"negative_suppliers":
                          {str(k2): round(float(v), 2) for k2, v in neg.head(top_neg).items()}},
                est_impact_usd=round(abs(total_neg), 2),
                one_line=(
                    f"{len(neg)} suppliers have negative total spend (refunds/credits/adjustments "
                    f"or errors) totaling ${total_neg:,.0f}; most negative is {neg.index[0]} "
                    f"at ${float(neg.iloc[0]):,.0f} — verify."
                ),
            )
        )

    pos = by_sup[by_sup > 0]
    if len(pos) > 2:
        mu, sd = float(pos.mean()), float(pos.std(ddof=0))
        if sd > 0:
            thresh = mu + k * sd
            outliers = pos[pos > thresh].sort_values(ascending=False).head(top_out)
            for sup, val in outliers.items():
                z = (float(val) - mu) / sd
                findings.append(
                    Finding(
                        type="negative_or_anomalous_spend",
                        severity=_impact_severity(float(val), soft=20_000_000.0, floor=0.3),
                        entities={"signal": "high_outlier", "supplier": str(sup)},
                        metrics={
                            "spend": round(float(val), 2),
                            "mean_spend": round(mu, 2),
                            "std_spend": round(sd, 2),
                            "z_score": round(z, 2),
                            "outlier_k": k,
                        },
                        evidence={"supplier": str(sup)},
                        est_impact_usd=round(float(val), 2),
                        one_line=(
                            f"{sup} spend ${float(val):,.0f} is a statistical outlier "
                            f"({z:.1f}σ above the ${mu:,.0f} mean) — concentration / data-quality check."
                        ),
                    )
                )
    return findings


# ---------------------------------------------------------------------------
# 5. new_and_churned_suppliers
# ---------------------------------------------------------------------------
def detect_new_and_churned_suppliers(df: pd.DataFrame, params: dict[str, Any]) -> list[Finding]:
    """Suppliers that appeared (prior ~0, current material) or churned (current ~0,
    prior material). Rare on this data but a meaningful base-change signal."""
    if not _has(df, "supplier", "spend", "prev_spend"):
        return _skip("new_and_churned_suppliers", "supplier", "spend", "prev_spend")
    near_zero = float(params.get("near_zero", 1.0))
    material = float(params.get("material", 10000.0))

    d = df.dropna(subset=["spend", "prev_spend"])
    findings: list[Finding] = []

    new = d[(d["prev_spend"].abs() <= near_zero) & (d["spend"] >= material)]
    for _, row in new.sort_values("spend", ascending=False).iterrows():
        cur = float(row["spend"])
        findings.append(
            Finding(
                type="new_and_churned_suppliers",
                severity=_impact_severity(cur, soft=1_000_000.0),
                entities={"supplier": str(row["supplier"]), "signal": "new"},
                metrics={"spend": round(cur, 2), "prev_spend": round(float(row["prev_spend"]), 2)},
                evidence={"row_id": str(row.get("row_id", ""))},
                est_impact_usd=round(cur, 2),
                one_line=(
                    f"{row['supplier']} appears NEW this year: ~$0 prior → ${cur:,.0f} current spend."
                ),
            )
        )

    churned = d[(d["spend"].abs() <= near_zero) & (d["prev_spend"] >= material)]
    for _, row in churned.sort_values("prev_spend", ascending=False).iterrows():
        prev = float(row["prev_spend"])
        findings.append(
            Finding(
                type="new_and_churned_suppliers",
                severity=_impact_severity(prev, soft=1_000_000.0),
                entities={"supplier": str(row["supplier"]), "signal": "churned"},
                metrics={"spend": round(float(row["spend"]), 2), "prev_spend": round(prev, 2)},
                evidence={"row_id": str(row.get("row_id", ""))},
                est_impact_usd=round(prev, 2),
                one_line=(
                    f"{row['supplier']} appears CHURNED: ${prev:,.0f} prior → ~$0 current spend."
                ),
            )
        )
    return findings
