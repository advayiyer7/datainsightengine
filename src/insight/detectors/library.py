"""The seven deterministic detectors.

Conventions:
  * Input ``df`` is the cleaned canonical frame from Tier 0. Columns relied upon:
    ``supplier``, ``item``, ``category``, ``quantity``, ``effective_unit_price``,
    ``total``, ``order_date``, ``lead_time_days``, ``row_id``.
  * Every Finding carries real numbers in ``metrics`` and proof in ``evidence``.
  * ``one_line`` is templated from the numbers — never LLM-written.
  * Severity is a 0-1 score; impact-bearing detectors scale it by dollar impact.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..findings import Finding


def _has(df: pd.DataFrame, *cols: str) -> bool:
    return all(c in df.columns for c in cols)


def _impact_severity(impact: float, *, soft: float = 50000.0, floor: float = 0.1) -> float:
    """Map a dollar impact to a 0-1 severity via a saturating curve."""
    if impact <= 0:
        return floor
    return float(min(1.0, floor + (1 - floor) * impact / (impact + soft)))


# ---------------------------------------------------------------------------
# 1. fragmented_orders
# ---------------------------------------------------------------------------
def detect_fragmented_orders(df: pd.DataFrame, params: dict[str, Any]) -> list[Finding]:
    """Same supplier + same item with >= ``min_orders`` orders inside any rolling
    ``window_days`` window — orders that could have been consolidated.

    Redundant cost is estimated as (orders_in_window - 1) * shipping_cost_per_order,
    summing the worst (densest) non-overlapping-ish window per supplier+item group.
    """
    min_orders = int(params.get("min_orders", 3))
    window_days = int(params.get("window_days", 30))
    ship = float(params.get("shipping_cost_per_order", 250.0))
    if not _has(df, "supplier", "item", "order_date"):
        return []

    findings: list[Finding] = []
    g = df.dropna(subset=["order_date"]).sort_values("order_date")
    # Scale guard: only (supplier,item) groups with >= min_orders rows can qualify.
    # Pre-filtering keeps the Python group loop to the few interesting groups.
    sizes = g.groupby(["supplier", "item"])["order_date"].transform("size")
    g = g[sizes >= min_orders]
    for (supplier, item), grp in g.groupby(["supplier", "item"], sort=False):
        if len(grp) < min_orders:
            continue
        dates = grp["order_date"].to_numpy()
        rows = grp["row_id"].tolist()
        n = len(grp)
        # Sliding window: for each start i, extend while within window_days.
        best_count = 0
        best_span = None
        i = 0
        win = np.timedelta64(window_days, "D")
        while i < n:
            j = i
            while j + 1 < n and (dates[j + 1] - dates[i]) <= win:
                j += 1
            count = j - i + 1
            if count > best_count:
                best_count = count
                best_span = (i, j)
            i += 1
        if best_count < min_orders or best_span is None:
            continue
        i, j = best_span
        window_rows = rows[i : j + 1]
        redundant = best_count - 1
        impact = redundant * ship
        span_days = int((dates[j] - dates[i]) / np.timedelta64(1, "D"))
        findings.append(
            Finding(
                type="fragmented_orders",
                severity=_impact_severity(impact, soft=5000.0),
                entities={"supplier": supplier, "item": item},
                metrics={
                    "orders": int(best_count),
                    "window_days": span_days,
                    "redundant_orders": int(redundant),
                    "est_extra_shipping": round(impact, 2),
                },
                evidence={"row_ids": window_rows},
                est_impact_usd=round(impact, 2),
                one_line=(
                    f"{int(best_count)} separate orders to {supplier} for {item} "
                    f"within {span_days} days; consolidating could save "
                    f"~${impact:,.0f} in shipping/handling."
                ),
            )
        )
    return findings


# ---------------------------------------------------------------------------
# 2. supplier_concentration
# ---------------------------------------------------------------------------
def detect_supplier_concentration(df: pd.DataFrame, params: dict[str, Any]) -> list[Finding]:
    """Per category, flag dominance: top supplier spend share > ``share_threshold``
    or a Herfindahl-Hirschman index above ``hhi_threshold``."""
    share_th = float(params.get("share_threshold", 0.5))
    hhi_th = float(params.get("hhi_threshold", 0.30))
    cat_col = "category" if "category" in df.columns else "item"
    if not _has(df, "supplier", "total") or cat_col not in df.columns:
        return []

    findings: list[Finding] = []
    for category, grp in df.groupby(cat_col, sort=False):
        spend = grp.groupby("supplier")["total"].sum()
        total = float(spend.sum())
        if total <= 0:
            continue
        shares = spend / total
        hhi = float((shares**2).sum())
        top_supplier = shares.idxmax()
        top_share = float(shares.max())
        if top_share < share_th and hhi < hhi_th:
            continue
        # Concentration risk severity blends top share and HHI.
        sev = float(min(1.0, 0.4 * top_share + 0.6 * hhi / max(hhi_th, 1e-9)))
        findings.append(
            Finding(
                type="supplier_concentration",
                severity=min(1.0, sev),
                entities={"category": category, "supplier": top_supplier},
                metrics={
                    "top_supplier_share": round(top_share, 3),
                    "hhi": round(hhi, 3),
                    "n_suppliers": int(spend.shape[0]),
                    "category_spend": round(total, 2),
                    "top_supplier_spend": round(float(spend.max()), 2),
                },
                evidence={"supplier_spend": {k: round(float(v), 2) for k, v in spend.items()}},
                est_impact_usd=None,
                one_line=(
                    f"In {category}, {top_supplier} holds {top_share:.0%} of "
                    f"${total:,.0f} spend (HHI {hhi:.2f}) across "
                    f"{int(spend.shape[0])} suppliers — concentration risk."
                ),
            )
        )
    return findings


# ---------------------------------------------------------------------------
# 3. maverick_price_variance
# ---------------------------------------------------------------------------
def detect_maverick_price_variance(df: pd.DataFrame, params: dict[str, Any]) -> list[Finding]:
    """Same item bought at materially different unit prices across suppliers.

    Benchmarking against the single cheapest *order* is misleading when ``item`` is
    a coarse category containing different products (one fluke cheap order sets an
    unrealistic floor). Instead we benchmark against the cheapest *supplier's mean*
    unit price for the item — a stable, defensible "best achievable" price — and
    estimate overpayment as spend above that benchmark. Requires >= ``min_orders``
    per supplier on both sides so a tiny sample can't drive the finding.
    """
    min_orders = int(params.get("min_orders", 3))
    var_pct = float(params.get("variance_pct", 0.15))
    max_var = params.get("max_variance_pct")  # None = no upper bound (default)
    max_var = float(max_var) if max_var is not None else None
    if not _has(df, "item", "supplier", "effective_unit_price", "quantity"):
        return []

    findings: list[Finding] = []
    d = df.dropna(subset=["effective_unit_price"])
    # Scale guard: an item needs >= 2 suppliers each with >= min_orders, so it needs
    # at least 2*min_orders rows total. Drop items that can't possibly qualify.
    isizes = d.groupby("item")["effective_unit_price"].transform("size")
    d = d[isizes >= 2 * min_orders]
    for item, grp in d.groupby("item", sort=False):
        sup = grp.groupby("supplier").agg(
            mean_price=("effective_unit_price", "mean"),
            orders=("effective_unit_price", "size"),
        )
        sup = sup[sup["orders"] >= min_orders]
        if len(sup) < 2:
            continue
        benchmark = float(sup["mean_price"].min())
        worst_mean = float(sup["mean_price"].max())
        if benchmark <= 0:
            continue
        spread = (worst_mean - benchmark) / benchmark
        if spread < var_pct:
            continue
        # Upper sanity bound: a spread this large almost always means the rows under
        # this "item" are not actually the same product (e.g. a generic item name
        # spanning contracts of wildly different size), not a real overpayment. Skip.
        if max_var is not None and spread > max_var:
            continue
        best_supplier = sup["mean_price"].idxmin()
        worst_supplier = sup["mean_price"].idxmax()
        # Overpayment: dollars spent above what the cheapest supplier averages.
        overpay = float(((grp["effective_unit_price"] - benchmark) * grp["quantity"]).clip(lower=0).sum())
        findings.append(
            Finding(
                type="maverick_price_variance",
                severity=_impact_severity(overpay, soft=200000.0),
                entities={"item": item, "best_supplier": best_supplier, "worst_supplier": worst_supplier},
                metrics={
                    "benchmark_unit_price": round(benchmark, 2),
                    "worst_supplier_mean_price": round(worst_mean, 2),
                    "spread_pct": round(spread, 3),
                    "suppliers_compared": int(len(sup)),
                    "orders": int(len(grp)),
                    "est_overpayment_vs_cheapest_supplier": round(overpay, 2),
                },
                evidence={
                    "supplier_mean_price": {k: round(float(v), 2) for k, v in sup["mean_price"].items()},
                    "supplier_orders": {k: int(v) for k, v in sup["orders"].items()},
                },
                est_impact_usd=round(overpay, 2),
                one_line=(
                    f"For {item}, {worst_supplier} averages ${worst_mean:,.2f}/unit vs "
                    f"{best_supplier}'s ${benchmark:,.2f} ({spread:.0%} higher); shifting volume "
                    f"to the cheapest supplier could save up to ~${overpay:,.0f}."
                ),
            )
        )
    return findings


# ---------------------------------------------------------------------------
# 4. tail_spend
# ---------------------------------------------------------------------------
def detect_tail_spend(df: pd.DataFrame, params: dict[str, Any]) -> list[Finding]:
    """Long tail of small orders that inflate processing cost. The tail is the set
    of orders below the ``spend_quantile`` value percentile; flagged when the tail
    is collectively below ``max_share`` of total spend yet many in count."""
    q = float(params.get("spend_quantile", 0.20))
    max_share = float(params.get("max_share", 0.05))
    proc = float(params.get("processing_cost_per_order", 100.0))
    if not _has(df, "total"):
        return []

    totals = df["total"].dropna()
    if len(totals) < 10:
        return []
    threshold = float(totals.quantile(q))
    tail = df[df["total"] <= threshold]
    grand = float(totals.sum())
    tail_spend = float(tail["total"].sum())
    share = tail_spend / grand if grand else 0.0
    if share > max_share or len(tail) == 0:
        return []
    impact = len(tail) * proc
    # Which suppliers dominate the tail (for the recommendation).
    if "supplier" in tail.columns:
        by_sup = tail.groupby("supplier").size().sort_values(ascending=False)
        tail_suppliers = {k: int(v) for k, v in by_sup.items()}
    else:
        tail_suppliers = {}
    return [
        Finding(
            type="tail_spend",
            severity=_impact_severity(impact, soft=5000.0),
            entities={},
            metrics={
                "tail_orders": int(len(tail)),
                "order_value_threshold": round(threshold, 2),
                "tail_spend": round(tail_spend, 2),
                "tail_share_of_spend": round(share, 4),
                "est_processing_cost": round(impact, 2),
            },
            evidence={"tail_suppliers": tail_suppliers, "row_ids": tail["row_id"].head(25).tolist()},
            est_impact_usd=round(impact, 2),
            one_line=(
                f"{int(len(tail))} small orders (each <= ${threshold:,.0f}) make up only "
                f"{share:.1%} of spend but cost ~${impact:,.0f} to process — consolidation candidate."
            ),
        )
    ]


# ---------------------------------------------------------------------------
# 5. single_source_risk
# ---------------------------------------------------------------------------
def detect_single_source_risk(df: pd.DataFrame, params: dict[str, Any]) -> list[Finding]:
    """Items sourced from exactly one supplier above ``min_spend`` — supply risk
    exposure weighted by spend."""
    min_spend = float(params.get("min_spend", 10000.0))
    if not _has(df, "item", "supplier", "total"):
        return []

    findings: list[Finding] = []
    d = df.dropna(subset=["total"])
    # Scale guard: only items whose total spend clears the threshold can qualify.
    ispend = d.groupby("item")["total"].transform("sum")
    d = d[ispend >= min_spend]
    for item, grp in d.groupby("item", sort=False):
        suppliers = grp["supplier"].dropna().unique()
        spend = float(grp["total"].sum())
        if len(suppliers) == 1 and spend >= min_spend:
            findings.append(
                Finding(
                    type="single_source_risk",
                    severity=_impact_severity(spend, soft=100000.0),
                    entities={"item": item, "supplier": suppliers[0]},
                    metrics={
                        "suppliers": 1,
                        "spend_at_risk": round(spend, 2),
                        "orders": int(len(grp)),
                    },
                    evidence={"row_ids": grp["row_id"].head(25).tolist()},
                    est_impact_usd=round(spend, 2),
                    one_line=(
                        f"{item} is sourced solely from {suppliers[0]} "
                        f"(${spend:,.0f} across {int(len(grp))} orders) — single-source supply risk."
                    ),
                )
            )
    return findings


# ---------------------------------------------------------------------------
# 6. timing_anomaly
# ---------------------------------------------------------------------------
def detect_timing_anomaly(df: pd.DataFrame, params: dict[str, Any]) -> list[Finding]:
    """Two timing signals:
       (a) lead-time outliers vs a supplier's own norm (|z| > ``lead_time_z``);
       (b) end-of-period clustering: > ``end_of_period_share`` of a supplier's
           orders land in the last month of a calendar quarter."""
    z_th = float(params.get("lead_time_z", 2.0))
    eop_share = float(params.get("end_of_period_share", 0.40))
    findings: list[Finding] = []

    # (a) lead-time outliers
    if _has(df, "supplier", "lead_time_days"):
        for supplier, grp in df.groupby("supplier", sort=False):
            lt = grp["lead_time_days"].dropna()
            if len(lt) < 5:
                continue
            mu, sd = float(lt.mean()), float(lt.std(ddof=0))
            if sd <= 0:
                continue
            z = (grp["lead_time_days"] - mu) / sd
            outliers = grp[z.abs() > z_th]
            if len(outliers) == 0:
                continue
            worst = outliers.loc[(outliers["lead_time_days"] - mu).abs().idxmax()]
            findings.append(
                Finding(
                    type="timing_anomaly",
                    severity=min(1.0, 0.3 + 0.1 * len(outliers)),
                    entities={"supplier": supplier, "signal": "lead_time_outlier"},
                    metrics={
                        "supplier_mean_lead_days": round(mu, 1),
                        "supplier_std_lead_days": round(sd, 1),
                        "outlier_orders": int(len(outliers)),
                        "worst_lead_days": round(float(worst["lead_time_days"]), 1),
                    },
                    evidence={"row_ids": outliers["row_id"].head(25).tolist()},
                    est_impact_usd=None,
                    one_line=(
                        f"{supplier} has {int(len(outliers))} lead-time outlier order(s) "
                        f"(worst {float(worst['lead_time_days']):.0f}d vs mean {mu:.0f}d)."
                    ),
                )
            )

    # (b) end-of-period clustering
    if _has(df, "supplier", "order_date"):
        d = df.dropna(subset=["order_date"]).copy()
        if len(d):
            month = d["order_date"].dt.month
            d["is_eop"] = month.isin([3, 6, 9, 12])
            for supplier, grp in d.groupby("supplier", sort=False):
                if len(grp) < 10:
                    continue
                share = float(grp["is_eop"].mean())
                if share > eop_share:
                    findings.append(
                        Finding(
                            type="timing_anomaly",
                            severity=min(1.0, share),
                            entities={"supplier": supplier, "signal": "end_of_period_spike"},
                            metrics={
                                "eop_order_share": round(share, 3),
                                "orders": int(len(grp)),
                                "eop_orders": int(grp["is_eop"].sum()),
                            },
                            evidence={"row_ids": grp[grp["is_eop"]]["row_id"].head(25).tolist()},
                            est_impact_usd=None,
                            one_line=(
                                f"{share:.0%} of {supplier}'s {int(len(grp))} orders fall in "
                                f"quarter-end months — possible budget-flush timing."
                            ),
                        )
                    )
    return findings


# ---------------------------------------------------------------------------
# 7. duplicate_order
# ---------------------------------------------------------------------------
def detect_duplicate_order(df: pd.DataFrame, params: dict[str, Any]) -> list[Finding]:
    """Near-identical orders: same supplier + item, total within ``amount_tol_pct``,
    placed within ``window_days`` of each other — candidate duplicate POs/payments."""
    window_days = int(params.get("window_days", 5))
    tol = float(params.get("amount_tol_pct", 0.01))
    if not _has(df, "supplier", "item", "total", "order_date"):
        return []

    findings: list[Finding] = []
    win = np.timedelta64(window_days, "D")
    g = df.dropna(subset=["order_date", "total"]).sort_values("order_date")
    # Scale guard: drop singleton (supplier,item) groups before the pairwise scan.
    sizes = g.groupby(["supplier", "item"])["order_date"].transform("size")
    g = g[sizes >= 2]
    # Bound the inner look-ahead so a dense group can't blow up to O(n^2); duplicates
    # are near-in-time, so comparing each order to the next LOOKAHEAD is sufficient.
    LOOKAHEAD = 200
    for (supplier, item), grp in g.groupby(["supplier", "item"], sort=False):
        if len(grp) < 2:
            continue
        rows = grp.reset_index(drop=True)
        dates = rows["order_date"].to_numpy()
        totals = rows["total"].to_numpy()
        ids = rows["row_id"].tolist()
        n = len(rows)
        for i in range(n):
            for j in range(i + 1, min(n, i + 1 + LOOKAHEAD)):
                if (dates[j] - dates[i]) > win:
                    break
                a, b = totals[i], totals[j]
                if a <= 0:
                    continue
                if abs(a - b) / a <= tol:
                    gap = int((dates[j] - dates[i]) / np.timedelta64(1, "D"))
                    findings.append(
                        Finding(
                            type="duplicate_order",
                            severity=_impact_severity(float(b), soft=20000.0, floor=0.3),
                            entities={"supplier": supplier, "item": item},
                            metrics={
                                "amount": round(float(b), 2),
                                "days_apart": gap,
                                "order_total_a": round(float(a), 2),
                                "order_total_b": round(float(b), 2),
                            },
                            evidence={"row_ids": [str(ids[i]), str(ids[j])]},
                            est_impact_usd=round(float(b), 2),
                            one_line=(
                                f"Two near-identical orders to {supplier} for {item} "
                                f"(~${float(b):,.0f}) placed {gap} day(s) apart — possible duplicate."
                            ),
                        )
                    )
    return findings
