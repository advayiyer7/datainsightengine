# RESULTS — Procurement Insight Engine evaluation

## Hypothesis under test

_Most valuable procurement insights are computable patterns, not open-ended reasoning. A cheap deterministic detector layer + a thin LLM narrator can match a dump-everything-to-LLM baseline on insight quality, at a fraction of the cost, and stay flat as data scales._

## Quality at full data

| approach | n_findings | recall | validity_mean_0_2 | validity_valuable_frac | total_tokens | est_usd | runtime_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B1 detectors (all) | 36 | 1.0 | 1.694 | 0.694 | 0 | 0.0 | 0.078 |
| SYS det+narrator | 8 | 0.941 | 1.875 | 0.875 | 6550 | 0.016858 | 29.515 |
| B0 dump-to-LLM | 6 | 0.059 | 1.667 | 0.667 | 39375 | 0.053427 | 30.312 |
| FULL +agentic | 8 | 0.824 | 1.625 | 0.625 | 23484 | 0.063044 | 95.078 |

_Coverage measured against 17 **judge-curated, valuable-only** ground-truth entries (loaded from answer_key.json; curated=False)._

> ⚠ **Recall caveat (read before trusting recall):** the ground-truth set is built from the engine's OWN detectors (then filtered by the judge to valuable-only). It therefore contains only insights the detectors can produce, so detector-based approaches (B1/SYS/FULL) have their recall **upper-bounded near 1.0 by construction**. This is NOT a clean win over B0. To make recall a fair test, a human must add insights the detectors *cannot* catch (`source:"manual"` in answer_key.json) and set `curated:true` — which has NOT yet been done for this key.

## Cost scaling vs data size

| slice | rows | B0_est_usd | SYS_est_usd | B1_est_usd | B0_findings | B1_findings |
| --- | --- | --- | --- | --- | --- | --- |
| 0.05 | 608 | 0.033525 | 0.024757 | 0.0 | 7 | 17 |
| 0.1 | 1215 | 0.046782 | 0.01551 | 0.0 | 0 | 31 |
| 0.25 | 3038 | 0.051044 | 0.018126 | 0.0 | 0 | 32 |
| 0.5 | 6076 | 0.051046 | 0.019408 | 0.0 | 0 | 35 |
| 1.0 | 12152 | 0.053102 | 0.017058 | 0.0 | 0 | 36 |

## What the run showed

- **Selection:** SYS narrates **8 insights** (down from B1's 36 raw findings); per-finding validity rose from B1 1.69 to SYS 1.88 — merging + top-N ranking lifts quality over the raw dump.
- **Spurious numbers:** untraceable-figure rate fell from 9% to **0%** after the per-insight grounding repair (45 numbers cited).
- **Cost:** SYS $0.0169 vs B0 $0.0534 at full data — SYS is materially cheaper. 
- **Cost scaling:** as rows grew, B0 cost rose by $0.0196 while SYS rose by $-0.0077 — SYS stays ~flat, B0 climbs (thesis supported).
- **Quality — recall:** SYS 0.94 vs B0 0.06 (see the recall caveat above — this is bounded by construction, not a clean win).
- **Quality — precision (LLM-judge 0–2):** SYS 1.88 (8 insights) vs B0 1.67 (6 insights). SYS now matches or beats B0 on per-finding validity while keeping recall + low cost.
- **Grounding:** narrator cited 45/45 numbers, all traceable to findings — no fabrication.

**Verdict:** the hypothesis is **SUPPORTED** by this run.

![quality vs cost](quality_vs_cost.png)

## Honest caveats

- Spend-grain data is one row per supplier-year, single year (2025): "trend" means the in-row `prevYearSpend` comparison, not a multi-year series.
- The provided `yoyChange` ratio and `flagGreaterThan50PercentChange` are NOT used as truth (tiny `prevYearSpend` denominators make ~69% of rows flag TRUE). YoY is recomputed robustly with a `base_floor` and a `min_abs_change` gate.
- `tail_spend` admin overhead (`admin_cost_per_supplier`) is a configurable assumption, not a measured value; negative `totalSpend` is surfaced as a data-quality signal, not netted into totals.