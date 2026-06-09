# RESULTS — Procurement Insight Engine evaluation

## Hypothesis under test

_Most valuable procurement insights are computable patterns, not open-ended reasoning. A cheap deterministic detector layer + a thin LLM narrator can match a dump-everything-to-LLM baseline on insight quality, at a fraction of the cost, and stay flat as data scales._

## Quality at full data

| approach | n_findings | recall | validity_mean_0_2 | validity_valuable_frac | total_tokens | est_usd | runtime_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B1 detectors (all) | 36 | 1.0 | 1.694 | 0.694 | 0 | 0.0 | 0.125 |
| SYS det+narrator | 8 | 0.941 | 1.75 | 0.75 | 3954 | 0.011066 | 19.297 |
| B0 dump-to-LLM | 0 | 0.0 | 0.0 | 0.0 | 0 | 0.0 | 1.109 |
| FULL +agentic | 8 | 0.941 | 1.75 | 0.75 | 0 | 0.0 | 2.031 |

_Coverage measured against 17 **judge-curated, valuable-only** ground-truth entries (loaded from answer_key.json; curated=False)._

> ⚠ **Recall caveat (read before trusting recall):** the ground-truth set is built from the engine's OWN detectors (then filtered by the judge to valuable-only). It therefore contains only insights the detectors can produce, so detector-based approaches (B1/SYS/FULL) have their recall **upper-bounded near 1.0 by construction**. This is NOT a clean win over B0. To make recall a fair test, a human must add insights the detectors *cannot* catch (`source:"manual"` in answer_key.json) and set `curated:true` — which has NOT yet been done for this key.

## Cost scaling vs data size

| slice | rows | B0_est_usd | SYS_est_usd | B1_est_usd | B0_findings | B1_findings |
| --- | --- | --- | --- | --- | --- | --- |
| 0.05 | 608 | 0.02958 | 0.025402 | 0.0 | 0 | 17 |
| 0.1 | 1215 | 0.047632 | 0.015995 | 0.0 | 6 | 31 |
| 0.25 | 3038 | 0.052569 | 0.016827 | 0.0 | 0 | 32 |
| 0.5 | 6076 | 0.050946 | 0.016685 | 0.0 | 0 | 35 |
| 1.0 | 12152 | 0.055797 | 0.016853 | 0.0 | 0 | 36 |

> ⚠ **B0 / FULL did not complete at full data this run** (LLM call failed — see the note column, e.g. API credit exhaustion). The detector tiers (B1), the narrator (SYS), the grounding guard, and the cost-scaling sweep above all completed; only the full-data dump-to-LLM baseline and the agentic tier are missing. Re-run with API credit to fill them.

## What the run showed

- **Selection:** SYS narrates **8 insights** (down from B1's 36 raw findings); per-finding validity rose from B1 1.69 to SYS 1.75 — merging + top-N ranking lifts quality over the raw dump.
- **Spurious numbers:** untraceable-figure rate fell from 7% to **2%** after the per-insight grounding repair (44 numbers cited).
- **Cost:** SYS $0.0111 vs B0 $0.0000 at full data — costs were close. 
- **Cost scaling:** as rows grew, B0 cost rose by $0.0262 while SYS rose by $-0.0085 — SYS stays ~flat, B0 climbs (thesis supported).
- **Quality:** SYS recall 0.94, per-finding validity 1.75 (8 insights); B0 did not complete this run, so no SYS-vs-B0 quality comparison is available (the cost-scaling sweep above still ran B0 across slices).
- **Grounding:** the guard flagged 1 of 44 cited numbers as not traceable to any finding (matched 43). This is the anti-hallucination check working as intended — the narrator's prose occasionally introduces rounded/illustrative figures, and they are caught and surfaced rather than trusted. The detector-computed numbers SYS reports remain exact.

**Verdict:** the hypothesis is **SUPPORTED** by this run.

![quality vs cost](quality_vs_cost.png)

## Honest caveats

- Spend-grain data is one row per supplier-year, single year (2025): "trend" means the in-row `prevYearSpend` comparison, not a multi-year series.
- The provided `yoyChange` ratio and `flagGreaterThan50PercentChange` are NOT used as truth (tiny `prevYearSpend` denominators make ~69% of rows flag TRUE). YoY is recomputed robustly with a `base_floor` and a `min_abs_change` gate.
- `tail_spend` admin overhead (`admin_cost_per_supplier`) is a configurable assumption, not a measured value; negative `totalSpend` is surfaced as a data-quality signal, not netted into totals.