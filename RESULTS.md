# RESULTS — Procurement Insight Engine evaluation

## Hypothesis under test

_Most valuable procurement insights are computable patterns, not open-ended reasoning. A cheap deterministic detector layer + a thin LLM narrator can match a dump-everything-to-LLM baseline on insight quality, at a fraction of the cost, and stay flat as data scales._

## Quality at full data

| approach | n_findings | recall | validity_mean_0_2 | validity_valuable_frac | total_tokens | est_usd | runtime_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B1 detectors (all) | 32 | 1.0 | 1.719 | 0.719 | 0 | 0.0 | 0.297 |
| SYS det+narrator | 8 | 1.0 | 1.625 | 0.625 | 4233 | 0.010997 | 13.515 |
| B0 dump-to-LLM | 8 | 0.318 | 1.625 | 0.625 | 36194 | 0.04551 | 19.953 |
| FULL +agentic | 8 | 1.0 | 1.75 | 0.75 | 18057 | 0.039745 | 48.735 |

_Coverage measured against 22 **judge-curated, valuable-only** ground-truth entries (loaded from answer_key.json; curated=False)._

> ⚠ **Recall caveat (read before trusting recall):** the ground-truth set is built from the engine's OWN detectors (then filtered by the judge to valuable-only). It therefore contains only insights the detectors can produce, so detector-based approaches (B1/SYS/FULL) have their recall **upper-bounded near 1.0 by construction**. This is NOT a clean win over B0. To make recall a fair test, a human must add insights the detectors *cannot* catch (`source:"manual"` in answer_key.json) and set `curated:true` — which has NOT yet been done for this key.

## Cost scaling vs data size

| slice | rows | B0_est_usd | SYS_est_usd | B1_est_usd | B0_findings | B1_findings |
| --- | --- | --- | --- | --- | --- | --- |
| 0.05 | 39 | 0.011798 | 0.019846 | 0.0 | 7 | 8 |
| 0.1 | 78 | 0.016108 | 0.010912 | 0.0 | 8 | 16 |
| 0.25 | 194 | 0.023674 | 0.011743 | 0.0 | 8 | 26 |
| 0.5 | 388 | 0.027809 | 0.010965 | 0.0 | 7 | 33 |
| 1.0 | 777 | 0.04728 | 0.010802 | 0.0 | 7 | 32 |

## What the run showed

- **Selection:** SYS narrates **8 insights** (down from B1's 32 raw findings); per-finding validity is comparable (B1 1.72 vs SYS 1.62); on this dataset the detector output is fairly uniform in quality, so selection's win is **fewer, focused insights at low cost** rather than higher per-item validity. A noisier dataset with a longer low-value tail would show a larger precision gain.
- **Spurious numbers:** the narrator produced **0 untraceable figures** out of 81 cited — the tightened 'verbatim numbers only' prompt prevented fabrication, so the per-insight repair was not needed.
- **Cost:** SYS $0.0110 vs B0 $0.0455 at full data — SYS is materially cheaper. 
- **Cost scaling:** as rows grew, B0 cost rose by $0.0355 while SYS rose by $-0.0090 — SYS stays ~flat, B0 climbs (thesis supported).
- **Quality — recall:** SYS 1.00 vs B0 0.32 (see the recall caveat above — this is bounded by construction, not a clean win).
- **Quality — precision (LLM-judge 0–2):** SYS 1.62 (8 insights) vs B0 1.62 (8 insights). SYS now matches or beats B0 on per-finding validity while keeping recall + low cost.
- **Grounding:** narrator cited 81/81 numbers, all traceable to findings — no fabrication.

**Verdict:** the hypothesis is **SUPPORTED** by this run.

![quality vs cost](quality_vs_cost.png)

## Honest caveats

- This dataset has no SKU column, so `item` falls back to `Item_Category` (5 broad categories). `maverick_price_variance` therefore benchmarks the cheapest *supplier average* per category rather than per-SKU prices — its dollar figures are an upper bound on savings, not a precise number.
- Shipping/handling and processing costs in `fragmented_orders`/`tail_spend` are configurable assumptions (`config.yaml`), not measured values.