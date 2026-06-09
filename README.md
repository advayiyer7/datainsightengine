# Cheap Procurement Insight Engine (supplier-spend grain)

Surfaces **valuable, actionable spend insights** from a supplier-spend CSV — e.g.
_"top 10 suppliers account for 23% of $2.3B spend (HHI 0.01)"_, _"Bartlett & Co spend
fell $309M (-64%) YoY"_, or _"6,293 suppliers each below $10k carry ~$3.1M in admin
overhead"_ — **cheaply**, without dumping the whole dataset into an LLM.

> **Data grain note.** This engine was retargeted from transaction-grain
> purchase-order data to **supplier-spend grain**: one row per supplier-year with
> `supplierName, year, totalSpend, prevYearSpend, yoyChange, flagGreaterThan50PercentChange`.
> The architecture (schema mapping, narrator top-N selection, grounding guard, agentic
> tier, judge-curated answer key, evaluation harness) is unchanged; only the **detector
> library** was replaced. The engine auto-detects the grain from the schema mapping
> (a mapped `spend` column ⇒ spend grain) and the transaction detectors are removed.

## The hypothesis this code tests

> Most valuable procurement insights are **computable patterns**, not open-ended
> reasoning. A cheap deterministic **detector** layer + a thin **LLM narrator** that
> only narrates the findings can match a "dump-everything-into-the-LLM" baseline on
> insight quality, at a fraction of the cost, and stay flat as data scales.

So this repo is the engine **plus an evaluation harness** that tries to prove or
break that hypothesis by comparing approaches on quality vs. cost.

## Architecture — three tiers

| Tier | Module | What it does | LLM? |
|------|--------|--------------|------|
| **0** Ingest | `insight.ingest` | Load CSV, fuzzy-map headers, detect grain, clean/normalize, report a **data-quality** summary (negatives, near-zero prevYearSpend, outliers, noisy-flag share). | no |
| **1** Detectors | `insight.detectors` | 5 deterministic pandas spend detectors → structured `Finding`s with real numbers + supplier-name evidence. | no |
| **2** Narrator | `insight.narrator` | Merges findings per subject, ranks, narrates only the **top-N** using **only** finding numbers; a grounding guard catches any invented figure. | thin/cheap |
| **3** Agentic | `insight.agentic` | Bounded, sandboxed loop that writes & runs pandas to discover spend patterns the fixed detectors miss. | optional |

Every tier speaks the same `Finding` schema (`insight.findings`):

```python
Finding(type, severity, entities, metrics, evidence, est_impact_usd, one_line, source)
```

### The 5 spend detectors (Tier 1)

1. `supplier_concentration` — top-N suppliers by spend, their cumulative share of total **positive** spend, and a Herfindahl-Hirschman index. Config: `top_n`.
2. `tail_spend` — count of suppliers below a spend threshold + combined share; quantifies admin/consolidation overhead. Config: `tail_threshold_usd`, `admin_cost_per_supplier`.
3. `yoy_spend_movers` — biggest YoY increases/decreases, computed **robustly**: `pct_change = (spend - prev_spend) / max(prev_spend, base_floor)`, surfaced only if `abs(change) ≥ min_abs_change` **and** `prev_spend ≥ base_floor` (excludes tiny-denominator noise); flags positive→negative sign flips. Config: `base_floor`, `min_abs_change`, `top_k`.
4. `negative_or_anomalous_spend` — negative `totalSpend` (refunds/credits/errors) and statistical high outliers (`> mean + k·std`). A data-quality + anomaly finding. Config: `outlier_k`.
5. `new_and_churned_suppliers` — prior ~0 with material current spend (new), or current ~0 with material prior (churned). Config: `near_zero`, `material`.

> The transaction-grain detectors (`fragmented_orders`, `maverick_price_variance`,
> `single_source_risk`, `timing_anomaly`, `duplicate_order`) were **removed** — they
> need items/quantities/dates this grain doesn't have. Each spend detector guards on
> its required columns and emits a `*_skipped` marker instead of crashing if run on
> the wrong grain.

**The provided `yoyChange` ratio and `flagGreaterThan50PercentChange` are NOT trusted**
(tiny `prevYearSpend` denominators make ~69% of rows flag TRUE) — YoY is recomputed
robustly. All thresholds live in `config.yaml`; 5 detectors are covered by unit tests
including the negative-spend and tiny-prevYearSpend edge cases (`tests/`).

## Setup

```bash
pip install -e .            # installs the `insight` CLI + deps
export ANTHROPIC_API_KEY=sk-...     # bash
# PowerShell:  $env:ANTHROPIC_API_KEY = "sk-..."
```

The key is read from the environment and **never hardcoded**. Without a key the
deterministic tiers (0, 1) and the grounding guard all still run; the LLM tiers
(narrator, B0 baseline, agentic, judge) degrade to a clearly-labeled fallback or
are skipped, so every command runs end-to-end offline.

### Dataset

The target is a **supplier-spend CSV** (`csvdata.csv`, ~12k suppliers, year 2025)
with columns `supplierName, year, totalSpend, prevYearSpend, yoyChange,
flagGreaterThan50PercentChange`. `config.yaml` already maps it:

```yaml
column_map:
  supplier: supplierName
  spend: totalSpend
  prev_spend: prevYearSpend
  yoy_change: yoyChange        # carried but NOT trusted (noisy)
  year: year
  spend_flag: flagGreaterThan50PercentChange   # NOT trusted (~69% TRUE)
```

A mapped `spend` column switches the engine to spend grain. Required fields:
`supplier`, `spend`, `prev_spend`. To use your own spend CSV, run
`insight ingest --data your.csv`, read the **resolved schema mapping + data-quality
report**, and correct any mis-guessed header under `column_map:`.

> **Known data-quality issues (handled, not hidden):** `totalSpend` includes
> **negative** values (refunds/credits/adjustments or errors, down to ~-$197k) —
> surfaced by `negative_or_anomalous_spend`, never netted away. `prevYearSpend` has
> near-zero (cents) values that make the provided `yoyChange` explode and flag ~69%
> of rows — so the **provided flag and raw ratio are ignored** and YoY is recomputed
> robustly. It is a **single year**, so "trend" means the in-row prevYearSpend
> comparison. `ingest` prints all of these counts.

## CLI

```bash
insight ingest   --data path.csv [--config config.yaml]      # load, map schema, clean, report
insight detect   --data path.csv [-o findings.json]          # Tier 1 -> findings.json
insight run      --data path.csv [--agentic]                 # full pipeline -> insights.md + insights.json
insight make-answer-key --data path.csv [-o answer_key.json] # ground-truth insight set for eval
insight evaluate --data path.csv [--slices 0.1,0.25,0.5,1.0] # B0/B1/SYS/FULL -> table + chart + RESULTS.md
```

## The evaluation harness

`insight evaluate` runs four approaches on the same data and compares them:

- **B0 — dump-to-LLM:** feed raw rows into the LLM, ask for insights. The expensive ceiling (samples if data exceeds the context budget).
- **B1 — detectors only:** Tier 1 findings, no LLM.
- **SYS — detectors + narrator:** the main proposal (Tiers 1+2).
- **FULL — + agentic:** Tiers 1+2+3.

It reports, into `eval_out/`:

- **`results_quality.csv`** — per-approach coverage (recall vs the answer key),
  validity (LLM-judge precision 0–2), token cost, USD, runtime.
- **`results_scaling.csv`** — how cost grows as you feed 10/25/50/100% of rows
  (the thesis: SYS stays ~flat while B0 climbs).
- **`quality_vs_cost.png`** — the headline chart (cost-scaling line + quality-vs-cost scatter).
- **`RESULTS.md`** — an auto-written summary that states whether the run
  **supported or broke** the hypothesis, plus honest caveats.

### Ground truth (and the recall caveat)

`insight make-answer-key` runs the detectors at **strict** thresholds, then runs the
**LLM-judge** and keeps only candidates scored *valuable* (2/2), writing them to
`answer_key.json` with `curated: false`. Review it, set `validity`/`include`, and —
crucially — **add insights the detectors cannot catch** (`source: "manual"`), then
set `curated: true`.

Why this matters: because the key is built from the engine's own detectors, recall
for detector-based approaches (B1/SYS/FULL) is **upper-bounded near 1.0 by
construction**. That is *not* a clean win over B0 — `evaluate` says so explicitly in
`RESULTS.md`. Recall only becomes a fair test once you add non-detector insights.

### Selection (top-N)

The narrator does **not** pass through every finding. It merges findings about the
same subject, ranks them by a configurable business-impact score
(`narrator.rank_formula`, default `severity × est_impact_usd`), and narrates only the
top `narrator.top_n` (default 8). The full finding set still lands in `findings.json`;
the report and the SYS/FULL numbers the evaluator scores are the selected top-N.

## How to read the results

- **Cost** should diverge: B1 = $0, SYS small and roughly flat across slices, B0
  climbing with row count.
- **Coverage** (recall) and **validity** (precision) should show SYS ≈ B0 quality
  — if SYS matches B0's insight quality far cheaper, the hypothesis holds.
- The grounding line tells you whether the narrator stayed honest: every cited
  number must trace back to a finding, or it's flagged.

A clean **negative** result is a valid outcome — if SYS can't match B0, `RESULTS.md`
says so rather than hiding it.

## Tests

```bash
pytest        # detector unit tests + grounding-guard tests
```

## Project layout

```
src/insight/
  config.py      ingest.py     findings.py    llm.py       jsonx.py
  detectors/     narrator.py   grounding.py   sandbox.py    selection.py
  agentic.py     baseline.py   judge.py       answer_key.py
  evaluate.py    cli.py
tests/           config.yaml   csvdata.csv    pyproject.toml
```
