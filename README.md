# Cheap Procurement Insight Engine

Surfaces **valuable, actionable procurement insights** from a purchase-order CSV —
e.g. _"you placed 7 separate orders to Delta_Logistics for Office Supplies in 27
days; consolidating could save ~$1,500 in shipping"_ — **cheaply**, without dumping
the whole dataset into an LLM.

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
| **0** Ingest | `insight.ingest` | Load CSV, fuzzy-map headers to a canonical schema (config override allowed), clean/normalize, report what changed. | no |
| **1** Detectors | `insight.detectors` | 7 deterministic pandas detectors → structured `Finding`s with real numbers + evidence. | no |
| **2** Narrator | `insight.narrator` | Ranks/merges findings and writes plain-language recommendations using **only** finding numbers; a grounding guard catches any invented figure. | thin/cheap |
| **3** Agentic | `insight.agentic` | Bounded, sandboxed loop that writes & runs pandas to discover novel patterns the fixed detectors miss. | optional |

Every tier speaks the same `Finding` schema (`insight.findings`):

```python
Finding(type, severity, entities, metrics, evidence, est_impact_usd, one_line, source)
```

### The 7 detectors (Tier 1)

1. `fragmented_orders` — same supplier+item, many orders in a rolling window → redundant shipping.
2. `supplier_concentration` — per-category spend share + Herfindahl index → dominance risk.
3. `maverick_price_variance` — same item bought cheaper elsewhere → overpayment vs the cheapest supplier's average.
4. `tail_spend` — long tail of tiny orders inflating processing cost.
5. `single_source_risk` — items sourced from exactly one supplier, weighted by spend.
6. `timing_anomaly` — lead-time outliers vs a supplier's norm + quarter-end clustering.
7. `duplicate_order` — near-identical orders within a short window → possible duplicate payment.

All thresholds live in `config.yaml`; ≥3 detectors are covered by unit tests with known answers (`tests/`).

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

Point the tool at any procurement / purchase-order CSV. A bundled example —
Kaggle's **"Procurement KPI Analysis Dataset"** (`Procurement KPI Analysis
Dataset.csv`, 777 rows) — is included. To use your own:

1. Download a procurement CSV (e.g. Kaggle "Procurement KPI Analysis Dataset" or
   "Company Purchasing Dataset").
2. Run `insight ingest --data your.csv` and read the **resolved schema mapping**.
3. If a column was mis-guessed, correct it in `config.yaml` under `column_map:`
   (`canonical_field: Your_Header`). Required fields: `supplier`, `item`,
   `quantity`, `order_date`, and one of `unit_price`/`total`.

> **Note on this dataset:** it has no SKU column, so `item` falls back to
> `Item_Category` (5 broad categories). `maverick_price_variance` therefore
> benchmarks the cheapest *supplier average* per category rather than per-SKU
> prices — its dollar figures are an **upper bound** on savings, not a precise
> number. This caveat is surfaced in `RESULTS.md` too.

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
  config.py      ingest.py     findings.py    llm.py
  detectors/     narrator.py   grounding.py   sandbox.py
  agentic.py     baseline.py   judge.py       answer_key.py
  evaluate.py    cli.py
tests/           config.yaml   pyproject.toml
```
