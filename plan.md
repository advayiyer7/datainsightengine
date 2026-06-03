# Claude Code Prompt — Cheap Procurement Insight Engine (detectors + LLM narrator + agentic discovery)

> Paste everything below the line into Claude Code, run from an **empty directory**. It's the full spec: the hypothesis being tested, the three-tier architecture, the evaluation harness, and a definition of done. Build it incrementally and **actually run it** against a real dataset — don't just emit files.

---

## What we're building and why

A system that reads a procurement / purchase-order dataset and automatically surfaces **valuable, actionable insights** (e.g. "you placed 4 separate orders to Supplier A for the same item in 18 days — consolidating would have saved ~$X in shipping") **cheaply** — without dumping the whole dataset into an LLM.

**The hypothesis this code is built to test:** *Most valuable procurement insights are computable patterns, not open-ended reasoning. A cheap deterministic detector layer + a thin LLM that only narrates the findings can match a "dump everything into the LLM" baseline on insight quality, at a fraction of the cost, and stay reliable as data scales.*

So the build is not just the engine — it's the engine **plus an evaluation harness that proves or breaks that hypothesis** by comparing approaches on quality vs. cost.

## Stack

- **Python 3.11+**, `pandas`, `numpy`
- **LLM:** Anthropic API via the `anthropic` Python SDK. Read the key from `ANTHROPIC_API_KEY` env var. Use `claude-haiku-4-5` for the cheap narrator/judge and allow a config flag to switch the agentic tier to a stronger model. **Never hardcode keys.**
- CLI via `argparse` or `typer`. Config via a simple `config.yaml`.
- `matplotlib` for the final quality-vs-cost chart.
- `pytest` for detector unit tests.
- Clean package layout, type hints, docstrings. A real `README.md`.

Keep the dataset path configurable — I will download a Kaggle procurement dataset (e.g. "Procurement KPI Analysis Dataset" or "Company Purchasing Dataset") and point the tool at the CSV. **Do not assume exact column names** — build a small schema-mapping step (below).

## Architecture — three tiers

### Tier 0 — Ingest & schema mapping
- Load the CSV. Detect/normalize the columns the detectors need: a transaction needs at minimum `supplier`, `item/sku`, `quantity`, `unit_price` (or total), and `order_date`. Optional: `category`, `shipping_cost`, `lead_time`, `delivery_date`, `risk_score`, `country`.
- Don't hardcode names. Implement `map_schema(df)` that uses fuzzy header matching + a config override (`config.yaml` `column_map:`) so I can correct mappings for whatever dataset I load. Print the resolved mapping and flag any required field it couldn't find.
- Clean: parse dates, coerce numerics, normalize supplier names (strip/case/whitespace; basic dedup of obvious variants), drop/flag unusable rows. Report what was cleaned.

### Tier 1 — Deterministic detector library (the core; pure pandas, no LLM)
Each detector is a function `detect(df, params) -> list[Finding]`. A **Finding** is a structured dataclass:
```
Finding(
  type: str,            # e.g. "fragmented_orders"
  severity: float,      # 0-1, for ranking
  entities: dict,       # {supplier, item, ...}
  metrics: dict,        # the numbers, e.g. {orders:4, window_days:18, est_extra_shipping:1240.0}
  evidence: dict,       # row ids / values that prove it — for grounding & verification
  est_impact_usd: float | None,
  one_line: str         # terse factual statement, numbers filled in (NOT LLM-written)
)
```
Implement these detectors (make thresholds configurable in `config.yaml`):
1. **fragmented_orders** — same supplier + same item, N≥k orders within a rolling window; estimate redundant shipping/handling cost.
2. **supplier_concentration** — spend share per supplier per category; flag categories where one supplier exceeds a share threshold (compute a Herfindahl-style concentration index). Surfaces "Supplier 1 is dominant."
3. **maverick_price_variance** — same item bought at materially different unit prices across orders/suppliers; quantify overpayment vs. the best observed price.
4. **tail_spend** — many tiny orders / long tail of low-value suppliers that inflate processing cost.
5. **single_source_risk** — items sourced from exactly one supplier (supply-risk exposure), weighted by spend.
6. **timing_anomaly** — orders clustered oddly in time, end-of-period spikes, or lead-time outliers vs. that supplier's norm.
7. **duplicate_payment / duplicate_order** — near-identical orders (same supplier/item/amount/date window) that may be errors.
Each detector returns Findings with real numbers + the evidence rows. Write **pytest unit tests** for at least 3 detectors using small hand-built DataFrames with a known correct answer.

### Tier 2 — LLM narrator + prioritizer (thin, cheap)
- Input: the **list of Findings only** (never the raw dataset). Typically 10–50 findings.
- The LLM: ranks them by business impact, deduplicates/merges related ones, and writes a short plain-language recommendation per top insight ("What / Why it matters / Est. impact / Suggested action").
- **Hard rule in the prompt: the model may only use numbers present in the Findings. It must not invent figures.** After generation, run a cheap programmatic check that every dollar amount / count in the narration appears in the source Finding's metrics; flag any that doesn't (this is the grounding/anti-hallucination guard).
- Output a clean ranked insights report (markdown + JSON).

### Tier 3 — Agentic discovery (optional, bounded, for novel patterns)
- A goal-directed loop for insights the fixed detectors don't cover. Given a broad goal ("find avoidable cost > $X" / "find anomalies in this data"), the agent **writes pandas code, executes it in a sandboxed exec environment over the dataframe, inspects the result, and iterates** until it surfaces candidate findings meeting the criteria or hits a cap.
- **Bounds (enforced, non-negotiable):** max N iterations, max wall-clock, max token budget — all from config. Run untrusted generated code in a restricted namespace (no file/network/system access; only the dataframe + pandas/numpy). Log every iteration's code + result.
- Output: new candidate Findings in the same schema, fed back through the Tier-2 narrator. Clearly label which insights came from detectors vs. the agent.

## The evaluation harness (this is half the point — build it as a first-class feature)

A `evaluate` command that runs these approaches on the same dataset and compares them:
- **B0 — Dump-to-LLM:** feed as much raw data as fits into the LLM context, ask for insights directly. The expensive ceiling. (If data exceeds context, sample/chunk and note it.)
- **B1 — Detectors only:** Tier 1 raw findings, no LLM.
- **SYS — Detectors + narrator:** Tiers 1+2 (the main proposal).
- **FULL — + agentic:** Tiers 1+2+3.

Metrics, reported in a table + chart:
- **Cost:** total tokens and est. USD per run (track input/output tokens per LLM call). Show how cost scales as you feed larger row-count slices (10%, 25%, 50%, 100%) — the thesis is SYS stays ~flat while B0 climbs.
- **Coverage (recall):** fraction of a **ground-truth insight set** that each approach surfaces. Generate that ground-truth set with a `make-answer-key` command that runs the detectors at strict thresholds on the full data and lets me hand-edit the resulting JSON — this is the Tier-1 "answer key" we validate against.
- **Validity (precision):** of the insights each approach surfaces, how many are real/non-trivial. Use an LLM-as-judge pass scoring each surfaced insight 0–2 (wrong / trivial / valuable) against the evidence, plus support manual override in the JSON.
- **Runtime.**

Produce: a results table (CSV), a **quality-vs-cost scatter/line chart** (the headline slide visual), and a short auto-written `RESULTS.md` summarizing what the run showed about the hypothesis.

## CLI (make these work)
```
insight ingest   --data path.csv [--config config.yaml]      # load, map schema, clean, report
insight detect   --data path.csv                              # Tier 1 -> findings.json
insight run      --data path.csv [--agentic]                  # full pipeline -> insights report (md+json)
insight make-answer-key --data path.csv                       # ground-truth insight set for eval
insight evaluate --data path.csv [--slices 0.1,0.25,0.5,1.0]  # B0/B1/SYS/FULL -> table + chart + RESULTS.md
```

## Definition of done
1. `pip install -e .` works; all commands run end to end on a real procurement CSV with **no crashes**; `pytest` passes.
2. Schema mapping handles unknown column names with fuzzy match + config override, and prints the resolved mapping.
3. All 7 detectors implemented with configurable thresholds and ≥3 covered by unit tests with known answers.
4. Narrator uses **only** Finding numbers; the programmatic grounding check catches any invented figure.
5. Agentic tier runs sandboxed with enforced iteration/time/token caps and logs every step.
6. `evaluate` produces the cost-vs-quality table, the headline chart, and an auto-written `RESULTS.md`.
7. `README.md` explains setup, the Kaggle dataset step, the hypothesis, and how to read the results.

## Build order (verify as you go — run real commands, don't just write code)
1. Scaffold package + config + README skeleton. 
2. Tier 0 ingest/schema/clean; test on a real downloaded CSV; print the mapping + cleaning report.
3. Tier 1 detectors one at a time, each with a unit test, each printing real findings on the dataset.
4. Tier 2 narrator + the grounding guard; produce a readable report.
5. `make-answer-key` + the `evaluate` harness with B0/B1/SYS; table + chart.
6. Tier 3 agentic layer (sandboxed, bounded) + FULL in eval.
7. Auto-write RESULTS.md; final pass on errors, tests, README.

Optimize for **cost transparency and honest evaluation** over feature count. If something is too expensive or unreliable, surface that in RESULTS.md rather than hiding it — a clean negative result is a valid outcome here.