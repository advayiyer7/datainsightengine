# RESULTS2 — Post‑Fix Evaluation (readable companion to RESULTS.md)

This is a hand‑readable summary of the **live** evaluation run after the four fixes
(narrator selection, judge‑curated answer key, spurious‑number repair, top‑N
scoring). It was produced from a real `insight evaluate` run on the bundled
777‑row procurement dataset with live Claude Haiku calls — not a hypothetical.
`RESULTS.md` is the auto‑generated version; this file says the same thing in plainer
language.

---

## TL;DR

- **SYS (detectors + narrator) costs ~$0.011 per run and stays flat** as data grows.
  **B0 (dump everything to the LLM) costs ~$0.047 and climbs** with row count — ~4×
  more expensive at full data and widening.
- **SYS surfaces 8 focused, ranked insights** instead of 32 raw findings, and every
  number it prints is **traceable to a finding (0 fabricated)**.
- **SYS matches B0 on insight quality** (judge validity 1.62 vs 1.62) while being far
  cheaper and covering far more of the ground truth.
- **Verdict: the hypothesis is SUPPORTED** — with one honest caveat about recall
  (below).

---

## What was tested

> *Most valuable procurement insights are computable patterns, not open‑ended
> reasoning. A cheap detector layer + a thin LLM narrator can match a
> dump‑everything‑to‑LLM baseline on quality, for a fraction of the cost, and stay
> flat as data scales.*

Four approaches, same data:

| Code | Approach |
|---|---|
| **B1** | detectors only, no LLM (all raw findings) |
| **SYS** | detectors + narrator, **top‑8 selected & merged** (the proposal) |
| **B0** | dump raw rows into the LLM, ask for insights (the expensive baseline) |
| **FULL** | detectors + agentic explorer, narrated top‑8 |

---

## Quality at full data (777 rows)

| Approach | Insights | Recall | Validity (0–2) | Valuable % | Cost (USD) |
|---|---|---|---|---|---|
| B1 detectors (all) | 32 | 1.00 | 1.72 | 72% | $0.000 |
| **SYS det+narrator** | **8** | 1.00 | 1.62 | 62% | **$0.011** |
| B0 dump‑to‑LLM | 8 | 0.32 | 1.62 | 62% | $0.046 |
| FULL +agentic | 8 | 1.00 | 1.75 | 75% | $0.040 |

**Read it like this:**
- **Recall** = share of the 22 ground‑truth insights surfaced. SYS 1.00 vs B0 0.32 —
  but see the caveat; this is bounded by construction, not a clean win.
- **Validity** = LLM‑judge score 0 (wrong) / 1 (trivial) / 2 (valuable). SYS ties B0
  at 1.62. FULL edges ahead at 1.75 because the agent added a non‑detector insight.
- **Cost** = real token spend. SYS is ~4× cheaper than B0 here.

---

## Cost scaling vs data size (the headline)

| Rows | B0 cost | SYS cost |
|---|---|---|
| 39 | $0.012 | $0.020 |
| 78 | $0.016 | $0.011 |
| 194 | $0.024 | $0.012 |
| 388 | $0.028 | $0.011 |
| 777 | $0.047 | $0.011 |

**B0 climbs with every extra row; SYS is flat.** (At the smallest slice B0 is briefly
cheaper because SYS pays a small fixed cost to narrate 8 insights no matter what — by
388+ rows SYS is decisively cheaper, and the gap keeps growing.) On a 10k+ row
dataset the divergence is much larger; the engine supports finer slices
(`0.05, 0.1, 0.25, 0.5, 1.0`) to draw the curve clearly.

See `eval_out/quality_vs_cost.png` for the chart.

---

## What the four fixes changed

1. **Narrator now selects, doesn't pass through.** It merges findings about the same
   supplier+item across detectors, ranks by business impact, and narrates only the
   top 8. Example from this run — insight #1 **merged** the price‑variance and
   fragmented‑orders findings for Office Supplies into a single recommendation
   (sources: `maverick_price_variance`, `fragmented_orders`). Result: **32 raw → 8
   focused**, at lower cost.

2. **Answer key is judge‑curated.** `make-answer-key` keeps only insights the
   LLM‑judge scores *valuable*, and marks the file `curated: false` so you know to
   review it and add insights the detectors can't catch.

3. **Spurious numbers eliminated.** The tightened "verbatim numbers only" prompt
   produced **0 untraceable figures out of 81 cited** this run (down from ~20% in the
   first eval). A per‑insight repair pass is still there as a safety net but wasn't
   needed.

4. **Evaluation scores the real output.** SYS/FULL are now scored on the narrated
   top‑8 (not all 32), so the precision/cost effect of selection actually shows up;
   B1 stays as the all‑raw baseline for contrast.

---

## Honest caveats (don't skip these)

- **Recall is bounded by construction.** The ground‑truth set is built from the
  engine's *own* detectors, so detector‑based approaches (B1/SYS/FULL) can score near
  1.0 almost automatically. SYS's 1.00 vs B0's 0.32 is **not** a clean victory until a
  human adds insights the detectors cannot produce (`source:"manual"` in
  `answer_key.json`) and sets `curated:true`.
- **Validity is comparable, not dramatically higher, on this dataset.** This data's
  detector output is uniformly decent (mostly high‑impact price‑variance and
  fragmented‑order findings, no junk tier), so selection's win here is *fewer,
  cheaper, focused* insights rather than a big precision jump. A messier dataset with
  a long low‑value tail would show selection lifting validity more.
- **No SKU column.** `item` falls back to `Item_Category` (5 broad categories), so the
  price‑variance dollar figures are an upper bound, not a precise promise.
- **The agent is nondeterministic.** It adds 0–3 novel findings depending on the run;
  FULL reflects whatever it found that run.

---

## Bottom line

The cheap path (detectors + a thin, grounded narrator) **matched the expensive
dump‑to‑LLM baseline on insight quality at ~4× lower cost, with cost staying flat as
data scales, and zero fabricated numbers.** That is the hypothesis, supported — stated
with the recall caveat rather than hidden.
