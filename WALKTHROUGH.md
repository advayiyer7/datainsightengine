# Walkthrough — In Plain English

A simple guide to what this project is, what I built, and what the results mean.
No jargon. If you only read one file, read this one.

---

## 1. The big idea (in one paragraph)

You have a spreadsheet of purchase orders (who you bought from, what, how much, when).
Buried in it are money-saving insights — like "you keep ordering the same thing in
tiny batches and paying extra shipping each time." The lazy way to find them is to
dump the whole spreadsheet into an AI and ask "what's wrong here?" — but that's
**expensive** and gets pricier as your data grows.

**This project tests a cheaper idea:** most of those insights are just *math patterns*
you can find with plain code (no AI). So we do the math with cheap code first, then
use a small, cheap AI **only** to write up the findings in nice English. The claim:
this matches the expensive "dump everything into AI" approach on quality, for a
fraction of the cost.

**The result: the claim mostly held up.** More on that below.

---

## 2. How the system is built (the "three tiers")

Think of it as an assembly line with three stations.

### Tier 0 — Read & tidy the spreadsheet
Every spreadsheet names its columns differently ("Supplier" vs "Vendor" vs "Seller").
This step **automatically figures out which column is which**, cleans up the data
(fixes dates, numbers, messy supplier names), and tells you what it did. If it guesses
a column wrong, you can correct it in one config file.

### Tier 1 — The "detectors" (cheap code, no AI)
Seven small programs, each hunting for one specific money problem:

| Detector | What it looks for | Plain-English example |
|---|---|---|
| Fragmented orders | Same thing ordered many times in a short window | "7 separate orders for office supplies in 27 days — combine them, save shipping" |
| Supplier concentration | One supplier dominating a category | "One vendor controls 62% of your MRO spend — risky" |
| Price variance | Paying different prices for the same thing | "Supplier A charges 21% more than Supplier B for the same category" |
| Tail spend | Lots of tiny orders that cost more to process than they're worth | "156 tiny orders cost ~$15,600 just to process" |
| Single-source risk | Something you can only buy from one place | "Only one supplier sells you this — no backup" |
| Timing anomaly | Weird timing, like quarter-end spending spikes | "Supplier X's orders cluster at quarter-end — budget flush?" |
| Duplicate order | Two near-identical orders close together | "Two $165k orders, 3 days apart — possible double-payment" |

Each detector outputs a tidy "finding" with **real numbers and the exact rows that
prove it**. No AI involved — this is fast and free.

### Tier 2 — The AI narrator (small, cheap AI)
The AI **never sees your raw spreadsheet** — it only sees the findings from Tier 1.
Its job: rank them by importance, merge duplicates, and write a short recommendation
for each ("What / Why it matters / Suggested action").

**The safety net:** there's a strict rule — the AI may only use numbers that already
exist in the findings. After it writes, a checker scans every dollar amount and
percentage and **flags anything the AI made up**. (This actually caught the AI
inventing a "$150k" figure during testing — the guard worked.)

### Tier 3 — The AI explorer (optional)
For weird patterns the seven detectors don't cover, an AI agent can **write its own
analysis code and run it** in a locked sandbox (it can't touch your files or the
internet — only the data). It's strictly capped: limited tries, time, and cost.

In the real run, this explorer found things the fixed detectors miss, like:
- 125 orders marked "compliant" that actually had **10–16% defect rates** hiding ~$913k in quality costs
- ~$163k in negotiated discounts that weren't consistently applied
- $6.99M of non-compliant spend concentrated with one supplier

---

## 3. The scoreboard (how we proved the idea)

The whole point was to **measure**, not just claim. So there's a built-in test that
runs four approaches on the same data and compares them:

- **B0** = dump everything into the AI (the expensive way we're testing against)
- **B1** = just the cheap detectors, no AI
- **SYS** = detectors + AI narrator (**our proposed approach**)
- **FULL** = everything including the AI explorer

It scores each on: **coverage** (did it find the known issues?), **precision** (are
the issues it found actually valuable? — judged 0–2 by an AI grader), **cost** (real
dollars), and **runtime**.

---

## 4. What actually happened (the results)

Ran live on a real 777-row procurement dataset. Here's the scoreboard:

| Approach | Found the known issues? (recall) | Quality per finding (0–2) | Cost |
|---|---|---|---|
| Detectors only (B1) | **100%** | 1.40 | **$0.00** |
| **Detectors + AI (SYS)** | **100%** | 1.40 | **$0.02** |
| Dump-everything-to-AI (B0) | 27% | 1.71 | $0.03 |
| Everything + explorer (FULL) | **100%** | 1.53 | $0.05 |

### The three takeaways

1. **💰 Cost stays flat as data grows (the main point).**
   As we fed more rows (78 → 777), the expensive "dump-to-AI" approach got **pricier
   and pricier**, while our approach **stayed roughly flat**. With bigger datasets the
   gap only widens. ✅ Claim supported.

2. **🎯 Our approach found WAY more (100% vs 27%).**
   The cheap detectors caught *every* known issue. The dump-everything-to-AI approach
   only caught about a quarter of them — it gets overwhelmed and overlooks things. ✅
   Claim supported.

3. **⚖️ The honest catch — the AI was "punchier" per item.**
   When the AI did surface an issue, its write-ups scored a bit higher on average
   (1.71 vs 1.40), because it naturally returns a few sharp insights instead of a long
   list. Our detector approach finds *everything* but that list includes some minor
   items that drag the average down. **The fix is easy** (only narrate the top items),
   and it's written up honestly in the results rather than hidden.

**Bottom line: the hypothesis was SUPPORTED.** You can match — and on coverage,
beat — the expensive approach for a fraction of the cost, and the cost stays flat as
your data grows.

> One honesty note baked into the report: this particular dataset has no product-code
> column, so the "price variance" detector compares broad categories rather than exact
> products. Its dollar figures are a *ceiling* (best case), not a precise promise.

---

## 5. Where to look in the repo

| If you want to see... | Open this |
|---|---|
| The headline results + chart | `RESULTS.md` and `eval_out/quality_vs_cost.png` |
| A finished insights report | `insights.md` |
| What the AI explorer tried, step by step | `agentic_log.json` |
| The raw detector output | `findings.json` |
| How to set it up and run it yourself | `README.md` |

### Try it yourself (5 commands)
```bash
pip install -e .                                              # install
insight ingest   --data "Procurement KPI Analysis Dataset.csv"   # read & tidy
insight detect   --data "Procurement KPI Analysis Dataset.csv"   # run the 7 detectors
insight run      --data "Procurement KPI Analysis Dataset.csv" --agentic   # full report
insight evaluate --data "Procurement KPI Analysis Dataset.csv"   # the scoreboard
```
(The AI parts need an `ANTHROPIC_API_KEY`; without one, the cheap detectors still run
fully and everything else degrades gracefully instead of crashing.)

---

## 6. Is it solid?

- **14 automated tests** check the detectors and the made-up-number guard. All pass.
- Every command was **actually run** on real data with live AI calls — these aren't
  hypothetical numbers.
- The AI sandbox was tested against attempts to read files, import code, and break
  out — all blocked.
