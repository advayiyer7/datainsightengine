"""Command-line interface.

    insight ingest   --data path.csv [--config config.yaml]
    insight detect   --data path.csv [-o findings.json]
    insight run      --data path.csv [--agentic]
    insight make-answer-key --data path.csv [-o answer_key.json]
    insight evaluate --data path.csv [--slices 0.1,0.25,0.5,1.0] [--no-agentic]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .agentic import discover
from .answer_key import build_answer_key, load_answer_key, write_answer_key
from .config import load_config
from .detectors import run_all
from .findings import findings_to_json
from .ingest import ingest
from .llm import LLMClient
from .narrator import narrate


def _reconfigure_stdout() -> None:
    """Make console output UTF-8 safe on Windows (em-dashes, ✓, etc.)."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except Exception:
            pass


def _load(args) -> tuple:
    cfg = load_config(args.config)
    df, sm, rep = ingest(args.data, column_map=cfg.get("column_map"))
    return cfg, df, sm, rep


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------
def cmd_ingest(args) -> int:
    cfg, df, sm, rep = _load(args)
    print(sm.render())
    print()
    print(rep.render())
    if sm.missing_required:
        print("\nERROR: required fields unmapped:", ", ".join(sm.missing_required))
        print("Add a column_map override in config.yaml to fix.")
        return 2
    print(f"\nOK: {len(df)} clean rows ready for detection.")
    return 0


def cmd_detect(args) -> int:
    cfg, df, sm, rep = _load(args)
    findings = run_all(df, cfg.get("detectors", {}))
    out = args.output or "findings.json"
    Path(out).write_text(findings_to_json(findings), encoding="utf-8")
    from collections import Counter
    by_type = Counter(f.type for f in findings)
    print(f"Detected {len(findings)} findings -> {out}")
    for t, n in by_type.most_common():
        print(f"  {t:<26} {n}")
    print("\nTop findings:")
    for f in findings[:8]:
        imp = f"${f.est_impact_usd:,.0f}" if f.est_impact_usd else "n/a"
        print(f"  [{f.severity:.2f}] {f.type:<24} {imp:>14}  {f.one_line}")
    return 0


def cmd_run(args) -> int:
    cfg, df, sm, rep = _load(args)
    findings = run_all(df, cfg.get("detectors", {}))
    client = LLMClient(cfg=cfg.get("llm", {}))
    if not client.available:
        print("(note: ANTHROPIC_API_KEY not set — narrator runs in deterministic fallback)\n")

    if args.agentic:
        ag = discover(df, client, cfg.get("agentic", {}))
        print(f"Agentic discovery: {ag.stopped_reason}; +{len(ag.findings)} candidate findings "
              f"over {ag.iterations_used} iteration(s).")
        Path("agentic_log.json").write_text(json.dumps(ag.to_dict(), indent=2, default=str), encoding="utf-8")
        findings = findings + ag.findings

    ncfg = cfg.get("narrator", {})
    narr = narrate(
        findings, client,
        top_n=int(ncfg.get("top_n", 8)),
        rank_formula=ncfg.get("rank_formula", "severity_x_impact"),
        null_impact_usd=float(ncfg.get("null_impact_usd", 20000.0)),
        max_per_type=ncfg.get("max_per_type"),
    )
    # The FULL finding set is persisted to findings.json; the report is the selected top-N.
    Path("findings.json").write_text(findings_to_json(findings), encoding="utf-8")
    Path("insights.md").write_text(narr.markdown, encoding="utf-8")
    Path("insights.json").write_text(
        json.dumps({"findings": [f.to_dict() for f in findings], "report": narr.to_dict()},
                   indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\nWrote findings.json ({len(findings)} findings), insights.md and insights.json "
          f"({len(narr.insights)} selected insights).")
    g = narr.grounding
    status = "all grounded" if g.ok else f"{len(g.unmatched)} ungrounded after repair"
    print(f"Grounding guard: {g.matched}/{g.total_numbers} numbers traced to findings — {status}")
    print(f"Spurious-number rate: {narr.spurious_rate_before:.0%} before repair -> "
          f"{narr.spurious_rate_after:.0%} after.")
    if not g.ok:
        for u in g.unmatched:
            print(f"  ungrounded: {u['text']} ({u['value']})")
    print(f"LLM tokens used: {client.usage.total_tokens} (est ${client.usage.est_usd:.4f})")
    print("\n--- report preview ---\n")
    print("\n".join(narr.markdown.splitlines()[:24]))
    return 0


def cmd_make_answer_key(args) -> int:
    cfg, df, sm, rep = _load(args)
    client = LLMClient(cfg=cfg.get("llm", {}))
    if not client.available:
        print("(note: ANTHROPIC_API_KEY not set — candidates kept unjudged; set a key to judge-filter)")
    entries, meta = build_answer_key(df, cfg.get("detectors", {}), client)
    out = args.output or "answer_key.json"
    write_answer_key(entries, out, meta)
    print(f"Wrote {len(entries)} ground-truth entries -> {out}")
    if meta.get("judged"):
        print(f"Judge kept {meta['kept_valuable']} valuable / dropped "
              f"{meta['dropped_trivial_or_wrong']} trivial-or-wrong of {meta['candidates']} candidates.")
    print("REVIEW REQUIRED: edit validity/include and ADD non-detector insights (source:\"manual\"), "
          "then set curated:true — otherwise detector recall is upper-bounded by construction.")
    from collections import Counter
    for t, n in Counter(e["type"] for e in entries).most_common():
        print(f"  {t:<26} {n}")
    return 0


def cmd_evaluate(args) -> int:
    from .evaluate import run_evaluation  # lazy import (pulls matplotlib)
    cfg, df, sm, rep = _load(args)
    slices = None
    if args.slices:
        slices = [float(x) for x in args.slices.split(",") if x.strip()]

    gold = None
    gold_curated = False
    key_path = args.answer_key or "answer_key.json"
    if Path(key_path).exists():
        gold, gold_curated = load_answer_key(key_path)
        print(f"Using answer key: {key_path} ({len(gold)} entries, curated={gold_curated})")
    else:
        print(f"No {key_path} found — generating an in-memory judge-curated key for this run.")

    client = LLMClient(cfg=cfg.get("llm", {}))
    if not client.available:
        print("(note: ANTHROPIC_API_KEY not set — B0/SYS/FULL run in fallback; LLM costs read $0)")

    out = run_evaluation(df, cfg, gold=gold, gold_curated=gold_curated, out_dir=args.out_dir,
                         slices=slices, run_agentic=not args.no_agentic)
    print(f"\nWrote {args.out_dir}/results_quality.csv, results_scaling.csv, RESULTS.md"
          + (f", {Path(out.chart_path).name}" if out.chart_path else " (chart skipped)"))
    print("\n--- quality table ---")
    cols = ["approach", "n_findings", "recall", "validity_mean_0_2", "est_usd", "runtime_s"]
    print(" | ".join(cols))
    for r in out.quality_rows:
        print(" | ".join(str(r[c]) for c in cols))
    print("\nSee RESULTS.md for the full write-up.")
    return 0


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="insight", description="Cheap procurement insight engine.")
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp):
        sp.add_argument("--data", required=True, help="path to procurement CSV")
        sp.add_argument("--config", default="config.yaml", help="path to config.yaml")

    sp = sub.add_parser("ingest", help="load, map schema, clean, report")
    add_common(sp); sp.set_defaults(func=cmd_ingest)

    sp = sub.add_parser("detect", help="Tier 1 detectors -> findings.json")
    add_common(sp); sp.add_argument("-o", "--output"); sp.set_defaults(func=cmd_detect)

    sp = sub.add_parser("run", help="full pipeline -> insights report (md+json)")
    add_common(sp); sp.add_argument("--agentic", action="store_true", help="enable Tier 3 agentic discovery")
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("make-answer-key", help="ground-truth insight set for eval")
    add_common(sp); sp.add_argument("-o", "--output"); sp.set_defaults(func=cmd_make_answer_key)

    sp = sub.add_parser("evaluate", help="B0/B1/SYS/FULL -> table + chart + RESULTS.md")
    add_common(sp)
    sp.add_argument("--slices", help="comma list, e.g. 0.1,0.25,0.5,1.0")
    sp.add_argument("--answer-key", help="path to answer_key.json (default: ./answer_key.json)")
    sp.add_argument("--out-dir", default="eval_out", help="output directory")
    sp.add_argument("--no-agentic", action="store_true", help="skip the FULL agentic tier")
    sp.set_defaults(func=cmd_evaluate)
    return p


def main(argv: list[str] | None = None) -> int:
    _reconfigure_stdout()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except FileNotFoundError as exc:
        print(f"ERROR: file not found: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
