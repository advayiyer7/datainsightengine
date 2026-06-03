"""Tier 3 — bounded agentic discovery.

A goal-directed loop for patterns the fixed detectors don't cover. Each iteration
the agent may (a) run a pandas snippet in the sandbox and inspect the result, or
(b) emit candidate Findings in the standard schema. The loop is hard-bounded by
iteration count, wall-clock seconds, and token budget — all from config — and every
step's code + result is logged.

Without an API key this tier is a no-op (it needs the LLM to write code) and says so.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .findings import Finding
from .llm import LLMClient
from .sandbox import run_sandboxed

SYSTEM = (
    "You are a data-analysis agent hunting for valuable, NON-OBVIOUS procurement insights "
    "in a pandas DataFrame named `df`. Fixed detectors already cover: fragmented orders, "
    "supplier concentration, price variance, tail spend, single-source risk, timing anomalies, "
    "and duplicate orders — look for something ELSE (e.g. defect-rate vs price, compliance gaps, "
    "seasonal supplier cost drift, negotiation savings left on the table).\n\n"
    "Each turn return ONLY JSON, one of:\n"
    "  {\"action\":\"run_code\",\"code\":\"<python>\",\"reason\":\"...\"}  -- explore; assign to `result`.\n"
    "  {\"action\":\"emit_findings\",\"findings\":[{...}],\"reason\":\"...\"}  -- report candidates.\n"
    "  {\"action\":\"stop\",\"reason\":\"...\"}\n\n"
    "Sandbox rules: only `df`, `pd`, `np` are available. NO imports, file, or network access. "
    "Assign your computed answer to a variable `result`.\n\n"
    "Each finding object MUST be: {\"type\":str,\"severity\":0-1 float,\"entities\":{...},"
    "\"metrics\":{...with real numbers you computed...},\"evidence\":{...},\"est_impact_usd\":number|null,"
    "\"one_line\":str}. Only include findings you actually verified with code. Prefer 1-4 strong findings."
)


@dataclass
class IterationLog:
    n: int
    action: str
    code: str = ""
    reason: str = ""
    exec_ok: bool | None = None
    exec_output: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.n,
            "action": self.action,
            "reason": self.reason,
            "code": self.code,
            "exec_ok": self.exec_ok,
            "exec_output": self.exec_output[:1500],
        }


@dataclass
class AgenticResult:
    findings: list[Finding] = field(default_factory=list)
    log: list[IterationLog] = field(default_factory=list)
    stopped_reason: str = ""
    iterations_used: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "stopped_reason": self.stopped_reason,
            "iterations_used": self.iterations_used,
            "n_findings": len(self.findings),
            "log": [l.to_dict() for l in self.log],
            "findings": [f.to_dict() for f in self.findings],
        }


def _extract_json(text: str) -> dict[str, Any]:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        s, e = text.find("{"), text.rfind("}")
        if s != -1 and e > s:
            return json.loads(text[s : e + 1])
        raise


def _schema_brief(df: pd.DataFrame) -> str:
    cols = []
    for c in df.columns:
        cols.append(f"{c} ({df[c].dtype})")
    sample = df.head(3).to_dict(orient="records")
    return "Columns: " + ", ".join(cols) + f"\nRows: {len(df)}\nSample: {json.dumps(sample, default=str)[:1200]}"


def discover(
    df: pd.DataFrame,
    client: LLMClient,
    cfg: dict[str, Any],
) -> AgenticResult:
    """Run the bounded discovery loop. ``cfg`` is the ``agentic:`` config block."""
    res = AgenticResult()
    if not client.available:
        res.stopped_reason = "no_api_key (agentic tier requires the LLM to write code)"
        return res

    max_iter = int(cfg.get("max_iterations", 6))
    max_wall = float(cfg.get("max_wall_clock_seconds", 120))
    max_tokens = int(cfg.get("max_tokens", 60000))
    goal = cfg.get("goal", "Find avoidable cost or anomalies.")
    model = client.cfg.get("agentic_model", client.cfg.get("narrator_model"))

    start = time.monotonic()
    tokens_at_start = client.usage.total_tokens
    transcript = f"GOAL: {goal}\n\nDATAFRAME:\n{_schema_brief(df)}\n"
    history: list[str] = []

    for i in range(1, max_iter + 1):
        # --- enforce bounds BEFORE spending another call ---
        if time.monotonic() - start > max_wall:
            res.stopped_reason = f"wall_clock_exceeded ({max_wall}s)"
            break
        if client.usage.total_tokens - tokens_at_start > max_tokens:
            res.stopped_reason = f"token_budget_exceeded ({max_tokens})"
            break

        remaining = max_iter - i
        nudge = (
            f"\n\nIterations remaining after this one: {remaining}. "
            + ("You are near the cap — if you have ANY evidence, use emit_findings NOW; "
               "do not keep exploring." if remaining <= 2 else
               "Explore if needed, but emit_findings as soon as you have verified a real pattern.")
        )
        prompt = transcript + "\n\nHISTORY:\n" + ("\n".join(history[-6:]) if history else "(none)") + \
            nudge + "\n\nReturn your next JSON action."
        try:
            raw = client.complete(prompt, system=SYSTEM, model=model, max_tokens=1500)
            action = _extract_json(raw)
        except Exception as exc:
            res.log.append(IterationLog(n=i, action="error", reason=f"LLM/parse error: {exc}"))
            res.stopped_reason = f"llm_error: {exc}"
            break

        kind = action.get("action", "stop")
        reason = str(action.get("reason", ""))

        if kind == "run_code":
            code = str(action.get("code", ""))
            ex = run_sandboxed(code, df)
            out = ex.result_repr or ex.stdout or ex.error
            res.log.append(IterationLog(n=i, action="run_code", code=code, reason=reason,
                                        exec_ok=ex.ok, exec_output=out))
            history.append(f"[iter {i}] ran code; ok={ex.ok}; output:\n{out[:1200]}")
            continue

        if kind == "emit_findings":
            for fd in action.get("findings", []) or []:
                try:
                    f = Finding.from_dict({**fd, "source": "agent"})
                    f.severity = float(max(0.0, min(1.0, f.severity)))
                    res.findings.append(f)
                except Exception:
                    continue
            res.log.append(IterationLog(n=i, action="emit_findings", reason=reason,
                                        exec_output=f"{len(action.get('findings', []) or [])} candidate(s)"))
            res.stopped_reason = "emitted_findings"
            break

        # stop / unknown
        res.log.append(IterationLog(n=i, action="stop", reason=reason))
        res.stopped_reason = reason or "agent_stopped"
        break
    else:
        res.stopped_reason = f"max_iterations_reached ({max_iter})"

    res.iterations_used = len(res.log)
    return res
