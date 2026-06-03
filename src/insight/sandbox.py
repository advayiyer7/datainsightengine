"""Restricted execution sandbox for agent-generated pandas code.

The agent (Tier 3) writes pandas snippets that we must run over the dataframe
without granting file/network/system access. Two layers of defense:

  1. Static screen: reject code containing forbidden tokens (imports, dunders,
     ``open``/``eval``/``exec``, os/sys/subprocess/socket, etc.) before running.
  2. Restricted namespace: execute with a curated ``__builtins__`` that omits
     ``__import__``, ``open``, ``eval``, ``exec`` and friends, exposing only ``df``,
     ``pd``, ``np`` and a safe subset of builtins.

This is not a security boundary against a determined adversary, but it blocks the
realistic failure modes of an LLM that wanders off-task. The dataframe is passed
as a copy so generated code cannot mutate the caller's data.
"""

from __future__ import annotations

import io
import re
from contextlib import redirect_stdout
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

_FORBIDDEN = [
    r"\bimport\b",
    r"\b__\w+__\b",      # dunder access (__builtins__, __class__, ...)
    r"\bopen\b",
    r"\beval\b",
    r"\bexec\b",
    r"\bcompile\b",
    r"\bgetattr\b",
    r"\bsetattr\b",
    r"\bglobals\b",
    r"\blocals\b",
    r"\bos\b",
    r"\bsys\b",
    r"\bsubprocess\b",
    r"\bsocket\b",
    r"\bShell\b",
    r"\binput\b",
    r"\bto_csv\b",
    r"\bto_pickle\b",
    r"\bread_csv\b",
]
_FORBIDDEN_RE = re.compile("|".join(_FORBIDDEN))

_SAFE_BUILTINS = {
    name: __builtins__[name] if isinstance(__builtins__, dict) else getattr(__builtins__, name)
    for name in [
        "abs", "min", "max", "sum", "len", "range", "sorted", "round", "enumerate",
        "zip", "map", "filter", "list", "dict", "set", "tuple", "float", "int", "str",
        "bool", "print", "any", "all", "reversed", "divmod", "pow", "repr",
    ]
}


@dataclass
class ExecResult:
    ok: bool
    stdout: str = ""
    result_repr: str = ""
    error: str = ""


def _violations(code: str) -> list[str]:
    return list({m.group(0) for m in _FORBIDDEN_RE.finditer(code)})


def run_sandboxed(code: str, df: pd.DataFrame, *, max_output_chars: int = 4000) -> ExecResult:
    """Execute ``code`` against a copy of ``df`` in a restricted namespace.

    Conventions for the agent: assign your answer to a variable named ``result``
    (a number, string, Series, or DataFrame) and/or ``print`` summaries. The repr
    of ``result`` (truncated) is returned for the agent to inspect next iteration.
    """
    bad = _violations(code)
    if bad:
        return ExecResult(ok=False, error=f"blocked tokens: {sorted(bad)}")

    namespace: dict[str, Any] = {
        "__builtins__": _SAFE_BUILTINS,
        "pd": pd,
        "np": np,
        "df": df.copy(),
        "result": None,
    }
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            exec(code, namespace)  # noqa: S102 — intentional, sandboxed above
    except Exception as exc:  # surface the error back to the agent, don't crash
        return ExecResult(ok=False, stdout=buf.getvalue()[:max_output_chars], error=f"{type(exc).__name__}: {exc}")

    result = namespace.get("result")
    rep = ""
    if result is not None:
        try:
            if isinstance(result, (pd.DataFrame, pd.Series)):
                rep = result.head(30).to_string()
            else:
                rep = repr(result)
        except Exception as exc:
            rep = f"<unreprable result: {exc}>"
    return ExecResult(
        ok=True,
        stdout=buf.getvalue()[:max_output_chars],
        result_repr=rep[:max_output_chars],
    )
