"""The :class:`Finding` schema shared across all tiers.

A Finding is the universal currency of the engine: detectors emit them, the
narrator consumes them, the agent produces them in the same shape, and the
evaluator compares them. It is deliberately serializable to plain JSON.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Finding:
    """A structured, evidence-backed observation about the data.

    Attributes:
        type: Stable machine key, e.g. ``"fragmented_orders"``.
        severity: 0-1 score used for ranking (higher = more important).
        entities: The subjects, e.g. ``{"supplier": "Gamma_Co", "item": "MRO"}``.
        metrics: The numbers behind the claim, e.g. ``{"orders": 4, "window_days": 18}``.
            Every figure cited by the narrator must trace back to here.
        evidence: Row ids / raw values that prove the finding, for grounding.
        est_impact_usd: Estimated dollar impact, or ``None`` if not quantifiable.
        one_line: Terse factual statement with numbers filled in. NOT LLM-written.
        source: Provenance — ``"detector"`` or ``"agent"``.
    """

    type: str
    severity: float
    entities: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)
    est_impact_usd: float | None = None
    one_line: str = ""
    source: str = "detector"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Finding":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})

    def key(self) -> str:
        """A coverage key used to match findings across approaches/answer-keys.

        Two findings about the same pattern + same entities collide here, which is
        exactly what we want for recall scoring.
        """
        ents = "|".join(f"{k}={v}" for k, v in sorted(self.entities.items()))
        return f"{self.type}::{ents}"


def findings_to_json(findings: list[Finding], indent: int = 2) -> str:
    return json.dumps([f.to_dict() for f in findings], indent=indent, default=str)


def findings_from_json(text: str) -> list[Finding]:
    return [Finding.from_dict(d) for d in json.loads(text)]
