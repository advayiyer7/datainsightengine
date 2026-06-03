"""Thin wrapper around the Anthropic API with token + cost accounting.

Design goals:
  * Never hardcode keys — read ``ANTHROPIC_API_KEY`` from the environment.
  * Track input/output tokens per call so the evaluator can report real cost.
  * Degrade gracefully: importing this module never requires a key or network.
    A clear error is raised only when an actual call is attempted without a key.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

try:  # the SDK is a hard dependency, but keep import failures legible
    import anthropic
except Exception:  # pragma: no cover
    anthropic = None  # type: ignore


class LLMUnavailable(RuntimeError):
    """Raised when an LLM call is requested but no key/SDK is available."""


@dataclass
class Usage:
    """Running tally of token usage and estimated cost across many calls."""

    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    price_per_mtok_input: float = 1.0
    price_per_mtok_output: float = 5.0

    def add(self, in_tok: int, out_tok: int) -> None:
        self.input_tokens += int(in_tok)
        self.output_tokens += int(out_tok)
        self.calls += 1

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def est_usd(self) -> float:
        return (
            self.input_tokens / 1_000_000 * self.price_per_mtok_input
            + self.output_tokens / 1_000_000 * self.price_per_mtok_output
        )

    def merge(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.calls += other.calls

    def to_dict(self) -> dict:
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "est_usd": round(self.est_usd, 6),
        }


@dataclass
class LLMClient:
    """A small client that wraps message creation and accumulates usage.

    Each instance carries its own :class:`Usage` tally; pass an ``llm`` config
    dict (from ``config.yaml``) to set models and pricing.
    """

    cfg: dict = field(default_factory=dict)
    usage: Usage = field(default_factory=Usage)
    _client: object | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.usage.price_per_mtok_input = float(self.cfg.get("price_per_mtok_input", 1.0))
        self.usage.price_per_mtok_output = float(self.cfg.get("price_per_mtok_output", 5.0))

    @property
    def available(self) -> bool:
        return anthropic is not None and bool(os.environ.get("ANTHROPIC_API_KEY"))

    def _ensure(self) -> object:
        if anthropic is None:
            raise LLMUnavailable(
                "The 'anthropic' package is not importable. Run `pip install anthropic`."
            )
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise LLMUnavailable(
                "ANTHROPIC_API_KEY is not set. Export your key, e.g.\n"
                "  PowerShell:  $env:ANTHROPIC_API_KEY = 'sk-...'\n"
                "  bash:        export ANTHROPIC_API_KEY=sk-..."
            )
        if self._client is None:
            self._client = anthropic.Anthropic()
        return self._client

    def complete(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> str:
        """Send a single-user-turn message and return the text, tallying usage."""
        client = self._ensure()
        model = model or self.cfg.get("narrator_model", "claude-haiku-4-5")
        max_tokens = max_tokens or int(self.cfg.get("max_output_tokens", 2048))
        kwargs = dict(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        if system:
            kwargs["system"] = system
        resp = client.messages.create(**kwargs)  # type: ignore[attr-defined]
        self.usage.add(resp.usage.input_tokens, resp.usage.output_tokens)
        parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
        return "".join(parts).strip()
