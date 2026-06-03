"""Cheap procurement insight engine.

Three tiers:
  Tier 0 — ingest & schema mapping (``insight.ingest``)
  Tier 1 — deterministic detector library (``insight.detectors``)
  Tier 2 — thin LLM narrator + grounding guard (``insight.narrator``)
  Tier 3 — bounded agentic discovery (``insight.agentic``)

Plus an evaluation harness (``insight.evaluate``) that compares approaches on
quality vs. cost.
"""

__version__ = "0.1.0"
