"""Tier 0 — ingest, schema mapping, and cleaning.

The detectors need a small canonical schema, but real CSVs use arbitrary headers.
:func:`map_schema` resolves messy headers to canonical fields via (1) explicit
config override, (2) a synonym table, (3) fuzzy string matching — in that order.
:func:`clean` then coerces types, normalizes supplier names, derives helper
columns (``total``, ``lead_time_days``), and reports what it changed.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# Canonical field -> synonyms (lowercased, non-alphanumeric stripped for matching).
CANONICAL_SYNONYMS: dict[str, list[str]] = {
    "supplier": ["supplier", "vendor", "supplier_name", "vendor_name", "seller", "merchant"],
    "item": ["item", "sku", "product", "material", "part", "item_name", "description", "item_category"],
    "category": ["category", "item_category", "commodity", "spend_category", "class", "type"],
    "quantity": ["quantity", "qty", "units", "order_quantity", "amount_units", "volume"],
    "unit_price": ["unit_price", "unitcost", "price", "unit_cost", "price_per_unit", "rate"],
    "negotiated_price": ["negotiated_price", "negotiated", "contract_price", "agreed_price"],
    "total": ["total", "total_price", "amount", "total_cost", "line_total", "extended_price", "spend"],
    "order_date": ["order_date", "po_date", "purchase_date", "date", "created_date", "orderdate"],
    "delivery_date": ["delivery_date", "received_date", "ship_date", "receipt_date", "deliverydate"],
    "lead_time": ["lead_time", "leadtime", "lead_time_days", "delivery_days"],
    "shipping_cost": ["shipping_cost", "freight", "shipping", "delivery_cost", "handling"],
    "risk_score": ["risk_score", "risk", "supplier_risk"],
    "country": ["country", "region", "origin", "location"],
    "order_status": ["order_status", "status", "po_status", "state"],
    "defective_units": ["defective_units", "defects", "defective", "rejected_units"],
    "compliance": ["compliance", "compliant", "is_compliant"],
    "po_id": ["po_id", "po", "order_id", "purchase_order", "id", "po_number", "ponumber"],
}

# Fields a detector run cannot proceed without. ``unit_price`` OR ``total`` suffices.
REQUIRED = ["supplier", "item", "quantity", "order_date"]
REQUIRED_PRICE = ["unit_price", "total"]  # at least one


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


@dataclass
class SchemaMapping:
    """Result of :func:`map_schema`: canonical field -> source column (or None)."""

    mapping: dict[str, str | None] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def get(self, field_name: str) -> str | None:
        return self.mapping.get(field_name)

    @property
    def missing_required(self) -> list[str]:
        miss = [f for f in REQUIRED if not self.mapping.get(f)]
        if not any(self.mapping.get(f) for f in REQUIRED_PRICE):
            miss.append("unit_price|total")
        return miss

    def render(self) -> str:
        lines = ["Resolved schema mapping (canonical <- source column):"]
        for canon in CANONICAL_SYNONYMS:
            src = self.mapping.get(canon)
            if src:
                lines.append(f"  {canon:<18} <- {src}")
        unresolved = [c for c in CANONICAL_SYNONYMS if not self.mapping.get(c)]
        if unresolved:
            lines.append("  (unmapped: " + ", ".join(unresolved) + ")")
        if self.missing_required:
            lines.append("  !! MISSING REQUIRED: " + ", ".join(self.missing_required))
        for n in self.notes:
            lines.append("  note: " + n)
        return "\n".join(lines)


def map_schema(df: pd.DataFrame, column_map: dict[str, str] | None = None) -> SchemaMapping:
    """Resolve canonical fields to source columns.

    Priority per field: explicit ``column_map`` override > exact synonym match >
    fuzzy match (cutoff 0.82). A source column is claimed by at most one field.
    """
    column_map = column_map or {}
    headers = list(df.columns)
    norm_to_header = {_norm(h): h for h in headers}
    result: dict[str, str | None] = {}
    claimed: set[str] = set()
    notes: list[str] = []

    # 1) explicit overrides win and are validated against real headers.
    for canon, src in column_map.items():
        if src in headers:
            result[canon] = src
            claimed.add(src)
        elif src is not None:
            notes.append(f"config column_map['{canon}']='{src}' not found in CSV headers; ignoring")

    # 2) synonym + fuzzy for everything still unresolved.
    for canon, synonyms in CANONICAL_SYNONYMS.items():
        if result.get(canon):
            continue
        chosen: str | None = None
        for syn in synonyms:
            h = norm_to_header.get(_norm(syn))
            if h and h not in claimed:
                chosen = h
                break
        if chosen is None:
            candidates = [_norm(s) for s in synonyms]
            pool = [n for n, h in norm_to_header.items() if h not in claimed]
            best, best_score = None, 0.0
            for cand in candidates:
                for n in pool:
                    score = difflib.SequenceMatcher(None, cand, n).ratio()
                    if score > best_score:
                        best, best_score = n, score
            if best is not None and best_score >= 0.82:
                chosen = norm_to_header[best]
        if chosen:
            result[canon] = chosen
            claimed.add(chosen)
        else:
            result[canon] = None

    return SchemaMapping(mapping=result, notes=notes)


@dataclass
class CleanReport:
    """What :func:`clean` changed, for transparency."""

    rows_in: int = 0
    rows_out: int = 0
    dropped_missing_required: int = 0
    coerced_numeric: dict[str, int] = field(default_factory=dict)
    bad_dates: int = 0
    supplier_variants_merged: int = 0
    derived: list[str] = field(default_factory=list)

    def render(self) -> str:
        lines = ["Cleaning report:"]
        lines.append(f"  rows in:  {self.rows_in}")
        lines.append(f"  rows out: {self.rows_out}  (dropped {self.rows_in - self.rows_out})")
        if self.dropped_missing_required:
            lines.append(f"  dropped (missing required field values): {self.dropped_missing_required}")
        if self.bad_dates:
            lines.append(f"  unparseable order_dates coerced to NaT: {self.bad_dates}")
        for col, n in self.coerced_numeric.items():
            if n:
                lines.append(f"  non-numeric values coerced in {col}: {n}")
        if self.supplier_variants_merged:
            lines.append(f"  supplier name variants normalized: {self.supplier_variants_merged}")
        if self.derived:
            lines.append("  derived columns: " + ", ".join(self.derived))
        return "\n".join(lines)


def _normalize_supplier(name: Any) -> Any:
    if not isinstance(name, str):
        return name
    s = name.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def clean(
    df: pd.DataFrame, sm: SchemaMapping
) -> tuple[pd.DataFrame, CleanReport]:
    """Return a canonicalized, cleaned DataFrame plus a :class:`CleanReport`.

    The returned frame has canonical column names (``supplier``, ``item``, ...),
    parsed dates, numeric prices/quantities, a normalized ``supplier`` column,
    and derived ``total`` / ``lead_time_days`` where computable.
    """
    rep = CleanReport(rows_in=len(df))
    out = pd.DataFrame(index=df.index)

    # Rename mapped source columns to canonical names.
    for canon, src in sm.mapping.items():
        if src and src in df.columns:
            out[canon] = df[src]

    # Preserve a row id for evidence.
    if "po_id" in out.columns:
        out["row_id"] = out["po_id"].astype(str)
    else:
        out["row_id"] = ["row_" + str(i) for i in range(len(out))]

    # Numerics. Strip currency symbols / thousands separators / parens-negatives
    # ("$1,234.50", "(45.00)") before coercion so prices written as money parse.
    for col in ["quantity", "unit_price", "negotiated_price", "total",
                "shipping_cost", "lead_time", "risk_score", "defective_units"]:
        if col in out.columns:
            # Non-numeric (object OR pandas-3.0 `str` dtype) may hold currency text
            # like "$1,234.00" or "(45.00)" — strip it before coercion.
            if not pd.api.types.is_numeric_dtype(out[col]):
                s = out[col].astype(str).str.strip()
                # "(45.00)" accounting-negative -> "-45.00"
                s = s.str.replace(r"^\((.*)\)$", r"-\1", regex=True)
                s = s.str.replace(r"[\$,]", "", regex=True).str.strip()
                num = pd.to_numeric(s, errors="coerce")
                before_bad = int((num.isna() & out[col].notna()).sum())
                out[col] = num
            else:
                out[col] = pd.to_numeric(out[col], errors="coerce")
                before_bad = 0
            rep.coerced_numeric[col] = before_bad

    # Dates.
    for col in ["order_date", "delivery_date"]:
        if col in out.columns:
            parsed = pd.to_datetime(out[col], errors="coerce")
            if col == "order_date":
                rep.bad_dates = int(parsed.isna().sum() - out[col].isna().sum())
            out[col] = parsed

    # Supplier normalization + obvious-variant dedup (case/whitespace-insensitive).
    if "supplier" in out.columns:
        out["supplier"] = out["supplier"].map(_normalize_supplier)
        canon_map: dict[str, str] = {}
        for name in out["supplier"].dropna().unique():
            k = _norm(name)
            canon_map.setdefault(k, name)
        merged = 0
        new_vals = []
        for name in out["supplier"]:
            if isinstance(name, str):
                target = canon_map[_norm(name)]
                if target != name:
                    merged += 1
                new_vals.append(target)
            else:
                new_vals.append(name)
        out["supplier"] = new_vals
        rep.supplier_variants_merged = merged

    # Effective unit price prefers a negotiated price when present.
    if "negotiated_price" in out.columns and "unit_price" in out.columns:
        out["effective_unit_price"] = out["negotiated_price"].fillna(out["unit_price"])
    elif "unit_price" in out.columns:
        out["effective_unit_price"] = out["unit_price"]

    # Derive total spend if not provided.
    if "total" not in out.columns and "quantity" in out.columns and "effective_unit_price" in out.columns:
        out["total"] = out["quantity"] * out["effective_unit_price"]
        rep.derived.append("total")
    elif "total" in out.columns:
        # Fill any missing totals from qty*price where possible.
        if "quantity" in out.columns and "effective_unit_price" in out.columns:
            fill = out["quantity"] * out["effective_unit_price"]
            out["total"] = out["total"].fillna(fill)

    # Derive lead time from dates if not provided.
    if "lead_time" not in out.columns and "order_date" in out.columns and "delivery_date" in out.columns:
        lt = (out["delivery_date"] - out["order_date"]).dt.days
        out["lead_time_days"] = lt.where(lt >= 0)
        rep.derived.append("lead_time_days")
    elif "lead_time" in out.columns:
        out["lead_time_days"] = out["lead_time"]

    # Drop rows missing required values for detection.
    required_present = [c for c in ["supplier", "item", "quantity", "order_date"] if c in out.columns]
    before = len(out)
    if required_present:
        mask = out[required_present].notna().all(axis=1)
        # also need some price signal
        if "total" in out.columns:
            mask &= out["total"].notna()
        out = out[mask].copy()
    rep.dropped_missing_required = before - len(out)
    rep.rows_out = len(out)

    out = out.reset_index(drop=True)
    return out, rep


def ingest(
    csv_path: str, column_map: dict[str, str] | None = None
) -> tuple[pd.DataFrame, SchemaMapping, CleanReport]:
    """Full Tier 0: read CSV, map schema, clean. Returns (df, mapping, report)."""
    raw = pd.read_csv(csv_path)
    sm = map_schema(raw, column_map=column_map)
    df, rep = clean(raw, sm)
    return df, sm, rep
