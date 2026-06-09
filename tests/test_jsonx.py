"""Tests for lenient LLM-JSON parsing (the cause of B0/agent eval failures)."""

from __future__ import annotations

import pytest

from insight.jsonx import loads_lenient, salvage_objects


def test_thousands_separator_commas_in_numbers():
    # The exact failure mode: large dollar figures written with commas.
    raw = '{"findings":[{"est_impact_usd": 309,389,563, "type":"x"}]}'
    out = loads_lenient(raw, array_key="findings")
    assert out["findings"][0]["est_impact_usd"] == 309389563


def test_strips_code_fences_and_prose():
    raw = 'Here you go:\n```json\n{"a": 1}\n```'
    assert loads_lenient(raw) == {"a": 1}


def test_extra_data_after_object():
    # Agent error mode: a valid object followed by trailing junk.
    raw = '{"action":"stop","reason":"done"} <- that is my answer'
    assert loads_lenient(raw)["action"] == "stop"


def test_commas_in_strings_preserved():
    raw = '{"supplier": "Bartlett and Company, Inc", "spend": 1,234}'
    out = loads_lenient(raw)
    assert out["supplier"] == "Bartlett and Company, Inc"
    assert out["spend"] == 1234


def test_salvage_truncated_array():
    raw = '{"insights":[{"id":0,"t":"a"},{"id":1,"t":"b"},{"id":2,"t":"trunc'
    objs = salvage_objects(raw)
    assert [o["id"] for o in objs] == [0, 1]


def test_unparseable_raises():
    with pytest.raises(Exception):
        loads_lenient("no json here at all")
