import copy
import xml.etree.ElementTree as ET

import pytest

from forgeml.reporting import bar_chart, validate_memory_plan


def test_chart_is_valid_svg_with_escaped_labels():
    svg = bar_chart(
        [{"name": "x<&", "latency": 2.0}],
        [("latency", "measured < latency", "#67e8f9")],
        title="Measured & checked",
        unit="ms",
        note="finite samples only",
    )
    root = ET.fromstring(svg)
    assert root.attrib["role"] == "img"
    assert "x&lt;&amp;" in svg
    assert "2.000" in svg


def test_transpose_alias_reconstruction():
    base = {"shape": [2, 4], "dtype": "float32", "device": "cpu"}
    transposed = {"shape": [4, 2], "dtype": "float32", "device": "cpu"}
    slot = {"shape": [8], "dtype": "float32", "device": "cpu"}
    validate_memory_plan(
        {
            "graph": {
                "nodes": [
                    {"name": "a", "op": "relu", "inputs": ["x"], "spec": base},
                    {
                        "name": "t",
                        "op": "transpose",
                        "inputs": ["a"],
                        "spec": transposed,
                    },
                    {"name": "out", "op": "relu", "inputs": ["t"], "spec": transposed},
                ],
                "outputs": ["out"],
            },
            "memory_plan": {
                "aliases": {"t": "a"},
                "naive_bytes": 64,
                "planned_bytes": 32,
                "slot_specs": {"0": slot},
                "allocations": {"a": {"slot": 0, "size_bytes": 32, "first": 0, "last": 2}},
            },
        }
    )


def test_memory_reconstruction_requires_identical_accounting_scope():
    spec = {"shape": [4], "dtype": "float32", "device": "cpu"}
    explanation = {
        "graph": {
            "nodes": [
                {"name": "a", "op": "relu", "inputs": ["x"], "spec": spec},
                {"name": "out", "op": "relu", "inputs": ["a"], "spec": spec},
            ],
            "outputs": ["out"],
        },
        "memory_plan": {
            "aliases": {},
            "naive_bytes": 16,
            "planned_bytes": 16,
            "slot_specs": {"0": spec},
            "allocations": {"a": {"slot": 0, "size_bytes": 16, "first": 0, "last": 1}},
        },
    }
    validate_memory_plan(explanation)
    wrong = copy.deepcopy(explanation)
    wrong["memory_plan"]["naive_bytes"] = 32
    with pytest.raises(ValueError, match="totals"):
        validate_memory_plan(wrong)
    overlap = copy.deepcopy(explanation)
    overlap["graph"]["nodes"].insert(1, {"name": "b", "op": "relu", "inputs": ["x"], "spec": spec})
    overlap["memory_plan"]["naive_bytes"] = 32
    overlap["memory_plan"]["allocations"]["a"]["last"] = 2
    overlap["memory_plan"]["allocations"]["b"] = {
        "slot": 0,
        "size_bytes": 16,
        "first": 1,
        "last": 1,
    }
    with pytest.raises(ValueError, match="overlapping"):
        validate_memory_plan(overlap)
