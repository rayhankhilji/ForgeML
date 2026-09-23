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


def test_memory_reconstruction_requires_identical_accounting_scope():
    spec = {"shape": [4], "dtype": "float32", "device": "cpu"}
    explanation = {
        "graph": {
            "nodes": [{"name": "a", "spec": spec}, {"name": "out", "spec": spec}],
            "outputs": ["out"],
        },
        "memory_plan": {
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
    overlap["graph"]["nodes"].insert(1, {"name": "b", "spec": spec})
    overlap["memory_plan"]["naive_bytes"] = 32
    overlap["memory_plan"]["allocations"]["b"] = {
        "slot": 0,
        "size_bytes": 16,
        "first": 1,
        "last": 2,
    }
    with pytest.raises(ValueError, match="overlapping"):
        validate_memory_plan(overlap)
