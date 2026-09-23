import json

import pytest
import torch

from forgeml import measurement


def test_summary_uses_raw_samples_and_linear_percentile():
    result = measurement.summarize([4.0, 1.0, 3.0, 2.0])
    assert result["samples_ms"] == [4.0, 1.0, 3.0, 2.0]
    assert result["median_ms"] == 2.5
    assert result["p95_ms"] == pytest.approx(3.85)
    assert result["count"] == 4
    assert result["min_ms"] == 1.0
    assert result["max_ms"] == 4.0


@pytest.mark.parametrize("samples", [[], [float("nan")], [float("inf")], [0.0], [-1.0]])
def test_invalid_latency_cannot_be_published(samples):
    with pytest.raises(ValueError):
        measurement.summarize(samples)


def test_measurement_rotates_variants_and_excludes_warmup(monkeypatch):
    calls = []
    ticks = iter(range(0, 100_000_000, 1_000_000))
    monkeypatch.setattr(measurement.time, "perf_counter_ns", lambda: next(ticks))
    report = measurement.measure_variants(
        {"a": lambda: calls.append("a"), "b": lambda: calls.append("b")},
        device="cpu",
        warmup=1,
        repeats=3,
    )
    assert calls == ["a", "b", "a", "b", "b", "a", "a", "b"]
    assert report["a"]["samples_ms"] == [1.0, 1.0, 1.0]
    assert report["b"]["samples_ms"] == [1.0, 1.0, 1.0]


def test_correctness_gate_rejects_wrong_or_nonfinite_values():
    expected = torch.ones(4)
    assert measurement.check_output(expected.clone(), expected)["max_absolute_error"] == 0.0
    with pytest.raises(AssertionError):
        measurement.check_output(torch.zeros(4), expected)
    with pytest.raises(AssertionError, match="non-finite"):
        measurement.check_output(torch.full((4,), float("nan")), expected)


def test_report_roundtrip_and_no_nan(tmp_path):
    path = tmp_path / "results" / "report.json"
    measurement.write_report({"samples": [1.0, 2.0]}, path)
    assert json.loads(path.read_text()) == {"samples": [1.0, 2.0]}
    with pytest.raises(ValueError):
        measurement.write_report({"wrong": float("nan")}, path)
    assert json.loads(path.read_text()) == {"samples": [1.0, 2.0]}


def test_autotune_cpu_does_not_invent_gpu_results():
    from forgeml import compile

    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.GELU()).eval()
    x = torch.randn(2, 4)
    compiled = compile(model, (x,))
    result = compiled.autotune(x, warmup=0, repeats=2)
    assert result["nodes"] == {}
    assert compiled.kernel_configs == {}


def test_autotune_excludes_incorrect_candidate(monkeypatch):
    from forgeml import autotune, compile, kernels

    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.GELU()).eval()
    x = torch.randn(2, 4)
    compiled = compile(model, (x,))
    compiled.kernel_plan = {node.name: "triton" for node in compiled.graph.nodes}

    def fake_matmul(a, b, bias=None, *, approximate="none", config=None, out=None):
        value = torch.nn.functional.gelu(a @ b + bias, approximate=approximate)
        return value + 10 if config == kernels.CANDIDATES[0] else value

    def fake_measure(variants, **kwargs):
        return {name: measurement.summarize([2.0, 2.0]) for name in variants}

    monkeypatch.setattr(kernels, "matmul", fake_matmul)
    monkeypatch.setattr(autotune, "measure_variants", fake_measure)
    report = autotune.tune_graph(compiled, (x,), warmup=0, repeats=2)
    node = next(iter(report["nodes"].values()))
    assert "candidate_0" in node["rejected"]
    assert "candidate_0" not in node["candidates"]
    assert node["winner"] == "candidate_1"
    assert list(compiled.kernel_configs.values()) == [kernels.CANDIDATES[1]]
