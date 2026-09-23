import copy

import pytest

from forgeml.measurement import summarize
from nebula.benchmarks import compare_reports


def point(tp=1, pp=1, latency=12.0):
    return {
        "suite": "nebula.generation",
        "measured": True,
        "topology": {"tp": tp, "pp": pp, "world_size": tp * pp},
        "model": {"hidden_size": 16},
        "settings": {"batch": 4, "new_tokens": 8},
        "environment": {
            "device": "cpu",
            "torch": "test-fixture",
            "torch_threads": 1,
            "platform": "synthetic-test-fixture",
            "git_commit": "fixture-not-a-measurement",
            "git_dirty": False,
        },
        "correctness": {
            "greedy_tokens_exact": True,
            "prefill": {"passed": True},
            "decode": {"passed": True},
        },
        "generation": summarize([latency, latency]),
    }


def test_strong_scaling_definitions():
    report = compare_reports([point(), point(tp=2, latency=8.0), point(tp=2, pp=2, latency=6.0)])
    points = report["points"]
    assert [row["speedup"] for row in points] == [1.0, 1.5, 2.0]
    assert [row["parallel_efficiency"] for row in points] == [1.0, 0.75, 0.5]


def test_scaling_rejects_workload_changes_and_fabricated_summaries():
    baseline = point()
    changed_batch = point(tp=2)
    changed_batch["settings"]["batch"] = 8
    with pytest.raises(ValueError, match="must match"):
        compare_reports([baseline, changed_batch])
    changed_median = point(tp=2)
    changed_median["generation"]["median_ms"] = 1.0
    with pytest.raises(ValueError, match="raw samples"):
        compare_reports([baseline, changed_median])
    unmeasured = point(tp=2)
    unmeasured["measured"] = False
    with pytest.raises(ValueError, match="measured"):
        compare_reports([baseline, unmeasured])
    failed = point(tp=2)
    failed["correctness"]["decode"]["passed"] = False
    with pytest.raises(ValueError, match="correctness"):
        compare_reports([baseline, failed])


def test_scaling_requires_unique_baseline_and_topologies():
    with pytest.raises(ValueError, match="baseline"):
        compare_reports([point(tp=2)])
    baseline = point()
    with pytest.raises(ValueError, match="baseline"):
        compare_reports([baseline, copy.deepcopy(baseline)])
    with pytest.raises(ValueError, match="duplicate topology"):
        compare_reports([baseline, point(tp=2), point(tp=2)])
    dirty = point()
    dirty["environment"]["git_dirty"] = True
    with pytest.raises(ValueError, match="clean recorded commit"):
        compare_reports([dirty])
