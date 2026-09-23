from __future__ import annotations

import json
import math
import os
import platform
import statistics
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch


def percentile(samples: Sequence[float], q: float) -> float:
    if not samples or not 0 <= q <= 1:
        raise ValueError("percentile requires samples and q in [0, 1]")
    values = sorted(samples)
    if not all(math.isfinite(v) for v in values):
        raise ValueError("samples must be finite")
    index = (len(values) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def summarize(samples_ms: Sequence[float]) -> dict[str, Any]:
    if not samples_ms or any(not math.isfinite(v) or v <= 0 for v in samples_ms):
        raise ValueError("latencies must be finite and positive")
    return {
        "samples_ms": list(samples_ms),
        "count": len(samples_ms),
        "median_ms": statistics.median(samples_ms),
        "p95_ms": percentile(samples_ms, 0.95),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
    }


def synchronize(device: torch.device | str) -> None:
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type != "cpu":
        raise ValueError("measurement supports CPU or CUDA, not other accelerators")


@torch.inference_mode()
def measure_variants(
    variants: Mapping[str, Callable[[], Any]],
    *,
    device: torch.device | str,
    warmup: int = 5,
    repeats: int = 25,
) -> dict[str, dict[str, Any]]:
    if not variants or warmup < 0 or repeats < 2:
        raise ValueError("provide variants, warmup >= 0 and repeats >= 2")
    names = list(variants)
    for name in names:
        for _ in range(warmup):
            variants[name]()
    synchronize(device)
    samples: dict[str, list[float]] = {name: [] for name in names}
    for iteration in range(repeats):
        offset = iteration % len(names)
        for name in names[offset:] + names[:offset]:
            synchronize(device)
            start = time.perf_counter_ns()
            result = variants[name]()
            synchronize(device)
            elapsed = (time.perf_counter_ns() - start) / 1_000_000
            samples[name].append(elapsed)
            del result
    return {name: summarize(values) for name, values in samples.items()}


def tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float16:
        return 1e-2, 1e-2
    if dtype == torch.bfloat16:
        return 5e-2, 5e-2
    if dtype == torch.float32:
        return 1e-4, 1e-4
    if dtype == torch.float64:
        return 1e-7, 1e-8
    return 0.0, 0.0


@torch.inference_mode()
def check_output(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float | bool]:
    if not isinstance(actual, torch.Tensor) or not isinstance(expected, torch.Tensor):
        raise TypeError("benchmark outputs must be tensors")
    rtol, atol = tolerances(expected.dtype)
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise AssertionError("non-finite output cannot pass a benchmark correctness gate")
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    delta = (actual.to(torch.float64) - expected.to(torch.float64)).abs()
    return {
        "passed": True,
        "rtol": rtol,
        "atol": atol,
        "max_absolute_error": delta.max().item() if delta.numel() else 0.0,
    }


def environment(device: torch.device | str) -> dict[str, Any]:
    device = torch.device(device)
    synchronize(device)
    metadata: dict[str, Any] = {
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "device": str(device),
        "cuda_runtime": torch.version.cuda,
        "timing": "synchronized_host_wall_clock",
        "percentile_method": "linear_interpolation_at_(n-1)*q",
    }
    root = Path(__file__).resolve().parents[2]
    try:
        metadata["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        metadata["git_dirty"] = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=root, text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        metadata["git_commit"] = None
        metadata["git_dirty"] = None
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        metadata.update(
            gpu_name=props.name,
            gpu_memory_bytes=props.total_memory,
            compute_capability=[props.major, props.minor],
            cuda_device_count=torch.cuda.device_count(),
        )
    return metadata


def write_report(report: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=destination.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
