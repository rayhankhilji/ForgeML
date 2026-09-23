from __future__ import annotations

import argparse
import json
import time

import torch
from torch import nn

from forgeml.compiler import compile
from forgeml.measurement import (
    check_output,
    environment,
    measure_variants,
    synchronize,
    write_report,
)

WORKLOADS = ((8, 64, 128, 32), (32, 128, 256, 64), (64, 256, 512, 128))


class ResidualMLP(nn.Module):
    def __init__(self, width: int, hidden: int):
        super().__init__()
        self.up = nn.Linear(width, hidden)
        self.activation = nn.GELU(approximate="tanh")
        self.down = nn.Linear(hidden, width)

    def forward(self, x):
        return self.down(self.activation(self.up(x))) + x


def make_model(width: int, hidden: int, output: int, residual: bool = False) -> nn.Module:
    if residual:
        return ResidualMLP(width, hidden).eval()
    return nn.Sequential(
        nn.Linear(width, hidden), nn.GELU(approximate="tanh"), nn.Linear(hidden, output)
    ).eval()


@torch.inference_mode()
def benchmark_compiler(
    *,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    backend: str = "torch",
    warmup: int = 5,
    repeats: int = 25,
    seed: int = 2026,
    autotune: bool = False,
) -> dict:
    target = torch.device(device)
    if target.type not in ("cpu", "cuda"):
        raise ValueError("benchmarks support CPU or CUDA")
    if target.type == "cpu" and dtype != torch.float32:
        raise ValueError("CPU benchmark workloads require float32")
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no CUDA device is available")
    if warmup < 0 or repeats < 2:
        raise ValueError("warmup must be >= 0 and repeats >= 2")
    torch.manual_seed(seed)
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    previous_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        rows = []
        for batch, width, hidden, output in WORKLOADS:
            for residual in (False, True):
                model = make_model(width, hidden, output, residual).to(device=target, dtype=dtype)
                x = torch.randn(batch, width, device=target, dtype=dtype)
                reference = model(x)
                synchronize(target)
                start = time.perf_counter_ns()
                unoptimized = compile(model, (x,), backend=backend, optimize=False)
                synchronize(target)
                unoptimized_compile_ms = (time.perf_counter_ns() - start) / 1_000_000
                start = time.perf_counter_ns()
                optimized = compile(model, (x,), backend=backend)
                synchronize(target)
                optimized_compile_ms = (time.perf_counter_ns() - start) / 1_000_000
                tuning = optimized.autotune(x) if autotune else None
                correctness = {
                    "unoptimized": check_output(unoptimized(x), reference),
                    "optimized": check_output(optimized(x), reference),
                }
                timings = measure_variants(
                    {
                        "eager": lambda model=model, x=x: model(x),
                        "unoptimized": lambda model=unoptimized, x=x: model(x),
                        "optimized": lambda model=optimized, x=x: model(x),
                    },
                    device=target,
                    warmup=warmup,
                    repeats=repeats,
                )
                eager_ms = timings["eager"]["median_ms"]
                for timing in timings.values():
                    timing["rows_per_second"] = batch * 1000 / timing["median_ms"]
                    timing["speedup_vs_eager"] = eager_ms / timing["median_ms"]
                rows.append(
                    {
                        "name": f"{'residual' if residual else 'mlp'}_{batch}_{width}_{hidden}",
                        "batch": batch,
                        "width": width,
                        "hidden": hidden,
                        "output": width if residual else output,
                        "dtype": str(dtype),
                        "correctness": correctness,
                        "timings": timings,
                        "compile_ms": {
                            "unoptimized": unoptimized_compile_ms,
                            "optimized": optimized_compile_ms,
                        },
                        "unoptimized": unoptimized.explain(),
                        "optimized": optimized.explain(),
                        "autotuning": tuning,
                    }
                )
        return {
            "schema_version": 1,
            "suite": "forgeml.compiler",
            "measured": True,
            "environment": environment(target),
            "settings": {
                "seed": seed,
                "warmup": warmup,
                "repeats": repeats,
                "backend": backend,
                "dtype": str(dtype),
                "autotune": autotune,
                "tf32": False,
                "gradients": False,
                "variant_order": "rotating_per_repetition",
            },
            "scope": {
                "latency": "whole_call_including_python_dispatch_and_output_ownership",
                "excluded": ["model_initialization", "compilation", "autotuning", "warmup"],
                "memory": "planned_intermediate_storage_not_allocator_peak",
                "warning": "CPU graph fusion does not imply CPU kernel fusion or speedup",
            },
            "workloads": rows,
        }
    finally:
        torch.set_float32_matmul_precision(previous_precision)
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32


def main() -> None:
    parser = argparse.ArgumentParser(prog="forgeml", description="Inspect and benchmark ForgeML")
    commands = parser.add_subparsers(dest="command", required=True)
    bench = commands.add_parser("benchmark")
    bench.add_argument("--device", default="cpu")
    bench.add_argument("--backend", choices=("torch", "triton", "auto"), default="torch")
    bench.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    bench.add_argument("--warmup", type=int, default=5)
    bench.add_argument("--repeats", type=int, default=25)
    bench.add_argument("--threads", type=int, default=1)
    bench.add_argument("--seed", type=int, default=2026)
    bench.add_argument("--autotune", action="store_true")
    bench.add_argument("--output")
    explain = commands.add_parser("explain")
    explain.add_argument("--mermaid", action="store_true")
    commands.add_parser("kernel-source")
    args = parser.parse_args()
    if args.command == "kernel-source":
        from forgeml.kernels import kernel_source

        print(kernel_source())
        return
    if args.command == "explain":
        torch.manual_seed(0)
        model = make_model(64, 128, 32)
        compiled = compile(model, (torch.randn(8, 64),))
        print(
            compiled.graph.to_mermaid()
            if args.mermaid
            else json.dumps(compiled.explain(), indent=2)
        )
        return
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    report = benchmark_compiler(
        device=args.device,
        dtype=getattr(torch, args.dtype),
        backend=args.backend,
        warmup=args.warmup,
        repeats=args.repeats,
        seed=args.seed,
        autotune=args.autotune,
    )
    if args.output:
        write_report(report, args.output)
        print(f"Measured {len(report['workloads'])} workloads; report: {args.output}")
        for row in report["workloads"]:
            result = row["timings"]["optimized"]
            print(
                f"{row['name']}: {result['median_ms']:.4f} ms median, "
                f"{result['speedup_vs_eager']:.3f}x eager"
            )
    else:
        print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
