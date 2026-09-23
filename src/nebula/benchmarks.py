from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch

from forgeml.measurement import (
    check_output,
    environment,
    measure_variants,
    summarize,
    synchronize,
    write_report,
)
from nebula.model import DecoderConfig, TinyDecoder
from nebula.parallel import ParallelContext
from nebula.runtime import Engine


@torch.inference_mode()
def reference_generate(model: TinyDecoder, prompts: torch.Tensor, count: int) -> list[list[int]]:
    tokens = prompts.clone()
    for _ in range(count):
        predicted = model(tokens)[:, -1].argmax(dim=-1, keepdim=True)
        tokens = torch.cat((tokens, predicted), dim=1)
    return tokens[:, prompts.shape[1] :].tolist()


@torch.inference_mode()
def benchmark_runtime(
    engine: Engine,
    context: ParallelContext,
    config: DecoderConfig,
    *,
    batch: int = 4,
    prompt_length: int = 8,
    new_tokens: int = 8,
    warmup: int = 2,
    repeats: int = 10,
    dtype: torch.dtype = torch.float32,
) -> dict:
    if batch < 1 or prompt_length < 1 or new_tokens < 1 or repeats < 2 or warmup < 0:
        raise ValueError("positive batch/prompt/tokens, repeats >= 2 and warmup >= 0 required")
    if prompt_length + new_tokens > config.max_seq_len:
        raise ValueError("workload exceeds context capacity")
    generator = torch.Generator(device="cpu").manual_seed(2026)
    tokens = torch.randint(config.vocab_size, (batch, prompt_length), generator=generator).to(
        context.device
    )
    prompts = tokens.tolist()
    ids = tuple(range(100, 100 + batch))
    reference = TinyDecoder(config).eval().to(device=context.device, dtype=dtype)
    expected_logits = reference(tokens)
    actual = engine.forward(tokens, ids, start_pos=0)
    prefill_correctness = check_output(actual, expected_logits)
    next_tokens = expected_logits[:, -1].argmax(dim=-1, keepdim=True)
    expected_decode = reference(torch.cat((tokens, next_tokens), dim=1))[:, -1:]
    actual_decode = engine.forward(next_tokens, ids, start_pos=prompt_length)
    decode_correctness = check_output(actual_decode, expected_decode)
    engine.release(ids)
    expected_tokens = reference_generate(reference, tokens, new_tokens)
    if engine.generate(prompts, new_tokens) != expected_tokens:
        raise AssertionError("distributed greedy tokens differ from the monolithic reference")
    del reference, expected_logits, actual, expected_decode, actual_decode
    totals = measure_variants(
        {"generation": lambda: engine.generate(prompts, new_tokens)},
        device=context.device,
        warmup=warmup,
        repeats=repeats,
    )["generation"]
    prefill_samples = []
    decode_samples = []
    for iteration in range(warmup + repeats):
        synchronize(context.device)
        start = time.perf_counter_ns()
        logits = engine.forward(tokens, ids, start_pos=0)
        synchronize(context.device)
        elapsed = (time.perf_counter_ns() - start) / 1_000_000
        if iteration >= warmup:
            prefill_samples.append(elapsed)
        predicted = logits[:, -1].argmax(dim=-1, keepdim=True)
        for step in range(new_tokens - 1):
            synchronize(context.device)
            start = time.perf_counter_ns()
            logits = engine.forward(predicted, ids, start_pos=prompt_length + step)
            synchronize(context.device)
            elapsed = (time.perf_counter_ns() - start) / 1_000_000
            if iteration >= warmup:
                decode_samples.append(elapsed)
            predicted = logits[:, -1].argmax(dim=-1, keepdim=True)
        engine.release(ids)
    totals["output_tokens_per_second"] = batch * new_tokens * 1000 / totals["median_ms"]
    element_bytes = torch.empty((), dtype=dtype).element_size()
    capacity_per_rank = (
        2
        * (config.num_layers // context.pp_size)
        * batch
        * config.max_seq_len
        * (config.num_heads // context.tp_size)
        * (config.hidden_size // config.num_heads)
        * element_bytes
    )
    return {
        "schema_version": 1,
        "suite": "nebula.generation",
        "measured": True,
        "environment": environment(context.device),
        "model": asdict(config),
        "topology": {
            "tp": context.tp_size,
            "pp": context.pp_size,
            "world_size": context.world_size,
        },
        "settings": {
            "batch": batch,
            "prompt_length": prompt_length,
            "new_tokens": new_tokens,
            "warmup": warmup,
            "repeats": repeats,
            "dtype": str(dtype),
            "prompt_seed": 2026,
            "tf32": False,
        },
        "correctness": {
            "prefill": prefill_correctness,
            "decode": decode_correctness,
            "greedy_tokens_exact": True,
        },
        "generation": totals,
        "prefill": summarize(prefill_samples),
        "decode_step": summarize(decode_samples) if decode_samples else None,
        "cache_capacity_per_rank_bytes_for_batch": capacity_per_rank,
        "scope": {
            "generation": "leader_wall_clock_including_commands_cache_allocation_sampling_release",
            "prefill": "leader_forward_call_including_commands_and_cache_allocation",
            "decode_step": "leader_forward_call_excluding_sampling_and_release",
            "decode_context_lengths": list(range(prompt_length + 1, prompt_length + new_tokens)),
            "memory": "analytical_KV_capacity_not_measured_allocator_peak",
            "network": "use_separate_per_rank_profiler_traces_not_inferred_from_wall_time",
            "pipeline": "blocking_sequential_stages_without_microbatch_overlap",
        },
    }


def compare_reports(reports: list[dict]) -> dict:
    if not reports:
        raise ValueError("provide at least a world-size-one baseline")
    baseline_candidates = [r for r in reports if r.get("topology", {}).get("world_size") == 1]
    if len(baseline_candidates) != 1:
        raise ValueError("exactly one measured world-size-one baseline required")
    baseline = baseline_candidates[0]

    def signature(report):
        env = report["environment"]
        return (
            report["model"],
            report["settings"],
            env["device"].split(":")[0],
            env["torch"],
            env["torch_threads"],
            env.get("gpu_name"),
            env.get("compute_capability"),
            env["platform"],
            env.get("git_commit"),
            env.get("git_dirty"),
        )

    expected_signature = signature(baseline)
    points = []
    topologies = set()
    base_median = summarize(baseline["generation"]["samples_ms"])["median_ms"]
    for report in reports:
        if report.get("suite") != "nebula.generation" or report.get("measured") is not True:
            raise ValueError("scaling accepts measured Nebula generation reports only")
        if signature(report) != expected_signature:
            raise ValueError("model, workload, precision, code, runtime and hardware must match")
        if report["environment"].get("git_dirty") is not False or not report["environment"].get(
            "git_commit"
        ):
            raise ValueError("scaling comparison requires reports from a clean recorded commit")
        if not report["correctness"]["greedy_tokens_exact"] or not all(
            report["correctness"][phase]["passed"] for phase in ("prefill", "decode")
        ):
            raise ValueError("all points must pass correctness gates")
        topology = report["topology"]
        world = topology["world_size"]
        if world < 1 or topology["tp"] * topology["pp"] != world:
            raise ValueError("invalid topology")
        key = (topology["tp"], topology["pp"])
        if key in topologies:
            raise ValueError("duplicate topology")
        topologies.add(key)
        timing = summarize(report["generation"]["samples_ms"])
        if not math.isclose(timing["median_ms"], report["generation"]["median_ms"], rel_tol=1e-12):
            raise ValueError("stored median does not agree with raw samples")
        speedup = base_median / timing["median_ms"]
        points.append(
            {
                **topology,
                "median_ms": timing["median_ms"],
                "p95_ms": timing["p95_ms"],
                "speedup": speedup,
                "parallel_efficiency": speedup / world,
                "samples": timing["count"],
            }
        )
    return {
        "schema_version": 1,
        "suite": "nebula.strong_scaling",
        "measured": True,
        "environment": baseline["environment"],
        "settings": baseline["settings"],
        "model": baseline["model"],
        "definition": "fixed_total_batch_and_tokens; S_p=T_1/T_p; E_p=S_p/p",
        "points": sorted(points, key=lambda point: (point["world_size"], point["pp"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m nebula.benchmarks")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("benchmark", "profile"):
        command = sub.add_parser(name)
        command.add_argument("--tp", type=int, default=1)
        command.add_argument("--pp", type=int, default=1)
        command.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
        command.add_argument("--batch", type=int, default=4)
        command.add_argument("--prompt-length", type=int, default=8)
        command.add_argument("--new-tokens", type=int, default=8)
        command.add_argument("--layers", type=int, default=4)
        command.add_argument("--hidden", type=int, default=128)
        command.add_argument("--heads", type=int, default=8)
        command.add_argument("--ff", type=int, default=256)
        command.add_argument("--warmup", type=int, default=2)
        command.add_argument("--repeats", type=int, default=10)
        command.add_argument("--threads", type=int, default=1)
        command.add_argument("--output", required=True)
    compare = sub.add_parser("compare")
    compare.add_argument("reports", nargs="+")
    compare.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "compare":
        report = compare_reports([json.loads(Path(path).read_text()) for path in args.reports])
        write_report(report, args.output)
        print(json.dumps(report, indent=2))
        return
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    config = DecoderConfig(
        hidden_size=args.hidden,
        num_heads=args.heads,
        num_layers=args.layers,
        intermediate_size=args.ff,
    )
    if args.batch < 1 or args.prompt_length < 1 or args.new_tokens < 1:
        parser.error("batch, prompt length and new tokens must be positive")
    if args.prompt_length + args.new_tokens > config.max_seq_len:
        parser.error("prompt plus generation exceeds context capacity")
    context = ParallelContext.from_env(tp_size=args.tp, pp_size=args.pp, device=args.device)
    engine = None
    try:
        engine = Engine(config, context=context, max_requests=max(args.batch, 32))
        if args.command == "profile":
            activities = [torch.profiler.ProfilerActivity.CPU]
            if context.device.type == "cuda":
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            with torch.profiler.profile(
                activities=activities, record_shapes=True, profile_memory=True
            ) as profiler:
                if context.rank == 0:
                    prompts = [list(range(args.prompt_length)) for _ in range(args.batch)]
                    engine.generate(prompts, args.new_tokens)
                    engine.close()
                else:
                    engine.serve()
            directory = Path(args.output)
            directory.mkdir(parents=True, exist_ok=True)
            profiler.export_chrome_trace(str(directory / f"rank-{context.rank}.json"))
        elif context.rank == 0:
            report = benchmark_runtime(
                engine,
                context,
                config,
                batch=args.batch,
                prompt_length=args.prompt_length,
                new_tokens=args.new_tokens,
                warmup=args.warmup,
                repeats=args.repeats,
            )
            engine.close()
            write_report(report, args.output)
            print(
                f"TP={args.tp} PP={args.pp}: {report['generation']['median_ms']:.3f} ms; "
                f"{report['generation']['output_tokens_per_second']:.1f} output tokens/s"
            )
        else:
            engine.serve()
    finally:
        if engine is not None:
            engine.close()
        context.close()


if __name__ == "__main__":
    main()
