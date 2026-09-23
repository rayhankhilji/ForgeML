from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING

import torch

from forgeml import kernels
from forgeml.measurement import check_output, measure_variants
from forgeml.ops import evaluate

if TYPE_CHECKING:
    from forgeml.compiler import CompiledModel


@torch.inference_mode()
def tune_graph(
    compiled: CompiledModel,
    inputs: tuple[torch.Tensor, ...],
    *,
    warmup: int = 3,
    repeats: int = 10,
) -> dict:
    if warmup < 0 or repeats < 2:
        raise ValueError("autotuning requires warmup >= 0 and repeats >= 2")
    compiled(*inputs)
    values = dict(compiled.graph.constants)
    values.update(zip(compiled.graph.inputs, inputs))
    records = {}
    chosen = {}
    for node in compiled.graph.nodes:
        args = tuple(values[name] for name in node.inputs)
        expected = evaluate(node.op, args, node.attrs)
        if compiled.kernel_plan.get(node.name) == "triton":
            bias = args[2] if node.op == "fused_linear_gelu" else None
            approximate = node.attrs.get("approximate", "none")
            variants = {}
            configs = {}
            rejected = {}
            errors = {}
            for index, config in enumerate(kernels.CANDIDATES):
                name = f"candidate_{index}"

                def launch(config=config, args=args, bias=bias, approximate=approximate):
                    return kernels.matmul(
                        args[0], args[1], bias, approximate=approximate, config=config
                    )

                actual = launch()
                try:
                    errors[name] = check_output(actual, expected)
                except AssertionError as exc:
                    rejected[name] = str(exc)
                    continue
                variants[name] = launch
                configs[name] = config
            if not variants:
                raise RuntimeError(f"no numerically valid kernel candidate for {node.name!r}")
            timings = measure_variants(
                variants, device=args[0].device, warmup=warmup, repeats=repeats
            )
            winner = min(timings, key=lambda name: timings[name]["median_ms"])
            chosen[node.name] = configs[winner]
            records[node.name] = {
                "winner": winner,
                "config": asdict(configs[winner]),
                "candidates": {
                    name: {
                        "config": asdict(configs[name]),
                        "timing": timing,
                        "correctness": errors[name],
                    }
                    for name, timing in timings.items()
                },
                "rejected": rejected,
                "input_shapes": [list(arg.shape) for arg in args],
                "input_strides": [list(arg.stride()) for arg in args],
                "dtype": str(args[0].dtype),
                "device": str(args[0].device),
                "approximate": approximate,
            }
        values[node.name] = expected
    compiled.kernel_configs.update(chosen)
    return {
        "scope": "this_compiled_model_instance_only",
        "timing": "synchronized_host_wall_clock_including_output_allocation",
        "warmup": warmup,
        "repeats": repeats,
        "nodes": records,
    }
