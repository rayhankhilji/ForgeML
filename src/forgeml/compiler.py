from __future__ import annotations

from dataclasses import asdict, dataclass, field

import torch

from forgeml import kernels
from forgeml.frontend import from_torch
from forgeml.ir import Graph, GraphError
from forgeml.kernels import DEFAULT_CONFIG, KernelConfig
from forgeml.memory import MemoryPlan, plan_memory
from forgeml.ops import evaluate
from forgeml.passes import PassRecord
from forgeml.passes import optimize as optimize_graph


@dataclass
class CompiledModel:
    graph: Graph
    original_graph: Graph
    memory_plan: MemoryPlan
    passes: list[PassRecord]
    backend: str
    kernel_plan: dict[str, str] = field(default_factory=dict)
    kernel_configs: dict[str, KernelConfig] = field(default_factory=dict)

    def autotune(self, *inputs: torch.Tensor, warmup: int = 3, repeats: int = 10) -> dict:
        from forgeml.autotune import tune_graph

        return tune_graph(self, inputs, warmup=warmup, repeats=repeats)

    def kernel_source(self) -> str:
        return kernels.kernel_source()

    def __call__(self, *inputs: torch.Tensor):
        names = list(self.graph.inputs)
        if len(inputs) != len(names):
            raise GraphError(f"expected {len(names)} inputs, got {len(inputs)}")
        for t in inputs:
            if not isinstance(t, torch.Tensor):
                raise GraphError("inputs must be torch.Tensor")
        values: dict[str, torch.Tensor] = {}
        for name, t in zip(names, inputs):
            spec = self.graph.inputs[name]
            if tuple(t.shape) != spec.shape:
                raise GraphError(f"input {name!r} shape {tuple(t.shape)} != expected {spec.shape}")
            if t.dtype != spec.dtype:
                raise GraphError(f"input {name!r} dtype {t.dtype} != expected {spec.dtype}")
            if str(t.device) != spec.device:
                raise GraphError(f"input {name!r} device {t.device} != expected {spec.device}")
            values[name] = t
        for name, t in self.graph.constants.items():
            values[name] = t
        plan = self.memory_plan
        slots: dict[int, torch.Tensor] = {}
        for slot, spec in plan.slot_specs.items():
            slots[slot] = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
        outputs = set(self.graph.outputs)
        consumers: dict[str, int] = {}
        for node in self.graph.nodes:
            for i in node.inputs:
                consumers[i] = consumers.get(i, 0) + 1
        remaining = dict(consumers)
        with torch.inference_mode():
            for node in self.graph.nodes:
                args = tuple(values[i] for i in node.inputs)
                is_output = node.name in outputs
                use_triton = self.kernel_plan.get(node.name) == "triton"
                if is_output:
                    if use_triton:
                        values[node.name] = self._run_triton(node, args, None)
                    else:
                        values[node.name] = evaluate(node.op, args, node.attrs).clone()
                else:
                    alloc = plan.allocations[node.name]
                    view = slots[alloc.slot][: alloc.size_bytes // node.spec.dtype.itemsize]
                    view = view.view(node.spec.shape)
                    if use_triton:
                        self._run_triton(node, args, view)
                    elif node.op == "matmul":
                        torch.matmul(args[0], args[1], out=view)
                    elif node.op == "add":
                        torch.add(args[0], args[1], out=view)
                    elif node.op == "mul":
                        torch.mul(args[0], args[1], out=view)
                    else:
                        view.copy_(evaluate(node.op, args, node.attrs))
                    values[node.name] = view
                for i in node.inputs:
                    if i in remaining:
                        remaining[i] -= 1
                        if remaining[i] == 0 and i not in outputs:
                            values.pop(i, None)
            result = []
            for name in self.graph.outputs:
                if name not in values:
                    raise GraphError(f"output {name!r} was not produced")
                result.append(values[name].clone())
        if self.graph.output_is_tuple:
            return tuple(result)
        return result[0]

    def _run_triton(
        self, node, args: tuple[torch.Tensor, ...], out: torch.Tensor | None
    ) -> torch.Tensor:
        config = self.kernel_configs.get(node.name, DEFAULT_CONFIG)
        if node.op == "matmul":
            return kernels.matmul(args[0], args[1], config=config, out=out)
        if node.op == "fused_linear_gelu":
            return kernels.matmul(
                args[0],
                args[1],
                args[2],
                approximate=node.attrs.get("approximate", "none"),
                config=config,
                out=out,
            )
        raise GraphError(f"no triton kernel for op {node.op!r}")

    def explain(self) -> dict:
        return {
            "backend": self.backend,
            "original_nodes": len(self.original_graph.nodes),
            "optimized_nodes": len(self.graph.nodes),
            "passes": [
                {
                    "name": p.name,
                    "nodes_before": p.nodes_before,
                    "nodes_after": p.nodes_after,
                }
                for p in self.passes
            ],
            "memory_plan": self.memory_plan.to_dict(),
            "graph": self.graph.to_dict(),
            "kernel_plan": dict(self.kernel_plan),
            "kernel_configs": {k: asdict(v) for k, v in self.kernel_configs.items()},
        }


def compile(
    model: torch.nn.Module | Graph,
    example_inputs: tuple[torch.Tensor, ...] = (),
    *,
    backend: str = "torch",
    optimize: bool = True,
) -> CompiledModel:
    if backend not in ("torch", "triton", "auto"):
        raise GraphError(f"unsupported backend {backend!r}; expected 'torch', 'triton', or 'auto'")
    if isinstance(model, Graph):
        model.validate()
        graph = model.clone()
    elif isinstance(model, torch.nn.Module):
        graph = from_torch(model, example_inputs)
    else:
        raise GraphError("compile expects an nn.Module or a Graph")
    original = graph.clone()
    if optimize:
        graph, records = optimize_graph(graph)
    else:
        graph.validate()
        records = []
    plan = plan_memory(graph)
    kernel_plan = kernels.select_kernels(graph, backend)
    return CompiledModel(graph, original, plan, records, backend, kernel_plan=kernel_plan)
