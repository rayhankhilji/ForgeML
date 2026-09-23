from __future__ import annotations

from dataclasses import dataclass

import torch

from forgeml.frontend import from_torch
from forgeml.ir import Graph, GraphError
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
                if is_output:
                    values[node.name] = evaluate(node.op, args, node.attrs).clone()
                else:
                    alloc = plan.allocations[node.name]
                    view = slots[alloc.slot][: alloc.size_bytes // node.spec.dtype.itemsize]
                    view = view.view(node.spec.shape)
                    if node.op in ("matmul", "add", "mul"):
                        if node.op == "matmul":
                            torch.matmul(args[0], args[1], out=view)
                        elif node.op == "add":
                            torch.add(args[0], args[1], out=view)
                        else:
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
        }


def compile(
    model: torch.nn.Module | Graph,
    example_inputs: tuple[torch.Tensor, ...] = (),
    *,
    backend: str = "torch",
    optimize: bool = True,
) -> CompiledModel:
    if backend != "torch":
        raise GraphError(f"unsupported backend {backend!r}; only 'torch' is available")
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
    return CompiledModel(graph, original, plan, records, backend)
