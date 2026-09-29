from __future__ import annotations

from dataclasses import asdict, dataclass, field

import torch

from forgeml import kernels
from forgeml.analysis import graph_analysis
from forgeml.frontend import from_torch
from forgeml.ir import Graph, GraphError
from forgeml.kernels import DEFAULT_CONFIG, KernelConfig
from forgeml.memory import OUT_CAPABLE_OPS, VIEW_LIKE_OPS, MemoryPlan, plan_memory
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

    def __post_init__(self) -> None:
        remaining: dict[str, int] = {}
        for node in self.graph.nodes:
            for i in node.inputs:
                remaining[i] = remaining.get(i, 0) + 1
        self._remaining_template = remaining
        self._outputs = frozenset(self.graph.outputs)
        self._produced = {node.name for node in self.graph.nodes}
        # The slot arena is allocated once and reused across calls. __call__ is
        # therefore not reentrant; concurrent calls must synchronize externally.
        self._slots: dict[int, torch.Tensor] = {
            slot: torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
            for slot, spec in self.memory_plan.slot_specs.items()
        }
        self._views: dict[str, torch.Tensor] = {
            node.name: self._slots[a.slot][: a.size_bytes // node.spec.dtype.itemsize].view(
                node.spec.shape
            )
            for node in self.graph.nodes
            if (a := self.memory_plan.allocations.get(node.name)) is not None
        }
        # Per-node closures pre-resolve dispatch: view materialization, output
        # ownership, slot-backed out= writes, and Triton selection are all
        # decided once at compile time rather than re-branched every call.
        self._steps = [(node, self._make_step(node)) for node in self.graph.nodes]

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)
        # Assigning a new kernel plan (e.g. forcing a backend for diagnostics
        # or tuning) must rebuild the precompiled dispatch table.
        if name == "kernel_plan" and "_views" in self.__dict__:
            object.__setattr__(
                self,
                "_steps",
                [(node, self._make_step(node)) for node in self.graph.nodes],
            )

    def _make_step(self, node):
        use_triton = self.kernel_plan.get(node.name) == "triton"
        is_output = node.name in self._outputs
        if node.op in VIEW_LIKE_OPS:
            if use_triton:
                raise GraphError(f"no triton kernel for op {node.op!r}")
            if is_output:
                return lambda args: self._evaluate(node, args).clone()
            return lambda args: self._evaluate(node, args)
        if is_output:
            if use_triton:
                return lambda args: self._run_triton(node, args, None)
            return lambda args: self._evaluate(node, args)
        if use_triton:
            if node.name in self._views:
                view = self._views[node.name]

                def run_triton_slot(args, node=node, view=view):
                    return self._run_triton(node, args, view)

                return run_triton_slot
            return lambda args: self._run_triton(node, args, None)
        if node.op not in OUT_CAPABLE_OPS:
            # Ops without an out= variant produce a fresh tensor whose
            # lifetime is managed by refcounting, not the slot arena.
            return lambda args: self._evaluate(node, args)
        view = self._views[node.name]
        if node.op == "matmul":
            return lambda args: torch.matmul(args[0], args[1], out=view)
        if node.op == "linear":
            if len(node.inputs) == 3:
                return lambda args: torch.addmm(args[2], args[0], args[1], out=view)
            return lambda args: torch.mm(args[0], args[1], out=view)
        if node.op == "add":
            return lambda args: torch.add(args[0], args[1], out=view)
        if node.op == "mul":
            return lambda args: torch.mul(args[0], args[1], out=view)
        if node.op == "relu":
            return lambda args: torch.clamp_min(args[0], 0, out=view)
        raise GraphError(f"no slot writer for op {node.op!r}")

    def autotune(self, *inputs: torch.Tensor, warmup: int = 3, repeats: int = 10) -> dict:
        from forgeml.autotune import tune_graph

        return tune_graph(self, inputs, warmup=warmup, repeats=repeats)

    def kernel_source(self) -> str:
        return kernels.kernel_source()

    def __call__(self, *inputs: torch.Tensor):
        names = list(self.graph.inputs)
        if len(inputs) != len(names):
            raise GraphError(f"expected {len(names)} inputs, got {len(inputs)}")
        values: dict[str, torch.Tensor] = {}
        for name, t in zip(names, inputs):
            if not isinstance(t, torch.Tensor):
                raise GraphError("inputs must be torch.Tensor")
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
        outputs = self._outputs
        remaining = dict(self._remaining_template)
        with torch.inference_mode():
            for node, run in self._steps:
                values[node.name] = run(tuple(values[i] for i in node.inputs))
                for i in node.inputs:
                    if i in remaining:
                        remaining[i] -= 1
                        if remaining[i] == 0 and i not in outputs:
                            values.pop(i, None)
            produced = self._produced
            result = []
            for name in self.graph.outputs:
                if name not in values:
                    raise GraphError(f"output {name!r} was not produced")
                # Node-produced outputs already own their storage; outputs that
                # name an input or constant must be cloned to avoid aliasing.
                result.append(values[name] if name in produced else values[name].clone())
        if self.graph.output_is_tuple:
            return tuple(result)
        return result[0]

    @staticmethod
    def _evaluate(node, args: tuple[torch.Tensor, ...]) -> torch.Tensor:
        try:
            return evaluate(node.op, args, node.attrs)
        except GraphError:
            raise
        except Exception as e:
            raise GraphError(f"node {node.name!r} ({node.op}) execution failed: {e}") from e

    def _run_triton(
        self, node, args: tuple[torch.Tensor, ...], out: torch.Tensor | None
    ) -> torch.Tensor:
        config = self.kernel_configs.get(node.name, DEFAULT_CONFIG)
        if node.op == "matmul":
            return kernels.matmul(args[0], args[1], config=config, out=out)
        if node.op == "linear":
            return kernels.matmul(
                args[0], args[1], args[2] if len(args) == 3 else None, config=config, out=out
            )
        if node.op == "fused_linear_gelu":
            return kernels.matmul(
                args[0],
                args[1],
                args[2] if len(args) == 3 else None,
                activation="gelu",
                approximate=node.attrs.get("approximate", "none"),
                config=config,
                out=out,
            )
        raise GraphError(f"no triton kernel for op {node.op!r}")

    def explain(self) -> dict:
        analysis = graph_analysis(self.graph)
        original_analysis = graph_analysis(self.original_graph)
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
            "analysis": analysis,
            "original_analysis": original_analysis,
            "optimization_delta": {
                "nodes": len(self.original_graph.nodes) - len(self.graph.nodes),
                "flops": original_analysis["flops"] - analysis["flops"],
                "logical_bytes": original_analysis["logical_bytes"] - analysis["logical_bytes"],
                "critical_path_ops": original_analysis["critical_path_ops"]
                - analysis["critical_path_ops"],
                "planned_bytes": self.memory_plan.naive_bytes - self.memory_plan.planned_bytes,
            },
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
