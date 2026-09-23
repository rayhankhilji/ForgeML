from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any

import torch


class GraphError(ValueError):
    pass


_SUPPORTED_DTYPES = {
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
    torch.bool,
}


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[int, ...]
    dtype: torch.dtype
    device: str

    def __post_init__(self) -> None:
        if not isinstance(self.shape, tuple) or not all(
            isinstance(d, int) and not isinstance(d, bool) for d in self.shape
        ):
            raise GraphError(f"shape must be a tuple of ints, got {self.shape!r}")
        if any(d <= 0 for d in self.shape):
            raise GraphError(f"shape dims must be positive integers, got {self.shape!r}")
        if not isinstance(self.dtype, torch.dtype):
            raise GraphError(f"dtype must be a torch.dtype, got {self.dtype!r}")
        if self.dtype not in _SUPPORTED_DTYPES:
            raise GraphError(f"unsupported dtype {self.dtype}")
        if not isinstance(self.device, str) or not self.device:
            raise GraphError(f"device must be a non-empty string, got {self.device!r}")
        try:
            torch.device(self.device)
        except (RuntimeError, TypeError) as e:
            raise GraphError(f"invalid device {self.device!r}") from e

    @property
    def nbytes(self) -> int:
        return math.prod(self.shape) * self.dtype.itemsize

    def to_dict(self) -> dict:
        return {
            "shape": list(self.shape),
            "dtype": str(self.dtype).replace("torch.", ""),
            "device": self.device,
        }


@dataclass(frozen=True)
class Node:
    name: str
    op: str
    inputs: tuple[str, ...]
    attrs: dict[str, Any]
    spec: TensorSpec


def _spec_of(t: torch.Tensor, device: str | None = None) -> TensorSpec:
    return TensorSpec(tuple(t.shape), t.dtype, device or str(t.device))


def _meta_tensor(spec: TensorSpec) -> torch.Tensor:
    return torch.empty(spec.shape, dtype=spec.dtype, device="meta")


def _check_attrs(op: str, attrs: dict[str, Any], arity: int) -> None:
    from forgeml.ops import SUPPORTED_OPS

    if op not in SUPPORTED_OPS:
        raise GraphError(f"unsupported op {op!r}")
    allowed = {
        "matmul": set(),
        "add": set(),
        "mul": set(),
        "relu": set(),
        "gelu": {"approximate"},
        "reshape": {"shape"},
        "transpose": {"dim0", "dim1"},
        "softmax": {"dim"},
        "fused_linear_gelu": {"approximate"},
    }[op]
    extra = set(attrs) - allowed
    if extra:
        raise GraphError(f"op {op!r} got unsupported attrs {sorted(extra)}")
    required = {
        "reshape": {"shape"},
        "transpose": {"dim0", "dim1"},
        "softmax": {"dim"},
    }.get(op, set())
    missing = required - set(attrs)
    if missing:
        raise GraphError(f"op {op!r} requires attrs {sorted(missing)}")


def infer_spec(op: str, input_specs: list[TensorSpec], attrs: dict[str, Any]) -> TensorSpec:
    from forgeml.ops import evaluate

    _check_attrs(op, attrs, len(input_specs))
    expected_arity = {
        "matmul": 2,
        "add": 2,
        "mul": 2,
        "relu": 1,
        "gelu": 1,
        "reshape": 1,
        "transpose": 1,
        "softmax": 1,
        "fused_linear_gelu": 3,
    }[op]
    if len(input_specs) != expected_arity:
        raise GraphError(f"op {op!r} expects {expected_arity} inputs, got {len(input_specs)}")
    if op == "matmul":
        a, b = input_specs
        if len(a.shape) != 2 or len(b.shape) != 2:
            raise GraphError("matmul requires rank-2 inputs")
        if a.dtype != b.dtype:
            raise GraphError(f"matmul dtype mismatch {a.dtype} vs {b.dtype}")
        if a.shape[1] != b.shape[0]:
            raise GraphError(f"matmul shape mismatch {a.shape} x {b.shape}")
    if (
        op in ("add", "mul")
        and input_specs[0].dtype != input_specs[1].dtype
        and input_specs[0].shape != ()
        and input_specs[1].shape != ()
    ):
        raise GraphError(f"{op} dtype mismatch")
    devices = {s.device for s in input_specs}
    if len(devices) > 1:
        scalar_ok = op in ("add", "mul") and any(
            s.shape == () and torch.device(s.device).type == "cpu" for s in input_specs
        )
        if not scalar_ok:
            raise GraphError(f"{op} device mismatch {sorted(devices)}")
    if op == "gelu" and attrs.get("approximate", "none") not in ("none", "tanh"):
        raise GraphError("gelu approximate must be 'none' or 'tanh'")
    if op == "reshape":
        shape = attrs["shape"]
        if not isinstance(shape, (tuple, list)) or not all(
            isinstance(d, int) and not isinstance(d, bool) and d > 0 for d in shape
        ):
            raise GraphError(f"reshape shape must be positive ints, got {shape!r}")
        if math.prod(shape) != math.prod(input_specs[0].shape):
            raise GraphError(f"reshape numel mismatch {input_specs[0].shape} -> {tuple(shape)}")
    if op == "transpose":
        rank = len(input_specs[0].shape)
        for key in ("dim0", "dim1"):
            d = attrs[key]
            if not isinstance(d, int) or isinstance(d, bool) or not -rank <= d < rank:
                raise GraphError(f"transpose {key}={d!r} out of range for rank {rank}")
    if op == "softmax":
        d = attrs["dim"]
        rank = len(input_specs[0].shape)
        if not isinstance(d, int) or isinstance(d, bool) or not -rank <= d < rank:
            raise GraphError(f"softmax dim={d!r} out of range for rank {rank}")
    if op == "fused_linear_gelu":
        a, b, bias = input_specs
        if len(a.shape) != 2 or len(b.shape) != 2 or len(bias.shape) != 1:
            raise GraphError("fused_linear_gelu requires rank-2 matmul inputs and 1-D bias")
        if a.shape[1] != b.shape[0] or bias.shape[0] != b.shape[1]:
            raise GraphError("fused_linear_gelu shape mismatch")
        if not (a.dtype == b.dtype == bias.dtype):
            raise GraphError("fused_linear_gelu dtype mismatch")
        if attrs.get("approximate", "none") not in ("none", "tanh"):
            raise GraphError("fused_linear_gelu approximate must be 'none' or 'tanh'")
    try:
        out = evaluate(op, tuple(_meta_tensor(s) for s in input_specs), dict(attrs))
    except GraphError:
        raise
    except Exception as e:
        raise GraphError(f"op {op!r} failed spec inference: {e}") from e
    if all(len(s.shape) == 0 for s in input_specs) and op in ("add", "mul"):
        device = next(
            (s.device for s in input_specs if torch.device(s.device).type != "cpu"),
            input_specs[0].device,
        )
    else:
        device = next(
            (s.device for s in input_specs if len(s.shape) > 0),
            input_specs[0].device,
        )
    return TensorSpec(tuple(out.shape), out.dtype, device)


def _mermaid_label(text: str) -> str:
    return text.replace("&", "#amp;").replace('"', "#quot;").replace("\n", "#32;").replace("\r", "")


@dataclass
class Graph:
    inputs: dict[str, TensorSpec]
    constants: dict[str, torch.Tensor]
    nodes: list[Node]
    outputs: tuple[str, ...]
    output_is_tuple: bool = False

    def specs(self) -> dict[str, TensorSpec]:
        specs = dict(self.inputs)
        for name, t in self.constants.items():
            specs[name] = _spec_of(t)
        for node in self.nodes:
            specs[node.name] = node.spec
        return specs

    def validate(self) -> None:
        names: set[str] = set()
        for name in list(self.inputs) + list(self.constants):
            if not isinstance(name, str) or not name:
                raise GraphError("value names must be non-empty strings")
            if name in names:
                raise GraphError(f"duplicate name {name!r}")
            names.add(name)
        for name, spec in self.inputs.items():
            if not isinstance(spec, TensorSpec):
                raise GraphError(f"input {name!r} spec must be a TensorSpec")
        env = dict(self.inputs)
        for name, t in self.constants.items():
            if not isinstance(t, torch.Tensor):
                raise GraphError(f"constant {name!r} is not a torch.Tensor")
            env[name] = _spec_of(t)
        for node in self.nodes:
            if not isinstance(node.name, str) or not node.name:
                raise GraphError("node names must be non-empty strings")
            if not isinstance(node.spec, TensorSpec):
                raise GraphError(f"node {node.name!r} spec must be a TensorSpec")
            if node.name in names:
                raise GraphError(f"duplicate name {node.name!r}")
            names.add(node.name)
            for inp in node.inputs:
                if inp not in env:
                    raise GraphError(f"node {node.name!r} input {inp!r} is not defined before use")
            try:
                spec = infer_spec(node.op, [env[i] for i in node.inputs], dict(node.attrs))
            except GraphError as e:
                raise GraphError(f"node {node.name!r}: {e}") from e
            if spec != node.spec:
                raise GraphError(
                    f"node {node.name!r} spec mismatch: declared {node.spec}, inferred {spec}"
                )
            env[node.name] = node.spec
        if not self.outputs:
            raise GraphError("graph outputs must be non-empty")
        if len(self.outputs) > 1 and not self.output_is_tuple:
            raise GraphError("multiple graph outputs require output_is_tuple=True")
        for name in self.outputs:
            if name not in env:
                raise GraphError(f"graph output {name!r} is not defined")

    def clone(self) -> Graph:
        return Graph(
            inputs=dict(self.inputs),
            constants={k: v.detach().clone() for k, v in self.constants.items()},
            nodes=[
                Node(n.name, n.op, tuple(n.inputs), copy.deepcopy(n.attrs), n.spec)
                for n in self.nodes
            ],
            outputs=tuple(self.outputs),
            output_is_tuple=self.output_is_tuple,
        )

    def to_dict(self) -> dict:
        return {
            "inputs": {k: v.to_dict() for k, v in self.inputs.items()},
            "constants": {k: _spec_of(v).to_dict() for k, v in self.constants.items()},
            "nodes": [
                {
                    "name": n.name,
                    "op": n.op,
                    "inputs": list(n.inputs),
                    "attrs": {
                        k: list(v) if isinstance(v, tuple) else v for k, v in n.attrs.items()
                    },
                    "spec": n.spec.to_dict(),
                }
                for n in self.nodes
            ],
            "outputs": list(self.outputs),
            "output_is_tuple": self.output_is_tuple,
        }

    def to_mermaid(self) -> str:
        names = list(self.inputs) + list(self.constants) + [n.name for n in self.nodes]
        ids = {name: f"v{i}" for i, name in enumerate(dict.fromkeys(names))}
        lines = ["graph TD"]
        for name, spec in self.inputs.items():
            lines.append(
                f'    {ids[name]}["{_mermaid_label(name)} input {list(spec.shape)} {spec.dtype}"]'
            )
        for name, t in self.constants.items():
            lines.append(
                f'    {ids[name]}["{_mermaid_label(name)} const {list(t.shape)} {t.dtype}"]'
            )
        for node in self.nodes:
            lines.append(
                f'    {ids[node.name]}["{_mermaid_label(node.name)}'
                f' {node.op} {list(node.spec.shape)} {node.spec.dtype}"]'
            )
            for inp in node.inputs:
                lines.append(f"    {ids[inp]} --> {ids[node.name]}")
        for name in self.outputs:
            lines.append(f"    {ids[name]}:::output")
        lines.append("    classDef output fill:#dfd")
        return "\n".join(lines)


class GraphBuilder:
    def __init__(self, inputs: dict[str, TensorSpec]):
        self._inputs = dict(inputs)
        for name, spec in self._inputs.items():
            if not isinstance(name, str) or not name:
                raise GraphError("input names must be non-empty strings")
            if not isinstance(spec, TensorSpec):
                raise GraphError(f"input {name!r} spec must be a TensorSpec")
        self._constants: dict[str, torch.Tensor] = {}
        self._nodes: list[Node] = []
        self._names = set(self._inputs)

    def _claim(self, name: str) -> None:
        if not isinstance(name, str) or not name:
            raise GraphError("names must be non-empty strings")
        if name in self._names:
            raise GraphError(f"duplicate name {name!r}")
        self._names.add(name)

    def _env(self) -> dict[str, TensorSpec]:
        env = dict(self._inputs)
        for k, v in self._constants.items():
            env[k] = _spec_of(v)
        for n in self._nodes:
            env[n.name] = n.spec
        return env

    def constant(self, name: str, value: torch.Tensor) -> str:
        self._claim(name)
        if not isinstance(value, torch.Tensor):
            raise GraphError(f"constant {name!r} must be a torch.Tensor")
        _spec_of(value)
        self._constants[name] = value.detach().clone()
        return name

    def add(self, name: str, op: str, inputs: tuple[str, ...], **attrs) -> str:
        self._claim(name)
        env = self._env()
        for inp in inputs:
            if inp not in env:
                raise GraphError(f"node {name!r} input {inp!r} is not defined")
        try:
            spec = infer_spec(op, [env[i] for i in inputs], attrs)
        except GraphError as e:
            self._names.discard(name)
            raise GraphError(f"node {name!r}: {e}") from e
        self._nodes.append(Node(name, op, tuple(inputs), dict(attrs), spec))
        return name

    def finish(self, outputs: tuple[str, ...], output_is_tuple: bool = False) -> Graph:
        graph = Graph(
            inputs=dict(self._inputs),
            constants=dict(self._constants),
            nodes=list(self._nodes),
            outputs=tuple(outputs),
            output_is_tuple=output_is_tuple or len(outputs) > 1,
        )
        graph.validate()
        return graph
