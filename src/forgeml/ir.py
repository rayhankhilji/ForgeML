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
            parsed = torch.device(self.device)
        except (RuntimeError, TypeError) as e:
            raise GraphError(f"invalid device {self.device!r}") from e
        # Canonicalize index-less CUDA devices so that "cuda" and "cuda:0"
        # refer to the same value everywhere specs are compared by string.
        if parsed.type == "cuda" and parsed.index is None:
            object.__setattr__(self, "device", "cuda:0")

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


def _check_attrs(op: str, attrs: dict[str, Any]) -> None:
    from forgeml.ops import SUPPORTED_OPS

    if op not in SUPPORTED_OPS:
        raise GraphError(f"unsupported op {op!r}")
    if not isinstance(attrs, dict):
        raise GraphError(f"op {op!r} attrs must be a dict, got {type(attrs).__name__}")
    allowed = {
        "matmul": set(),
        "linear": set(),
        "add": set(),
        "mul": set(),
        "relu": set(),
        "gelu": {"approximate"},
        "reshape": {"shape"},
        "transpose": {"dim0", "dim1"},
        "softmax": {"dim"},
        "narrow": {"dim", "start", "length"},
        "fused_linear_gelu": {"approximate"},
        "layer_norm": {"normalized_shape", "eps"},
        "sdpa": {"is_causal", "scale"},
        "conv2d": {"stride", "padding", "dilation", "groups"},
        "embedding": set(),
    }[op]
    extra = set(attrs) - allowed
    if extra:
        raise GraphError(f"op {op!r} got unsupported attrs {sorted(extra)}")
    required = {
        "reshape": {"shape"},
        "transpose": {"dim0", "dim1"},
        "softmax": {"dim"},
        "narrow": {"dim", "start", "length"},
        "layer_norm": {"normalized_shape", "eps"},
        "sdpa": {"is_causal"},
        "conv2d": {"stride", "padding", "dilation", "groups"},
    }.get(op, set())
    missing = required - set(attrs)
    if missing:
        raise GraphError(f"op {op!r} requires attrs {sorted(missing)}")


def _pair(value: Any, name: str, *, positive: bool) -> tuple[int, int]:
    if isinstance(value, int) and not isinstance(value, bool):
        pair = (value, value)
    elif (
        isinstance(value, (tuple, list))
        and len(value) == 2
        and all(isinstance(v, int) and not isinstance(v, bool) for v in value)
    ):
        pair = tuple(value)
    else:
        raise GraphError(f"conv2d {name} must be an int or pair of ints, got {value!r}")
    minimum = 1 if positive else 0
    if any(v < minimum for v in pair):
        qualifier = "positive" if positive else "non-negative"
        raise GraphError(f"conv2d {name} values must be {qualifier}, got {pair!r}")
    return pair


_FLOAT_ONLY_OPS = {"gelu", "softmax", "fused_linear_gelu", "layer_norm", "sdpa", "conv2d"}


def infer_spec(op: str, input_specs: list[TensorSpec], attrs: dict[str, Any]) -> TensorSpec:
    from forgeml.ops import evaluate

    _check_attrs(op, attrs)
    expected_arity = {
        "matmul": (2,),
        "linear": (2, 3),
        "add": (2,),
        "mul": (2,),
        "relu": (1,),
        "gelu": (1,),
        "reshape": (1,),
        "transpose": (1,),
        "softmax": (1,),
        "narrow": (1,),
        "fused_linear_gelu": (2, 3),
        "layer_norm": (1, 3),
        "sdpa": (3,),
        "conv2d": (2, 3),
        "embedding": (2,),
    }[op]
    if len(input_specs) not in expected_arity:
        choices = "/".join(str(v) for v in expected_arity)
        raise GraphError(f"op {op!r} expects {choices} inputs, got {len(input_specs)}")
    if op in _FLOAT_ONLY_OPS and any(not s.dtype.is_floating_point for s in input_specs):
        raise GraphError(f"op {op!r} requires floating-point inputs")
    if op == "matmul":
        a, b = input_specs
        if len(a.shape) != 2 or len(b.shape) != 2:
            raise GraphError("matmul requires rank-2 inputs")
        if a.dtype != b.dtype:
            raise GraphError(f"matmul dtype mismatch {a.dtype} vs {b.dtype}")
        if a.dtype == torch.bool:
            raise GraphError("matmul does not support bool inputs")
        if a.shape[1] != b.shape[0]:
            raise GraphError(f"matmul shape mismatch {a.shape} x {b.shape}")
    if op == "linear":
        x, weight = input_specs[:2]
        if len(x.shape) != 2 or len(weight.shape) != 2:
            raise GraphError("linear requires rank-2 input and weight tensors")
        if x.dtype != weight.dtype:
            raise GraphError(f"linear dtype mismatch {x.dtype} vs {weight.dtype}")
        if x.dtype == torch.bool:
            raise GraphError("linear does not support bool inputs")
        if x.shape[1] != weight.shape[0]:
            raise GraphError(f"linear shape mismatch {x.shape} x {weight.shape}")
        if len(input_specs) == 3:
            bias = input_specs[2]
            if len(bias.shape) != 1 or bias.shape[0] != weight.shape[1]:
                raise GraphError("linear bias must be a 1-D tensor of output width")
            if bias.dtype != x.dtype:
                raise GraphError("linear bias dtype mismatch")
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
    if op == "narrow":
        x = input_specs[0]
        rank = len(x.shape)
        for key in ("dim", "start", "length"):
            v = attrs[key]
            if not isinstance(v, int) or isinstance(v, bool):
                raise GraphError(f"narrow {key} must be an int, got {v!r}")
        dim = attrs["dim"] % rank if rank else attrs["dim"]
        if rank == 0 or not -rank <= attrs["dim"] < rank:
            raise GraphError(f"narrow dim={attrs['dim']!r} out of range for rank {rank}")
        if (
            attrs["length"] <= 0
            or attrs["start"] < 0
            or attrs["start"] + attrs["length"] > x.shape[dim]
        ):
            raise GraphError(
                f"narrow range start={attrs['start']} length={attrs['length']} "
                f"exceeds dim {dim} of size {x.shape[dim]}"
            )
    if op == "fused_linear_gelu":
        a, b = input_specs[:2]
        if len(a.shape) != 2 or len(b.shape) != 2:
            raise GraphError("fused_linear_gelu requires rank-2 matmul inputs")
        if a.shape[1] != b.shape[0]:
            raise GraphError("fused_linear_gelu shape mismatch")
        if len(input_specs) == 3:
            bias = input_specs[2]
            if len(bias.shape) != 1 or bias.shape[0] != b.shape[1]:
                raise GraphError("fused_linear_gelu bias must be a 1-D output-width tensor")
            if bias.dtype != a.dtype:
                raise GraphError("fused_linear_gelu bias dtype mismatch")
        if a.dtype != b.dtype:
            raise GraphError("fused_linear_gelu dtype mismatch")
        if attrs.get("approximate", "none") not in ("none", "tanh"):
            raise GraphError("fused_linear_gelu approximate must be 'none' or 'tanh'")
    if op == "layer_norm":
        x = input_specs[0]
        normalized = attrs["normalized_shape"]
        if isinstance(normalized, int) and not isinstance(normalized, bool):
            normalized = (normalized,)
        if (
            not isinstance(normalized, (tuple, list))
            or not normalized
            or not all(isinstance(d, int) and not isinstance(d, bool) and d > 0 for d in normalized)
        ):
            raise GraphError(
                f"layer_norm normalized_shape must be positive ints, got {normalized!r}"
            )
        normalized = tuple(normalized)
        if len(x.shape) < len(normalized) or x.shape[-len(normalized) :] != normalized:
            raise GraphError(f"layer_norm shape {x.shape} does not end with {normalized!r}")
        if len(input_specs) == 3:
            weight, bias = input_specs[1:]
            if weight.shape != normalized or bias.shape != normalized:
                raise GraphError("layer_norm weight/bias must match normalized_shape")
            if not (x.dtype == weight.dtype == bias.dtype):
                raise GraphError("layer_norm dtype mismatch")
        eps = attrs["eps"]
        if (
            not isinstance(eps, (int, float))
            or isinstance(eps, bool)
            or not math.isfinite(float(eps))
            or float(eps) <= 0
        ):
            raise GraphError("layer_norm eps must be a finite positive number")
    if op == "sdpa":
        q, k, v = input_specs
        if not (len(q.shape) == len(k.shape) == len(v.shape) == 4):
            raise GraphError("sdpa requires rank-4 [batch, heads, sequence, dim] tensors")
        if q.shape[0:2] != k.shape[0:2] or k.shape != v.shape or q.shape[3] != k.shape[3]:
            raise GraphError("sdpa batch/head/key-value shapes are incompatible")
        if not (q.dtype == k.dtype == v.dtype):
            raise GraphError("sdpa dtype mismatch")
        if not isinstance(attrs["is_causal"], bool):
            raise GraphError("sdpa is_causal must be a bool")
        scale = attrs.get("scale")
        if scale is not None and (
            not isinstance(scale, (int, float))
            or isinstance(scale, bool)
            or not math.isfinite(float(scale))
            or float(scale) <= 0
        ):
            raise GraphError("sdpa scale must be a finite positive number or None")
    if op == "conv2d":
        x, weight = input_specs[:2]
        if len(x.shape) != 4 or len(weight.shape) != 4:
            raise GraphError("conv2d requires NCHW input and OIHW weight tensors")
        groups = attrs["groups"]
        if not isinstance(groups, int) or isinstance(groups, bool) or groups <= 0:
            raise GraphError("conv2d groups must be a positive integer")
        if (
            x.shape[1] % groups
            or weight.shape[0] % groups
            or x.shape[1] != weight.shape[1] * groups
        ):
            raise GraphError("conv2d channels/groups are incompatible")
        if len(input_specs) == 3:
            bias = input_specs[2]
            if len(bias.shape) != 1 or bias.shape[0] != weight.shape[0]:
                raise GraphError("conv2d bias must be a 1-D tensor of output channels")
            if bias.dtype != x.dtype:
                raise GraphError("conv2d bias dtype mismatch")
        if x.dtype != weight.dtype:
            raise GraphError("conv2d dtype mismatch")
        _pair(attrs["stride"], "stride", positive=True)
        _pair(attrs["padding"], "padding", positive=False)
        _pair(attrs["dilation"], "dilation", positive=True)
    if op == "embedding":
        indices, weight = input_specs
        if indices.dtype != torch.int64:
            raise GraphError("embedding indices must be int64")
        if len(weight.shape) != 2:
            raise GraphError("embedding weight must be rank-2 [vocab, dim]")
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
            if not isinstance(node.attrs, dict):
                raise GraphError(
                    f"node {node.name!r} attrs must be a dict, got {type(node.attrs).__name__}"
                )
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
        if not isinstance(self.output_is_tuple, bool):
            raise GraphError("output_is_tuple must be a bool")
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
