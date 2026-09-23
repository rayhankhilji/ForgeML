from __future__ import annotations

import math
import operator
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.fx import symbolic_trace

from forgeml.ir import Graph, GraphBuilder, GraphError, TensorSpec


class UnsupportedOperator(GraphError):
    pass


def _spec(t: torch.Tensor) -> TensorSpec:
    return TensorSpec(tuple(t.shape), t.dtype, str(t.device))


def _check_module_state(model: nn.Module) -> None:
    for m in model.modules():
        if m.training and isinstance(
            m, (nn.modules.dropout._DropoutNd, nn.modules.batchnorm._BatchNorm)
        ):
            raise GraphError(
                f"module {type(m).__name__} is in training mode; call model.eval() before compiling"
            )


def _resolve_attr(root: nn.Module, target: str) -> torch.Tensor:
    obj: Any = root
    for part in target.split("."):
        if not isinstance(obj, (nn.Module, torch.Tensor)) or not hasattr(obj, part):
            raise GraphError(f"cannot resolve attribute {target!r}")
        obj = getattr(obj, part)
    if not isinstance(obj, torch.Tensor):
        raise GraphError(f"attribute {target!r} is not a tensor")
    return obj


def _static_shape(raw: Any, in_shape: tuple[int, ...], where: str) -> tuple[int, ...]:
    if isinstance(raw, (tuple, list)):
        dims = list(raw)
    else:
        raise UnsupportedOperator(f"{where} requires a static shape tuple/list")
    if not all(isinstance(d, int) and not isinstance(d, bool) for d in dims):
        raise UnsupportedOperator(f"{where} shape dims must be ints, got {dims!r}")
    negs = sum(1 for d in dims if d == -1)
    if negs > 1:
        raise UnsupportedOperator(f"{where} allows at most one -1 dim, got {dims!r}")
    if any(d == 0 or d < -1 for d in dims):
        raise UnsupportedOperator(f"{where} dims must be positive or -1, got {dims!r}")
    if negs == 1:
        known = math.prod(d for d in dims if d != -1)
        total = math.prod(in_shape)
        if known == 0 or total % known:
            raise GraphError(f"{where} cannot infer -1 dim for input {in_shape}")
        dims[dims.index(-1)] = total // known
    return tuple(dims)


class _Lowerer:
    def __init__(self, model: nn.Module, example_inputs: tuple[torch.Tensor, ...]):
        self.model = model
        self.example_inputs = example_inputs
        self.builder: GraphBuilder | None = None
        self.env: dict[str, str] = {}
        self.const_tensors: dict[str, torch.Tensor] = {}
        self.specs: dict[str, TensorSpec] = {}
        self._counter = 0

    def _fresh(self, hint: str) -> str:
        self._counter += 1
        return f"{hint}_{self._counter}"

    def _arg(self, a: Any) -> str:
        import torch.fx

        if isinstance(a, torch.fx.Node):
            if a.name not in self.env:
                raise GraphError(f"fx node {a.name!r} has no produced value")
            return self.env[a.name]
        if isinstance(a, (int, float, bool)):
            t = torch.tensor(a)
            name = self._fresh("scalar")
            self.builder.constant(name, t)
            self.const_tensors[name] = t
            self.specs[name] = _spec(t)
            return name
        raise UnsupportedOperator(f"unsupported argument {a!r} of type {type(a).__name__}")

    def _bind(self, node, names: tuple[str, ...], defaults: dict | None = None) -> dict:
        if len(node.args) > len(names):
            raise UnsupportedOperator(f"too many arguments at {node.name!r}")
        bound = dict(zip(names, node.args))
        for key, value in node.kwargs.items():
            if key not in names or key in bound:
                raise UnsupportedOperator(
                    f"unsupported or repeated argument {key!r} at {node.name!r}"
                )
            bound[key] = value
        for key, value in (defaults or {}).items():
            bound.setdefault(key, value)
        if set(names) - set(bound):
            raise UnsupportedOperator(f"missing argument at {node.name!r}")
        return bound

    def _linear(
        self, out_name: str, x: str, weight: torch.Tensor, bias: torch.Tensor | None
    ) -> str:
        w_name = self._fresh(f"{out_name}_weight_t")
        w_t = weight.t().contiguous()
        self.builder.constant(w_name, w_t)
        self.const_tensors[w_name] = w_t
        mm = self.builder.add(self._fresh(f"{out_name}_matmul"), "matmul", (x, w_name))
        if bias is None:
            return mm
        b_name = self._fresh(f"{out_name}_bias")
        b_c = bias.contiguous()
        self.builder.constant(b_name, b_c)
        self.const_tensors[b_name] = b_c
        return self.builder.add(out_name, "add", (mm, b_name))

    def lower(self) -> Graph:
        inputs = {f"input_{i}": _spec(t) for i, t in enumerate(self.example_inputs)}
        try:
            traced = symbolic_trace(self.model)
        except Exception as e:
            raise UnsupportedOperator(f"symbolic trace failed: {e}") from e
        self.builder = GraphBuilder(inputs)
        self.specs = dict(inputs)
        placeholders = [n for n in traced.graph.nodes if n.op == "placeholder"]
        if len(placeholders) != len(self.example_inputs):
            raise GraphError(
                f"model expects {len(placeholders)} inputs, "
                f"got {len(self.example_inputs)} example inputs"
            )
        for i, ph in enumerate(placeholders):
            self.env[ph.name] = f"input_{i}"
        for node in traced.graph.nodes:
            if node.op == "placeholder":
                continue
            if node.op == "output":
                return self._finish(node)
            gname = self._lower_node(traced, node)
            self.env[node.name] = gname
            self.specs[gname] = self.builder._env()[gname]
        raise GraphError("traced graph has no output node")

    def _lower_node(self, traced, node) -> str:
        name = node.name
        if node.op == "get_attr":
            t = _resolve_attr(traced, node.target)
            t = t.detach().clone().contiguous()
            self.builder.constant(name, t)
            self.const_tensors[name] = t
            return name
        if node.op == "call_module":
            mod = traced
            for part in node.target.split("."):
                mod = getattr(mod, part)
            return self._lower_module(name, mod, node)
        if node.op == "call_function":
            return self._lower_function(name, node)
        if node.op == "call_method":
            return self._lower_method(name, node)
        raise UnsupportedOperator(f"unsupported fx node op {node.op!r} at {name!r}")

    def _lower_module(self, name: str, mod: nn.Module, node) -> str:
        bound = self._bind(node, ("input",))
        x = self._arg(bound["input"])
        if isinstance(mod, nn.Identity):
            return x
        if isinstance(mod, nn.Linear):
            weight = mod.weight.detach().clone()
            bias = None if mod.bias is None else mod.bias.detach().clone()
            return self._linear(name, x, weight, bias)
        if isinstance(mod, nn.GELU):
            return self.builder.add(name, "gelu", (x,), approximate=mod.approximate)
        if isinstance(mod, nn.ReLU):
            if mod.inplace:
                raise UnsupportedOperator(f"in-place ReLU at {name!r} is not supported")
            return self.builder.add(name, "relu", (x,))
        if isinstance(mod, nn.Softmax):
            if mod.dim is None:
                raise UnsupportedOperator(f"Softmax at {name!r} has no static dim")
            return self.builder.add(name, "softmax", (x,), dim=mod.dim)
        raise UnsupportedOperator(f"unsupported module {type(mod).__name__} at {name!r}")

    def _lower_function(self, name: str, node) -> str:
        target = node.target
        if target in (torch.matmul, torch.mm):
            bound = self._bind(node, ("input", "other"))
            return self.builder.add(
                name, "matmul", (self._arg(bound["input"]), self._arg(bound["other"]))
            )
        if target is operator.matmul:
            bound = self._bind(node, ("input", "other"))
            return self.builder.add(
                name, "matmul", (self._arg(bound["input"]), self._arg(bound["other"]))
            )
        if target in (operator.add, torch.add):
            bound = self._bind(node, ("input", "other", "alpha", "out"), {"alpha": 1, "out": None})
            if "out" in node.kwargs or bound["out"] is not None:
                raise UnsupportedOperator(f"add out= at {name!r} is not supported")
            if bound["alpha"] != 1:
                raise UnsupportedOperator(f"add alpha!=1 at {name!r} is not supported")
            return self.builder.add(
                name, "add", (self._arg(bound["input"]), self._arg(bound["other"]))
            )
        if target in (operator.mul, torch.mul):
            bound = self._bind(node, ("input", "other", "out"), {"out": None})
            if "out" in node.kwargs or bound["out"] is not None:
                raise UnsupportedOperator(f"mul out= at {name!r} is not supported")
            return self.builder.add(
                name, "mul", (self._arg(bound["input"]), self._arg(bound["other"]))
            )
        if target is F.relu:
            bound = self._bind(node, ("input", "inplace"), {"inplace": False})
            if bound["inplace"] is not False:
                raise UnsupportedOperator(f"in-place relu at {name!r} is not supported")
            return self.builder.add(name, "relu", (self._arg(bound["input"]),))
        if target is torch.relu:
            bound = self._bind(node, ("input",))
            return self.builder.add(name, "relu", (self._arg(bound["input"]),))
        if target is F.gelu or target is getattr(torch, "gelu", None):
            bound = self._bind(node, ("input", "approximate"), {"approximate": "none"})
            if bound["approximate"] not in ("none", "tanh"):
                raise UnsupportedOperator(f"gelu approximate {bound['approximate']!r} at {name!r}")
            return self.builder.add(
                name, "gelu", (self._arg(bound["input"]),), approximate=bound["approximate"]
            )
        if target is F.linear:
            bound = self._bind(node, ("input", "weight", "bias"), {"bias": None})
            x = self._arg(bound["input"])
            w = self._arg(bound["weight"])
            if w not in self.const_tensors:
                raise UnsupportedOperator(f"linear weight at {name!r} must be constant")
            bias = None
            if bound["bias"] is not None:
                bias_name = self._arg(bound["bias"])
                if bias_name not in self.const_tensors:
                    raise UnsupportedOperator(f"linear bias at {name!r} must be constant")
                bias = self.const_tensors[bias_name]
            return self._linear(name, x, self.const_tensors[w], bias)
        if target in (torch.reshape, torch.Tensor.reshape, torch.Tensor.view):
            bound = self._bind(node, ("input", "shape"))
            x = self._arg(bound["input"])
            shape = _static_shape(bound["shape"], self.specs[x].shape, "reshape")
            return self.builder.add(name, "reshape", (x,), shape=shape)
        if target is torch.transpose:
            bound = self._bind(node, ("input", "dim0", "dim1"))
            return self.builder.add(
                name,
                "transpose",
                (self._arg(bound["input"]),),
                dim0=bound["dim0"],
                dim1=bound["dim1"],
            )
        if target in (F.softmax, torch.softmax):
            bound = self._bind(
                node,
                ("input", "dim", "dtype", "_stacklevel"),
                {"dim": None, "dtype": None, "_stacklevel": 3},
            )
            if bound["dtype"] is not None:
                raise UnsupportedOperator(f"softmax dtype= at {name!r} is not supported")
            if not isinstance(bound["dim"], int) or isinstance(bound["dim"], bool):
                raise UnsupportedOperator(f"softmax at {name!r} needs a static int dim")
            return self.builder.add(name, "softmax", (self._arg(bound["input"]),), dim=bound["dim"])
        raise UnsupportedOperator(
            f"unsupported function {getattr(target, '__name__', target)!r} at {name!r}"
        )

    def _lower_method(self, name: str, node) -> str:
        method = node.target
        if method in ("matmul", "mm"):
            bound = self._bind(node, ("input", "other"))
            return self.builder.add(
                name, "matmul", (self._arg(bound["input"]), self._arg(bound["other"]))
            )
        if method in ("view", "reshape"):
            if node.kwargs:
                raise UnsupportedOperator(f"{method} kwargs at {name!r} are not supported")
            if len(node.args) < 2:
                raise UnsupportedOperator(f"{method} at {name!r} needs a shape")
            x = self._arg(node.args[0])
            raw = node.args[1:]
            if len(raw) == 1 and isinstance(raw[0], (tuple, list)):
                dims = list(raw[0])
            else:
                dims = list(raw)
            shape = _static_shape(tuple(dims), self.specs[x].shape, method)
            return self.builder.add(name, "reshape", (x,), shape=shape)
        if method == "transpose":
            bound = self._bind(node, ("input", "dim0", "dim1"))
            return self.builder.add(
                name,
                "transpose",
                (self._arg(bound["input"]),),
                dim0=bound["dim0"],
                dim1=bound["dim1"],
            )
        if method == "t":
            bound = self._bind(node, ("input",))
            x = self._arg(bound["input"])
            if len(self.specs[x].shape) != 2:
                raise UnsupportedOperator(f"t() at {name!r} requires a rank-2 tensor")
            return self.builder.add(name, "transpose", (x,), dim0=0, dim1=1)
        if method == "softmax":
            bound = self._bind(node, ("input", "dim", "dtype"), {"dtype": None})
            if bound["dtype"] is not None:
                raise UnsupportedOperator(f"softmax dtype= at {name!r} is not supported")
            if not isinstance(bound["dim"], int) or isinstance(bound["dim"], bool):
                raise UnsupportedOperator(f"softmax at {name!r} needs a static int dim")
            return self.builder.add(name, "softmax", (self._arg(bound["input"]),), dim=bound["dim"])
        raise UnsupportedOperator(f"unsupported method {method!r} at {name!r}")

    def _finish(self, output_node) -> Graph:
        import torch.fx

        result = output_node.args[0]
        if isinstance(result, torch.fx.Node):
            outputs = (self.env[result.name],)
            is_tuple = False
        elif isinstance(result, tuple):
            names = []
            for r in result:
                if not isinstance(r, torch.fx.Node):
                    raise UnsupportedOperator(
                        "only flat tuples of tensors are supported as outputs"
                    )
                names.append(self.env[r.name])
            outputs = tuple(names)
            is_tuple = True
        else:
            raise UnsupportedOperator(
                f"unsupported output type {type(result).__name__}; "
                "only a tensor or flat tuple of tensors is supported"
            )
        return self.builder.finish(outputs, output_is_tuple=is_tuple)


def from_torch(model: torch.nn.Module, example_inputs: tuple[torch.Tensor, ...]) -> Graph:
    if not isinstance(model, nn.Module):
        raise GraphError("from_torch expects an nn.Module")
    for i, t in enumerate(example_inputs):
        if not isinstance(t, torch.Tensor):
            raise GraphError(f"example input {i} is not a tensor")
    _check_module_state(model)
    with torch.no_grad():
        return _Lowerer(model, tuple(example_inputs)).lower()


def _onnx_attr_dict(node) -> dict:
    import onnx

    return {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}


def _onnx_tensor_to_torch(tp) -> torch.Tensor:
    import numpy as np
    import onnx

    if tp.data_location == onnx.TensorProto.EXTERNAL:
        raise UnsupportedOperator(f"tensor {tp.name!r} uses external data, which is not supported")
    arr = onnx.numpy_helper.to_array(tp)
    return torch.from_numpy(np.array(arr, copy=True, order="C"))


_ONNX_OPS = {
    "MatMul",
    "Add",
    "Mul",
    "Relu",
    "Gelu",
    "Reshape",
    "Transpose",
    "Softmax",
    "Constant",
    "Identity",
    "Gemm",
}


def _onnx_check_attrs(node, attrs: dict, allowed: set[str]) -> None:
    extra = set(attrs) - allowed
    if extra:
        raise UnsupportedOperator(
            f"{node.op_type} node {node.name!r} has unsupported attrs {sorted(extra)}"
        )


def _onnx_dims(vi) -> list:
    dims = []
    for d in vi.type.tensor_type.shape.dim:
        if d.HasField("dim_value"):
            dims.append(d.dim_value)
        else:
            dims.append(None)
    return dims


def _onnx_dtype(vi) -> torch.dtype:
    import onnx

    elem = vi.type.tensor_type.elem_type
    np_dtype = onnx.helper.tensor_dtype_to_np_dtype(elem)
    return torch.from_numpy(__import__("numpy").empty(0, dtype=np_dtype)).dtype


def _onnx_check_arity(node, count: int) -> None:
    if len(node.input) != count:
        raise GraphError(
            f"ONNX {node.op_type} node {node.name!r} expects {count} inputs, got {len(node.input)}"
        )


def from_onnx(model_or_path, example_inputs: tuple[torch.Tensor, ...] | None = None) -> Graph:
    import onnx

    if isinstance(model_or_path, onnx.ModelProto):
        model = model_or_path
    else:
        model = onnx.load_model(str(model_or_path), load_external_data=False)
    opset = None
    for imp in model.opset_import:
        if imp.domain in ("", "ai.onnx"):
            opset = imp.version
        else:
            raise UnsupportedOperator(f"unsupported ONNX domain {imp.domain!r}")
    if opset is None or not 13 <= opset <= 22:
        raise UnsupportedOperator(f"unsupported ONNX opset {opset}")
    g = model.graph

    if example_inputs is not None:
        devices = {str(t.device) for t in example_inputs}
        if len(devices) != 1:
            raise GraphError("example inputs must all be on the same device")
        target_device = devices.pop()
    else:
        target_device = None

    const_tensors: dict[str, torch.Tensor] = {}
    for init in g.initializer:
        if init.name in const_tensors:
            raise GraphError(f"duplicate ONNX initializer {init.name!r}")
        t = _onnx_tensor_to_torch(init)
        if target_device is not None:
            t = t.to(target_device)
        const_tensors[init.name] = t

    builder_inputs: dict[str, TensorSpec] = {}
    onnx_inputs = [vi for vi in g.input if vi.name not in const_tensors]
    if len({vi.name for vi in onnx_inputs}) != len(onnx_inputs):
        raise GraphError("duplicate ONNX input names")
    if example_inputs is not None:
        if len(example_inputs) != len(onnx_inputs):
            raise GraphError(
                f"ONNX model expects {len(onnx_inputs)} inputs, "
                f"got {len(example_inputs)} example inputs"
            )
        for vi, t in zip(onnx_inputs, example_inputs):
            dims = _onnx_dims(vi)
            if len(dims) != t.dim():
                raise GraphError(
                    f"example input for {vi.name!r} rank {t.dim()} != ONNX rank {len(dims)}"
                )
            for got, meta in zip(t.shape, dims):
                if meta is not None and meta != got:
                    raise GraphError(
                        f"example input for {vi.name!r} dim {got} != fixed ONNX dim {meta}"
                    )
            if t.dtype != _onnx_dtype(vi):
                raise GraphError(
                    f"example input for {vi.name!r} dtype {t.dtype} != ONNX dtype {_onnx_dtype(vi)}"
                )
            builder_inputs[vi.name] = _spec(t)
    else:
        for vi in onnx_inputs:
            dims = _onnx_dims(vi)
            if any(d is None or d <= 0 for d in dims):
                raise GraphError(
                    f"ONNX input {vi.name!r} has a dynamic or unknown dimension; "
                    "pass example_inputs to fix shapes"
                )
            builder_inputs[vi.name] = TensorSpec(tuple(dims), _onnx_dtype(vi), "cpu")
    builder = GraphBuilder(builder_inputs)
    for name, t in const_tensors.items():
        builder.constant(name, t)
    env: dict[str, str] = {}
    for name in list(builder_inputs) + list(const_tensors):
        env[name] = name
    counter = [0]

    def fresh(hint: str) -> str:
        counter[0] += 1
        return f"{hint}_{counter[0]}"

    def spec_of_graph_name(gname: str) -> TensorSpec:
        if gname in builder_inputs:
            return builder_inputs[gname]
        if gname in const_tensors:
            return _spec(const_tensors[gname])
        for nd in builder._nodes:
            if nd.name == gname:
                return nd.spec
        raise GraphError(f"unknown value {gname!r}")

    def add_scalar(value: float, like: TensorSpec, hint: str) -> str:
        t = torch.tensor(value, dtype=like.dtype, device=like.device)
        name = fresh(hint)
        builder.constant(name, t)
        const_tensors[name] = t
        env[name] = name
        return name

    def set_out(onnx_name: str, gname: str) -> None:
        if onnx_name in env:
            raise GraphError(f"duplicate ONNX value {onnx_name!r}")
        env[onnx_name] = gname

    for node in g.node:
        if node.domain not in ("", "ai.onnx"):
            raise UnsupportedOperator(
                f"unsupported ONNX domain {node.domain!r} on node {node.name!r}"
            )
        if node.op_type not in _ONNX_OPS:
            raise UnsupportedOperator(f"unsupported ONNX op {node.op_type!r} on node {node.name!r}")
        attrs = _onnx_attr_dict(node)
        ins = []
        for i in node.input:
            if not i:
                continue
            if i not in env:
                raise GraphError(f"ONNX node {node.name!r} input {i!r} has no producer")
            ins.append(env[i])
        outs = [o for o in node.output if o]
        if not outs:
            raise GraphError(f"ONNX node {node.name!r} has no outputs")
        if len(outs) > 1:
            raise UnsupportedOperator(f"ONNX node {node.name!r} produces multiple outputs")
        out = outs[0]
        if out in env:
            raise GraphError(f"duplicate ONNX output {out!r} on node {node.name!r}")

        if node.op_type == "Constant":
            _onnx_check_attrs(node, attrs, {"value"})
            _onnx_check_arity(node, 0)
            if "value" not in attrs:
                raise UnsupportedOperator(
                    f"Constant node {node.name!r} lacks a 'value' tensor attr"
                )
            t = _onnx_tensor_to_torch(attrs["value"])
            if target_device is not None:
                t = t.to(target_device)
            builder.constant(out, t)
            const_tensors[out] = t
            env[out] = out
            continue
        if node.op_type == "Identity":
            _onnx_check_attrs(node, attrs, set())
            _onnx_check_arity(node, 1)
            set_out(out, ins[0])
            continue
        if node.op_type == "MatMul":
            _onnx_check_attrs(node, attrs, set())
            _onnx_check_arity(node, 2)
            set_out(out, builder.add(out, "matmul", tuple(ins)))
            continue
        if node.op_type in ("Add", "Mul"):
            _onnx_check_attrs(node, attrs, set())
            _onnx_check_arity(node, 2)
            set_out(out, builder.add(out, node.op_type.lower(), tuple(ins)))
            continue
        if node.op_type == "Relu":
            _onnx_check_attrs(node, attrs, set())
            _onnx_check_arity(node, 1)
            set_out(out, builder.add(out, "relu", tuple(ins)))
            continue
        if node.op_type == "Gelu":
            if opset < 20:
                raise UnsupportedOperator("ONNX Gelu requires opset >= 20")
            _onnx_check_attrs(node, attrs, {"approximate"})
            _onnx_check_arity(node, 1)
            approx = attrs.get("approximate", "none")
            if isinstance(approx, bytes):
                approx = approx.decode("utf-8")
            if approx not in ("none", "tanh"):
                raise UnsupportedOperator(f"unsupported Gelu approximate {approx!r}")
            set_out(out, builder.add(out, "gelu", tuple(ins), approximate=approx))
            continue
        if node.op_type == "Softmax":
            _onnx_check_attrs(node, attrs, {"axis"})
            _onnx_check_arity(node, 1)
            set_out(out, builder.add(out, "softmax", tuple(ins), dim=attrs.get("axis", -1)))
            continue
        if node.op_type == "Transpose":
            _onnx_check_attrs(node, attrs, {"perm"})
            _onnx_check_arity(node, 1)
            rank = len(spec_of_graph_name(ins[0]).shape)
            perm = list(attrs.get("perm", reversed(range(rank))))
            if sorted(perm) != list(range(rank)):
                raise UnsupportedOperator(f"Transpose node {node.name!r} has invalid perm {perm}")
            moved = [i for i, p in enumerate(perm) if p != i]
            if not moved:
                set_out(out, ins[0])
                continue
            if len(moved) == 2 and perm[moved[0]] == moved[1]:
                set_out(
                    out,
                    builder.add(out, "transpose", (ins[0],), dim0=moved[0], dim1=moved[1]),
                )
                continue
            raise UnsupportedOperator(
                f"Transpose node {node.name!r} perm {perm} is not a single swap"
            )
        if node.op_type == "Reshape":
            _onnx_check_attrs(node, attrs, {"allowzero"})
            _onnx_check_arity(node, 2)
            if attrs.get("allowzero", 0) != 0:
                raise UnsupportedOperator(
                    f"Reshape node {node.name!r} allowzero=1 is not supported"
                )
            shape_t = const_tensors.get(env[node.input[1]])
            if shape_t is None:
                raise UnsupportedOperator(
                    f"Reshape node {node.name!r} shape input must be constant"
                )
            if shape_t.dtype != torch.int64 or shape_t.dim() != 1:
                raise UnsupportedOperator(
                    f"Reshape node {node.name!r} shape must be a 1-D int64 tensor"
                )
            raw = [int(v) for v in shape_t.cpu().tolist()]
            in_shape = spec_of_graph_name(ins[0]).shape
            resolved = []
            for i, d in enumerate(raw):
                if d == 0:
                    if i >= len(in_shape):
                        raise GraphError(
                            f"Reshape node {node.name!r} zero dim index {i} "
                            f"out of range for input rank {len(in_shape)}"
                        )
                    resolved.append(in_shape[i])
                else:
                    resolved.append(d)
            if sum(1 for d in resolved if d == -1) > 1:
                raise UnsupportedOperator(f"Reshape node {node.name!r} has more than one -1 dim")
            if -1 in resolved:
                known = 1
                for d in resolved:
                    if d != -1:
                        if d <= 0:
                            raise UnsupportedOperator(
                                f"Reshape node {node.name!r} has invalid dim {d}"
                            )
                        known *= d
                total = 1
                for d in in_shape:
                    total *= d
                if total % known:
                    raise GraphError(f"Reshape node {node.name!r} cannot infer -1 dim")
                resolved[resolved.index(-1)] = total // known
            if any(d <= 0 for d in resolved):
                raise UnsupportedOperator(
                    f"Reshape node {node.name!r} resolved to non-positive dims"
                )
            set_out(out, builder.add(out, "reshape", (ins[0],), shape=tuple(resolved)))
            continue
        if node.op_type == "Gemm":
            _onnx_check_attrs(node, attrs, {"alpha", "beta", "transA", "transB"})
            _onnx_check_arity(node, len(node.input))
            alpha = float(attrs.get("alpha", 1.0))
            beta = float(attrs.get("beta", 1.0))
            trans_a = int(attrs.get("transA", 0))
            trans_b = int(attrs.get("transB", 0))
            if trans_a not in (0, 1) or trans_b not in (0, 1):
                raise UnsupportedOperator(f"Gemm node {node.name!r} transA/transB must be 0 or 1")
            if len(ins) not in (2, 3):
                raise GraphError(f"Gemm node {node.name!r} expects 2 or 3 inputs")
            a, b = ins[0], ins[1]
            if trans_a:
                a = builder.add(fresh(f"{out}_transA"), "transpose", (a,), dim0=0, dim1=1)
            if trans_b:
                b = builder.add(fresh(f"{out}_transB"), "transpose", (b,), dim0=0, dim1=1)
            mm = builder.add(fresh(f"{out}_matmul"), "matmul", (a, b))
            if alpha != 1.0:
                c = add_scalar(alpha, spec_of_graph_name(ins[0]), f"{out}_alpha")
                mm = builder.add(fresh(f"{out}_alpha_mul"), "mul", (mm, c))
            if len(ins) == 3 and beta != 0.0:
                c_term = ins[2]
                if beta != 1.0:
                    bc = add_scalar(beta, spec_of_graph_name(ins[2]), f"{out}_beta")
                    c_term = builder.add(fresh(f"{out}_beta_mul"), "mul", (ins[2], bc))
                set_out(out, builder.add(out, "add", (mm, c_term)))
            else:
                set_out(out, mm)
            continue
        raise UnsupportedOperator(f"unhandled ONNX op {node.op_type!r}")

    graph_outputs = [o.name for o in g.output]
    if not graph_outputs:
        raise GraphError("ONNX graph has no outputs")
    if len(set(graph_outputs)) != len(graph_outputs):
        raise GraphError("duplicate ONNX graph outputs")
    missing = [o for o in graph_outputs if o not in env]
    if missing:
        raise GraphError(f"ONNX outputs {missing} are not produced")
    for vi in g.output:
        dims = _onnx_dims(vi)
        got = spec_of_graph_name(env[vi.name])
        if len(dims) != len(got.shape):
            raise GraphError(
                f"ONNX output {vi.name!r} rank {len(dims)} != inferred {len(got.shape)}"
            )
        for meta, actual in zip(dims, got.shape):
            if meta is not None and meta != actual:
                raise GraphError(f"ONNX output {vi.name!r} dim {meta} != inferred {actual}")
        if got.dtype != _onnx_dtype(vi):
            raise GraphError(
                f"ONNX output {vi.name!r} dtype {got.dtype} != declared {_onnx_dtype(vi)}"
            )
    outputs = tuple(env[o] for o in graph_outputs)
    return builder.finish(outputs, output_is_tuple=len(outputs) > 1)
