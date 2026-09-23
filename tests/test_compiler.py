import json

import pytest
import torch
from torch import nn

from forgeml import compile
from forgeml.frontend import from_torch
from forgeml.ir import GraphError


def make_mlp():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(8, 16), nn.GELU(approximate="tanh"), nn.Linear(16, 4)).eval()


def test_mlp_eager_parity():
    m = make_mlp()
    x = torch.randn(4, 8)
    c = compile(m, (x,))
    torch.testing.assert_close(c(x), m(x))


def test_backend_rejection():
    with pytest.raises(GraphError, match="backend"):
        compile(make_mlp(), (torch.randn(4, 8),), backend="tensorrt")


def test_compiled_outputs_independent_across_calls():
    m = make_mlp()
    x = torch.randn(4, 8)
    c = compile(m, (x,))
    out1 = c(x)
    out1_copy = out1.clone()
    out2 = c(x + 1.0)
    torch.testing.assert_close(out1, out1_copy)
    assert out1.data_ptr() != out2.data_ptr()


def test_explain_is_json_serializable():
    c = compile(make_mlp(), (torch.randn(4, 8),))
    s = json.dumps(c.explain())
    assert "fused_linear_gelu" in s
    e = c.explain()
    assert e["optimized_nodes"] <= e["original_nodes"]
    assert e["memory_plan"]["planned_bytes"] <= e["memory_plan"]["naive_bytes"]


def test_noncontiguous_input_accepted():
    m = make_mlp()
    x = torch.randn(8, 4).t()
    assert not x.is_contiguous() and x.shape == (4, 8)
    c = compile(m, (x,))
    torch.testing.assert_close(c(x), m(x))


def test_optimize_flag_off():
    m = make_mlp()
    x = torch.randn(4, 8)
    c = compile(m, (x,), optimize=False)
    assert c.passes == []
    torch.testing.assert_close(c(x), m(x))


def test_explicit_graph_compiles():
    m = make_mlp()
    x = torch.randn(4, 8)
    g = from_torch(m, (x,))
    c = compile(g)
    torch.testing.assert_close(c(x), m(x))


def test_output_alias_not_clobbered_by_slot_reuse():
    from forgeml.ir import GraphBuilder, TensorSpec

    def spec(*shape):
        return TensorSpec(tuple(shape), torch.float32, "cpu")

    b = GraphBuilder({"x": spec(4)})
    b.add("a", "relu", ("x",))
    b.add("b", "reshape", ("a",), shape=(2, 2))
    b.add("c", "relu", ("a",))
    b.constant("two", torch.full((4,), 2.0))
    b.add("d", "mul", ("c", "two"))
    b.add("out", "relu", ("d",))
    g = b.finish(("b", "out"))
    c = compile(g, optimize=False)
    plan = c.memory_plan
    assert plan.allocations["d"].slot == plan.allocations["a"].slot
    x = torch.randn(4)
    out_b, out_d = c(x)
    torch.testing.assert_close(out_b, torch.relu(x).reshape(2, 2))
    torch.testing.assert_close(out_d, torch.relu(torch.relu(x) * 2))


def test_explicit_graph_constants_snapshot_optimize_off():
    from forgeml.ir import GraphBuilder, TensorSpec

    b = GraphBuilder({"x": TensorSpec((2,), torch.float32, "cpu")})
    b.constant("c", torch.ones(2))
    b.add("y", "add", ("x", "c"))
    g = b.finish(("y",))
    c = compile(g, optimize=False)
    g.constants["c"].fill_(99.0)
    out = c(torch.zeros(2))
    torch.testing.assert_close(out, torch.ones(2))
