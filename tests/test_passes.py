import torch

from forgeml.ir import GraphBuilder, TensorSpec
from forgeml.passes import (
    eliminate_dead_nodes,
    fold_constants,
    fuse_linear_gelu,
    optimize,
    schedule,
)


def spec(*shape):
    return TensorSpec(tuple(shape), torch.float32, "cpu")


def test_dce_removes_unreachable_nodes_and_consts():
    b = GraphBuilder({"x": spec(2, 3)})
    b.constant("dead_w", torch.ones(3, 3))
    b.constant("live_w", torch.ones(3, 4))
    b.add("dead", "matmul", ("x", "dead_w"))
    b.add("y", "matmul", ("x", "live_w"))
    g = b.finish(("y",))
    out = eliminate_dead_nodes(g)
    assert [n.name for n in out.nodes] == ["y"]
    assert "dead_w" not in out.constants
    assert "x" in out.inputs  # public inputs kept


def test_fold_constants_produces_tensor_no_node():
    b = GraphBuilder({"x": spec(2)})
    b.constant("a", torch.ones(2))
    b.constant("c", torch.full((2,), 3.0))
    b.add("ac", "add", ("a", "c"))
    b.add("y", "mul", ("x", "ac"))
    g = b.finish(("y",))
    out = fold_constants(g)
    assert [n.name for n in out.nodes] == ["y"]
    assert isinstance(out.constants["ac"], torch.Tensor)
    torch.testing.assert_close(out.constants["ac"], torch.full((2,), 4.0))


def _linear_gelu_graph():
    b = GraphBuilder({"x": spec(2, 4)})
    b.constant("w", torch.randn(4, 8))
    b.constant("bias", torch.randn(8))
    b.add("mm", "matmul", ("x", "w"))
    b.add("lin", "add", ("mm", "bias"))
    b.add("y", "gelu", ("lin",), approximate="tanh")
    return b.finish(("y",))


def test_fusion_3_to_1():
    out = fuse_linear_gelu(_linear_gelu_graph())
    assert len(out.nodes) == 1
    n = out.nodes[0]
    assert n.op == "fused_linear_gelu"
    assert n.name == "y"
    assert n.attrs["approximate"] == "tanh"


def test_fusion_skipped_when_shared_or_output():
    b = GraphBuilder({"x": spec(2, 4)})
    b.constant("w", torch.randn(4, 8))
    b.constant("bias", torch.randn(8))
    b.add("mm", "matmul", ("x", "w"))
    b.add("lin", "add", ("mm", "bias"))
    b.add("y", "gelu", ("lin",))
    g = b.finish(("y", "lin"))
    out = fuse_linear_gelu(g)
    assert len(out.nodes) == 3

    b = GraphBuilder({"x": spec(2, 4)})
    b.constant("w", torch.randn(4, 8))
    b.constant("bias", torch.randn(8))
    b.add("mm", "matmul", ("x", "w"))
    b.add("lin", "add", ("mm", "bias"))
    b.add("other", "add", ("mm", "bias"))
    b.add("y", "gelu", ("lin",))
    g = b.finish(("y", "other"))
    out = fuse_linear_gelu(g)
    assert len(out.nodes) == 4


def test_fusion_skipped_for_non_1d_bias():
    b = GraphBuilder({"x": spec(2, 4)})
    b.constant("w", torch.randn(4, 8))
    b.constant("bias", torch.randn(2, 8))
    b.add("mm", "matmul", ("x", "w"))
    b.add("lin", "add", ("mm", "bias"))
    b.add("y", "gelu", ("lin",))
    g = b.finish(("y",))
    assert len(fuse_linear_gelu(g).nodes) == 3


def test_schedule_respects_dependencies_and_is_deterministic():
    b = GraphBuilder({"x": spec(4), "z": spec(4)})
    b.add("a", "relu", ("x",))
    b.add("c", "mul", ("z", "z"))
    b.add("y", "add", ("a", "c"))
    g = b.finish(("y",))
    s1 = schedule(g)
    s2 = schedule(g)
    assert [n.name for n in s1.nodes] == [n.name for n in s2.nodes]
    pos = {n.name: i for i, n in enumerate(s1.nodes)}
    for n in s1.nodes:
        for i in n.inputs:
            if i in pos:
                assert pos[i] < pos[n.name]


def test_optimize_returns_records_and_preserves_original():
    g = _linear_gelu_graph()
    out, records = optimize(g)
    assert [r.name for r in records] == [
        "eliminate_dead_nodes",
        "fold_constants",
        "eliminate_dead_nodes",
        "fuse_linear_gelu",
        "schedule",
    ]
    assert len(out.nodes) == 1
    assert len(g.nodes) == 3  # original untouched


def test_fold_skips_oversized_output_without_evaluating(monkeypatch):
    from forgeml import passes

    monkeypatch.setattr(passes, "_FOLD_LIMIT_BYTES", 1)

    def boom(*a, **k):
        raise AssertionError("evaluate called over the fold cap")

    monkeypatch.setattr(passes, "evaluate", boom)
    b = GraphBuilder({"x": spec(2)})
    b.constant("a", torch.ones(2))
    b.constant("c", torch.ones(2))
    b.add("ac", "add", ("a", "c"))
    b.add("y", "mul", ("x", "ac"))
    g = b.finish(("y",))
    out = passes.fold_constants(g)
    assert [n.name for n in out.nodes] == ["ac", "y"]
