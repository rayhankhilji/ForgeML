import torch

from forgeml.ir import GraphBuilder, TensorSpec
from forgeml.passes import (
    canonicalize_linear,
    common_subexpression_elimination,
    eliminate_dead_nodes,
    fold_constants,
    fuse_linear_gelu,
    optimize,
    schedule,
    simplify_algebra,
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


def test_matmul_bias_add_canonicalizes_to_linear():
    b = GraphBuilder({"x": spec(2, 4)})
    b.constant("w", torch.randn(4, 8))
    b.constant("bias", torch.randn(8))
    b.add("mm", "matmul", ("x", "w"))
    b.add("y", "add", ("bias", "mm"))
    g = b.finish(("y",))
    out = canonicalize_linear(g)
    assert [n.op for n in out.nodes] == ["linear"]
    assert out.nodes[0].name == "y"
    assert out.nodes[0].inputs == ("x", "w", "bias")


def test_matmul_bias_add_stays_when_projection_is_shared_or_output():
    b = GraphBuilder({"x": spec(2, 4)})
    b.constant("w", torch.randn(4, 8))
    b.constant("bias", torch.randn(8))
    b.add("mm", "matmul", ("x", "w"))
    b.add("y", "add", ("mm", "bias"))
    b.add("z", "add", ("mm", "bias"))
    g = b.finish(("y", "z"))
    out = canonicalize_linear(g)
    assert [n.op for n in out.nodes] == ["matmul", "add", "add"]

    b = GraphBuilder({"x": spec(2, 4)})
    b.constant("w", torch.randn(4, 8))
    b.constant("bias", torch.randn(8))
    b.add("mm", "matmul", ("x", "w"))
    b.add("y", "add", ("mm", "bias"))
    g = b.finish(("mm", "y"))
    out = canonicalize_linear(g)
    assert [n.op for n in out.nodes] == ["matmul", "add"]


def test_linear_gelu_fusion_preserves_unbiased_and_biased_forms():
    b = GraphBuilder({"x": spec(2, 4), "z": spec(2, 4)})
    b.constant("w", torch.randn(4, 8))
    b.constant("bias", torch.randn(8))
    b.add("lin", "linear", ("x", "w", "bias"))
    b.add("act", "gelu", ("lin",), approximate="none")
    b.add("plain", "linear", ("z", "w"))
    b.add("out", "add", ("act", "plain"))
    out = fuse_linear_gelu(b.finish(("out",)))
    assert [n.op for n in out.nodes] == ["fused_linear_gelu", "linear", "add"]
    assert out.nodes[0].inputs == ("x", "w", "bias")
    assert out.nodes[1].inputs == ("z", "w")


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


def test_simplify_inverse_transposes_and_identity_reshape():
    b = GraphBuilder({"x": spec(2, 4)})
    b.add("t1", "transpose", ("x",), dim0=0, dim1=1)
    b.add("t2", "transpose", ("t1",), dim0=0, dim1=1)
    b.add("same", "reshape", ("t2",), shape=(2, 4))
    b.add("y", "relu", ("same",))
    out = simplify_algebra(b.finish(("y",)))
    assert [n.op for n in out.nodes] == ["relu"]
    assert out.nodes[0].inputs == ("x",)


def test_common_subexpression_eliminates_duplicate_pure_nodes():
    b = GraphBuilder({"x": spec(2, 4), "z": spec(2, 4)})
    b.add("a", "add", ("x", "z"))
    b.add("b", "add", ("x", "z"))
    b.add("y", "mul", ("a", "b"))
    g = b.finish(("y",))
    out = common_subexpression_elimination(g)
    assert [n.name for n in out.nodes] == ["a", "y"]
    assert out.nodes[-1].inputs == ("a", "a")


def test_cse_preserves_multiple_output_semantics():
    b = GraphBuilder({"x": spec(2)})
    b.add("a", "relu", ("x",))
    b.add("b", "relu", ("x",))
    g = b.finish(("a", "b"))
    out = common_subexpression_elimination(g)
    assert len(out.nodes) == 1
    assert out.outputs == ("a", "a")


def test_optimize_returns_records_and_preserves_original():
    g = _linear_gelu_graph()
    out, records = optimize(g)
    assert [r.name for r in records] == [
        "eliminate_dead_nodes",
        "fold_constants",
        "eliminate_dead_nodes",
        "simplify_algebra",
        "common_subexpression_elimination",
        "canonicalize_linear",
        "fuse_linear_gelu",
        "fuse_shared_projections",
        "eliminate_dead_nodes",
        "schedule",
    ]
    assert len(out.nodes) == 1
    assert len(g.nodes) == 3  # original untouched


def test_fuse_linear_gelu_fuses_bare_matmul():
    b = GraphBuilder({"x": spec(2, 4)})
    b.constant("w", torch.randn(4, 8))
    b.add("mm", "matmul", ("x", "w"))
    b.add("y", "gelu", ("mm",))
    g = b.finish(("y",))
    out = fuse_linear_gelu(g)
    assert [n.op for n in out.nodes] == ["fused_linear_gelu"]
    assert len(out.nodes[0].inputs) == 2


def test_fuse_shared_projections_emits_wide_linear_and_narrows():
    from forgeml import compile
    from forgeml.passes import fuse_shared_projections

    torch.manual_seed(0)
    b = GraphBuilder({"x": spec(2, 8)})
    b.constant("wq", torch.randn(8, 4))
    b.constant("wk", torch.randn(8, 4))
    b.constant("wv", torch.randn(8, 4))
    b.add("q", "linear", ("x", "wq"))
    b.add("k", "linear", ("x", "wk"))
    b.add("v", "linear", ("x", "wv"))
    b.add("yk", "add", ("q", "k"))
    b.add("z", "add", ("yk", "v"))
    g = b.finish(("z",))
    out = fuse_shared_projections(g)
    ops = [n.op for n in out.nodes]
    assert ops.count("linear") == 1
    assert ops.count("narrow") == 3
    x = torch.randn(2, 8)
    torch.testing.assert_close(compile(out, optimize=False)(x), compile(g, optimize=False)(x))


def test_fuse_shared_projections_with_biases_and_output_member():
    from forgeml import compile
    from forgeml.passes import fuse_shared_projections

    torch.manual_seed(1)
    b = GraphBuilder({"x": spec(2, 8)})
    b.constant("wq", torch.randn(8, 4))
    b.constant("wk", torch.randn(8, 4))
    b.constant("bq", torch.randn(4))
    b.constant("bk", torch.randn(4))
    b.add("q", "linear", ("x", "wq", "bq"))
    b.add("k", "linear", ("x", "wk", "bk"))
    g = b.finish(("q", "k"))
    out = fuse_shared_projections(g)
    ops = [n.op for n in out.nodes]
    assert ops.count("linear") == 1 and ops.count("narrow") == 2
    assert out.outputs == ("q", "k")
    x = torch.randn(2, 8)
    want = compile(g, optimize=False)(x)
    got = compile(out, optimize=False)(x)
    for a, e in zip(got, want):
        torch.testing.assert_close(a, e)


def test_cse_dedupes_commutative_operands():
    b = GraphBuilder({"x": spec(4), "y": spec(4)})
    b.add("a", "mul", ("x", "y"))
    b.add("bb", "mul", ("y", "x"))
    g = b.finish(("a", "bb"))
    out = common_subexpression_elimination(g)
    assert len(out.nodes) == 1
    assert out.outputs == ("a", "a")


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
