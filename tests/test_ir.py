import pytest
import torch

from forgeml.ir import Graph, GraphBuilder, GraphError, Node, TensorSpec


def spec(*shape):
    return TensorSpec(tuple(shape), torch.float32, "cpu")


def test_tensorspec_rejects_nonpositive_dims():
    with pytest.raises(GraphError):
        TensorSpec((0, 4), torch.float32, "cpu")
    with pytest.raises(GraphError):
        TensorSpec((-1, 4), torch.float32, "cpu")


def test_tensorspec_nbytes():
    assert TensorSpec((2, 3), torch.float32, "cpu").nbytes == 24
    assert TensorSpec((4,), torch.float16, "cpu").nbytes == 8


def test_builder_rejects_duplicate_names():
    b = GraphBuilder({"x": spec(2, 3)})
    with pytest.raises(GraphError):
        b.constant("x", torch.ones(1))
    b.constant("c", torch.ones(1))
    with pytest.raises(GraphError):
        b.add("c", "relu", ("x",))


def test_builder_rejects_missing_input():
    b = GraphBuilder({"x": spec(2, 3)})
    with pytest.raises(GraphError, match="not defined"):
        b.add("y", "relu", ("nope",))


def test_builder_rejects_unknown_op():
    b = GraphBuilder({"x": spec(2, 3)})
    with pytest.raises(GraphError, match="unsupported op"):
        b.add("y", "sin", ("x",))


def test_builder_rejects_shape_mismatch():
    b = GraphBuilder({"x": spec(2, 3), "w": spec(5, 3)})
    with pytest.raises(GraphError, match="matmul shape mismatch"):
        b.add("y", "matmul", ("x", "w"))


def test_builder_infers_specs():
    b = GraphBuilder({"x": spec(2, 3)})
    b.constant("w", torch.ones(3, 4))
    b.add("mm", "matmul", ("x", "w"))
    g = b.finish(("mm",))
    assert g.specs()["mm"] == TensorSpec((2, 4), torch.float32, "cpu")


def test_linear_and_unbiased_fused_specs():
    b = GraphBuilder({"x": spec(2, 3)})
    b.constant("w", torch.ones(3, 4))
    b.constant("bias", torch.ones(4))
    b.add("lin", "linear", ("x", "w", "bias"))
    b.add("act", "fused_linear_gelu", ("x", "w"), approximate="tanh")
    g = b.finish(("lin", "act"))
    assert g.specs()["lin"].shape == (2, 4)
    assert g.specs()["act"].shape == (2, 4)
    with pytest.raises(GraphError, match="linear bias"):
        b.add("bad", "linear", ("x", "w", "w"))


def test_explicit_graph_validation():
    g = Graph(
        inputs={"x": spec(2, 3)},
        constants={},
        nodes=[Node("y", "relu", ("missing",), {}, spec(2, 3))],
        outputs=("y",),
    )
    with pytest.raises(GraphError, match="not defined"):
        g.validate()


def test_explicit_graph_rejects_wrong_spec():
    g = Graph(
        inputs={"x": spec(2, 3)},
        constants={},
        nodes=[Node("y", "relu", ("x",), {}, spec(9, 9))],
        outputs=("y",),
    )
    with pytest.raises(GraphError, match="spec mismatch"):
        g.validate()


def test_graph_requires_nonempty_outputs():
    g = Graph(inputs={"x": spec(2)}, constants={}, nodes=[], outputs=())
    with pytest.raises(GraphError, match="non-empty"):
        g.validate()


def test_graph_rejects_duplicate_names():
    g = Graph(
        inputs={"x": spec(2)},
        constants={"x": torch.ones(2)},
        nodes=[],
        outputs=("x",),
    )
    with pytest.raises(GraphError, match="duplicate"):
        g.validate()


def test_graph_topological_order():
    g = Graph(
        inputs={"x": spec(2, 3)},
        constants={},
        nodes=[
            Node("b", "relu", ("a",), {}, spec(2, 3)),
            Node("a", "relu", ("x",), {}, spec(2, 3)),
        ],
        outputs=("b",),
    )
    with pytest.raises(GraphError, match="not defined"):
        g.validate()


def test_clone_independent():
    b = GraphBuilder({"x": spec(2)})
    b.constant("c", torch.ones(2))
    b.add("y", "add", ("x", "c"))
    g = b.finish(("y",))
    c = g.clone()
    c.constants["c"].fill_(9.0)
    assert g.constants["c"].sum().item() == 2.0
    assert c is not g


def test_to_dict_and_mermaid():
    b = GraphBuilder({"x": spec(2)})
    b.add("y", "relu", ("x",))
    g = b.finish(("y",))
    d = g.to_dict()
    assert d["nodes"][0]["op"] == "relu"
    assert d["inputs"]["x"]["shape"] == [2]
    m = g.to_mermaid()
    assert "relu" in m and "-->" in m


def test_invalid_device_string():
    with pytest.raises(GraphError):
        TensorSpec((2,), torch.float32, "not-a-device")


def test_device_mismatch_rejected():
    b = GraphBuilder(
        {
            "x": TensorSpec((2, 4), torch.float32, "cpu"),
            "w": TensorSpec((4, 8), torch.float32, "cuda:0"),
        }
    )
    with pytest.raises(GraphError, match="device mismatch"):
        b.add("y", "matmul", ("x", "w"))


def test_cpu_scalar_allowed_with_other_device():
    b = GraphBuilder({"x": TensorSpec((2, 4), torch.float32, "cuda:0")})
    b.constant("s", torch.tensor(2.0))
    b.add("y", "add", ("x", "s"))
    g = b.finish(("y",))
    assert g.nodes[0].spec.device == "cuda:0"


def test_multiple_outputs_require_tuple_flag():
    g = Graph(
        inputs={"x": spec(2)},
        constants={},
        nodes=[Node("y", "relu", ("x",), {}, spec(2))],
        outputs=("x", "y"),
        output_is_tuple=False,
    )
    with pytest.raises(GraphError, match="output_is_tuple"):
        g.validate()


def test_builder_multi_output_sets_tuple_flag():
    b = GraphBuilder({"x": spec(2)})
    b.add("y", "relu", ("x",))
    g = b.finish(("x", "y"))
    assert g.output_is_tuple


def test_missing_required_attr_grapherror():
    b = GraphBuilder({"x": spec(2, 4)})
    with pytest.raises(GraphError, match="requires attrs"):
        b.add("y", "softmax", ("x",))


def test_mermaid_unique_ids_and_label_escaping():
    g = Graph(
        inputs={"a-b": spec(2), "a_b": spec(2)},
        constants={},
        nodes=[Node('x"y\n1', "add", ("a-b", "a_b"), {}, spec(2))],
        outputs=('x"y\n1',),
    )
    g.validate()
    m = g.to_mermaid()
    assert "a-b" in m and "a_b" in m
    assert "#quot;" in m
    assert '"x"y' not in m
    import re as _re

    ids = _re.findall(r"^    (v\d+)\[", m, _re.MULTILINE)
    assert len(ids) == len(set(ids)) == 3


def test_neural_operator_specs():
    b = GraphBuilder(
        {
            "x": spec(2, 4),
            "q": spec(1, 2, 3, 4),
            "k": spec(1, 2, 3, 4),
            "v": spec(1, 2, 3, 4),
            "image": spec(2, 3, 8, 8),
            "ids": TensorSpec((2, 5), torch.int64, "cpu"),
        }
    )
    b.constant("ln_w", torch.ones(4))
    b.constant("ln_b", torch.zeros(4))
    b.constant("conv_w", torch.randn(6, 3, 3, 3))
    b.constant("emb_w", torch.randn(10, 7))
    b.add("ln", "layer_norm", ("x", "ln_w", "ln_b"), normalized_shape=(4,), eps=1e-5)
    b.add("attn", "sdpa", ("q", "k", "v"), is_causal=True)
    b.add(
        "conv",
        "conv2d",
        ("image", "conv_w"),
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
    )
    b.add("emb", "embedding", ("ids", "emb_w"))
    g = b.finish(("ln", "attn", "conv", "emb"))
    assert g.specs()["attn"].shape == (1, 2, 3, 4)
    assert g.specs()["conv"].shape == (2, 6, 8, 8)
    assert g.specs()["emb"].shape == (2, 5, 7)


def test_layer_norm_accepts_scalar_normalized_shape():
    b = GraphBuilder({"x": spec(2, 4)})
    b.add("ln", "layer_norm", ("x",), normalized_shape=4, eps=1e-5)
    g = b.finish(("ln",))
    assert g.specs()["ln"].shape == (2, 4)


def test_neural_operator_rejections():
    b = GraphBuilder({"x": spec(2, 4), "q": spec(1, 2, 3, 4), "image": spec(2, 3, 8, 8)})
    with pytest.raises(GraphError, match="does not end"):
        b.add("ln", "layer_norm", ("x",), normalized_shape=(5,), eps=1e-5)
    with pytest.raises(GraphError, match="rank-4"):
        b.add("attn", "sdpa", ("q", "q", "x"), is_causal=True)
    b.constant("bad_conv", torch.ones(4, 2, 3, 3))
    with pytest.raises(GraphError, match="channels/groups"):
        b.add(
            "conv",
            "conv2d",
            ("image", "bad_conv"),
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
        )
    b.constant("emb_w", torch.ones(5, 3))
    with pytest.raises(GraphError, match="int64"):
        b.add("emb", "embedding", ("x", "emb_w"))


def test_narrow_spec_and_bounds():
    b = GraphBuilder({"x": spec(2, 8)})
    b.add("n", "narrow", ("x",), dim=1, start=2, length=3)
    g = b.finish(("n",))
    assert g.specs()["n"].shape == (2, 3)
    b2 = GraphBuilder({"x": spec(2, 8)})
    b2.add("n", "narrow", ("x",), dim=-1, start=0, length=4)
    assert b2.finish(("n",)).specs()["n"].shape == (2, 4)
    with pytest.raises(GraphError, match="narrow range"):
        b3 = GraphBuilder({"x": spec(2, 8)})
        b3.add("n", "narrow", ("x",), dim=1, start=6, length=4)
    with pytest.raises(GraphError, match="narrow dim"):
        b4 = GraphBuilder({"x": spec(2, 8)})
        b4.add("n", "narrow", ("x",), dim=2, start=0, length=1)
    with pytest.raises(GraphError, match="narrow range"):
        b5 = GraphBuilder({"x": spec(2, 8)})
        b5.add("n", "narrow", ("x",), dim=1, start=0, length=0)


def test_float_only_ops_reject_integer_inputs():
    b = GraphBuilder({"x": TensorSpec((2, 4), torch.int64, "cpu")})
    with pytest.raises(GraphError, match="floating-point"):
        b.add("g", "gelu", ("x",))
    with pytest.raises(GraphError, match="floating-point"):
        b.add("s", "softmax", ("x",), dim=1)


def test_node_attrs_must_be_dict():
    g = Graph(
        inputs={"x": spec(2)},
        constants={},
        nodes=[Node("y", "relu", ("x",), None, spec(2))],
        outputs=("y",),
    )
    with pytest.raises(GraphError, match="attrs must be a dict"):
        g.validate()


def test_indexless_cuda_device_normalizes():
    assert TensorSpec((2,), torch.float32, "cuda").device == "cuda:0"
    assert TensorSpec((2,), torch.float32, "cuda:1").device == "cuda:1"
    assert TensorSpec((2,), torch.float32, "cpu").device == "cpu"
