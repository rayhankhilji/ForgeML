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
