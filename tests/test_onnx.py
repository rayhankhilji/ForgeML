import numpy as np
import pytest
import torch

onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper, numpy_helper

from forgeml.compiler import compile
from forgeml.frontend import UnsupportedOperator, from_onnx
from forgeml.ir import GraphError


def make_model(nodes, inputs, outputs, initializers=(), opset=20, domain=""):
    g = helper.make_graph(nodes, "g", inputs, outputs, initializer=list(initializers))
    m = helper.make_model(g, opset_imports=[helper.make_opsetid(domain, opset)])
    m.ir_version = 10
    return m


def vi(name, dims, dtype=TensorProto.FLOAT):
    return helper.make_tensor_value_info(name, dtype, dims)


def test_matmul_add_gelu_parity():
    w = numpy_helper.from_array(np.random.randn(4, 8).astype(np.float32), "w")
    bias = numpy_helper.from_array(np.random.randn(8).astype(np.float32), "bias")
    nodes = [
        helper.make_node("MatMul", ["x", "w"], ["mm"]),
        helper.make_node("Add", ["mm", "bias"], ["lin"]),
        helper.make_node("Gelu", ["lin"], ["y"]),
    ]
    m = make_model(nodes, [vi("x", [2, 4])], [vi("y", [2, 8])], [w, bias])
    g = from_onnx(m)
    x = torch.randn(2, 4)
    c = compile(g)
    expected = torch.nn.functional.gelu(
        x @ torch.tensor(numpy_helper.to_array(w)) + torch.tensor(numpy_helper.to_array(bias))
    )
    torch.testing.assert_close(c(x), expected)


def test_gemm_transpose_alpha_beta():
    a = np.random.randn(2, 4).astype(np.float32)
    b = np.random.randn(8, 4).astype(np.float32)  # transB -> (4,8)
    cb = np.random.randn(8).astype(np.float32)
    inits = [numpy_helper.from_array(t, n) for t, n in ((b, "b"), (cb, "c"))]
    nodes = [helper.make_node("Gemm", ["a", "b", "c"], ["y"], alpha=2.0, beta=0.5, transB=1)]
    m = make_model(nodes, [vi("a", [2, 4])], [vi("y", [2, 8])], inits)
    g = from_onnx(m)
    c = compile(g)
    xt = torch.from_numpy(a)
    expected = 2.0 * (xt @ torch.from_numpy(b).t()) + 0.5 * torch.from_numpy(cb)
    torch.testing.assert_close(c(xt), expected)


def test_reshape_zero_copies_input_dim():
    shape = numpy_helper.from_array(np.array([0, -1], dtype=np.int64), "shape")
    nodes = [helper.make_node("Reshape", ["x", "shape"], ["y"], allowzero=0)]
    m = make_model(nodes, [vi("x", [2, 6])], [vi("y", [2, 6])], [shape])
    g = from_onnx(m)
    reshape_node = next(n for n in g.nodes if n.op == "reshape")
    assert reshape_node.attrs["shape"] == (2, 6)


def test_example_inputs_override_dims():
    nodes = [helper.make_node("Relu", ["x"], ["y"])]
    m = make_model(nodes, [vi("x", ["batch", 4])], [vi("y", ["batch", 4])])
    x = torch.randn(3, 4)
    g = from_onnx(m, (x,))
    assert g.inputs["x"].shape == (3, 4)


def test_dynamic_dims_rejected_without_examples():
    nodes = [helper.make_node("Relu", ["x"], ["y"])]
    m = make_model(nodes, [vi("x", ["batch", 4])], [vi("y", ["batch", 4])])
    with pytest.raises(GraphError, match="dynamic"):
        from_onnx(m)


def test_unsupported_op_rejected():
    nodes = [helper.make_node("Conv", ["x", "w"], ["y"])]
    w = numpy_helper.from_array(np.zeros((1, 1, 3, 3), dtype=np.float32), "w")
    m = make_model(nodes, [vi("x", [1, 1, 5, 5])], [vi("y", [1, 1, 3, 3])], [w])
    with pytest.raises(UnsupportedOperator, match="Conv"):
        from_onnx(m)


def test_nondefault_domain_rejected():
    nodes = [helper.make_node("Relu", ["x"], ["y"])]
    m = make_model(nodes, [vi("x", [4])], [vi("y", [4])])
    m.opset_import.add(domain="com.microsoft", version=1)
    with pytest.raises(UnsupportedOperator, match="domain"):
        from_onnx(m)


def test_external_data_rejected():
    t = onnx.TensorProto()
    t.name = "w"
    t.dims.extend([2, 2])
    t.data_type = TensorProto.FLOAT
    t.data_location = onnx.TensorProto.EXTERNAL
    t.external_data.add(key="location", value="weights.bin")
    nodes = [helper.make_node("Add", ["x", "w"], ["y"])]
    m = make_model(nodes, [vi("x", [2, 2])], [vi("y", [2, 2])], [t])
    with pytest.raises(UnsupportedOperator, match="external data"):
        from_onnx(m)


def test_tuple_output_preserved():
    w = numpy_helper.from_array(np.eye(4, dtype=np.float32), "w")
    nodes = [
        helper.make_node("MatMul", ["x", "w"], ["mm"]),
        helper.make_node("Relu", ["mm"], ["y"]),
    ]
    m = make_model(nodes, [vi("x", [2, 4])], [vi("y", [2, 4]), vi("mm", [2, 4])], [w])
    g = from_onnx(m)
    assert g.output_is_tuple and len(g.outputs) == 2


def test_allowzero1_rejected():
    shape = numpy_helper.from_array(np.array([0, -1], dtype=np.int64), "shape")
    nodes = [helper.make_node("Reshape", ["x", "shape"], ["y"], allowzero=1)]
    m = make_model(nodes, [vi("x", [2, 6])], [vi("y", [2, 6])], [shape])
    with pytest.raises(UnsupportedOperator, match="allowzero"):
        from_onnx(m)


def test_scalar_constant_shape():
    c = numpy_helper.from_array(np.array(2.5, dtype=np.float32), "c")
    nodes = [helper.make_node("Add", ["x", "c"], ["y"])]
    m = make_model(nodes, [vi("x", [4])], [vi("y", [4])], [c])
    g = from_onnx(m)
    assert g.constants["c"].shape == ()
    x = torch.ones(4)
    torch.testing.assert_close(compile(g)(x), x + 2.5)


def test_static_dim_override_rejected():
    nodes = [helper.make_node("Relu", ["x"], ["y"])]
    m = make_model(nodes, [vi("x", [2, 4])], [vi("y", [2, 4])])
    with pytest.raises(GraphError, match="fixed ONNX dim"):
        from_onnx(m, (torch.randn(3, 4),))


def test_duplicate_output_rejected():
    w = numpy_helper.from_array(np.eye(2, dtype=np.float32), "w")
    nodes = [
        helper.make_node("Relu", ["x"], ["y"]),
        helper.make_node("Identity", ["y"], ["y"]),
    ]
    m = make_model(nodes, [vi("x", [2, 2])], [vi("y", [2, 2])], [w])
    with pytest.raises(GraphError, match="duplicate"):
        from_onnx(m)


def test_output_metadata_mismatch_rejected():
    nodes = [helper.make_node("Relu", ["x"], ["y"])]
    m = make_model(nodes, [vi("x", [2, 4])], [vi("y", [2, 8])])
    with pytest.raises(GraphError, match="ONNX output"):
        from_onnx(m)
