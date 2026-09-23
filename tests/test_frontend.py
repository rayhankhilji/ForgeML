import pytest
import torch
from torch import nn

from forgeml.compiler import compile
from forgeml.frontend import UnsupportedOperator, from_torch
from forgeml.ir import GraphError


class LinearOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(8, 4)

    def forward(self, x):
        return self.lin(x)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 4))

    def forward(self, x):
        return self.net(x)


class SkipAdd(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(8, 8)

    def forward(self, x):
        return x + torch.relu(self.lin(x))


class Buffered(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("scale", torch.full((4,), 2.0))
        self.lin = nn.Linear(8, 4)

    def forward(self, x):
        return self.lin(x) * self.scale


class ScalarOps(nn.Module):
    def forward(self, x):
        return (x + 1.5) * 2


class Shapes(nn.Module):
    def forward(self, x):
        y = x.view(4, 2)
        return torch.softmax(y.transpose(0, 1), dim=1)


class TupleOut(nn.Module):
    def forward(self, x):
        return torch.relu(x), torch.neg(x) if False else x * 2


class SingleTupleOut(nn.Module):
    def forward(self, x):
        return (torch.relu(x),)


class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)


class InPlace(nn.Module):
    def forward(self, x):
        return x.relu_()


class ControlFlow(nn.Module):
    def forward(self, x):
        if x.sum() > 0:
            return x
        return -x


def parity(model, *xs):
    model.eval()
    g = from_torch(model, xs)
    c = compile(g)
    expected = model(*xs)
    actual = c(*xs)
    if isinstance(expected, tuple):
        for e, a in zip(expected, actual):
            torch.testing.assert_close(a, e)
    else:
        torch.testing.assert_close(actual, expected)
    return g


def test_linear_parity():
    parity(LinearOnly(), torch.randn(2, 8))


def test_mlp_parity_gelu_exact_and_tanh():
    parity(MLP(), torch.randn(2, 8))
    m = nn.Sequential(nn.Linear(8, 16), nn.GELU(approximate="tanh"), nn.Linear(16, 4))
    m.eval()
    parity(m, torch.randn(2, 8))


def test_root_linear_module():
    lin = nn.Linear(8, 4)
    lin.eval()
    with torch.no_grad():
        lin.bias.copy_(torch.arange(4, dtype=torch.float32) + 1)
    x = torch.randn(2, 8)
    g = from_torch(lin, (x,))
    ops = [n.op for n in g.nodes]
    assert "matmul" in ops and "add" in ops
    c = compile(g)
    torch.testing.assert_close(c(x), lin(x))


def test_root_linear_no_bias():
    lin = nn.Linear(8, 4, bias=False)
    lin.eval()
    x = torch.randn(2, 8)
    c = compile(lin, (x,))
    torch.testing.assert_close(c(x), lin(x))


def test_functional_linear_bias_variants():
    class FL(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.randn(4, 8))
            self.b = nn.Parameter(torch.randn(4))

        def forward(self, x):
            return torch.nn.functional.linear(x, self.w, self.b)

    parity(FL(), torch.randn(2, 8))

    class FLNone(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.randn(4, 8))

        def forward(self, x):
            return torch.nn.functional.linear(x, self.w, None)

    parity(FLNone(), torch.randn(2, 8))

    class FLKw(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.randn(4, 8))
            self.b = nn.Parameter(torch.randn(4))

        def forward(self, x):
            return torch.nn.functional.linear(input=x, weight=self.w, bias=self.b)

    parity(FLKw(), torch.randn(2, 8))


def test_skip_add_branch():
    parity(SkipAdd(), torch.randn(2, 8))


def test_buffers_become_constants():
    g = parity(Buffered(), torch.randn(2, 8))
    assert any(t.eq(2.0).all() for t in g.constants.values())


def test_scalar_add_mul_promotion():
    g = parity(ScalarOps(), torch.randn(4))
    assert g.nodes  # scalar became a constant


def test_reshape_transpose_softmax():
    parity(Shapes(), torch.randn(2, 4))


def test_tensor_vs_tuple_outputs():
    g = from_torch(LinearOnly(), (torch.randn(2, 8),))
    assert not g.output_is_tuple
    g2 = from_torch(TupleOut(), (torch.randn(4),))
    assert g2.output_is_tuple and len(g2.outputs) == 2
    g3 = from_torch(SingleTupleOut(), (torch.randn(4),))
    assert g3.output_is_tuple and len(g3.outputs) == 1


def test_input_count_and_type_errors():
    c = compile(LinearOnly(), (torch.randn(2, 8),))
    with pytest.raises(GraphError, match="expected 1 inputs"):
        c(torch.randn(2, 8), torch.randn(2, 8))
    with pytest.raises(GraphError, match="shape"):
        c(torch.randn(3, 8))
    with pytest.raises(GraphError, match="dtype"):
        c(torch.randn(2, 8, dtype=torch.float64))


def test_unsupported_sin():
    with pytest.raises(UnsupportedOperator, match="sin"):
        from_torch(Sin(), (torch.randn(4),))


def test_inplace_op_rejected():
    with pytest.raises(UnsupportedOperator):
        from_torch(InPlace(), (torch.randn(4),))


def test_dynamic_control_flow_rejected():
    with pytest.raises(GraphError):
        from_torch(ControlFlow(), (torch.randn(4),))


def test_training_mode_dropout_rejected():
    m = nn.Sequential(nn.Linear(8, 8), nn.Dropout(0.5))
    m.train()
    with pytest.raises(GraphError, match="training mode"):
        from_torch(m, (torch.randn(2, 8),))


def test_weight_snapshot_isolated_from_mutation():
    m = LinearOnly()
    x = torch.randn(2, 8)
    c = compile(m, (x,))
    before = c(x)
    with torch.no_grad():
        m.lin.weight.fill_(0.0)
        m.lin.bias.fill_(0.0)
    after = c(x)
    torch.testing.assert_close(before, after)


def test_functional_linear_and_mm():
    class F(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.randn(4, 8))

        def forward(self, x):
            return torch.nn.functional.linear(x, self.w) @ self.w

    parity(F(), torch.randn(2, 8))


def test_reshape_minus_one_and_torch_reshape():
    class R(nn.Module):
        def forward(self, x):
            return x.reshape((2, -1))

    parity(R(), torch.randn(2, 4))

    class R2(nn.Module):
        def forward(self, x):
            return torch.reshape(x, (2, 4))

    parity(R2(), torch.randn(4, 2))


def test_gelu_approximate_kwarg():
    class G(nn.Module):
        def forward(self, x):
            return torch.nn.functional.gelu(x, approximate="tanh")

    parity(G(), torch.randn(4))


def test_relu_inplace_positional_rejected():
    class R(nn.Module):
        def forward(self, x):
            return torch.nn.functional.relu(x, True)

    with pytest.raises(UnsupportedOperator):
        from_torch(R(), (torch.randn(4),))


def test_softmax_kwarg_and_dtype_reject():
    class S(nn.Module):
        def forward(self, x):
            return x.softmax(dim=1)

    parity(S(), torch.randn(2, 4))

    class SD(nn.Module):
        def forward(self, x):
            return torch.softmax(x, dim=1, dtype=torch.float64)

    with pytest.raises(UnsupportedOperator):
        from_torch(SD(), (torch.randn(2, 4),))


def test_t_rank3_rejected():
    class T(nn.Module):
        def forward(self, x):
            return x.t()

    with pytest.raises(UnsupportedOperator, match="rank-2"):
        from_torch(T(), (torch.randn(2, 3, 4),))


def test_scalar_int_promotion():
    class S(nn.Module):
        def forward(self, x):
            return x + 3

    g = parity(S(), torch.randn(4))
    assert g.nodes[0].spec.dtype == torch.float32
