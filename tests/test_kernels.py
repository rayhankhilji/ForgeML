import pytest
import torch
from torch import nn

from forgeml.compiler import compile
from forgeml.ir import GraphBuilder, GraphError, TensorSpec
from forgeml.kernels import (
    CANDIDATES,
    DEFAULT_CONFIG,
    KernelConfig,
    available,
    kernel_source,
    matmul,
    select_kernels,
)

gpu = pytest.mark.gpu
_has_cuda = torch.cuda.is_available()


def cuda_required():
    if not _has_cuda:
        pytest.skip("CUDA not available")
    pytest.importorskip("triton")


def spec(*shape, device="cpu"):
    return TensorSpec(tuple(shape), torch.float32, device)


_DTYPES_TOL = [
    (torch.float32, 1e-4, 1e-4),
    (torch.float16, 1e-2, 1e-2),
    (torch.bfloat16, 5e-2, 5e-2),
]


def _check_dtype(dtype):
    if dtype == torch.bfloat16 and torch.cuda.get_device_capability() < (8, 0):
        pytest.skip("bf16 requires capability >= 8.0")


def test_lazy_import_and_source():
    assert available() is (
        _has_cuda and __import__("importlib").util.find_spec("triton") is not None
    )
    src = kernel_source()
    assert "@triton.jit" in src and "_matmul_kernel" in src
    assert isinstance(DEFAULT_CONFIG, KernelConfig)
    assert len(CANDIDATES) == 4


def test_config_validation_cpu():
    a = torch.randn(4, 4)
    with pytest.raises(GraphError, match="CUDA"):
        matmul(a, a)
    with pytest.raises(GraphError):
        matmul(a, a, config=KernelConfig(block_m=8))


def test_select_kernels_cpu_auto_and_triton_error():
    b = GraphBuilder({"x": spec(2, 4)})
    b.constant("w", torch.randn(4, 8))
    b.add("mm", "matmul", ("x", "w"))
    g = b.finish(("mm",))
    plan = select_kernels(g, "auto")
    assert plan == {"mm": "torch"}
    assert select_kernels(g, "torch") == {"mm": "torch"}
    with pytest.raises(GraphError, match="triton"):
        select_kernels(g, "triton")
    with pytest.raises(GraphError, match="backend"):
        select_kernels(g, "bogus")


def test_compile_auto_on_cpu_parity():
    m = nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 4)).eval()
    x = torch.randn(2, 8)
    c = compile(m, (x,), backend="auto")
    assert set(c.kernel_plan.values()) == {"torch"}
    torch.testing.assert_close(c(x), m(x))


def test_compile_triton_backend_cpu_error():
    m = nn.Linear(8, 4).eval()
    with pytest.raises(GraphError):
        compile(m, (torch.randn(2, 8),), backend="triton")


def test_scalar_device_choice():
    b = GraphBuilder(
        {
            "s_cpu": TensorSpec((), torch.float32, "cpu"),
            "s_gpu": TensorSpec((), torch.float32, "cuda:0"),
        }
    )
    b.add("y", "add", ("s_cpu", "s_gpu"))
    g = b.finish(("y",))
    assert g.nodes[0].spec.device == "cuda:0"


@gpu
@pytest.mark.parametrize("dtype,rtol,atol", _DTYPES_TOL)
@pytest.mark.parametrize("m,n,k", [(1, 17, 9), (31, 65, 33), (64, 64, 64)])
def test_matmul_gpu(dtype, rtol, atol, m, n, k):
    cuda_required()
    _check_dtype(dtype)
    a = torch.randn(m, k, dtype=dtype, device="cuda")
    b = torch.randn(k, n, dtype=dtype, device="cuda")
    got = matmul(a, b)
    expected = torch.matmul(a.float(), b.float()).to(dtype)
    torch.testing.assert_close(got.float(), expected.float(), rtol=rtol, atol=atol)


@gpu
@pytest.mark.parametrize("dtype,rtol,atol", _DTYPES_TOL)
@pytest.mark.parametrize("approximate", ["none", "tanh"])
def test_bias_gelu_gpu(dtype, rtol, atol, approximate):
    cuda_required()
    _check_dtype(dtype)
    a = torch.randn(16, 32, dtype=dtype, device="cuda")
    b = torch.randn(32, 24, dtype=dtype, device="cuda")
    bias = torch.randn(24, dtype=dtype, device="cuda")
    got = matmul(a, b, bias, approximate=approximate)
    expected = torch.nn.functional.gelu(
        (a.float() @ b.float()).to(dtype) + bias, approximate=approximate
    )
    torch.testing.assert_close(got.float(), expected.float(), rtol=rtol, atol=atol)


@gpu
@pytest.mark.parametrize("approximate", ["none", "tanh"])
def test_strided_bias_gpu(approximate):
    cuda_required()
    a = torch.randn(16, 32, device="cuda")
    b = torch.randn(32, 24, device="cuda")
    bias = torch.randn(48, device="cuda")[::2]
    got = matmul(a, b, bias, approximate=approximate)
    expected = torch.nn.functional.gelu(a @ b + bias, approximate=approximate)
    torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-4)


@gpu
def test_noncontiguous_inputs_gpu():
    cuda_required()
    a = torch.randn(32, 16, device="cuda").t()
    b = torch.randn(24, 32, device="cuda").t()
    got = matmul(a, b)
    torch.testing.assert_close(got, a @ b, rtol=1e-4, atol=1e-4)


@gpu
def test_out_buffer_and_overlap_reject():
    cuda_required()
    a = torch.randn(16, 16, device="cuda")
    b = torch.randn(16, 16, device="cuda")
    out = torch.empty(16, 16, device="cuda")
    got = matmul(a, b, out=out)
    assert got.data_ptr() == out.data_ptr()
    torch.testing.assert_close(out, a @ b, rtol=1e-4, atol=1e-4)
    with pytest.raises(GraphError, match="storage"):
        matmul(a, b, out=a)


@gpu
def test_compiled_mixed_plan_gpu():
    cuda_required()

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(16, 16)

        def forward(self, x):
            return torch.relu(self.lin(x)) + x

    m = M().eval().cuda()
    x = torch.randn(4, 16, device="cuda")
    c = compile(m, (x,), backend="auto")
    assert "triton" in c.kernel_plan.values()
    assert "torch" in c.kernel_plan.values()
    torch.testing.assert_close(c(x), m(x), rtol=1e-4, atol=1e-4)
