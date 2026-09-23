import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _matmul_kernel(
    A,
    B,
    Bias,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    SAM: tl.constexpr,
    SAK: tl.constexpr,
    SBK: tl.constexpr,
    SBN: tl.constexpr,
    SBI: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    ACT: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    ks = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for block in range(tl.cdiv(K, BK)):
        kk = block * BK + ks
        a = tl.load(
            A + rows[:, None] * SAM + kk[None, :] * SAK,
            (rows[:, None] < M) & (kk[None, :] < K),
            other=0,
        )
        b = tl.load(
            B + kk[:, None] * SBK + cols[None, :] * SBN,
            (kk[:, None] < K) & (cols[None, :] < N),
            other=0,
        )
        acc += tl.dot(a, b, input_precision="ieee")
    value = acc.to(C.dtype.element_ty).to(tl.float32)
    if HAS_BIAS:
        bias = tl.load(Bias + cols * SBI, cols < N, other=0).to(tl.float32)
        value = (value + bias[None, :]).to(C.dtype.element_ty).to(tl.float32)
    if ACT == 1:
        value = 0.5 * value * (1.0 + libdevice.erf(value * 0.7071067811865476))
    elif ACT == 2:
        value = (
            0.5
            * value
            * (
                1.0
                + libdevice.tanh(0.7978845608028654 * (value + 0.044715 * value * value * value))
            )
        )
    tl.store(
        C + rows[:, None] * N + cols[None, :],
        value,
        (rows[:, None] < M) & (cols[None, :] < N),
    )


def launch(a, b, bias, out, approximate, config):
    import triton

    m, k = a.shape
    _, n = b.shape
    act = 0 if bias is None else (1 if approximate == "none" else 2)
    grid = (triton.cdiv(m, config.block_m), triton.cdiv(n, config.block_n))
    _matmul_kernel[grid](
        a,
        b,
        bias if bias is not None else a,
        out,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        bias.stride(0) if bias is not None else 0,
        bias is not None,
        act,
        config.block_m,
        config.block_n,
        config.block_k,
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )
    return out
