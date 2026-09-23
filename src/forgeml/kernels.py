from __future__ import annotations

import importlib.resources
import importlib.util
from dataclasses import dataclass

import torch

from forgeml.ir import Graph, GraphError


@dataclass(frozen=True)
class KernelConfig:
    block_m: int = 32
    block_n: int = 64
    block_k: int = 32
    num_warps: int = 4
    num_stages: int = 2


DEFAULT_CONFIG = KernelConfig()
CANDIDATES = (
    KernelConfig(16, 32, 32, 4, 2),
    KernelConfig(32, 64, 32, 4, 2),
    KernelConfig(64, 64, 32, 4, 3),
    KernelConfig(32, 128, 32, 8, 3),
)

_TILES = {16, 32, 64, 128}
_WARPS = {4, 8}
_STAGES = {2, 3, 4}
_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


def available() -> bool:
    return torch.cuda.is_available() and importlib.util.find_spec("triton") is not None


def _check_config(config: KernelConfig) -> None:
    if not isinstance(config, KernelConfig):
        raise GraphError(f"config must be a KernelConfig, got {type(config).__name__}")
    for name in ("block_m", "block_n", "block_k"):
        v = getattr(config, name)
        if not isinstance(v, int) or v not in _TILES:
            raise GraphError(f"{name} must be one of {sorted(_TILES)}, got {v!r}")
    if config.num_warps not in _WARPS:
        raise GraphError(f"num_warps must be one of {sorted(_WARPS)}")
    if config.num_stages not in _STAGES:
        raise GraphError(f"num_stages must be one of {sorted(_STAGES)}")


def matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    approximate: str = "none",
    config: KernelConfig = DEFAULT_CONFIG,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise GraphError("triton matmul requires CUDA")
    if importlib.util.find_spec("triton") is None:
        raise GraphError("triton is not installed")
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        raise GraphError("matmul inputs must be tensors")
    if not a.is_cuda or not b.is_cuda or a.device != b.device:
        raise GraphError("matmul inputs must be CUDA tensors on the same device")
    if a.dtype not in _DTYPES or a.dtype != b.dtype:
        raise GraphError(f"matmul dtype {a.dtype} not supported or mismatched")
    if a.dim() != 2 or b.dim() != 2:
        raise GraphError("matmul requires rank-2 inputs")
    m, k = a.shape
    k2, n = b.shape
    if k != k2 or m <= 0 or n <= 0 or k <= 0:
        raise GraphError(f"matmul incompatible shapes {a.shape} x {b.shape}")
    if approximate not in ("none", "tanh"):
        raise GraphError(f"approximate must be 'none' or 'tanh', got {approximate!r}")
    if bias is not None and (
        not isinstance(bias, torch.Tensor)
        or not bias.is_cuda
        or bias.device != a.device
        or bias.dtype != a.dtype
        or bias.dim() != 1
        or bias.shape[0] != n
    ):
        raise GraphError("bias must be a 1-D CUDA tensor of output width matching dtype/device")
    _check_config(config)
    if a.dtype == torch.bfloat16 and torch.cuda.get_device_capability(a.device) < (8, 0):
        raise GraphError("bf16 matmul requires CUDA capability >= 8.0")
    if out is None:
        out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    else:
        if (
            not isinstance(out, torch.Tensor)
            or tuple(out.shape) != (m, n)
            or out.dtype != a.dtype
            or out.device != a.device
            or not out.is_contiguous()
        ):
            raise GraphError(
                "out must be a contiguous tensor of shape (M, N) matching dtype/device"
            )
        out_ptr = out.untyped_storage().data_ptr()
        for src in (a, b, bias):
            if src is not None and src.untyped_storage().data_ptr() == out_ptr:
                raise GraphError("out must not share storage with inputs")
    from forgeml._triton import launch

    with torch.cuda.device(a.device):
        return launch(a, b, bias, out, approximate, config)


def select_kernels(graph: Graph, backend: str) -> dict[str, str]:
    if backend == "torch":
        return {n.name: "torch" for n in graph.nodes}
    if backend == "auto":
        plan = {}
        can = available()
        specs = graph.specs()
        for node in graph.nodes:
            if (
                can
                and node.op in ("matmul", "fused_linear_gelu")
                and node.spec.dtype in _DTYPES
                and all(torch.device(specs[i].device).type == "cuda" for i in node.inputs)
            ):
                if node.spec.dtype == torch.bfloat16 and torch.cuda.get_device_capability(
                    torch.device(specs[node.inputs[0]].device)
                ) < (8, 0):
                    plan[node.name] = "torch"
                    continue
                plan[node.name] = "triton"
            else:
                plan[node.name] = "torch"
        return plan
    if backend == "triton":
        if not available():
            raise GraphError("backend 'triton' requires CUDA and the triton package")
        specs = graph.specs()
        if not all(torch.device(s.device).type == "cuda" for s in graph.inputs.values()):
            raise GraphError("backend 'triton' requires CUDA graph inputs")
        plan = {}
        for node in graph.nodes:
            if node.op in ("matmul", "fused_linear_gelu") and node.spec.dtype in _DTYPES:
                plan[node.name] = "triton"
            else:
                plan[node.name] = "torch"
        return plan
    raise GraphError(f"unsupported backend {backend!r}; expected 'torch', 'triton', or 'auto'")


def kernel_source() -> str:
    return importlib.resources.files("forgeml").joinpath("_triton.py").read_text()
