from __future__ import annotations

import torch
import torch.nn.functional as F

SUPPORTED_OPS = {
    "matmul",
    "add",
    "mul",
    "relu",
    "gelu",
    "reshape",
    "transpose",
    "softmax",
    "fused_linear_gelu",
}


def evaluate(op: str, args: tuple[torch.Tensor, ...], attrs: dict) -> torch.Tensor:
    if op == "matmul":
        return torch.matmul(args[0], args[1])
    if op == "add":
        return torch.add(args[0], args[1])
    if op == "mul":
        return torch.mul(args[0], args[1])
    if op == "relu":
        return F.relu(args[0])
    if op == "gelu":
        return F.gelu(args[0], approximate=attrs.get("approximate", "none"))
    if op == "reshape":
        return torch.reshape(args[0], tuple(attrs["shape"]))
    if op == "transpose":
        return torch.transpose(args[0], attrs["dim0"], attrs["dim1"])
    if op == "softmax":
        return torch.softmax(args[0], dim=attrs["dim"])
    if op == "fused_linear_gelu":
        out = torch.add(torch.matmul(args[0], args[1]), args[2])
        return F.gelu(out, approximate=attrs.get("approximate", "none"))
    raise ValueError(f"unsupported op {op!r}")
