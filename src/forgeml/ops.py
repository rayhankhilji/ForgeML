from __future__ import annotations

import torch
import torch.nn.functional as F

SUPPORTED_OPS = {
    "matmul",
    "linear",
    "add",
    "mul",
    "relu",
    "gelu",
    "reshape",
    "transpose",
    "softmax",
    "narrow",
    "fused_linear_gelu",
    "layer_norm",
    "sdpa",
    "conv2d",
    "embedding",
}


def evaluate(op: str, args: tuple[torch.Tensor, ...], attrs: dict) -> torch.Tensor:
    if op == "matmul":
        return torch.matmul(args[0], args[1])
    if op == "linear":
        if len(args) == 3:
            return torch.addmm(args[2], args[0], args[1])
        return torch.mm(args[0], args[1])
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
    if op == "narrow":
        return torch.narrow(args[0], attrs["dim"] % args[0].dim(), attrs["start"], attrs["length"])
    if op == "fused_linear_gelu":
        if len(args) == 3:
            out = torch.addmm(args[2], args[0], args[1])
        else:
            out = torch.mm(args[0], args[1])
        return F.gelu(out, approximate=attrs.get("approximate", "none"))
    if op == "layer_norm":
        weight = args[1] if len(args) == 3 else None
        bias = args[2] if len(args) == 3 else None
        normalized = attrs["normalized_shape"]
        if isinstance(normalized, int):
            normalized = (normalized,)
        return F.layer_norm(args[0], tuple(normalized), weight, bias, attrs["eps"])
    if op == "sdpa":
        return F.scaled_dot_product_attention(
            args[0],
            args[1],
            args[2],
            dropout_p=0.0,
            is_causal=attrs["is_causal"],
            scale=attrs.get("scale"),
        )
    if op == "conv2d":
        return F.conv2d(
            args[0],
            args[1],
            args[2] if len(args) == 3 else None,
            stride=attrs["stride"],
            padding=attrs["padding"],
            dilation=attrs["dilation"],
            groups=attrs["groups"],
        )
    if op == "embedding":
        return F.embedding(args[0], args[1])
    raise ValueError(f"unsupported op {op!r}")
