from __future__ import annotations

import math
from collections import Counter

from forgeml.ir import Graph, Node, TensorSpec


def _numel(spec: TensorSpec) -> int:
    return math.prod(spec.shape)


def _node_flops(node: Node, inputs: list[TensorSpec]) -> int:
    out = node.spec
    elements = _numel(out)
    if node.op == "matmul":
        m, k = inputs[0].shape
        _, n = inputs[1].shape
        return 2 * m * k * n
    if node.op == "fused_linear_gelu":
        m, k = inputs[0].shape
        _, n = inputs[1].shape
        return 2 * m * k * n + elements + 8 * elements
    if node.op in ("add", "mul", "relu"):
        return elements
    if node.op == "gelu":
        return 8 * elements
    if node.op == "softmax":
        return 5 * elements
    if node.op == "layer_norm":
        return 6 * elements
    if node.op == "sdpa":
        _, heads, q_len, head_dim = inputs[0].shape
        _, _, kv_len, _ = inputs[1].shape
        batch = inputs[0].shape[0]
        pairs = batch * heads * q_len * kv_len
        if node.attrs.get("is_causal"):
            full = batch * heads * q_len * kv_len
            causal = batch * heads * ((q_len * (q_len + 1)) // 2)
            pairs = min(full, causal)
        return 2 * pairs * (2 * head_dim) + 5 * pairs
    if node.op == "conv2d":
        _, _, out_h, out_w = out.shape
        out_channels, in_per_group, kernel_h, kernel_w = inputs[1].shape
        batch = inputs[0].shape[0]
        mac_terms = out_channels * in_per_group * kernel_h * kernel_w
        flops = 2 * batch * out_h * out_w * mac_terms
        return flops + (elements if len(inputs) == 3 else 0)
    return 0


def graph_analysis(graph: Graph) -> dict:
    graph.validate()
    specs = graph.specs()
    node_rows = []
    depths: dict[str, int] = {}
    total_flops = 0
    total_bytes = 0
    for node in graph.nodes:
        input_specs = [specs[name] for name in node.inputs]
        flops = _node_flops(node, input_specs)
        input_bytes = sum(specs[name].nbytes for name in dict.fromkeys(node.inputs))
        logical_bytes = input_bytes + node.spec.nbytes
        depth = 1 + max((depths.get(name, 0) for name in node.inputs), default=0)
        depths[node.name] = depth
        row = {
            "name": node.name,
            "op": node.op,
            "flops": flops,
            "logical_bytes": logical_bytes,
            "arithmetic_intensity": flops / max(logical_bytes, 1),
            "depth": depth,
            "output_shape": list(node.spec.shape),
        }
        node_rows.append(row)
        total_flops += flops
        total_bytes += logical_bytes
    op_counts = Counter(node.op for node in graph.nodes)
    return {
        "model": "static_operator_cost_bounds_not_measured_hardware_counters",
        "nodes": len(graph.nodes),
        "op_counts": dict(sorted(op_counts.items())),
        "flops": total_flops,
        "logical_bytes": total_bytes,
        "arithmetic_intensity": total_flops / max(total_bytes, 1),
        "critical_path_ops": max(depths.values(), default=0),
        "input_bytes": sum(spec.nbytes for spec in graph.inputs.values()),
        "constant_bytes": sum(t.numel() * t.element_size() for t in graph.constants.values()),
        "output_bytes": sum(specs[name].nbytes for name in graph.outputs),
        "dominant_nodes": sorted(
            node_rows,
            key=lambda row: (row["flops"], row["logical_bytes"], row["name"]),
            reverse=True,
        )[:5],
        "per_node": node_rows,
    }
