from __future__ import annotations

from dataclasses import dataclass

import torch

from forgeml.ir import Graph, GraphError, Node
from forgeml.ops import evaluate

_FOLD_LIMIT_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class PassRecord:
    name: str
    nodes_before: int
    nodes_after: int


def _rebuild(graph: Graph, nodes: list[Node], constants: dict) -> Graph:
    out = Graph(
        inputs=dict(graph.inputs),
        constants=constants,
        nodes=nodes,
        outputs=tuple(graph.outputs),
        output_is_tuple=graph.output_is_tuple,
    )
    out.validate()
    return out


def eliminate_dead_nodes(graph: Graph) -> Graph:
    needed: set[str] = set(graph.outputs)
    kept: list[Node] = []
    for node in reversed(graph.nodes):
        if node.name in needed:
            kept.append(node)
            needed.update(node.inputs)
    kept.reverse()
    constants = {k: v for k, v in graph.constants.items() if k in needed}
    return _rebuild(graph, kept, constants)


def fold_constants(graph: Graph) -> Graph:
    constants = dict(graph.constants)
    nodes: list[Node] = []
    env = dict(constants)
    for node in graph.nodes:
        if all(i in env for i in node.inputs):
            if node.spec.nbytes > _FOLD_LIMIT_BYTES:
                nodes.append(node)
                continue
            args = tuple(env[i] for i in node.inputs)
            try:
                with torch.inference_mode():
                    result = evaluate(node.op, args, dict(node.attrs))
            except Exception as e:
                raise GraphError(f"constant folding failed at node {node.name!r}: {e}") from e
            constants[node.name] = result
            env[node.name] = result
            continue
        nodes.append(node)
    return _rebuild(graph, nodes, constants)


def fuse_linear_gelu(graph: Graph) -> Graph:
    consumers: dict[str, list[str]] = {}
    for node in graph.nodes:
        for i in node.inputs:
            consumers.setdefault(i, []).append(node.name)
    by_name = {n.name: n for n in graph.nodes}
    fused: dict[str, Node] = {}
    remove: set[str] = set()
    for node in graph.nodes:
        if node.op != "gelu" or len(node.inputs) != 1:
            continue
        add = by_name.get(node.inputs[0])
        if add is None or add.op != "add":
            continue
        if add.name in graph.outputs or len(consumers.get(add.name, ())) != 1:
            continue
        mm = None
        bias = None
        for cand, other in ((add.inputs[0], add.inputs[1]), (add.inputs[1], add.inputs[0])):
            n = by_name.get(cand)
            if n is not None and n.op == "matmul" and other in graph.constants:
                mm, bias = n, other
                break
        if mm is None:
            continue
        if mm.name in graph.outputs or len(consumers.get(mm.name, ())) != 1:
            continue
        bias_t = graph.constants[bias]
        if bias_t.dim() != 1 or bias_t.shape[0] != mm.spec.shape[1]:
            continue
        if len(mm.spec.shape) != 2:
            continue
        fused[node.name] = Node(
            name=node.name,
            op="fused_linear_gelu",
            inputs=(mm.inputs[0], mm.inputs[1], bias),
            attrs={"approximate": node.attrs.get("approximate", "none")},
            spec=node.spec,
        )
        remove.update((mm.name, add.name))
    if not fused:
        return _rebuild(graph, list(graph.nodes), dict(graph.constants))
    nodes = [fused.get(n.name, n) for n in graph.nodes if n.name not in remove]
    used = {i for n in nodes for i in n.inputs}
    constants = {k: v for k, v in graph.constants.items() if k in used or k in graph.outputs}
    return _rebuild(graph, nodes, constants)


def schedule(graph: Graph) -> Graph:
    outputs = set(graph.outputs)
    index = {n.name: i for i, n in enumerate(graph.nodes)}
    node_names = set(index)
    consumers: dict[str, set[str]] = {}
    remaining: dict[str, set[str]] = {}
    for node in graph.nodes:
        for i in set(node.inputs):
            consumers.setdefault(i, set()).add(node.name)
        remaining[node.name] = {i for i in node.inputs if i in node_names}
    specs = graph.specs()
    remaining_consumers = {k: len(v) for k, v in consumers.items() if k in node_names}

    def freed_bytes(node: Node) -> int:
        total = 0
        for i in set(node.inputs):
            if i in outputs or i not in node_names:
                continue
            if remaining_consumers.get(i, 0) == 1:
                total += specs[i].nbytes
        return total

    ready = [n for n in graph.nodes if not remaining[n.name]]
    order: list[Node] = []
    by_name = {n.name: n for n in graph.nodes}
    while ready:
        best = max(ready, key=lambda n: (freed_bytes(n) - n.spec.nbytes, -index[n.name]))
        ready.remove(best)
        order.append(best)
        for consumer in consumers.get(best.name, ()):
            remaining[consumer].discard(best.name)
            if not remaining[consumer]:
                ready.append(by_name[consumer])
        for i in set(best.inputs):
            if i in remaining_consumers:
                remaining_consumers[i] -= 1
    if len(order) != len(graph.nodes):
        raise GraphError("graph contains a cycle")
    return _rebuild(graph, order, dict(graph.constants))


def optimize(graph: Graph) -> tuple[Graph, list[PassRecord]]:
    graph.validate()
    work = graph.clone()
    records: list[PassRecord] = []
    for name, fn in (
        ("eliminate_dead_nodes", eliminate_dead_nodes),
        ("fold_constants", fold_constants),
        ("eliminate_dead_nodes", eliminate_dead_nodes),
        ("fuse_linear_gelu", fuse_linear_gelu),
        ("schedule", schedule),
    ):
        before = len(work.nodes)
        work = fn(work)
        work.validate()
        records.append(PassRecord(name, before, len(work.nodes)))
    return work, records
