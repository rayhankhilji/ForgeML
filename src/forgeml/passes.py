from __future__ import annotations

from dataclasses import dataclass

import torch

from forgeml.ir import Graph, GraphError, Node
from forgeml.memory import plan_memory
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


def _resolve(aliases: dict[str, str], name: str) -> str:
    while aliases.get(name, name) != name:
        name = aliases[name]
    return name


def simplify_algebra(graph: Graph) -> Graph:
    aliases: dict[str, str] = {}
    nodes: list[Node] = []
    by_name: dict[str, Node] = {}
    specs = graph.specs()
    for node in graph.nodes:
        inputs = tuple(_resolve(aliases, i) for i in node.inputs)
        node = Node(node.name, node.op, inputs, dict(node.attrs), node.spec)
        if node.op == "reshape" and specs[node.inputs[0]].shape == node.spec.shape:
            aliases[node.name] = node.inputs[0]
            continue
        if node.op == "transpose":
            source = by_name.get(node.inputs[0])
            if (
                source is not None
                and source.op == "transpose"
                and node.attrs["dim0"] == source.attrs["dim0"]
                and node.attrs["dim1"] == source.attrs["dim1"]
            ):
                aliases[node.name] = _resolve(aliases, source.inputs[0])
                continue
        aliases[node.name] = node.name
        by_name[node.name] = node
        nodes.append(node)
    outputs = tuple(_resolve(aliases, name) for name in graph.outputs)
    rewritten = Graph(
        inputs=dict(graph.inputs),
        constants=dict(graph.constants),
        nodes=nodes,
        outputs=outputs,
        output_is_tuple=graph.output_is_tuple,
    )
    rewritten.validate()
    return eliminate_dead_nodes(rewritten)


def _freeze(value):
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in value.items()))
    return value


def common_subexpression_elimination(graph: Graph) -> Graph:
    aliases: dict[str, str] = {}
    seen: dict[tuple, str] = {}
    nodes: list[Node] = []
    for node in graph.nodes:
        inputs = tuple(_resolve(aliases, i) for i in node.inputs)
        key = (node.op, inputs, _freeze(node.attrs), node.spec)
        if key in seen:
            aliases[node.name] = seen[key]
            continue
        rewritten = Node(node.name, node.op, inputs, dict(node.attrs), node.spec)
        seen[key] = node.name
        aliases[node.name] = node.name
        nodes.append(rewritten)
    outputs = tuple(_resolve(aliases, name) for name in graph.outputs)
    used = {i for n in nodes for i in n.inputs} | set(outputs)
    constants = {k: v for k, v in graph.constants.items() if k in used}
    rewritten = Graph(
        inputs=dict(graph.inputs),
        constants=constants,
        nodes=nodes,
        outputs=outputs,
        output_is_tuple=graph.output_is_tuple,
    )
    rewritten.validate()
    return rewritten


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


def _topological_order(graph: Graph, score) -> list[Node]:
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
        best = max(ready, key=lambda n: score(n, index[n.name], freed_bytes(n)))
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
    return order


def schedule(graph: Graph) -> Graph:
    candidates = (
        lambda node, index, freed: (freed - node.spec.nbytes, -index),
        lambda node, index, freed: (freed - 2 * node.spec.nbytes, -index),
        lambda node, index, freed: (freed - node.spec.nbytes // 2, -index),
        lambda node, index, freed: (-node.spec.nbytes, freed, -index),
        lambda node, index, freed: (-index,),
        lambda node, index, freed: (index,),
    )
    orders = []
    seen = set()
    for score in candidates:
        order = _topological_order(graph, score)
        names = tuple(n.name for n in order)
        if names not in seen:
            seen.add(names)
            orders.append(order)
    best = min(
        orders,
        key=lambda order: (
            plan_memory(_rebuild(graph, order, dict(graph.constants))).planned_bytes,
            tuple(node.name for node in order),
        ),
    )
    return _rebuild(graph, best, dict(graph.constants))


def optimize(graph: Graph) -> tuple[Graph, list[PassRecord]]:
    graph.validate()
    work = graph.clone()
    records: list[PassRecord] = []
    for name, fn in (
        ("eliminate_dead_nodes", eliminate_dead_nodes),
        ("fold_constants", fold_constants),
        ("eliminate_dead_nodes", eliminate_dead_nodes),
        ("simplify_algebra", simplify_algebra),
        ("common_subexpression_elimination", common_subexpression_elimination),
        ("fuse_linear_gelu", fuse_linear_gelu),
        ("eliminate_dead_nodes", eliminate_dead_nodes),
        ("schedule", schedule),
    ):
        before = len(work.nodes)
        work = fn(work)
        work.validate()
        records.append(PassRecord(name, before, len(work.nodes)))
    return work, records
