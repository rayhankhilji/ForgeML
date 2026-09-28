from __future__ import annotations

from dataclasses import dataclass

import torch

from forgeml.ir import Graph, GraphError, Node, TensorSpec
from forgeml.memory import VIEW_LIKE_OPS, plan_memory
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
            # The byte cap bounds compile-time memory; view ops are exempt
            # because they materialize nothing, and input bytes are capped to
            # bound fold compute on large operands.
            input_bytes = sum(
                env[i].numel() * env[i].element_size() for i in dict.fromkeys(node.inputs)
            )
            if node.op not in VIEW_LIKE_OPS and (
                node.spec.nbytes > _FOLD_LIMIT_BYTES or input_bytes > _FOLD_LIMIT_BYTES
            ):
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
    used = {i for n in nodes for i in n.inputs} | set(graph.outputs)
    constants = {k: v for k, v in constants.items() if k in used}
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
        if node.op == "reshape":
            source = by_name.get(node.inputs[0])
            if source is not None and source.op == "reshape":
                node = Node(node.name, "reshape", source.inputs, dict(node.attrs), node.spec)
        if node.op == "transpose":
            rank = len(node.spec.shape)
            dims = frozenset((node.attrs["dim0"] % rank, node.attrs["dim1"] % rank))
            if len(dims) == 1:
                aliases[node.name] = node.inputs[0]
                continue
            source = by_name.get(node.inputs[0])
            if (
                source is not None
                and source.op == "transpose"
                and frozenset((source.attrs["dim0"] % rank, source.attrs["dim1"] % rank)) == dims
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
        key_inputs = tuple(sorted(inputs)) if node.op in ("add", "mul") else inputs
        key = (node.op, key_inputs, _freeze(node.attrs), node.spec)
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


def canonicalize_linear(graph: Graph) -> Graph:
    consumers: dict[str, list[str]] = {}
    for node in graph.nodes:
        for i in node.inputs:
            consumers.setdefault(i, []).append(node.name)
    by_name = {n.name: n for n in graph.nodes}
    specs = graph.specs()
    rewritten: dict[str, Node] = {}
    remove: set[str] = set()
    for node in graph.nodes:
        if node.op != "add":
            continue
        for candidate, bias in ((node.inputs[0], node.inputs[1]), (node.inputs[1], node.inputs[0])):
            matmul = by_name.get(candidate)
            bias_spec = specs.get(bias)
            if (
                matmul is None
                or matmul.op != "matmul"
                or matmul.name in graph.outputs
                or len(consumers.get(matmul.name, ())) != 1
                or bias_spec is None
                or len(bias_spec.shape) != 1
                or bias_spec.shape[0] != matmul.spec.shape[1]
            ):
                continue
            rewritten[node.name] = Node(
                name=node.name,
                op="linear",
                inputs=matmul.inputs + (bias,),
                attrs={},
                spec=node.spec,
            )
            remove.add(matmul.name)
            break
    nodes = [rewritten.get(n.name, n) for n in graph.nodes if n.name not in remove]
    used = {i for n in nodes for i in n.inputs}
    constants = {k: v for k, v in graph.constants.items() if k in used or k in graph.outputs}
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
        producer = by_name.get(node.inputs[0])
        if (
            producer is not None
            and producer.op in ("linear", "matmul")
            and producer.name not in graph.outputs
            and len(consumers.get(producer.name, ())) == 1
        ):
            fused[node.name] = Node(
                name=node.name,
                op="fused_linear_gelu",
                inputs=producer.inputs,
                attrs={"approximate": node.attrs.get("approximate", "none")},
                spec=node.spec,
            )
            remove.add(producer.name)
            continue
        add = producer
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


def fuse_shared_projections(graph: Graph) -> Graph:
    index = {n.name: i for i, n in enumerate(graph.nodes)}
    specs = graph.specs()
    groups: dict[tuple[str, int], list[Node]] = {}
    for node in graph.nodes:
        if node.op != "linear" or node.inputs[1] not in graph.constants:
            continue
        if len(node.inputs) == 3 and node.inputs[2] not in graph.constants:
            continue
        groups.setdefault((node.inputs[0], len(node.inputs)), []).append(node)
    groups = {k: v for k, v in groups.items() if len(v) >= 2}
    if not groups:
        return _rebuild(graph, list(graph.nodes), dict(graph.constants))
    taken = set(graph.inputs) | set(graph.constants) | set(index)
    constants = dict(graph.constants)
    counter = 0

    def fresh(hint: str) -> str:
        nonlocal counter
        while True:
            counter += 1
            candidate = f"{hint}_{counter}"
            if candidate not in taken:
                taken.add(candidate)
                return candidate

    emit_at: dict[str, list[Node]] = {}
    remove: set[str] = set()
    for members in groups.values():
        members.sort(key=lambda n: index[n.name])
        leader = members[0]
        weight_dtypes = {constants[m.inputs[1]].dtype for m in members}
        bias_dtypes = (
            {constants[m.inputs[2]].dtype for m in members} if len(leader.inputs) == 3 else set()
        )
        if len(weight_dtypes) != 1 or len(bias_dtypes) > 1:
            continue
        x_spec = specs[leader.inputs[0]]
        weight_name = fresh(f"{leader.name}_shared_weight")
        constants[weight_name] = torch.cat(
            [constants[m.inputs[1]] for m in members], dim=1
        ).contiguous()
        inputs = [leader.inputs[0], weight_name]
        if len(leader.inputs) == 3:
            bias_name = fresh(f"{leader.name}_shared_bias")
            constants[bias_name] = torch.cat([constants[m.inputs[2]] for m in members])
            inputs.append(bias_name)
        total = sum(m.spec.shape[1] for m in members)
        wide_name = fresh(f"{leader.name}_shared")
        wide_spec = TensorSpec((x_spec.shape[0], total), x_spec.dtype, x_spec.device)
        emitted = [Node(wide_name, "linear", tuple(inputs), {}, wide_spec)]
        offset = 0
        for member in members:
            emitted.append(
                Node(
                    member.name,
                    "narrow",
                    (wide_name,),
                    {"dim": 1, "start": offset, "length": member.spec.shape[1]},
                    member.spec,
                )
            )
            offset += member.spec.shape[1]
            remove.add(member.name)
        emit_at[leader.name] = emitted
    nodes: list[Node] = []
    for node in graph.nodes:
        if node.name in emit_at:
            nodes.extend(emit_at[node.name])
        elif node.name not in remove:
            nodes.append(node)
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
        for consumer in sorted(consumers.get(best.name, ())):
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
        ("canonicalize_linear", canonicalize_linear),
        ("fuse_linear_gelu", fuse_linear_gelu),
        ("fuse_shared_projections", fuse_shared_projections),
        ("eliminate_dead_nodes", eliminate_dead_nodes),
        ("schedule", schedule),
    ):
        before = len(work.nodes)
        work = fn(work)
        work.validate()
        records.append(PassRecord(name, before, len(work.nodes)))
    return work, records
