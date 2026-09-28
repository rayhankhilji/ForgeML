from __future__ import annotations

from dataclasses import dataclass

from forgeml.ir import Graph, TensorSpec

VIEW_LIKE_OPS = {"reshape"}


@dataclass(frozen=True)
class Allocation:
    slot: int
    size_bytes: int
    first: int
    last: int


@dataclass
class MemoryPlan:
    allocations: dict[str, Allocation]
    aliases: dict[str, str]
    slot_specs: dict[int, TensorSpec]
    naive_bytes: int
    planned_bytes: int

    def to_dict(self) -> dict:
        return {
            "aliases": dict(self.aliases),
            "allocations": {
                k: {
                    "slot": a.slot,
                    "size_bytes": a.size_bytes,
                    "first": a.first,
                    "last": a.last,
                }
                for k, a in self.allocations.items()
            },
            "slot_specs": {str(k): v.to_dict() for k, v in self.slot_specs.items()},
            "naive_bytes": self.naive_bytes,
            "planned_bytes": self.planned_bytes,
        }


def plan_memory(graph: Graph) -> MemoryPlan:
    graph.validate()
    outputs = set(graph.outputs)
    index = {n.name: i for i, n in enumerate(graph.nodes)}
    by_name = {n.name: n for n in graph.nodes}
    consumers: dict[str, list[str]] = {}
    for node in graph.nodes:
        for i in set(node.inputs):
            if i in index:
                consumers.setdefault(i, []).append(node.name)

    memo: dict[str, int] = {}

    def logical_last(name: str) -> int:
        if name in memo:
            return memo[name]
        last = index[name]
        for consumer_name in consumers.get(name, ()):
            consumer = by_name[consumer_name]
            contribution = (
                logical_last(consumer_name)
                if consumer.op in VIEW_LIKE_OPS
                else index[consumer_name]
            )
            last = max(last, contribution)
        memo[name] = last
        return last

    aliases: dict[str, str] = {}
    allocations: dict[str, Allocation] = {}
    slot_specs: dict[int, TensorSpec] = {}
    slot_last: dict[int, int] = {}
    naive = 0
    next_slot = 0
    for node in graph.nodes:
        if node.name in outputs:
            continue
        naive += node.spec.nbytes
        if node.op in VIEW_LIKE_OPS:
            aliases[node.name] = node.inputs[0]
            continue
        first = index[node.name]
        last = logical_last(node.name)
        need = node.spec.nbytes
        best = None
        for slot, spec in slot_specs.items():
            if slot_last[slot] >= first:
                continue
            if spec.dtype != node.spec.dtype or spec.device != node.spec.device:
                continue
            capacity = spec.shape[0] * spec.dtype.itemsize
            if capacity >= need and (
                best is None or capacity < slot_specs[best].shape[0] * node.spec.dtype.itemsize
            ):
                best = slot
        if best is None:
            best = next_slot
            next_slot += 1
            elements = -(-need // node.spec.dtype.itemsize)
            slot_specs[best] = TensorSpec((elements,), node.spec.dtype, node.spec.device)
        slot_last[best] = last
        allocations[node.name] = Allocation(best, need, first, last)
    planned = sum(s.shape[0] * s.dtype.itemsize for s in slot_specs.values())
    return MemoryPlan(allocations, aliases, slot_specs, naive, planned)
