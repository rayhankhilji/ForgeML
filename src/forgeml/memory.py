from __future__ import annotations

from dataclasses import dataclass

from forgeml.ir import Graph, TensorSpec


@dataclass(frozen=True)
class Allocation:
    slot: int
    size_bytes: int
    first: int
    last: int


@dataclass
class MemoryPlan:
    allocations: dict[str, Allocation]
    slot_specs: dict[int, TensorSpec]
    naive_bytes: int
    planned_bytes: int

    def to_dict(self) -> dict:
        return {
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
    last_use: dict[str, int] = {}
    for node in graph.nodes:
        for i in node.inputs:
            if i in index:
                last_use[i] = max(last_use.get(i, 0), index[node.name])
    allocations: dict[str, Allocation] = {}
    slot_specs: dict[int, TensorSpec] = {}
    slot_last: dict[int, int] = {}
    naive = 0
    next_slot = 0
    for node in graph.nodes:
        if node.name in outputs:
            continue
        naive += node.spec.nbytes
        first = index[node.name]
        last = last_use.get(node.name, first)
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
    return MemoryPlan(allocations, slot_specs, naive, planned)
