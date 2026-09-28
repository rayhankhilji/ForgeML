import torch

from forgeml.ir import GraphBuilder, TensorSpec
from forgeml.memory import plan_memory


def spec(*shape):
    return TensorSpec(tuple(shape), torch.float32, "cpu")


def test_chain_reuses_slot_with_strict_nonoverlap():
    b = GraphBuilder({"x": spec(4), "z": spec(4)})
    b.add("a", "relu", ("x",))
    b.add("b", "relu", ("a",))
    b.add("c", "relu", ("z",))
    b.add("d", "relu", ("c",))
    b.add("y", "add", ("b", "d"))
    g = b.finish(("y",))
    plan = plan_memory(g)
    assert plan.allocations["c"].slot == plan.allocations["a"].slot
    assert "y" not in plan.allocations
    for alloc in plan.allocations.values():
        assert alloc.first <= alloc.last
    slots = [(a.slot, a.first, a.last) for a in plan.allocations.values()]
    for i in range(len(slots)):
        for j in range(i + 1, len(slots)):
            if slots[i][0] == slots[j][0]:
                lo = max(slots[i][1], slots[j][1])
                hi = min(slots[i][2], slots[j][2])
                assert lo > hi or (slots[i][2] < slots[j][1] or slots[j][2] < slots[i][1])


def test_view_like_values_extend_source_lifetime_without_new_slot():
    b = GraphBuilder({"x": spec(8), "z": spec(2, 4)})
    b.add("a", "relu", ("x",))
    b.add("r", "reshape", ("a",), shape=(2, 4))
    b.add("c", "relu", ("z",))
    b.add("y", "add", ("r", "c"))
    g = b.finish(("y",))
    plan = plan_memory(g)
    assert plan.aliases == {"r": "a"}
    assert set(plan.allocations) == {"a", "c"}
    assert plan.allocations["a"].last == 3
    assert plan.allocations["a"].slot != plan.allocations["c"].slot
    assert plan.planned_bytes == 2 * 8 * 4


def test_transpose_view_extends_source_lifetime_without_new_slot():
    b = GraphBuilder({"x": spec(2, 4), "z": spec(4, 2)})
    b.add("a", "relu", ("x",))
    b.add("t", "transpose", ("a",), dim0=0, dim1=1)
    b.add("c", "relu", ("z",))
    b.add("y", "add", ("t", "c"))
    g = b.finish(("y",))
    plan = plan_memory(g)
    assert plan.aliases == {"t": "a"}
    assert set(plan.allocations) == {"a", "c"}
    assert plan.allocations["a"].last == 3
    assert plan.allocations["a"].slot != plan.allocations["c"].slot


def test_naive_and_planned_bytes():
    b = GraphBuilder({"x": spec(8), "z": spec(8)})
    b.add("a", "relu", ("x",))
    b.add("b", "relu", ("a",))
    b.add("c", "relu", ("z",))
    b.add("y", "add", ("b", "c"))
    g = b.finish(("y",))
    plan = plan_memory(g)
    assert plan.naive_bytes == 3 * 8 * 4
    assert plan.planned_bytes <= plan.naive_bytes
    assert plan.planned_bytes == 2 * 8 * 4
    d = plan.to_dict()
    assert d["allocations"]["a"]["slot"] == 0


def test_overlapping_lifetimes_get_distinct_slots():
    b = GraphBuilder({"x": spec(4), "z": spec(4)})
    b.add("a", "relu", ("x",))
    b.add("b", "relu", ("z",))
    b.add("y", "add", ("a", "b"))
    g = b.finish(("y",))
    plan = plan_memory(g)
    assert plan.allocations["a"].slot != plan.allocations["b"].slot


def test_output_only_graph_zero_bytes():
    b = GraphBuilder({"x": spec(4)})
    b.add("y", "relu", ("x",))
    g = b.finish(("y",))
    plan = plan_memory(g)
    assert plan.naive_bytes == 0 and plan.planned_bytes == 0
