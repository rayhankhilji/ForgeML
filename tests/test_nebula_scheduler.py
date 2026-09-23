import asyncio
import threading

import pytest

from nebula import (
    AsyncBatcher,
    DecoderConfig,
    Engine,
    EngineFailed,
    ParallelContext,
    QueueFull,
    RequestRouter,
    ServiceClosed,
)


class FakeEngine:
    max_requests = 64

    def __init__(self, delay=None, fail=None):
        self.batches = []
        self.delay = delay
        self.fail = fail
        self.healthy = True
        self.closed = False
        self.config = DecoderConfig(
            vocab_size=64,
            hidden_size=16,
            num_heads=4,
            num_layers=2,
            intermediate_size=32,
            max_seq_len=32,
        )

    def generate(self, prompts, max_new_tokens):
        self.batches.append((tuple(tuple(p) for p in prompts), max_new_tokens))
        if self.delay is not None:
            self.delay.wait(5)
        if self.fail is not None:
            raise self.fail
        return [[1] * max_new_tokens for _ in prompts]

    def close(self):
        self.closed = True


def run(coro):
    return asyncio.run(coro)


def test_coalesces_compatible_requests():
    async def go():
        e = FakeEngine()
        b = AsyncBatcher(e, max_batch_size=4, batch_window_ms=20)
        outs = await asyncio.gather(b.submit([1, 2], 2), b.submit([3, 4], 2), b.submit([5, 6], 2))
        assert outs == [[1, 1]] * 3
        assert len(e.batches) == 1 and len(e.batches[0][0]) == 3
        assert b.pending == 0
        await b.close()

    run(go())


def test_incompatible_requests_not_mixed():
    async def go():
        e = FakeEngine()
        b = AsyncBatcher(e, max_batch_size=4, batch_window_ms=10)
        outs = await asyncio.gather(b.submit([1, 2], 2), b.submit([1, 2], 5))
        assert outs == [[1, 1], [1] * 5]
        assert len(e.batches) == 2
        assert e.batches[0][1] != e.batches[1][1]
        await b.close()

    run(go())


def test_queue_full_and_fifo():
    async def go():
        gate = threading.Event()
        e = FakeEngine(delay=gate)
        b = AsyncBatcher(e, max_batch_size=1, max_queue_size=2, batch_window_ms=1)
        t1 = asyncio.create_task(b.submit([1], 1))
        await asyncio.sleep(0)
        t2 = asyncio.create_task(b.submit([2], 1))
        await asyncio.sleep(0)
        with pytest.raises(QueueFull):
            await b.submit([3], 1)
        gate.set()
        assert await t1 == [1]
        assert await t2 == [1]
        assert [p[0][0][0] for p in e.batches] == [1, 2]
        await b.close()

    run(go())


def test_cancellation_retires_outstanding():
    async def go():
        gate = threading.Event()
        e = FakeEngine(delay=gate)
        b = AsyncBatcher(e, max_batch_size=1, max_queue_size=4, batch_window_ms=5)
        t1 = asyncio.create_task(b.submit([1], 1))
        await asyncio.sleep(0)
        with pytest.raises(asyncio.TimeoutError):
            await b.submit([2], 1, timeout_s=0.01)
        gate.set()
        await t1
        await asyncio.sleep(0.05)
        assert b.pending == 0
        await b.close()

    run(go())


def test_close_settles_futures():
    async def go():
        e = FakeEngine()
        b = AsyncBatcher(e, max_batch_size=1, batch_window_ms=50)
        t = asyncio.create_task(b.submit([1], 1))
        await asyncio.sleep(0.01)
        await b.close()
        with pytest.raises(ServiceClosed):
            await t
        with pytest.raises(ServiceClosed):
            await b.submit([1], 1)
        assert e.closed

    run(go())


@pytest.mark.parametrize("failure", [EngineFailed, KeyError, AssertionError])
def test_engine_failure_marks_unhealthy_no_retry(failure):
    async def go():
        e = FakeEngine(fail=failure("boom"))
        b = AsyncBatcher(e, batch_window_ms=1)
        with pytest.raises(failure):
            await asyncio.wait_for(b.submit([1], 1), 10)
        assert not b.healthy
        assert len(e.batches) == 1
        with pytest.raises(EngineFailed):
            await b.submit([1], 1)
        await b.close()

    run(go())


def test_router_balances_and_skips_failed():
    async def go():
        good = FakeEngine()
        bad = FakeEngine(fail=EngineFailed("dead"))
        b1 = AsyncBatcher(good, batch_window_ms=1)
        b2 = AsyncBatcher(bad, batch_window_ms=1)
        r = RequestRouter([b1, b2])
        assert await r.generate([1], 1) == [1]
        b2._healthy = False
        for _ in range(3):
            assert await r.generate([1], 1) == [1]
        assert len(good.batches) == 4
        await r.close()

    run(go())


def test_router_queuefull_moves_to_next_replica():
    async def go():
        gate = threading.Event()
        e1 = FakeEngine(delay=gate)
        e2 = FakeEngine()
        b1 = AsyncBatcher(e1, max_batch_size=1, max_queue_size=1, batch_window_ms=1)
        b2 = AsyncBatcher(e2, max_batch_size=1, batch_window_ms=1)
        r = RequestRouter([b1, b2])
        t = asyncio.create_task(r.generate([1], 1))
        await asyncio.sleep(0)
        assert await r.generate([2], 1) == [1]
        assert len(e2.batches) == 1
        gate.set()
        await t
        await r.close()

    run(go())


def test_router_no_healthy_raises():
    async def go():
        e = FakeEngine()
        b = AsyncBatcher(e)
        b._healthy = False
        r = RequestRouter([b])
        with pytest.raises(EngineFailed):
            await r.generate([1], 1)

    run(go())


def test_real_engine_integration():
    async def go():
        cfg = DecoderConfig(
            vocab_size=32,
            hidden_size=16,
            num_heads=4,
            num_layers=2,
            intermediate_size=32,
            max_seq_len=16,
            seed=17,
        )
        ctx = ParallelContext.from_env()
        e = Engine(cfg, context=ctx, max_requests=8)
        b = AsyncBatcher(e, max_batch_size=4, batch_window_ms=5)
        outs = await asyncio.gather(b.submit([1, 2, 3], 2), b.submit([4, 5, 6], 2))
        assert all(len(o) == 2 for o in outs)
        slow = Engine(cfg, context=ParallelContext.from_env(), max_requests=8)
        want = slow.generate([[1, 2, 3], [4, 5, 6]], 2)
        assert outs == want
        assert e.decoder.cache.active_requests == 0
        await b.close()
        slow.close()

    run(go())


def test_batcher_invalid_config():
    e = FakeEngine()
    for kwargs in (
        {"max_batch_size": 0},
        {"max_queue_size": 0},
        {"max_batch_size": True},
        {"batch_window_ms": -1.0},
        {"batch_window_ms": float("inf")},
        {"max_batch_size": 1000},
    ):
        with pytest.raises(ValueError):
            AsyncBatcher(e, **kwargs)


def test_wrong_result_count_fails_all():
    class ShortEngine(FakeEngine):
        def generate(self, prompts, max_new_tokens):
            self.batches.append(prompts)
            return []

    async def go():
        e = ShortEngine()
        b = AsyncBatcher(e, batch_window_ms=1)
        outs = await asyncio.gather(b.submit([1], 1), b.submit([2], 1), return_exceptions=True)
        assert all(isinstance(o, EngineFailed) for o in outs)
        assert not b.healthy
        await b.close()

    run(go())


def test_close_drains_inflight():
    async def go():
        gate = threading.Event()
        e = FakeEngine(delay=gate)
        b = AsyncBatcher(e, max_batch_size=2, batch_window_ms=1)
        t = asyncio.create_task(b.submit([1], 1))
        await asyncio.sleep(0.05)
        closer = asyncio.create_task(b.close())
        await asyncio.sleep(0.01)
        gate.set()
        await asyncio.wait_for(closer, 10)
        assert await t == [1]
        assert e.closed

    run(go())
