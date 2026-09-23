from __future__ import annotations

import asyncio
import itertools
import logging
import math
from collections import deque

from nebula.runtime import Engine, EngineFailed, _validate_prompt


class QueueFull(RuntimeError):
    pass


class ServiceClosed(RuntimeError):
    pass


class AsyncBatcher:
    def __init__(
        self,
        engine: Engine,
        *,
        max_batch_size: int = 8,
        max_queue_size: int = 64,
        batch_window_ms: float = 5.0,
    ):
        for name, v in (
            ("max_batch_size", max_batch_size),
            ("max_queue_size", max_queue_size),
        ):
            if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            not isinstance(batch_window_ms, (int, float))
            or isinstance(batch_window_ms, bool)
            or not math.isfinite(batch_window_ms)
            or batch_window_ms < 0
        ):
            raise ValueError("batch_window_ms must be a finite non-negative number")
        if max_batch_size > engine.max_requests:
            raise ValueError("max_batch_size exceeds engine max_requests")
        self.engine = engine
        self.max_batch_size = max_batch_size
        self.max_queue_size = max_queue_size
        self.batch_window_ms = batch_window_ms
        self._queue: deque = deque()
        self._outstanding = 0
        self._closed = False
        self._healthy = True
        self._worker: asyncio.Task | None = None
        self._wake: asyncio.Event | None = None
        self._inflight = 0
        self._engine_closed = False

    @property
    def pending(self) -> int:
        return self._outstanding

    @property
    def healthy(self) -> bool:
        return self._healthy and not self._closed and self.engine.healthy

    async def submit(
        self, prompt: list[int], max_new_tokens: int, *, timeout_s: float | None = None
    ) -> list[int]:
        if self._closed:
            raise ServiceClosed("batcher is closed")
        if not self._healthy or not self.engine.healthy:
            raise EngineFailed("batcher engine is not healthy")
        if not isinstance(prompt, (list, tuple)):
            raise TypeError("prompt must be a list of token ids")
        if timeout_s is not None and (
            not isinstance(timeout_s, (int, float))
            or isinstance(timeout_s, bool)
            or not math.isfinite(timeout_s)
            or timeout_s <= 0
        ):
            raise ValueError("timeout_s must be a finite positive number")
        copied = list(prompt)
        _validate_prompt(self.engine.config, copied, max_new_tokens)
        if self._outstanding >= self.max_queue_size:
            raise QueueFull("request queue is full")
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        self._queue.append((copied, max_new_tokens, fut))
        self._outstanding += 1
        if self._worker is None or self._worker.done():
            self._wake = asyncio.Event()
            self._worker = loop.create_task(self._run())
        self._wake.set()
        try:
            if timeout_s is None:
                return await asyncio.shield(fut)
            return await asyncio.wait_for(asyncio.shield(fut), timeout_s)
        except (TimeoutError, asyncio.CancelledError):
            fut.cancel()
            raise

    def _retire(self, fut) -> None:
        self._outstanding -= 1

    async def _run(self) -> None:
        while True:
            if not self._queue:
                if self._closed:
                    return
                self._wake.clear()
                if not self._queue:
                    await self._wake.wait()
                    continue
            await asyncio.sleep(self.batch_window_ms / 1000.0)
            batch = []
            key = None
            rest = deque()
            while self._queue:
                item = self._queue.popleft()
                k = (len(item[0]), item[1])
                if key is None:
                    key = k
                if k == key and len(batch) < self.max_batch_size:
                    batch.append(item)
                else:
                    rest.append(item)
            self._queue = rest
            live = [it for it in batch if not it[2].cancelled()]
            for it in batch:
                if it not in live:
                    self._retire(it[2])
            if not live:
                continue
            prompts = [it[0] for it in live]
            max_new = live[0][1]
            self._inflight += len(live)
            try:
                results = await asyncio.to_thread(self.engine.generate, prompts, max_new)
                if not isinstance(results, list) or len(results) != len(live):
                    raise EngineFailed("engine result count does not match the submitted batch")
            except Exception as e:
                logging.getLogger(__name__).exception("Nebula batch execution failed")
                self._healthy = False
                self._inflight -= len(live)
                for it in live:
                    if not it[2].done():
                        it[2].set_exception(e)
                    self._retire(it[2])
                while self._queue:
                    it = self._queue.popleft()
                    if not it[2].done():
                        it[2].set_exception(e)
                    self._retire(it[2])
                return
            self._inflight -= len(live)
            for it, res in zip(live, results):
                if not it[2].done():
                    it[2].set_result(res)
                self._retire(it[2])
            if self._closed and not self._queue:
                return

    async def close(self) -> None:
        if not self._closed:
            self._closed = True
            while self._queue:
                it = self._queue.popleft()
                if not it[2].done():
                    it[2].set_exception(ServiceClosed("batcher closed"))
                self._retire(it[2])
            if self._wake is not None:
                self._wake.set()
        if self._worker is not None:
            await asyncio.shield(self._worker)
            self._worker = None
        if not self._engine_closed:
            self._engine_closed = True
            await asyncio.to_thread(self.engine.close)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.close()


class RequestRouter:
    def __init__(self, replicas: list[AsyncBatcher]):
        if not replicas:
            raise ValueError("replicas must be non-empty")
        self.replicas = list(replicas)
        self._counter = itertools.count()

    async def generate(
        self, prompt: list[int], max_new_tokens: int, *, timeout_s: float | None = None
    ) -> list[int]:
        healthy = [r for r in self.replicas if r.healthy]
        if not healthy:
            raise EngineFailed("no healthy replicas")
        start = next(self._counter) % len(healthy)
        ordered = list(enumerate(healthy[start:] + healthy[:start]))
        ordered.sort(key=lambda pair: (pair[1].pending, pair[0]))
        last_err = None
        for _, r in ordered:
            try:
                return await r.submit(prompt, max_new_tokens, timeout_s=timeout_s)
            except QueueFull as e:
                last_err = e
                continue
        raise last_err or EngineFailed("no replica accepted the request")

    async def close(self) -> None:
        results = await asyncio.gather(*(r.close() for r in self.replicas), return_exceptions=True)
        for res in results:
            if isinstance(res, BaseException):
                raise res
