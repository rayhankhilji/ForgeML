import tempfile
import time
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from nebula import DecoderConfig, Engine, EngineFailed, ParallelContext, TinyDecoder

pytestmark = [
    pytest.mark.distributed,
    pytest.mark.skipif(
        not (dist.is_available() and dist.is_gloo_available()),
        reason="torch.distributed gloo unavailable",
    ),
]

CFG = DecoderConfig(
    vocab_size=32,
    hidden_size=16,
    num_heads=4,
    num_layers=4,
    intermediate_size=32,
    max_seq_len=16,
    seed=17,
)


def _ref_generate(model, prompt, n):
    out = []
    cur = list(prompt)
    for _ in range(n):
        with torch.no_grad():
            logits = model(torch.tensor([cur], dtype=torch.long))
        nxt = int(logits[0, -1].argmax())
        out.append(nxt)
        cur.append(nxt)
    return out


def _worker(rank, world, tp, pp, store_path):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{store_path}",
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=20),
    )
    ctx = ParallelContext.from_env(tp_size=tp, pp_size=pp, timeout_s=20)
    engine = Engine(CFG, context=ctx, max_requests=4)
    try:
        if rank == 0:
            ref = TinyDecoder(CFG).eval()
            toks = torch.tensor([[1, 2, 3], [4, 5, 6]])
            with torch.no_grad():
                want = ref(toks)
            got = engine.forward(toks, (21, 22), 0)
            torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)
            got2 = engine.forward(torch.tensor([[7], [8]]), (21, 22), 3)
            with torch.no_grad():
                want2 = ref(torch.tensor([[1, 2, 3, 7], [4, 5, 6, 8]]))[:, -1:, :]
            torch.testing.assert_close(got2, want2, rtol=1e-4, atol=1e-5)
            got3 = engine.forward(torch.tensor([[9], [9]]), (21, 22), 4)
            with torch.no_grad():
                want3 = ref(torch.tensor([[1, 2, 3, 7, 9], [4, 5, 6, 8, 9]]))[:, -1:, :]
            torch.testing.assert_close(got3, want3, rtol=1e-4, atol=1e-5)
            engine.release((21, 22))
            assert engine.decoder.cache.active_requests == 0
            prompts = [[1, 2, 3], [5, 6]]
            gen = engine.generate(prompts, 2)
            want_gen = [_ref_generate(ref, p, 2) for p in prompts]
            assert gen == want_gen
            assert engine.decoder.cache.active_requests == 0
        else:
            engine.serve()
    finally:
        engine.close()
        ctx.close()
        dist.destroy_process_group()


def _run(world, tp, pp):
    with tempfile.TemporaryDirectory() as d:
        procs = mp.start_processes(
            _worker,
            args=(world, tp, pp, f"{d}/store"),
            nprocs=world,
            join=False,
            start_method="spawn",
        )
        deadline = time.time() + 60
        while time.time() < deadline:
            if all(p.exitcode is not None for p in procs.processes):
                break
            time.sleep(0.2)
        else:
            for p in procs.processes:
                if p.is_alive():
                    p.terminate()
            for p in procs.processes:
                p.join()
            pytest.fail("distributed workers did not finish within 60s")
        for p in procs.processes:
            assert p.exitcode == 0


def test_tp2_pp1():
    _run(world=2, tp=2, pp=1)


def test_tp1_pp2():
    _run(world=2, tp=1, pp=2)


def test_tp2_pp2():
    _run(world=4, tp=2, pp=2)


def _failing_worker(rank, store_path):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{store_path}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=10),
    )
    ctx = ParallelContext.from_env(tp_size=2, pp_size=1, timeout_s=10)
    engine = Engine(CFG, context=ctx, max_requests=4)
    if rank == 0:
        try:
            engine.forward(torch.tensor([[1, 2, 3]]), (1,), 0)
        except EngineFailed:
            assert not engine.healthy
            assert engine.decoder.cache.active_requests == 0
            return
        raise AssertionError("forward did not fail after peer death")
    raise RuntimeError("deliberate worker failure")


def test_worker_failure_propagates():
    with tempfile.TemporaryDirectory() as d:
        procs = mp.start_processes(
            _failing_worker,
            args=(f"{d}/store",),
            nprocs=2,
            join=False,
            start_method="spawn",
        )
        deadline = time.time() + 45
        while time.time() < deadline:
            if all(p.exitcode is not None for p in procs.processes):
                break
            time.sleep(0.2)
        else:
            for p in procs.processes:
                if p.is_alive():
                    p.terminate()
            for p in procs.processes:
                p.join()
            pytest.fail("distributed workers did not finish within 45s")
        for p in procs.processes:
            p.join()
        assert procs.processes[0].exitcode == 0
        assert procs.processes[1].exitcode != 0
