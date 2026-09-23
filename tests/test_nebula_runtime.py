import pytest
import torch

from nebula import DecoderConfig, Engine, EngineFailed, ParallelContext, TinyDecoder


def cfg():
    return DecoderConfig(
        vocab_size=32,
        hidden_size=16,
        num_heads=4,
        num_layers=2,
        intermediate_size=32,
        max_seq_len=16,
        seed=17,
    )


def engine(**kw):
    return Engine(cfg(), context=ParallelContext.from_env(), max_requests=4, **kw)


def ref_generate(model, prompt, n):
    out = []
    cur = list(prompt)
    for _ in range(n):
        with torch.no_grad():
            logits = model(torch.tensor([cur], dtype=torch.long))
        nxt = int(logits[0, -1].argmax())
        out.append(nxt)
        cur.append(nxt)
    return out


def test_forward_parity_prefill():
    e = engine()
    ref = TinyDecoder(cfg()).eval()
    toks = torch.tensor([[1, 2, 3], [4, 5, 6]])
    with torch.no_grad():
        want = ref(toks)
    got = e.forward(toks, (11, 12), 0)
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)
    assert e.decoder.cache.active_requests == 2
    e.close()


def test_decode_step_parity():
    e = engine()
    ref = TinyDecoder(cfg()).eval()
    toks = torch.tensor([[1, 2, 3]])
    e.forward(toks, (5,), 0)
    step = torch.tensor([[7]])
    got = e.forward(step, (5,), 3)
    with torch.no_grad():
        want = ref(torch.tensor([[1, 2, 3, 7]]))[:, -1:, :]
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)
    e.close()


def test_generate_matches_slow_reference():
    e = engine()
    ref = TinyDecoder(cfg()).eval()
    prompts = [[1, 2, 3], [4, 5], [9, 9, 9]]
    got = e.generate(prompts, 3)
    want = [ref_generate(ref, p, 3) for p in prompts]
    assert got == want
    assert all(len(g) == 3 for g in got)
    assert e.decoder.cache.active_requests == 0
    e.close()


def test_zero_max_new_no_cache():
    e = engine()
    assert e.generate([[1, 2]], 0) == [[]]
    assert e.decoder.cache.active_requests == 0
    e.close()


def test_invalid_request_does_not_poison():
    e = engine()
    with pytest.raises(ValueError):
        e.forward(torch.tensor([[1, 2]]), (1,), 15)
    with pytest.raises(ValueError):
        e.forward(torch.tensor([[1, 99]]), (1,), 0)
    with pytest.raises(ValueError):
        e.forward(torch.tensor([[1]]), (1, 2), 0)
    with pytest.raises(ValueError):
        e.forward(torch.tensor([[1]]), (1 << 62,), 0)
    assert e.healthy
    out = e.forward(torch.tensor([[1, 2]]), (3,), 0)
    assert out.shape == (1, 2, 32)
    e.close()


def test_closed_engine_rejects():
    e = engine()
    e.close()
    with pytest.raises(EngineFailed):
        e.forward(torch.tensor([[1]]), (1,), 0)
    with pytest.raises(EngineFailed):
        e.generate([[1]], 1)


@pytest.mark.parametrize("failure", [RuntimeError, KeyError, AssertionError])
def test_failed_forward_poisons_engine(monkeypatch, failure):
    e = engine()
    e.forward(torch.tensor([[1, 2]]), (8,), 0)

    def boom(*a, **k):
        raise failure("simulated failure")

    monkeypatch.setattr(e.decoder, "forward", boom)
    with pytest.raises(EngineFailed):
        e.forward(torch.tensor([[3]]), (8,), 2)
    assert not e.healthy
    assert e.decoder.cache.active_requests == 0
    with pytest.raises(EngineFailed):
        e.forward(torch.tensor([[1]]), (9,), 0)
    e.close()


def test_release():
    e = engine()
    e.forward(torch.tensor([[1, 2]]), (1,), 0)
    e.release((1,))
    assert e.decoder.cache.active_requests == 0
    e.close()


def test_noncontiguous_tokens_parity():
    e = engine()
    ref = TinyDecoder(cfg()).eval()
    base = torch.tensor([[1, 2, 3, 9], [4, 5, 6, 9]])
    toks = base[:, :3]
    assert not toks.is_contiguous()
    with torch.no_grad():
        want = ref(toks.contiguous())
    got = e.forward(toks, (31, 32), 0)
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)
    e.close()


def test_release_overflow_rejected_before_broadcast():
    e = engine()
    calls = []
    monkey = lambda t: calls.append(t)
    e._bcast = monkey
    with pytest.raises(ValueError):
        e.release((1 << 63,))
    assert calls == []
    assert e.healthy
    e.close()


def test_float_start_pos_rejected_healthy():
    e = engine()
    with pytest.raises(TypeError):
        e.forward(torch.tensor([[1]]), (1,), 0.0)
    assert e.healthy
    e.forward(torch.tensor([[1]]), (1,), 0)
    e.close()


def test_bcast_failure_poisons_and_close_skips(monkeypatch):
    e = engine()
    calls = []

    def boom(t):
        calls.append(t)
        raise RuntimeError("transport dead")

    monkeypatch.setattr(e, "_world", lambda: True)
    monkeypatch.setattr(e, "_bcast", boom)
    with pytest.raises(EngineFailed):
        e.forward(torch.tensor([[1]]), (1,), 0)
    assert not e.healthy
    assert e.decoder.cache.active_requests == 0
    e.close()
    assert len(calls) == 1
