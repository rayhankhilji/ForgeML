import pytest
import torch

from nebula.cache import CacheError, KVCache


def make(**kw):
    args = {
        "layer_ids": (0, 1),
        "num_heads": 2,
        "head_dim": 4,
        "max_seq_len": 8,
        "max_requests": 4,
        "device": torch.device("cpu"),
        "dtype": torch.float32,
    }
    args.update(kw)
    return KVCache(**args)


def kv(b, t, v=1.0):
    return torch.full((b, 2, t, 4), v)


def test_append_then_decode_returns_prefix():
    c = make()
    c.reserve((7, 9))
    k0, _v0 = c.append(0, (7, 9), kv(2, 3, 1.0), kv(2, 3, 2.0), 0)
    assert k0.shape == (2, 2, 3, 4)
    c.append(1, (7, 9), kv(2, 3), kv(2, 3), 0)
    k1, v1 = c.append(0, (7, 9), kv(2, 1, 5.0), kv(2, 1, 6.0), 3)
    assert k1.shape == (2, 2, 4, 4)
    assert (k1[:, :, :3] == 1.0).all()
    assert (k1[:, :, 3] == 5.0).all()
    assert (v1[:, :, 3] == 6.0).all()


def test_reserve_rejects_duplicates_and_capacity_without_mutation():
    c = make(max_requests=2)
    with pytest.raises(CacheError):
        c.reserve((1, 1))
    assert c.active_requests == 0
    c.reserve((1, 2))
    with pytest.raises(CacheError):
        c.reserve((3, 4))
    assert c.active_requests == 2
    with pytest.raises(CacheError):
        c.reserve((1,))


def test_append_validation_no_mutation():
    c = make()
    c.reserve((5,))
    with pytest.raises(CacheError):
        c.append(0, (5, 5), kv(2, 1), kv(2, 1), 0)
    with pytest.raises(CacheError):
        c.append(0, (5,), kv(2, 1), kv(2, 1), 1)
    with pytest.raises(CacheError):
        c.append(0, (5,), kv(1, 9), kv(1, 9), 0)
    with pytest.raises(CacheError):
        c.append(0, (5,), kv(1, 1), kv(1, 2), 0)
    with pytest.raises(CacheError):
        c.append(0, (99,), kv(1, 1), kv(1, 1), 0)
    with pytest.raises(CacheError):
        c.append(2, (5,), kv(1, 1), kv(1, 1), 0)
    c.append(0, (5,), kv(1, 2), kv(1, 2), 0)
    with pytest.raises(CacheError):
        c.append(0, (5,), kv(1, 1), kv(1, 1), 0)
    with pytest.raises(CacheError):
        c.append(0, (5,), kv(1, 7), kv(1, 7), 2)
    k, _ = c.append(1, (5,), kv(1, 2, 3.0), kv(1, 2), 0)
    assert k.shape == (1, 2, 2, 4)


def test_release_idempotent_and_reuse():
    c = make(max_requests=1)
    c.reserve((3,))
    c.append(0, (3,), kv(1, 1), kv(1, 1), 0)
    c.release((3,))
    c.release((3,))
    assert c.active_requests == 0
    c.reserve((4,))
    assert c.active_requests == 1


def test_clear():
    c = make()
    c.reserve((1, 2))
    c.clear()
    assert c.active_requests == 0


def test_validate_positions():
    c = make()
    c.reserve((1,))
    c.validate((1,), 0, 2)
    c.append(0, (1,), kv(1, 2), kv(1, 2), 0)
    with pytest.raises(CacheError):
        c.validate((1,), 2, 1)
    c.append(1, (1,), kv(1, 2), kv(1, 2), 0)
    c.validate((1,), 2, 1)
