import pytest
import torch

from nebula import DecoderConfig, TinyDecoder


def small():
    return DecoderConfig(
        vocab_size=32,
        hidden_size=16,
        num_heads=4,
        num_layers=2,
        intermediate_size=32,
        max_seq_len=16,
        seed=17,
    )


def test_forward_shape_and_finite():
    m = TinyDecoder(small()).eval()
    out = m(torch.tensor([[1, 2, 3], [4, 5, 6]]))
    assert out.shape == (2, 3, 32)
    assert torch.isfinite(out).all()


def test_deterministic_seed():
    a = TinyDecoder(small()).eval()
    b = TinyDecoder(small()).eval()
    x = torch.tensor([[1, 2, 3]])
    torch.testing.assert_close(a(x), b(x))


def test_external_rng_unchanged():
    torch.manual_seed(123)
    before = torch.rand(4)
    torch.manual_seed(123)
    TinyDecoder(small())
    after = torch.rand(4)
    assert torch.equal(before, after)


def test_causal_prefix_invariance():
    m = TinyDecoder(small()).eval()
    with torch.no_grad():
        short = m(torch.tensor([[1, 2, 3]]))
        long = m(torch.tensor([[1, 2, 3, 7, 8]]))
    torch.testing.assert_close(short, long[:, :3], rtol=1e-5, atol=1e-6)


def test_invalid_config():
    with pytest.raises(ValueError):
        DecoderConfig(vocab_size=0)
    with pytest.raises(ValueError):
        DecoderConfig(hidden_size=10, num_heads=3)
    with pytest.raises(ValueError):
        DecoderConfig(num_layers=-1)


def test_invalid_tokens():
    m = TinyDecoder(small()).eval()
    with pytest.raises(ValueError):
        m(torch.tensor([[32]]))
    with pytest.raises(ValueError):
        m(torch.tensor([[-1]]))
    with pytest.raises(ValueError):
        m(torch.zeros(1, 0, dtype=torch.long))
    with pytest.raises(ValueError):
        m(torch.zeros(1, 17, dtype=torch.long))
    with pytest.raises(ValueError):
        m(torch.zeros(1, 3, dtype=torch.float32))
