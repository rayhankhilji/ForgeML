from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class DecoderConfig:
    vocab_size: int = 256
    hidden_size: int = 128
    num_heads: int = 8
    num_layers: int = 4
    intermediate_size: int = 256
    max_seq_len: int = 128
    seed: int = 17

    def __post_init__(self):
        for name in (
            "vocab_size",
            "hidden_size",
            "num_heads",
            "num_layers",
            "intermediate_size",
            "max_seq_len",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")


class DecoderBlock(nn.Module):
    def __init__(self, config: DecoderConfig):
        super().__init__()
        h = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = h // config.num_heads
        self.ln1 = nn.LayerNorm(h, eps=1e-5)
        self.ln2 = nn.LayerNorm(h, eps=1e-5)
        self.q = nn.Linear(h, h, bias=False)
        self.k = nn.Linear(h, h, bias=False)
        self.v = nn.Linear(h, h, bias=False)
        self.o = nn.Linear(h, h, bias=False)
        self.up = nn.Linear(h, config.intermediate_size, bias=False)
        self.down = nn.Linear(config.intermediate_size, h, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, h = x.shape
        hd = self.head_dim
        n = self.num_heads
        res = x
        h1 = self.ln1(x)
        q = self.q(h1).view(b, t, n, hd).transpose(1, 2)
        k = self.k(h1).view(b, t, n, hd).transpose(1, 2)
        v = self.v(h1).view(b, t, n, hd).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(hd)
        mask = torch.full((t, t), float("-inf"), device=x.device).triu(1)
        scores = scores + mask
        attn = torch.softmax(scores.float(), dim=-1).to(x.dtype)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(b, t, h)
        x = res + self.o(out)
        res = x
        h2 = self.ln2(x)
        x = res + self.down(F.gelu(self.up(h2), approximate="tanh"))
        return x


class TinyDecoder(nn.Module):
    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.config = config
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed)
            self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
            self.position_embedding = nn.Embedding(config.max_seq_len, config.hidden_size)
            self.blocks = nn.ModuleList(DecoderBlock(config) for _ in range(config.num_layers))
            self.final_norm = nn.LayerNorm(config.hidden_size, eps=1e-5)
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if not isinstance(tokens, torch.Tensor) or tokens.dtype != torch.long:
            raise ValueError("tokens must be a long tensor")
        if tokens.dim() != 2 or tokens.shape[0] == 0 or tokens.shape[1] == 0:
            raise ValueError("tokens must have shape [B, T] with B, T >= 1")
        if tokens.shape[1] > self.config.max_seq_len:
            raise ValueError("sequence length exceeds max_seq_len")
        if (tokens < 0).any() or (tokens >= self.config.vocab_size).any():
            raise ValueError("token ids out of range")
        _, t = tokens.shape
        x = self.token_embedding(tokens) + self.position_embedding(
            torch.arange(t, device=tokens.device)
        )
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.final_norm(x))
