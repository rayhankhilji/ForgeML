from __future__ import annotations

import time

import torch
import torch.nn.functional as F
from torch import nn

from forgeml.compiler import compile
from forgeml.measurement import (
    check_output,
    environment,
    measure_variants,
    synchronize,
)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int = 128,
        num_heads: int = 8,
        intermediate_size: int = 256,
    ):
        super().__init__()
        if hidden_size <= 0 or num_heads <= 0 or hidden_size % num_heads:
            raise ValueError("hidden size must be positive and divisible by heads")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.q = nn.Linear(hidden_size, hidden_size)
        self.k = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, hidden_size)
        self.o = nn.Linear(hidden_size, hidden_size)
        self.up = nn.Linear(hidden_size, intermediate_size)
        self.down = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, hidden = x.shape
        residual = x
        normalized = self.ln1(x).reshape(batch * seq, hidden)
        q = self.q(normalized).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(normalized).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(normalized).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attended = attended.transpose(1, 2).reshape(batch * seq, hidden)
        x = residual + self.o(attended).reshape(batch, seq, hidden)
        residual = x
        normalized = self.ln2(x).reshape(batch * seq, hidden)
        return residual + self.down(F.gelu(self.up(normalized), approximate="tanh")).reshape(
            batch, seq, hidden
        )


class MultimodalFusion(nn.Module):
    def __init__(
        self,
        batch_size: int = 4,
        image_size: int = 32,
        image_channels: int = 3,
        text_tokens: int = 16,
        vocab_size: int = 256,
        text_dim: int = 32,
        fusion_dim: int = 64,
        classes: int = 10,
    ):
        super().__init__()
        if min(batch_size, image_size, image_channels, text_tokens, vocab_size) <= 0:
            raise ValueError("multimodal dimensions must be positive")
        if text_dim <= 0 or fusion_dim <= 0 or classes <= 0:
            raise ValueError("text, fusion and class dimensions must be positive")
        self.batch_size = batch_size
        self.text_tokens = text_tokens
        self.text_dim = text_dim
        self.fusion_dim = fusion_dim
        self.classes = classes
        self.vision = nn.Conv2d(image_channels, 16, kernel_size=3, padding=1)
        self.vision_head = nn.Linear(16 * image_size * image_size, fusion_dim)
        self.text_embedding = nn.Embedding(vocab_size, text_dim)
        self.text_norm = nn.LayerNorm(text_dim)
        self.text_head = nn.Linear(text_dim, fusion_dim)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(fusion_dim, classes),
        )

    def forward(self, image: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        vision = F.gelu(self.vision(image)).reshape(self.batch_size, -1)
        vision = self.vision_head(vision).reshape(self.batch_size, 1, self.fusion_dim)
        text = self.text_embedding(tokens).reshape(
            self.batch_size * self.text_tokens, self.text_dim
        )
        text = self.text_head(self.text_norm(text)).reshape(
            self.batch_size, self.text_tokens, self.fusion_dim
        )
        fused = (vision + text).reshape(self.batch_size * self.text_tokens, self.fusion_dim)
        return self.fusion(fused).reshape(self.batch_size, self.text_tokens, self.classes)


def _workload(name: str, model: nn.Module, inputs: tuple[torch.Tensor, ...]) -> dict:
    return {"name": name, "model": model, "inputs": inputs}


@torch.inference_mode()
def benchmark_neural(
    *,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    backend: str = "torch",
    warmup: int = 3,
    repeats: int = 15,
    seed: int = 2026,
    autotune: bool = False,
) -> dict:
    target = torch.device(device)
    if target.type not in ("cpu", "cuda"):
        raise ValueError("neural benchmarks support CPU or CUDA")
    if target.type == "cpu" and dtype != torch.float32:
        raise ValueError("CPU neural benchmarks require float32")
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no CUDA device is available")
    if warmup < 0 or repeats < 2:
        raise ValueError("warmup must be >= 0 and repeats >= 2")
    torch.manual_seed(seed)
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    previous_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        transformer = TransformerBlock().to(device=target, dtype=dtype).eval()
        hidden = torch.randn(4, 24, 128, device=target, dtype=dtype)
        multimodal = MultimodalFusion().to(device=target, dtype=dtype).eval()
        image = torch.randn(4, 3, 32, 32, device=target, dtype=dtype)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        tokens = torch.randint(256, (4, 16), generator=generator).to(target)
        workloads = (
            _workload("transformer_block_4x24x128", transformer, (hidden,)),
            _workload("vision_text_fusion_4x32_16", multimodal, (image, tokens)),
        )
        rows = []
        for workload in workloads:
            model = workload["model"]
            inputs = workload["inputs"]
            reference = model(*inputs)
            synchronize(target)
            start = time.perf_counter_ns()
            unoptimized = compile(model, inputs, backend=backend, optimize=False)
            synchronize(target)
            unoptimized_compile_ms = (time.perf_counter_ns() - start) / 1_000_000
            start = time.perf_counter_ns()
            optimized = compile(model, inputs, backend=backend)
            synchronize(target)
            optimized_compile_ms = (time.perf_counter_ns() - start) / 1_000_000
            tuning = optimized.autotune(*inputs) if autotune else None
            correctness = {
                "unoptimized": check_output(unoptimized(*inputs), reference),
                "optimized": check_output(optimized(*inputs), reference),
            }
            timings = measure_variants(
                {
                    "eager": lambda model=model, inputs=inputs: model(*inputs),
                    "unoptimized": lambda model=unoptimized, inputs=inputs: model(*inputs),
                    "optimized": lambda model=optimized, inputs=inputs: model(*inputs),
                },
                device=target,
                warmup=warmup,
                repeats=repeats,
            )
            eager_ms = timings["eager"]["median_ms"]
            for timing in timings.values():
                timing["speedup_vs_eager"] = eager_ms / timing["median_ms"]
            rows.append(
                {
                    "name": workload["name"],
                    "dtype": str(dtype),
                    "correctness": correctness,
                    "timings": timings,
                    "compile_ms": {
                        "unoptimized": unoptimized_compile_ms,
                        "optimized": optimized_compile_ms,
                    },
                    "unoptimized": unoptimized.explain(),
                    "optimized": optimized.explain(),
                    "autotuning": tuning,
                }
            )
        return {
            "schema_version": 1,
            "suite": "forgeml.neural",
            "measured": True,
            "environment": environment(target),
            "settings": {
                "seed": seed,
                "warmup": warmup,
                "repeats": repeats,
                "backend": backend,
                "dtype": str(dtype),
                "autotune": autotune,
                "tf32": False,
                "gradients": False,
                "variant_order": "rotating_per_repetition",
            },
            "scope": {
                "latency": "whole_call_including_python_dispatch_and_output_ownership",
                "excluded": ["model_initialization", "compilation", "autotuning", "warmup"],
                "memory": "planned_intermediate_storage_not_allocator_peak",
                "transformer": "single_prefill_block_with_causal_sdpa_not_incremental_decode",
                "multimodal": "late_additive_fusion_not_a_pretrained_foundation_model",
            },
            "workloads": rows,
        }
    finally:
        torch.set_float32_matmul_precision(previous_precision)
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
