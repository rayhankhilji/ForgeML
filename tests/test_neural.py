import json

import pytest
import torch

from forgeml.analysis import graph_analysis
from forgeml.compiler import compile
from forgeml.frontend import from_torch
from forgeml.neural import MultimodalFusion, TransformerBlock, benchmark_neural


def test_transformer_analysis_counts_attention_work():
    model = TransformerBlock(hidden_size=32, num_heads=4, intermediate_size=64).eval()
    x = torch.randn(2, 5, 32)
    graph = from_torch(model, (x,))
    analysis = graph_analysis(graph)
    assert analysis["op_counts"]["sdpa"] == 1
    assert analysis["op_counts"]["layer_norm"] == 2
    attention = next(row for row in analysis["per_node"] if row["op"] == "sdpa")
    assert attention["flops"] > 0
    assert 0 < analysis["critical_path_ops"] <= len(graph.nodes)


def test_multimodal_explain_is_serializable():
    model = MultimodalFusion(
        batch_size=2,
        image_size=8,
        text_tokens=4,
        vocab_size=16,
        text_dim=8,
        fusion_dim=16,
        classes=3,
    ).eval()
    image = torch.randn(2, 3, 8, 8)
    tokens = torch.randint(0, 16, (2, 4))
    compiled = compile(model, (image, tokens))
    torch.testing.assert_close(compiled(image, tokens), model(image, tokens), rtol=1e-4, atol=1e-4)
    encoded = json.dumps(compiled.explain())
    assert '"conv2d"' in encoded
    assert '"embedding"' in encoded
    assert compiled.explain()["analysis"]["op_counts"]["conv2d"] == 1


def test_neural_benchmark_contract():
    report = benchmark_neural(warmup=0, repeats=2, seed=7)
    assert report["suite"] == "forgeml.neural"
    assert report["measured"] is True
    assert [row["name"] for row in report["workloads"]] == [
        "transformer_block_4x24x128",
        "vision_text_fusion_4x32_16",
    ]
    for row in report["workloads"]:
        assert row["correctness"]["optimized"]["passed"] is True
        assert len(row["timings"]["optimized"]["samples_ms"]) == 2
        assert row["optimized"]["analysis"]["flops"] > 0


def test_neural_benchmark_rejects_bad_request():
    with pytest.raises(ValueError, match="repeats"):
        benchmark_neural(repeats=1)
    with pytest.raises(ValueError, match="float32"):
        benchmark_neural(device="cpu", dtype=torch.float16)
