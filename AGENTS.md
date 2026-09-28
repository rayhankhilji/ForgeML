# ForgeML — agent notes

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev,onnx]"
```

Note: this host is truly Intel — `uname -m` = x86_64, `hw.optional.arm64`
absent, `arch -arm64` unavailable — so no native-ARM env is possible.
`torch==2.14.0` has no `macosx_*_x86_64` wheel; the local `.venv` runs
`torch==2.2.2` + `numpy==1.26.4`. The `pyproject.toml` pins are declared but
have NOT been validated on this Intel host. GitHub Actions on Linux validated
these exact modern dependency pins, compiler tests and package build at commit
4ee14c9; use CI for the current full-suite result.

## Commands

```bash
.venv/bin/python -m pytest tests/ -q          # GPU tests auto-skip without CUDA+triton
.venv/bin/python -m pytest tests/ -q -m gpu   # GPU-only run (requires CUDA GPU + triton)
.venv/bin/ruff check src tests examples
.venv/bin/ruff format --check src tests examples
.venv/bin/python examples/compile_mlp.py
.venv/bin/python -m forgeml.benchmarks benchmark --suite neural --warmup 5 --repeats 25 --threads 1 --output /tmp/neural-cpu.json
```

## Contracts

- Public API: `forgeml.compile`, `from_torch`, `from_onnx`, `Graph`,
  `GraphBuilder`, `TensorSpec` (see `src/forgeml/` for exact signatures).
- Inference only: all compiled execution runs under `torch.inference_mode`;
  parameters are detached clones captured at compile time. Never mutate the
  source model or example inputs. Reject training-mode models with active
  Dropout/BatchNorm.
- Shapes are static: every dimension must be a positive integer; zero/dynamic
  dims fail in `TensorSpec`/`Graph.validate()`.
- Operator semantics are explicit and bounded: `matmul`, `linear`, `add`, `mul`,
  `relu`, `gelu`, `reshape`, `transpose`, `softmax`, `fused_linear_gelu`,
  `layer_norm`,
  `sdpa`, `conv2d`, `embedding`, and `narrow` (static slice; emitted only by
  `fuse_shared_projections`). SDPA is inference-only: zero dropout, no
  `attn_mask`, no `enable_gqa`. Conv2d is NCHW/OIHW with zero padding only.
  Embedding lookup requires int64 indices and rejects padding/max-norm/sparse
  module options. LayerNorm requires static trailing normalized dimensions.
  The core `linear` operator is rank-2; the FX frontend wraps higher-rank or
  rank-1 activations in static reshape views around that core operation.
- ONNX remains restricted to the default domain/opset 13–22: MatMul, Add, Mul,
  Relu, Gelu, Reshape, Transpose, Softmax, Constant, Identity, Gemm, Conv,
  LayerNormalization, and axis-0 Gather. Reject unsupported attrs/domains.
- Backends: `torch` (default), `triton`, and `auto`. `triton`/`auto` map only
  eligible CUDA `matmul`/`linear`/`fused_linear_gelu` nodes to the Triton kernel in
  `src/forgeml/_triton.py` (mixed execution with torch fallback for the rest);
  `triton` requires CUDA + the triton package and errors clearly otherwise.
  No silent fallback if a selected Triton launch fails. The Triton path is
  untested on this Intel host (all GPU tests skip); do not claim it validated.
- Do not claim CPU hardware fusion for `fused_linear_gelu` (it is an op-level
  fusion), nor that `MemoryPlan.planned_bytes` is real allocator peak memory.
  `reshape`/`transpose`/`narrow` intermediates are view/borrowed values; their
  producers' physical lifetimes must cover all view consumers before a slot may
  be reused. The slot arena covers only `out=`-capable ops (`matmul`, `linear`,
  `add`, `mul`, `relu`); other intermediates are refcounted temporaries.
  `CompiledModel.__call__` reuses a persistent arena and is NOT reentrant.
- Graph analysis in `explain()` is a static cost model, not hardware counters:
  report FLOP/logical-byte/arithmetic-intensity/critical-path bounds as modeled
  values and preserve `optimization_delta` semantics.
- No arbitrary `exec`/`eval` anywhere; ONNX attributes are data, not code.
- `explain()` output must stay JSON-serializable with no tensor values.
- Nebula uses rank-zero command authority; nonzero ranks call `Engine.serve()`.
  At engine init, ranks broadcast a SHA-256 config/dtype/max_requests digest
  and must agree before the decoder is built (EngineFailed on mismatch).
  Run distributed correctness with `pytest -q -m distributed`; CPU uses Gloo,
  CUDA uses NCCL. Never claim CPU multiprocess tests validate GPU scaling.
- Runtime errors invalidate caches and fail the engine; do not retry partially
  executed requests. Batcher admission counts queued and in-flight requests.
- Run examples with `python examples/nebula_generate.py --tokens 2` and
  `python examples/nebula_batching.py`. Limit Torch/OMP threads for small tests.
- Benchmark reports are evidence: preserve raw samples and source provenance.
  Render figures from frozen captures rather than re-running their inputs.
