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
have NOT been validated in this environment.

## Commands

```bash
.venv/bin/python -m pytest tests/ -q          # GPU tests auto-skip without CUDA+triton
.venv/bin/python -m pytest tests/ -q -m gpu   # GPU-only run (requires CUDA GPU + triton)
.venv/bin/ruff check src/forgeml tests examples/compile_mlp.py
.venv/bin/ruff format --check src/forgeml tests examples/compile_mlp.py
.venv/bin/python examples/compile_mlp.py
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
- Backends: `torch` (default), `triton`, and `auto`. `triton`/`auto` map
  eligible CUDA `matmul`/`fused_linear_gelu` nodes to the Triton kernel in
  `src/forgeml/_triton.py` (mixed execution with torch fallback for the rest);
  `triton` requires CUDA + the triton package and errors clearly otherwise.
  No silent fallback if a selected Triton launch fails. The Triton path is
  untested on this Intel host (all GPU tests skip); do not claim it validated.
- Do not claim CPU hardware fusion for `fused_linear_gelu` (it is an op-level
  fusion), nor that `MemoryPlan.planned_bytes` is real allocator peak memory.
- No arbitrary `exec`/`eval` anywhere; ONNX attributes are data, not code.
- `explain()` output must stay JSON-serializable with no tensor values.
