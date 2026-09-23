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
.venv/bin/python -m pytest tests/ -q
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
- Backend is `torch` only this milestone; other backend names must fail
  clearly. GPU works naturally when inputs and constants share the device.
- Do not claim CPU hardware fusion for `fused_linear_gelu` (it is an op-level
  fusion), nor that `MemoryPlan.planned_bytes` is real allocator peak memory.
- No arbitrary `exec`/`eval` anywhere; ONNX attributes are data, not code.
- `explain()` output must stay JSON-serializable with no tensor values.
