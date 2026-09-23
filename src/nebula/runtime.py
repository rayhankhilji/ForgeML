from __future__ import annotations

import contextlib
import logging
import math
import threading

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.profiler import record_function

from nebula.cache import CacheError, KVCache
from nebula.model import DecoderConfig, TinyDecoder
from nebula.parallel import ParallelContext

CMD_STOP = 0
CMD_STEP = 1
CMD_RELEASE = 2
_GENERATE_ID_BASE = 1 << 62
_MAX_REQUEST_ID = (1 << 62) - 1
_DEFAULT_CONFIG = DecoderConfig()
_LOG = logging.getLogger(__name__)


class EngineFailed(RuntimeError):
    pass


class _ShardBlock(nn.Module):
    def __init__(self, h: int, f: int):
        super().__init__()
        self.ln1_w = nn.Parameter(torch.empty(h), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.empty(h), requires_grad=False)
        self.ln2_w = nn.Parameter(torch.empty(h), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.empty(h), requires_grad=False)
        self.q = nn.Parameter(torch.empty(0, h), requires_grad=False)
        self.k = nn.Parameter(torch.empty(0, h), requires_grad=False)
        self.v = nn.Parameter(torch.empty(0, h), requires_grad=False)
        self.o = nn.Parameter(torch.empty(h, 0), requires_grad=False)
        self.up = nn.Parameter(torch.empty(0, h), requires_grad=False)
        self.down = nn.Parameter(torch.empty(h, 0), requires_grad=False)


class ShardedDecoder(nn.Module):
    def __init__(
        self,
        config: DecoderConfig,
        context: ParallelContext,
        *,
        dtype: torch.dtype = torch.float32,
        max_requests: int = 32,
    ):
        super().__init__()
        if config.num_heads % context.tp_size:
            raise ValueError("num_heads must be divisible by tp_size")
        if config.hidden_size % context.tp_size:
            raise ValueError("hidden_size must be divisible by tp_size")
        if config.intermediate_size % context.tp_size:
            raise ValueError("intermediate_size must be divisible by tp_size")
        if config.num_layers % context.pp_size:
            raise ValueError("num_layers must be divisible by pp_size")
        self.config = config
        self.context = context
        self.dtype = dtype
        device = context.device
        tp = context.tp_size
        ref = TinyDecoder(config)
        ref = ref.to(dtype=dtype)
        h = config.hidden_size
        f = config.intermediate_size
        layers_per_stage = config.num_layers // context.pp_size
        self.owned_layer_ids = tuple(
            range(
                context.pp_rank * layers_per_stage,
                (context.pp_rank + 1) * layers_per_stage,
            )
        )
        row_lo = context.tp_rank * h // tp
        row_hi = (context.tp_rank + 1) * h // tp
        f_lo = context.tp_rank * f // tp
        f_hi = (context.tp_rank + 1) * f // tp
        self.blocks = nn.ModuleList()
        for lid in self.owned_layer_ids:
            src = ref.blocks[lid]
            blk = _ShardBlock(h, f)
            with torch.no_grad():
                blk.ln1_w.copy_(src.ln1.weight)
                blk.ln1_b.copy_(src.ln1.bias)
                blk.ln2_w.copy_(src.ln2.weight)
                blk.ln2_b.copy_(src.ln2.bias)
                blk.q = nn.Parameter(src.q.weight[row_lo:row_hi].clone(), requires_grad=False)
                blk.k = nn.Parameter(src.k.weight[row_lo:row_hi].clone(), requires_grad=False)
                blk.v = nn.Parameter(src.v.weight[row_lo:row_hi].clone(), requires_grad=False)
                blk.o = nn.Parameter(src.o.weight[:, row_lo:row_hi].clone(), requires_grad=False)
                blk.up = nn.Parameter(src.up.weight[f_lo:f_hi].clone(), requires_grad=False)
                blk.down = nn.Parameter(src.down.weight[:, f_lo:f_hi].clone(), requires_grad=False)
            self.blocks.append(blk)
        if context.pp_rank == 0:
            self.token_embedding = nn.Parameter(
                ref.token_embedding.weight.clone(), requires_grad=False
            )
            self.position_embedding = nn.Parameter(
                ref.position_embedding.weight.clone(), requires_grad=False
            )
        if context.pp_rank == context.pp_size - 1:
            self.final_norm_w = nn.Parameter(ref.final_norm.weight.clone(), requires_grad=False)
            self.final_norm_b = nn.Parameter(ref.final_norm.bias.clone(), requires_grad=False)
            self.lm_head = nn.Parameter(ref.lm_head.weight.clone(), requires_grad=False)
        del ref
        self.to(device=device, dtype=dtype)
        self.num_heads_local = config.num_heads // tp
        self.head_dim = h // config.num_heads
        self.cache = KVCache(
            layer_ids=self.owned_layer_ids,
            num_heads=self.num_heads_local,
            head_dim=self.head_dim,
            max_seq_len=config.max_seq_len,
            max_requests=max_requests,
            device=device,
            dtype=dtype,
        )

    def _attention(self, blk, x, request_ids, start_pos, layer_id):
        ctx = self.context
        b, t, h = x.shape
        hl = self.num_heads_local
        hd = self.head_dim
        h1 = F.layer_norm(x, (h,), blk.ln1_w, blk.ln1_b, 1e-5)
        q = F.linear(h1, blk.q).view(b, t, hl, hd).transpose(1, 2)
        k = F.linear(h1, blk.k).view(b, t, hl, hd).transpose(1, 2)
        v = F.linear(h1, blk.v).view(b, t, hl, hd).transpose(1, 2)
        with record_function("nebula::kv_append"):
            k_all, v_all = self.cache.append(
                layer_id, request_ids, k.contiguous(), v.contiguous(), start_pos
            )
        end = start_pos + t
        scores = torch.matmul(q, k_all.transpose(-1, -2)) / math.sqrt(hd)
        key_pos = torch.arange(end, device=x.device)
        query_pos = start_pos + torch.arange(t, device=x.device)
        scores = scores.masked_fill(key_pos[None, :] > query_pos[:, None], float("-inf"))
        attn = torch.softmax(scores.float(), dim=-1).to(x.dtype)
        out = torch.matmul(attn, v_all).transpose(1, 2).reshape(b, t, hl * hd)
        partial = F.linear(out, blk.o)
        ctx.all_reduce(partial)
        return partial

    def _feed_forward(self, blk, x):
        h = self.config.hidden_size
        h2 = F.layer_norm(x, (h,), blk.ln2_w, blk.ln2_b, 1e-5)
        mid = F.gelu(F.linear(h2, blk.up), approximate="tanh")
        partial = F.linear(mid, blk.down)
        self.context.all_reduce(partial)
        return partial

    def forward(
        self, tokens: torch.Tensor, request_ids: tuple[int, ...], start_pos: int
    ) -> torch.Tensor:
        ctx = self.context
        b, t = tokens.shape
        h = self.config.hidden_size
        if start_pos == 0:
            self.cache.reserve(request_ids)
        if ctx.pp_rank == 0:
            pos = torch.arange(start_pos, start_pos + t, device=tokens.device)
            x = F.embedding(tokens, self.token_embedding) + F.embedding(
                pos, self.position_embedding
            )
        else:
            x = torch.empty(b, t, h, dtype=self.dtype, device=ctx.device)
            with record_function("nebula::pp_recv"):
                dist.recv(x, src=ctx.rank - ctx.tp_size)
        for blk, lid in zip(self.blocks, self.owned_layer_ids):
            with record_function("nebula::attention"):
                attn = self._attention(blk, x, request_ids, start_pos, lid)
            x = x + attn
            with record_function("nebula::feed_forward"):
                x = x + self._feed_forward(blk, x)
        if ctx.pp_rank != ctx.pp_size - 1:
            with record_function("nebula::pp_send"):
                dist.send(x.contiguous(), dst=ctx.rank + ctx.tp_size)
            logits = torch.empty(b, t, self.config.vocab_size, dtype=self.dtype, device=ctx.device)
        else:
            x = F.layer_norm(x, (h,), self.final_norm_w, self.final_norm_b, 1e-5)
            logits = F.linear(x, self.lm_head)
        if ctx.world_size > 1:
            dist.broadcast(logits, src=(ctx.pp_size - 1) * ctx.tp_size)
        return logits


def _validate_prompt(config: DecoderConfig, prompt: list[int], max_new: int) -> None:
    if not isinstance(prompt, (list, tuple)) or not prompt:
        raise ValueError("prompt must be a non-empty list of token ids")
    for tok in prompt:
        if not isinstance(tok, int) or isinstance(tok, bool):
            raise TypeError("prompt token ids must be ints")
        if tok < 0 or tok >= config.vocab_size:
            raise ValueError(f"token id {tok} out of range")
    if not isinstance(max_new, int) or isinstance(max_new, bool) or max_new < 0:
        raise ValueError("max_new_tokens must be a non-negative int")
    if len(prompt) + max_new > config.max_seq_len:
        raise ValueError("prompt length + max_new_tokens exceeds max_seq_len")


class Engine:
    def __init__(
        self,
        config: DecoderConfig = _DEFAULT_CONFIG,
        *,
        context: ParallelContext | None = None,
        dtype: torch.dtype = torch.float32,
        max_requests: int = 32,
    ):
        self._owns_context = context is None
        if context is None:
            context = ParallelContext.from_env()
        if not isinstance(max_requests, int) or isinstance(max_requests, bool) or max_requests <= 0:
            raise ValueError("max_requests must be a positive integer")
        if dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError(f"unsupported dtype {dtype}")
        if context.device.type != "cuda" and dtype != torch.float32:
            raise ValueError("low-precision dtype requires CUDA")
        self.config = config
        self.context = context
        self.max_requests = max_requests
        self.decoder = ShardedDecoder(config, context, dtype=dtype, max_requests=max_requests)
        self._healthy = True
        self._closed = False
        self._lock = threading.RLock()
        self._gen_counter = _GENERATE_ID_BASE
        if dist.is_available() and dist.is_initialized():
            self._comm_device = (
                context.device if dist.get_backend() == "nccl" else torch.device("cpu")
            )
        else:
            self._comm_device = torch.device("cpu")

    @property
    def healthy(self) -> bool:
        return self._healthy and not self._closed

    @property
    def _is_leader(self) -> bool:
        return self.context.rank == 0

    def _world(self) -> bool:
        return self.context.world_size > 1 and dist.is_initialized()

    def _validate_request(
        self,
        tokens: torch.Tensor,
        request_ids: tuple[int, ...],
        start_pos: int,
        internal: bool = False,
    ) -> None:
        if not isinstance(tokens, torch.Tensor):
            raise TypeError("tokens must be a torch.Tensor")
        if tokens.dtype != torch.long:
            raise ValueError("tokens must be a long tensor")
        if tokens.dim() != 2 or tokens.shape[0] == 0 or tokens.shape[1] == 0:
            raise ValueError("tokens must have shape [B, T] with B, T >= 1")
        if tokens.shape[0] != len(request_ids):
            raise ValueError("request_ids must match batch size")
        if tokens.shape[0] > self.max_requests:
            raise ValueError("batch exceeds max_requests")
        if start_pos < 0 or start_pos + tokens.shape[1] > self.config.max_seq_len:
            raise ValueError("sequence exceeds max_seq_len")
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("duplicate request ids")
        for r in request_ids:
            if not isinstance(r, int) or isinstance(r, bool) or r < 0:
                raise ValueError(f"invalid request id {r!r}")
            if r >= 1 << 63:
                raise ValueError("request id exceeds int64 range")
            if r > _MAX_REQUEST_ID and not internal:
                raise ValueError("request id exceeds manual range")
        if not isinstance(start_pos, int) or isinstance(start_pos, bool):
            raise TypeError("start_pos must be an integer")
        if (tokens < 0).any() or (tokens >= self.config.vocab_size).any():
            raise ValueError("token ids out of range")

    def _bcast(self, tensor: torch.Tensor) -> None:
        if self._world():
            dist.broadcast(tensor, src=0)

    def _invalidate(self) -> None:
        self._healthy = False
        self.decoder.cache.clear()

    def _collective_forward(
        self, tokens: torch.Tensor, request_ids: tuple[int, ...], start_pos: int
    ) -> torch.Tensor:
        err = 0
        try:
            if start_pos == 0:
                self.decoder.cache._check_ids(request_ids)
                for r in request_ids:
                    if r in self.decoder.cache._entries:
                        raise CacheError(f"request id {r} already reserved")
                if (
                    len(self.decoder.cache._entries) + len(request_ids)
                    > self.decoder.cache.max_requests
                ):
                    raise CacheError("cache capacity exceeded")
            else:
                self.decoder.cache.validate(request_ids, start_pos, tokens.shape[1])
        except CacheError:
            err = 1
        if self._world():
            flag = torch.tensor([err], dtype=torch.int64, device=self._comm_device)
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            err = int(flag.item())
        if err:
            raise EngineFailed("cache preflight failed")
        with torch.inference_mode():
            return self.decoder.forward(tokens, request_ids, start_pos)

    def forward(
        self,
        tokens: torch.Tensor,
        request_ids: tuple[int, ...],
        start_pos: int = 0,
        _internal: bool = False,
    ) -> torch.Tensor:
        with self._lock:
            if not self._healthy or self._closed:
                raise EngineFailed("engine is not healthy")
            self._validate_request(tokens, request_ids, start_pos, internal=_internal)
            if self._world() and not self._is_leader:
                raise EngineFailed("forward must be invoked on rank 0")
            try:
                if self._world():
                    header = torch.tensor(
                        [CMD_STEP, tokens.shape[0], tokens.shape[1], start_pos],
                        dtype=torch.int64,
                        device=self._comm_device,
                    )
                    self._bcast(header)
                    ids = torch.tensor(
                        list(request_ids), dtype=torch.int64, device=self._comm_device
                    )
                    self._bcast(ids)
                    wire = tokens.to(self._comm_device).contiguous()
                    self._bcast(wire)
                return self._collective_forward(
                    tokens.to(self.context.device), request_ids, start_pos
                )
            except Exception as exc:
                self._invalidate()
                raise EngineFailed("engine execution failed") from exc

    def serve(self) -> None:
        if self._is_leader or not self._world():
            raise EngineFailed("serve() is for non-zero ranks")
        try:
            while True:
                header = torch.zeros(4, dtype=torch.int64, device=self._comm_device)
                dist.broadcast(header, src=0)
                cmd, batch, time_, start_pos = (int(v) for v in header.tolist())
                if cmd == CMD_STOP:
                    if batch != 0 or time_ != 0 or start_pos != 0:
                        raise EngineFailed("malformed STOP header")
                    return
                if cmd == CMD_STEP:
                    if (
                        batch <= 0
                        or batch > self.max_requests
                        or time_ <= 0
                        or time_ > self.config.max_seq_len
                        or start_pos < 0
                        or start_pos + time_ > self.config.max_seq_len
                    ):
                        raise EngineFailed("malformed STEP header")
                    ids = torch.zeros(batch, dtype=torch.int64, device=self._comm_device)
                    dist.broadcast(ids, src=0)
                    tokens = torch.zeros(batch, time_, dtype=torch.int64, device=self._comm_device)
                    dist.broadcast(tokens, src=0)
                    self._collective_forward(
                        tokens.to(self.context.device),
                        tuple(int(i) for i in ids.tolist()),
                        start_pos,
                    )
                elif cmd == CMD_RELEASE:
                    if batch < 0 or batch > self.max_requests or time_ != 0 or start_pos != 0:
                        raise EngineFailed("malformed RELEASE header")
                    ids = torch.zeros(batch, dtype=torch.int64, device=self._comm_device)
                    dist.broadcast(ids, src=0)
                    self.decoder.cache.release(tuple(int(i) for i in ids.tolist()))
                else:
                    raise EngineFailed(f"unknown command {cmd}")
        except Exception as exc:
            self._invalidate()
            raise EngineFailed("worker execution failed") from exc

    def release(self, request_ids: tuple[int, ...]) -> None:
        with self._lock:
            if self._closed or not self._healthy:
                raise EngineFailed("engine is not healthy")
            if self._world() and not self._is_leader:
                raise EngineFailed("release must be invoked on rank 0")
            for r in request_ids:
                if not isinstance(r, int) or isinstance(r, bool) or r < 0 or r >= 1 << 63:
                    raise ValueError(f"invalid request id {r!r}")
            if len(request_ids) > self.max_requests:
                raise ValueError("release batch exceeds max_requests")
            if not request_ids:
                return
            try:
                if self._world():
                    header = torch.tensor(
                        [CMD_RELEASE, len(request_ids), 0, 0],
                        dtype=torch.int64,
                        device=self._comm_device,
                    )
                    self._bcast(header)
                    ids = torch.tensor(
                        list(request_ids), dtype=torch.int64, device=self._comm_device
                    )
                    self._bcast(ids)
                self.decoder.cache.release(request_ids)
            except Exception as exc:
                self._invalidate()
                raise EngineFailed("engine execution failed") from exc

    def generate(self, prompts: list[list[int]], max_new_tokens: int) -> list[list[int]]:
        with self._lock:
            if not self._healthy or self._closed:
                raise EngineFailed("engine is not healthy")
            if not self._is_leader and self._world():
                raise EngineFailed("generate must be invoked on rank 0")
            if (
                not isinstance(max_new_tokens, int)
                or isinstance(max_new_tokens, bool)
                or max_new_tokens < 0
            ):
                raise ValueError("max_new_tokens must be a non-negative int")
            for p in prompts:
                _validate_prompt(self.config, p, max_new_tokens)
            if len(prompts) > self.max_requests:
                raise ValueError("batch exceeds max_requests")
            if not prompts:
                return []
            if max_new_tokens == 0:
                return [[] for _ in prompts]
            results: list[list[int]] = [[] for _ in prompts]
            cohorts: dict[int, list[int]] = {}
            for i, p in enumerate(prompts):
                cohorts.setdefault(len(p), []).append(i)
            issued: list[int] = []
            try:
                for plen, idxs in cohorts.items():
                    ids = []
                    for _ in idxs:
                        while self._gen_counter in self.decoder.cache._entries:
                            self._gen_counter += 1
                        if self._gen_counter >= 1 << 63:
                            raise EngineFailed("request id space exhausted")
                        ids.append(self._gen_counter)
                        self._gen_counter += 1
                    issued.extend(ids)
                    tokens = torch.tensor([prompts[i] for i in idxs], dtype=torch.long)
                    logits = self.forward(tokens, tuple(ids), 0, _internal=True)
                    cont = [[] for _ in idxs]
                    nxt = logits[:, -1, :].argmax(-1)
                    for j, tok in enumerate(nxt.tolist()):
                        cont[j].append(tok)
                    for step in range(1, max_new_tokens):
                        step_tokens = nxt.view(-1, 1)
                        logits = self.forward(
                            step_tokens, tuple(ids), plen + step - 1, _internal=True
                        )
                        nxt = logits[:, -1, :].argmax(-1)
                        for j, tok in enumerate(nxt.tolist()):
                            cont[j].append(tok)
                    for j, i in enumerate(idxs):
                        results[i] = cont[j]
            finally:
                if issued and self._healthy:
                    self.release(tuple(issued))
                elif issued:
                    self.decoder.cache.clear()
            return results

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                if self._world() and self._is_leader and self._healthy:
                    header = torch.tensor(
                        [CMD_STOP, 0, 0, 0],
                        dtype=torch.int64,
                        device=self._comm_device,
                    )
                    with contextlib.suppress(Exception):
                        self._bcast(header)
                self.decoder.cache.clear()
                self._healthy = False
            finally:
                if self._owns_context:
                    with contextlib.suppress(Exception):
                        self.context.close()
