from __future__ import annotations

import torch


class CacheError(RuntimeError):
    pass


class KVCache:
    def __init__(
        self,
        *,
        layer_ids: tuple[int, ...],
        num_heads: int,
        head_dim: int,
        max_seq_len: int,
        max_requests: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        for name, v in (
            ("num_heads", num_heads),
            ("head_dim", head_dim),
            ("max_seq_len", max_seq_len),
            ("max_requests", max_requests),
        ):
            if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
                raise CacheError(f"{name} must be a positive integer")
        if (
            not layer_ids
            or any(not isinstance(i, int) or isinstance(i, bool) or i < 0 for i in layer_ids)
            or len(set(layer_ids)) != len(layer_ids)
        ):
            raise CacheError("layer_ids must be unique non-negative integers")
        if not isinstance(device, torch.device):
            raise CacheError("device must be a torch.device")
        if not isinstance(dtype, torch.dtype):
            raise CacheError("dtype must be a torch.dtype")
        self.layer_ids = tuple(layer_ids)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.max_requests = max_requests
        self.device = device
        self.dtype = dtype
        self._entries: dict[int, dict[int, list]] = {}

    @property
    def active_requests(self) -> int:
        return len(self._entries)

    def _check_ids(self, request_ids: tuple[int, ...]) -> None:
        if not request_ids:
            raise CacheError("request_ids must be non-empty")
        if len(set(request_ids)) != len(request_ids):
            raise CacheError("duplicate request ids")
        for r in request_ids:
            if not isinstance(r, int) or isinstance(r, bool) or r < 0 or r >= 1 << 63:
                raise CacheError(f"invalid request id {r!r}")

    def reserve(self, request_ids: tuple[int, ...]) -> None:
        self._check_ids(request_ids)
        for r in request_ids:
            if r in self._entries:
                raise CacheError(f"request id {r} already reserved")
        if len(self._entries) + len(request_ids) > self.max_requests:
            raise CacheError("cache capacity exceeded")
        for r in request_ids:
            self._entries[r] = {
                lid: [
                    torch.empty(
                        self.num_heads,
                        self.max_seq_len,
                        self.head_dim,
                        device=self.device,
                        dtype=self.dtype,
                    ),
                    torch.empty(
                        self.num_heads,
                        self.max_seq_len,
                        self.head_dim,
                        device=self.device,
                        dtype=self.dtype,
                    ),
                    0,
                ]
                for lid in self.layer_ids
            }

    def validate(self, request_ids: tuple[int, ...], start_pos: int, count: int) -> None:
        self._check_ids(request_ids)
        for r in request_ids:
            if r not in self._entries:
                raise CacheError(f"request id {r} is not reserved")
        for name, v in (("start_pos", start_pos), ("count", count)):
            if not isinstance(v, int) or isinstance(v, bool):
                raise CacheError(f"{name} must be an integer")
        if count <= 0:
            raise CacheError("count must be positive")
        if start_pos < 0 or start_pos + count > self.max_seq_len:
            raise CacheError("positions out of range")
        for r in request_ids:
            for lid, (_, _, length) in self._entries[r].items():
                if length != start_pos:
                    raise CacheError(
                        f"layer {lid} for request {r} at {length}, expected {start_pos}"
                    )

    def append(
        self,
        layer_id: int,
        request_ids: tuple[int, ...],
        key: torch.Tensor,
        value: torch.Tensor,
        start_pos: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if layer_id not in self.layer_ids:
            raise CacheError(f"layer {layer_id} not owned by this cache")
        self._check_ids(request_ids)
        for r in request_ids:
            if r not in self._entries:
                raise CacheError(f"request id {r} is not reserved")
        if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
            raise CacheError("key/value must be torch.Tensor")
        if not isinstance(start_pos, int) or isinstance(start_pos, bool):
            raise CacheError("start_pos must be an integer")
        if key.dim() != 4 or value.dim() != 4:
            raise CacheError("key/value must be [B, heads, T, head_dim]")
        b, heads, count, hd = key.shape
        if (
            b != len(request_ids)
            or heads != self.num_heads
            or hd != self.head_dim
            or value.shape != key.shape
        ):
            raise CacheError(f"bad key/value shape {tuple(key.shape)}")
        if count <= 0 or start_pos < 0 or start_pos + count > self.max_seq_len:
            raise CacheError("positions out of range")
        if key.dtype != self.dtype or value.dtype != self.dtype:
            raise CacheError("key/value dtype mismatch")
        if key.device != self.device or value.device != self.device:
            raise CacheError("key/value device mismatch")
        for r in request_ids:
            for lid, (_, _, length) in self._entries[r].items():
                if lid == layer_id:
                    if length != start_pos:
                        raise CacheError(
                            f"layer {layer_id} request {r} at {length}, expected {start_pos}"
                        )
                elif length not in (start_pos, start_pos + count):
                    raise CacheError(
                        f"layer {lid} request {r} at {length} while appending "
                        f"layer {layer_id} at {start_pos}+{count}"
                    )
        end = start_pos + count
        ks, vs = [], []
        for i, r in enumerate(request_ids):
            kbuf, vbuf, _ = self._entries[r][layer_id]
            kbuf[:, start_pos:end].copy_(key[i])
            vbuf[:, start_pos:end].copy_(value[i])
            self._entries[r][layer_id][2] = end
            ks.append(kbuf[:, :end])
            vs.append(vbuf[:, :end])
        return torch.stack(ks), torch.stack(vs)

    def release(self, request_ids: tuple[int, ...]) -> None:
        for r in request_ids:
            self._entries.pop(r, None)

    def clear(self) -> None:
        self._entries.clear()
