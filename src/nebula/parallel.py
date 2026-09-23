from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist
from torch.profiler import record_function


@dataclass
class ParallelContext:
    tp_size: int
    pp_size: int
    rank: int
    world_size: int
    tp_rank: int
    pp_rank: int
    device: torch.device
    tp_group: Any
    owns_process_group: bool

    @classmethod
    def from_env(
        cls,
        *,
        tp_size: int = 1,
        pp_size: int = 1,
        device: str = "cpu",
        timeout_s: float = 30.0,
    ) -> ParallelContext:
        for name, v in (("tp_size", tp_size), ("pp_size", pp_size)):
            if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            not isinstance(timeout_s, (int, float))
            or isinstance(timeout_s, bool)
            or not math.isfinite(timeout_s)
            or timeout_s <= 0
        ):
            raise ValueError("timeout_s must be a finite positive number")
        if device not in ("cpu", "cuda") and not (
            isinstance(device, str) and device.startswith("cuda:")
        ):
            raise ValueError("device must be 'cpu' or 'cuda'")
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            owns = False
            backend = dist.get_backend()
            if device == "cpu" and backend != "gloo":
                raise ValueError(f"preinitialized backend {backend!r} incompatible with cpu")
            if device != "cpu" and backend != "nccl":
                raise ValueError(f"preinitialized backend {backend!r} incompatible with cuda")
        else:
            env_world = os.environ.get("WORLD_SIZE")
            if env_world is None:
                world_size = 1
                rank = 0
                owns = False
            else:
                world_size = int(env_world)
                rank = int(os.environ.get("RANK", "0"))
                owns = True
        if world_size != tp_size * pp_size:
            raise ValueError(f"world_size {world_size} != tp_size {tp_size} * pp_size {pp_size}")
        dev = torch.device(device)
        if dev.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA device requested but unavailable")
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            if dev.index is not None and world_size > 1 and dev.index != local_rank:
                raise ValueError(f"device index {dev.index} != LOCAL_RANK {local_rank}")
            index = dev.index if dev.index is not None else local_rank
            dev = torch.device("cuda", index)
            torch.cuda.set_device(dev)
        if world_size > 1 and owns:
            if not dist.is_available():
                raise RuntimeError("torch.distributed is not available")
            backend = "nccl" if dev.type == "cuda" else "gloo"
            if dev.type == "cpu" and not dist.is_gloo_available():
                raise RuntimeError("gloo backend unavailable")
            if dev.type == "cuda" and not dist.is_nccl_available():
                raise RuntimeError("nccl backend unavailable")
            dist.init_process_group(backend=backend, timeout=timedelta(seconds=timeout_s))
        tp_group = None
        if world_size > 1:
            my_group = None
            for stage in range(pp_size):
                ranks = list(range(stage * tp_size, (stage + 1) * tp_size))
                g = dist.new_group(ranks, timeout=timedelta(seconds=timeout_s))
                if rank in ranks:
                    my_group = g
            tp_group = my_group
        return cls(
            tp_size=tp_size,
            pp_size=pp_size,
            rank=rank,
            world_size=world_size,
            tp_rank=rank % tp_size,
            pp_rank=rank // tp_size,
            device=dev,
            tp_group=tp_group,
            owns_process_group=owns,
        )

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.tp_size > 1 and self.tp_group is not None:
            with record_function("nebula::tp_all_reduce"):
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=self.tp_group)
        return tensor

    def close(self) -> None:
        if self.owns_process_group and dist.is_available() and dist.is_initialized():
            try:
                dist.destroy_process_group()
            finally:
                self.owns_process_group = False
