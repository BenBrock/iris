# SPDX-License-Identifier: MIT
"""Fixed PyTorch XPU heap with intra-node Level Zero IPC mappings."""

from contextlib import ExitStack
from functools import lru_cache
from importlib.resources import files
import struct

import numpy as np
import torch
import torch.distributed as dist

from iris.host.distributed.fd_passing import managed_fd, recv_fd, send_fd
from .torch_allocator import TorchPoolAllocator


@lru_cache(maxsize=1)
def _ipc_extension():
    from torch.utils.cpp_extension import load

    source = files("iris.drivers.local").joinpath("level_zero.sycl")
    return load("iris_level_zero", [str(source)], extra_ldflags=["-lze_loader"])


def _exchange(value, error):
    """Propagate a local setup error before any rank proceeds to the next IPC phase."""
    entries = [None] * dist.get_world_size()
    dist.all_gather_object(entries, (value, error))
    for rank, (_, message) in enumerate(entries):
        if message is not None:
            raise RuntimeError(f"Intel XPU IPC setup failed on rank {rank}: {message}")
    return [value for value, _ in entries]


class XpuAllocator(TorchPoolAllocator):
    device_type = "xpu"

    def map_heap(self, connections):
        """Map the fixed pool once. Keep the allocation offsets separate from the handles used for close."""
        bases = np.zeros(self.num_ranks, dtype=np.uint64)
        bases[self.cur_rank] = self.get_base_address()
        if self.num_ranks == 1:
            return bases

        with ExitStack() as exports, ExitStack() as mappings:
            metadata, error = None, None
            try:
                self._ipc = _ipc_extension().LevelZeroIpc(torch.xpu.current_stream(self.device_id).sycl_queue)
                handle, offset = self._ipc.export_heap(self.get_base_address(), self.heap_size)
                exports.callback(self._ipc.release_export, handle)
                # Each socket carries just one FD. The opaque handle bytes
                # must be preserved, except for its FD which is process-local.
                fd = struct.unpack_from("=i", handle)[0]
                for sock in connections.values():
                    send_fd(sock, fd)
                metadata = (handle, offset, self.heap_size)
            except Exception as exc:
                error = str(exc)
            peers = _exchange(metadata, error)
            if any(size != self.heap_size for _, _, size in peers):
                raise ValueError("XPU ranks must use the same symmetric heap size")

            try:
                for peer, sock in connections.items():
                    peer_handle, peer_offset, _ = peers[peer]
                    peer_fd, _ = recv_fd(sock)
                    with managed_fd(peer_fd):
                        base = self._ipc.open(peer_handle, peer_fd)
                    mappings.callback(self._ipc.close, base)
                    bases[peer] = base + peer_offset
            except Exception as exc:
                error = str(exc)
            _exchange(None, error)
            self._mappings = mappings.pop_all()
        return bases

    def close(self):
        if getattr(self, "_mappings", None):
            torch.xpu.synchronize(self.device_id)
            self._mappings.close()
            self._mappings = None
