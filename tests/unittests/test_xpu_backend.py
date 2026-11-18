# SPDX-License-Identifier: MIT

import pytest
import torch
import torch.distributed as dist
import triton
import triton.language as tl

import iris


pytestmark = pytest.mark.skipif(not getattr(torch.version, "xpu", None), reason="requires Intel XPU")


@triton.jit
def _ring(context, data, counter, previous, rank: tl.constexpr, size: tl.constexpr):
    ctx = iris.DeviceContext.initialize(context, rank, size)
    peer = (rank + 1) % size
    value = ctx.load(data + 1, from_rank=peer)
    tl.store(previous, value)
    ctx.atomic_add(counter, 1, to_rank=peer, scope="sys")
    ctx.atomic_sub(counter + 1, 2, to_rank=peer, scope="sys")


def test_xpu_peer_heap():
    if not dist.is_initialized() or dist.get_world_size() < 2:
        pytest.skip("run with at least two XPU ranks")
    rank = dist.get_rank()
    size = dist.get_world_size()
    device = torch.device("xpu", torch.xpu.current_device())
    ctx = iris.Iris(1 << 20, device=device)
    assert ctx.get_device() == device
    source = torch.full((2,), rank + 10, dtype=torch.int32, device=device)
    data = ctx.as_symmetric(source)
    counter = ctx.zeros(2, dtype=torch.int32)
    previous = torch.empty_like(counter)
    assert ctx.is_symmetric(data)
    assert not ctx.is_symmetric(source)
    peer = (rank + 1) % size
    offset = data[1:].data_ptr() - ctx.get_heap_bases()[rank].item()
    assert ctx.heap.translate(data[1:].data_ptr(), rank, peer) == ctx.get_heap_bases()[peer].item() + offset
    empty, empty_bases = ctx.allocate_symmetric(0)
    assert empty.numel() == 0
    assert empty_bases.tolist() == [0] * size
    ctx.barrier()
    _ring[(1,)](ctx.get_device_context(), data, counter, previous, rank, size)
    ctx.device_barrier()
    assert previous[0].item() == (rank + 1) % size + 10
    assert counter.tolist() == [1, -2]
    ctx.barrier()


def test_xpu_collectives():
    if not dist.is_initialized() or dist.get_world_size() < 2:
        pytest.skip("run with at least two XPU ranks")
    ctx = iris.Iris(1 << 21)
    size = ctx.get_num_ranks()
    src = ctx.full((32, 64), ctx.get_rank() + 1, dtype=torch.float32)
    gathered = ctx.zeros((size * 32, 64), dtype=src.dtype)
    reduced = ctx.zeros_like(src)
    ctx.barrier()
    ctx.ccl.all_gather(gathered, src)
    ctx.ccl.all_reduce(reduced, src)
    for rank, part in enumerate(gathered.chunk(size)):
        torch.testing.assert_close(part, torch.full_like(part, rank + 1))
    torch.testing.assert_close(reduced, torch.full_like(src, size * (size + 1) // 2))
    ctx.barrier()

    # One tile per rank. Reduce-scatter writes just the assigned tile in
    # each rank's full-sized output, and all-to-all exchanges column chunks.
    src = ctx.full((32, 64 * size), ctx.get_rank() + 1, dtype=torch.float32)
    exchanged = ctx.zeros_like(src)
    scattered = ctx.zeros_like(src)
    ctx.barrier()
    ctx.ccl.all_to_all(exchanged, src)
    ctx.ccl.reduce_scatter(scattered, src)
    for rank, (received, tile) in enumerate(zip(exchanged.chunk(size, dim=1), scattered.chunk(size, dim=1))):
        torch.testing.assert_close(received, torch.full_like(received, rank + 1))
        expected_sum = size * (size + 1) // 2 if rank == ctx.get_rank() else 0
        torch.testing.assert_close(tile, torch.full_like(tile, expected_sum))
    ctx.barrier()


def test_xpu_rejects_amd_engines_and_allocators(monkeypatch):
    with pytest.raises(ValueError, match="SDMA"):
        iris.Iris(1 << 20, engines=("sdma",))
    if not dist.is_initialized():
        pytest.skip("requires a process group")
    monkeypatch.setenv("IRIS_ALLOCATOR", "vmem_chunked")
    with pytest.raises(ValueError, match="torch allocator"):
        iris.Iris(1 << 20)


def test_xpu_heap_size_mismatch():
    if not dist.is_initialized() or dist.get_world_size() < 2:
        pytest.skip("run with at least two XPU ranks")
    with pytest.raises(ValueError, match="same symmetric heap size"):
        iris.Iris((1 << 20) + dist.get_rank() * 1024)
    # All ranks can establish another IPC mesh after the failed construction.
    ctx = iris.Iris(1 << 20)
    assert ctx.zeros(1).item() == 0
    ctx.barrier()


def test_xpu_fd_failure_reaches_every_rank(monkeypatch):
    if not dist.is_initialized() or dist.get_world_size() < 2:
        pytest.skip("run with at least two XPU ranks")
    from iris.host.memory.allocators import xpu_allocator

    if dist.get_rank() == 0:

        def fail_send(*args):
            raise OSError("injected FD transfer failure")

        monkeypatch.setattr(xpu_allocator, "send_fd", fail_send)
    with pytest.raises(RuntimeError, match="rank 0: injected FD transfer failure"):
        iris.Iris(1 << 20)


def test_xpu_bfloat16_atomic_diagnostic():
    from triton.compiler.errors import CompilationError

    if triton.runtime.driver.active.get_current_target().arch["arch"] != "pvc":
        pytest.skip("target supports BF16 atomic addition")
    if not dist.is_initialized():
        pytest.skip("requires a process group")
    ctx = iris.Iris(1 << 20)
    data = ctx.zeros(2, dtype=torch.bfloat16)
    with pytest.raises(CompilationError) as exc:
        _ring[(1,)](ctx.get_device_context(), data, data, data, ctx.get_rank(), ctx.get_num_ranks())
    assert "Use FP32 for BF16 atomic accumulation" in str(exc.getrepr())
