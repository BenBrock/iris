# Installation Guide

This guide covers how to install Iris on your system using various methods.

## Overview

Iris uses PyTorch and Triton. For AMD GPUs, use the ROCm prerequisites below. For Intel GPUs, see [Intel XPU](#intel-xpu).

## Prerequisites

### System Requirements

- Linux operating system (Ubuntu 22.04+)
- AMD GPU with ROCm 6.3.1+ support (MI300X, MI350X, MI355X, or other ROCm-compatible GPUs)

### Required Software

**Minimum working requirements based on the Docker setup:**

- Python 3.10+
- PyTorch 2.0+ (ROCm version)
- ROCm 6.3.1+ HIP runtime
- Git
- Triton (suggested commit: [dd5823453bcc7973eabadb65f9d827c43281c434](https://github.com/triton-lang/triton/tree/dd5823453bcc7973eabadb65f9d827c43281c434))

**Note**: These versions represent the minimum working configuration. Using different versions may cause compatibility issues.

### Gluon Backend Requirements (Experimental)

To use the experimental Gluon APIs, additional requirements apply:

- ROCm 7.0+
- Triton (required commit: [aafec417bded34db6308f5b3d6023daefae43905](https://github.com/triton-lang/triton/tree/aafec417bded34db6308f5b3d6023daefae43905) or later)

## Installation Methods
### 1. Direct Installation from Git

For a quick installation directly from the repository:

```shell
pip install git+https://github.com/ROCm/iris.git
```

### 2. Using Docker Compose

The easiest way to get started if you don't have the dependencies installed is using Docker Compose:

```shell
# Clone the repository
git clone https://github.com/ROCm/iris.git
cd iris

# Start the development container
docker compose up --build -d

# Attach to the running container
docker attach iris-dev

# Install Iris in development mode
cd iris && pip install -e .
```

### 3. Manual Docker Setup

If you prefer to build and run Docker containers manually:

```shell
# Build the Docker image
./docker/build.sh

# Run the container
./docker/run.sh

# Install Iris in development mode
pip install -e .
```

For scheduler-managed clusters, see the [SLURM guide](slurm.md).


### 4. Apptainer/Singularity

For HPC environments or systems where Docker is not available:

```shell
# Build the Apptainer image
./apptainer/build.sh

# Run the container
./apptainer/run.sh

# Install Iris in development mode
pip install -e .
```


## Intel XPU

Iris supports a fixed symmetric heap on peer-accessible Intel GPUs in one Linux host.
The backend uses the PyTorch XPU stream’s Level Zero context to export and open IPC
handles. Iris exchanges the Linux file descriptors over Unix sockets and keeps the
peer mappings alive for the lifetime of the heap.

The tested stack is Intel Data Center GPU Max 1550, PyTorch 2.13.0+xpu, and
triton-xpu 3.7.2. Install a matching XPU build of PyTorch and Triton, the Intel
oneAPI DPC++ compiler (`icpx`), and the Level Zero development headers and loader.
Activate the compiler environment in the shell used to launch all ranks. The first
multi-rank initialization builds a small SYCL extension in PyTorch’s extension cache.

From the checkout, install the core RMA dependencies into that environment and run
the hardware tests.

```bash
python -m pip install numpy requests ninja pytest
python -m pip install --no-deps -e .
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
python tests/run_tests_distributed.py tests/unittests/test_xpu_backend.py --num_ranks 2 -v
```

On multi-tile GPUs, flat device enumeration exposes each tile as an XPU device.
Launch one rank per participating device with `torchrun`. Initialize a process
group before constructing Iris. XCCL and Gloo can both coordinate the ranks.

```python
import os
import torch
import torch.distributed as dist
import iris

device = torch.device("xpu", int(os.environ["LOCAL_RANK"]))
torch.xpu.set_device(device)
dist.init_process_group("xccl", device_id=device)
ctx = iris.Iris(1 << 20, device=device)
src = ctx.full((32, 64), ctx.get_rank() + 1, dtype=torch.float32)
dst = ctx.zeros_like(src)
ctx.barrier()
ctx.ccl.all_reduce(dst, src)
ctx.barrier()
dist.destroy_process_group()
```

Use `allocator_type="torch"` (the default) for Triton RMA, remote atomics, and
Triton collectives. Use FP32 accumulators for BF16 atomic addition on GPU Max.
The VMem allocators, AMD SDMA engine, hardware tracing,
and tritonBLAS fused GEMMs are outside the Intel backend’s supported features.

## Optional: SDMA copy engine

Shader load/store, atomics, and collectives work with the base install. The SDMA
copy engine (`use_copy_engine=True` in device code, and host-initiated `put`,
`put_tile`, `put_tiles`, and `quiet`) needs [rocm-xio](https://github.com/ROCm/rocm-xio),
which is an optional extra:

```shell
pip install "iris[sdma] @ git+https://github.com/ROCm/iris.git"
# or, from a checkout
pip install -e ".[sdma]"
```

To add SDMA to an existing Iris install, rerun the install with the extra. If an
older rocm-xio is already installed, run `pip uninstall rocm-xio` first: every
rocm-xio commit has the same version number, so pip keeps the installed one.

Installing rocm-xio does not turn SDMA on; see [Iris Class](../reference/host/class.md)
for enabling it.

## Selecting the AMD runtime for fabric communication

When multiple ROCm versions are installed, set `LD_LIBRARY_PATH` before starting
Python so it points to the SDK used by PyTorch. The AMD fabric driver first loads
`libamdhip64.so` and `libamd_smi.so` through that search path, then falls back to
system-library discovery if those names are unavailable. This prevents a stale
system SONAME from selecting a different HIP major version alongside PyTorch's
runtime. Mixing HIP runtimes in one process can abort during device initialization.

## Next Steps

Once you have Iris running with any of these methods:

- Explore the [Examples](../reference/examples.md) directory
- Learn about the [Programming Model](../conceptual/programming-model.md)
- For batch-scheduled environments, see [Running Iris on SLURM](slurm.md)
