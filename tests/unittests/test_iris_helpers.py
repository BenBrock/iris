# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import torch
import iris
from iris import tensor_creation


def test_get_device():
    shmem = iris.iris(1 << 20)

    # Test that get_device returns the correct device
    device = shmem.get_device()
    assert str(device) == shmem.device

    assert device.type == ("xpu" if getattr(torch.version, "xpu", None) else "cuda")
    assert device.index is not None


def test_device_validation():
    shmem = iris.iris(1 << 20)
    iris_device = shmem.get_device()

    # Test valid devices
    assert tensor_creation.is_valid_device(None, iris_device)  # None is always valid
    assert tensor_creation.is_valid_device(shmem.device, iris_device)  # Same device
    assert tensor_creation.is_valid_device(torch.device(shmem.device), iris_device)  # PyTorch device object

    assert tensor_creation.is_valid_device(iris_device.type, iris_device)

    # Test invalid devices
    assert not tensor_creation.is_valid_device("cpu", iris_device)  # CPU is always invalid
    assert not tensor_creation.is_valid_device("mps", iris_device)  # MPS is always invalid

    other_device = torch.device(iris_device.type, iris_device.index + 1)
    assert not tensor_creation.is_valid_device(other_device, iris_device)
