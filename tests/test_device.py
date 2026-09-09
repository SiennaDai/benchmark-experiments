import pytest
import torch

from train_platform import resolve_device


def test_cpu_and_auto_device_resolution():
    assert resolve_device("cpu") == torch.device("cpu")
    expected = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    assert resolve_device("auto") == expected


def test_bad_and_unavailable_devices_are_rejected():
    with pytest.raises(ValueError, match="to_device"):
        resolve_device("mps")
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA is unavailable"):
            resolve_device("cuda:0")
