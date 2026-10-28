"""
device.py -- pick a device DGL can actually use.

WHY THIS EXISTS
---------------
`torch.cuda.is_available()` answers a question about *torch*, not about DGL.
On Windows those answers differ: PyTorch publishes CUDA wheels for Windows,
DGL does not. The DGL wheel index

    https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html

contains only `manylinux1_x86_64` builds (25 wheels, zero for Windows). So a
Windows machine can have `torch 2.3.0+cu121` with `cuda True` and still fail
the moment a graph is moved to the GPU:

    DGLError: Device API cuda is not enabled. Please install the cuda version of dgl.

That is exactly what happened when the Python 3.10 reference environment was
built here. Rather than crash mid-pipeline -- after data preparation, which
takes minutes on the real dataset -- we probe DGL once and fall back to CPU with
a warning.

GPU training of DGL models on Windows requires WSL2 or native Linux.
"""

import warnings

import torch

_cached = None


def dgl_supports_cuda() -> bool:
    """Return True only if a DGL graph can actually live on the GPU."""
    if not torch.cuda.is_available():
        return False
    try:
        import dgl

        g = dgl.graph((torch.tensor([0]), torch.tensor([1])), num_nodes=2)
        g.to("cuda:0")
        return True
    except Exception:
        return False


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Return `cuda:0` when torch *and* DGL both work there, else `cpu`."""
    global _cached
    if _cached is not None:
        return _cached

    if not prefer_cuda:
        _cached = torch.device("cpu")
        return _cached

    if torch.cuda.is_available() and not dgl_supports_cuda():
        warnings.warn(
            "torch reports CUDA but DGL was built without CUDA support "
            "(DGL publishes no Windows CUDA wheels). Falling back to CPU, "
            "which is correct but slow. Use WSL2 or Linux for GPU training.",
            RuntimeWarning,
            stacklevel=2,
        )
        _cached = torch.device("cpu")
        return _cached

    _cached = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return _cached


def describe() -> dict:
    """Device facts for the run manifest -- so a run records what it ran on."""
    dev = get_device()
    return {
        "device": str(dev),
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "dgl_cuda": dgl_supports_cuda() if torch.cuda.is_available() else False,
        "gpu_name": (torch.cuda.get_device_name(0)
                     if torch.cuda.is_available() else None),
    }