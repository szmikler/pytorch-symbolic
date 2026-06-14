#  Copyright (c) 2022 Szymon Mikler

from __future__ import annotations

import torch
from torch import nn


def get_parameter_count(model: nn.Module, only_trainable=False):
    """Get the number of parameters of a model."""
    cnt = 0
    for param in model.parameters():
        if only_trainable and not param.requires_grad:
            continue
        cnt += param.shape.numel()
    return cnt


def get_parameter_shapes(model: nn.Module):
    """Get the shapes of parameters of a model."""
    shapes = []
    for param in model.parameters():
        shapes.append(tuple(param.shape))
    return shapes


def model_similar(a: nn.Module, b: nn.Module):
    """Check whether two models have the same number of parameters and the same shapes of parameters."""
    if get_parameter_count(a) != get_parameter_count(b):
        return False

    if sorted(get_parameter_shapes(a)) != sorted(get_parameter_shapes(b)):
        return False
    return True


def hash_torch_tensor(tensor: torch.Tensor):
    """Hash the tensor by its raw bytes, dtype and shape."""
    try:
        tensor_bytes = tensor.detach().cpu().contiguous().numpy().tobytes()
    except (RuntimeError, TypeError):
        # Numpy can be unavailable and some dtypes cannot be converted to numpy.
        # Fall back to the much slower string representation.
        tensor_as_string = str(tensor.flatten().tolist())
        return hash(tensor_as_string)
    return hash((str(tensor.dtype), tuple(tensor.shape), tensor_bytes))


def models_have_corresponding_parameters(a: nn.Module, b: nn.Module):
    """Check whether two models' parameters have identical hash values.

    Parameter order does not matter.
    So if two models have identical parameters but in different order, this will still return True.
    """
    hashes_a = [hash_torch_tensor(p) for p in a.parameters()]
    hashes_b = [hash_torch_tensor(p) for p in b.parameters()]
    return set(hashes_a) == set(hashes_b)
