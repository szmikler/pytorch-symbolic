#  Copyright (c) 2022 Szymon Mikler

import torch
from torch import nn

from pytorch_symbolic import model_tools


def test_hash_equal_for_equal_tensors():
    a = torch.rand(16, 16)
    b = a.clone()
    assert model_tools.hash_torch_tensor(a) == model_tools.hash_torch_tensor(b)


def test_hash_differs_for_different_values():
    a = torch.zeros(4, 4)
    b = torch.ones(4, 4)
    assert model_tools.hash_torch_tensor(a) != model_tools.hash_torch_tensor(b)


def test_corresponding_parameters_roundtrip():
    torch.manual_seed(42)
    a = nn.Linear(8, 4)
    torch.manual_seed(42)
    b = nn.Linear(8, 4)
    assert model_tools.models_have_corresponding_parameters(a, b)

    torch.manual_seed(43)
    c = nn.Linear(8, 4)
    assert not model_tools.models_have_corresponding_parameters(a, c)
