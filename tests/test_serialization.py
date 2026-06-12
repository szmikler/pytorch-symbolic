#  Copyright (c) 2022 Szymon Mikler

import copy
import pickle

import torch
from torch import nn

from pytorch_symbolic import Input, SymbolicModel


def create_model(enable_forward_codegen=None):
    torch.manual_seed(42)
    inputs = Input(shape=(8,))
    x = nn.Linear(8, 16)(inputs)
    x = nn.ReLU()(x)
    outputs = nn.Linear(16, 4)(x)
    return SymbolicModel(inputs=inputs, outputs=outputs, enable_forward_codegen=enable_forward_codegen)


def test_pickle_roundtrip():
    model = create_model()
    data = torch.rand(5, 8)

    restored = pickle.loads(pickle.dumps(model))
    assert torch.equal(model(data), restored(data))


def test_pickle_roundtrip_codegen():
    model = create_model(enable_forward_codegen=True)
    data = torch.rand(5, 8)

    restored = pickle.loads(pickle.dumps(model))
    assert torch.equal(model(data), restored(data))
    # The restored model must use the generated forward, not fall back to graph replay
    assert restored.forward.__func__ is not SymbolicModel.forward


def test_pickle_roundtrip_no_codegen():
    model = create_model(enable_forward_codegen=False)
    data = torch.rand(5, 8)

    restored = pickle.loads(pickle.dumps(model))
    assert torch.equal(model(data), restored(data))


def test_pickle_roundtrip_detached():
    model = create_model()
    detached = model.detach_from_graph()
    data = torch.rand(5, 8)

    restored = pickle.loads(pickle.dumps(detached))
    assert torch.equal(detached(data), restored(data))


def test_deepcopy_model():
    model = create_model()
    data = torch.rand(5, 8)

    model_copy = copy.deepcopy(model)
    assert torch.equal(model(data), model_copy(data))


def test_private_attribute_probe_does_not_create_nodes():
    x = Input(shape=(8,))
    assert len(x.children) == 0

    assert not hasattr(x, "_nonexistent_attribute")
    assert not hasattr(x, "__wrapped__")
    assert len(x.children) == 0
