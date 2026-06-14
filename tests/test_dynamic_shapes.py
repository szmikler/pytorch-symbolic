#  Copyright (c) 2022 Szymon Mikler

import torch
from torch import nn

from pytorch_symbolic import Input, SymbolicModel


def test_none_in_shape():
    x = Input(shape=(None, 128))
    assert x.shape == (1, None, 128)
    assert tuple(x.v.shape) == (1, 16, 128)
    assert x.features == 128


def test_dynamic_size_hint():
    x = Input(shape=(3, None, None), dynamic_size_hint=32)
    assert tuple(x.v.shape) == (1, 3, 32, 32)
    assert x.shape == (1, 3, None, None)
    assert x.C == 3
    assert x.H is None
    assert x.W is None


def test_batch_shape_with_none():
    x = Input(batch_shape=(None, 10))
    assert x.shape == (None, 10)
    assert x.batch_size is None
    assert not x.batch_size_known

    y = Input(batch_shape=(4, 10))
    assert tuple(y.shape) == (4, 10)
    assert y.batch_size == 4
    assert y.batch_size_known


def test_static_input_unchanged():
    x = Input(shape=(28, 28))
    assert isinstance(x.shape, torch.Size)
    assert tuple(x.shape) == (1, 28, 28)
    assert x.batch_size == 1


def test_model_runs_with_varied_sizes():
    inputs = Input(shape=(3, None, None))
    outputs = nn.Conv2d(3, 8, 3, padding=1)(inputs)
    model = SymbolicModel(inputs=inputs, outputs=outputs)
    for h, w in [(16, 16), (32, 48), (7, 9)]:
        data = torch.rand(2, 3, h, w)
        assert tuple(model(data).shape) == (2, 8, h, w)

    inputs = Input(shape=(None, 64))
    outputs = nn.Linear(64, 8)(inputs)
    model = SymbolicModel(inputs=inputs, outputs=outputs)
    for length in [1, 5, 100]:
        data = torch.rand(2, length, 64)
        assert tuple(model(data).shape) == (2, length, 8)


def test_summary_and_input_shape():
    inputs = Input(shape=(3, None, None))
    outputs = nn.Conv2d(3, 8, 3, padding=1)(inputs)
    model = SymbolicModel(inputs=inputs, outputs=outputs)

    assert model.input_shape == (1, 3, None, None)
    model.summary()


def test_shape_index_fails_loudly():
    x = Input(shape=(None, 64))
    assert x.shape[1] is None

    # Using a dynamic dimension as a layer size must fail at once
    # instead of silently baking in the placeholder size
    raised = False
    try:
        nn.Linear(x.shape[1], 4)
    except TypeError:
        raised = True
    assert raised
