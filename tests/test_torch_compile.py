#  Copyright (c) 2022 Szymon Mikler

import platform
import sys

import torch
from torch import nn

from pytorch_symbolic import Input, SymbolicModel


def torch_version_at_least(major, minor):
    version = torch.__version__.split("+")[0].split(".")
    return (int(version[0]), int(version[1])) >= (major, minor)


def compile_supported():
    # torch.compile is not supported on Windows and requires Python >= 3.8
    if platform.system() == "Windows":
        return False
    if sys.version_info < (3, 8):
        return False
    return hasattr(torch, "compile")


def export_supported():
    # torch.export with the stable .module() API is available since torch 2.4
    return compile_supported() and torch_version_at_least(2, 4)


def create_model(enable_forward_codegen=None):
    torch.manual_seed(42)
    inputs = Input(shape=(8,))
    x = nn.Linear(8, 16)(inputs)
    x = nn.ReLU()(x)
    outputs = nn.Linear(16, 4)(x)
    return SymbolicModel(inputs=inputs, outputs=outputs, enable_forward_codegen=enable_forward_codegen)


def test_torch_compile_matches_eager():
    if not compile_supported():
        return
    model = create_model()
    compiled_model = torch.compile(model)

    data = torch.rand(4, 8)
    assert torch.allclose(model(data), compiled_model(data))


def test_torch_compile_no_codegen_matches_eager():
    if not compile_supported():
        return
    model = create_model(enable_forward_codegen=False)
    compiled_model = torch.compile(model, backend="eager")

    data = torch.rand(4, 8)
    assert torch.allclose(model(data), compiled_model(data))


def test_torch_compile_detached_model():
    if not compile_supported():
        return
    model = create_model()
    detached_model = model.detach_from_graph()
    compiled_model = torch.compile(detached_model, backend="eager")

    data = torch.rand(4, 8)
    assert torch.allclose(model(data), compiled_model(data))


def test_torch_export_matches_eager():
    if not export_supported():
        return
    model = create_model()

    data = torch.rand(4, 8)
    exported_program = torch.export.export(model, (data,))
    assert torch.allclose(model(data), exported_program.module()(data))
