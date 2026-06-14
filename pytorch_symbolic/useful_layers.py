#  Copyright (c) 2022 Szymon Mikler

from collections.abc import Callable

import torch
from torch import nn


class LambdaOpLayer(nn.Module):
    def __init__(self, op: Callable):
        super().__init__()
        self.forward = op


class NamedLambdaOpLayer(nn.Module):
    def __init__(self, op: Callable, name: str):
        super().__init__()
        self.forward = op
        self.name = name

    def __repr__(self):
        return self.name

    def _get_name(self):
        return self.name


class CallbackLayer(nn.Module):
    def __init__(self, op: Callable):
        """Layer that returns its inputs, but executes a callable ``op`` on them before returning.

        This can be used for debugging or logging purposes. Accepts only one argument.

        Example::

            x = CallbackLayer(print)(x)

        It does not change anything in x, but prints its value.
        """
        super().__init__()
        self.callback = op

    def forward(self, arg):
        self.callback(arg)
        return arg


class AddOpLayer(nn.Module):
    @staticmethod
    def forward(a, b):
        return a + b


class SubOpLayer(nn.Module):
    @staticmethod
    def forward(a, b):
        return a - b


class MulOpLayer(nn.Module):
    @staticmethod
    def forward(a, b):
        return a * b


class ModOpLayer(nn.Module):
    @staticmethod
    def forward(a, b):
        return torch.remainder(a, b)


class MatmulOpLayer(nn.Module):
    @staticmethod
    def forward(a, b):
        return a @ b


class ConcatLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, *tensors):
        return torch.cat(tensors=tensors, dim=self.dim)


class StackLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, *tensors):
        return torch.stack(tensors=tensors, dim=self.dim)


class ReshapeLayer(nn.Module):
    def __init__(self, shape, batch_size_included=False):
        super().__init__()
        if not batch_size_included:
            self.shape = (-1, *shape)
        else:
            self.shape = shape

    def forward(self, tensor):
        return torch.reshape(input=tensor, shape=self.shape)


class ViewCopyLayer(nn.Module):
    def __init__(self, shape, batch_size_included=False):
        super().__init__()
        if not batch_size_included:
            self.shape = (-1, *shape)
        else:
            self.shape = shape

    def forward(self, tensor):
        return torch.view_copy(input=tensor, size=self.shape)


class AggregateLayer(nn.Module):
    def __init__(self, op: Callable, dim=None, keepdim=False):
        super().__init__()
        self.op = op

        self.dim = dim
        self.keepdim = keepdim
        if self.dim is None:
            self.forward = self.forward_nodim

    def forward(self, tensor):
        return self.op(input=tensor, dim=self.dim, keepdim=self.keepdim)

    def forward_nodim(self, tensor):
        return self.op(input=tensor)


class UnpackLayer(nn.Module):
    def forward(self, *args):
        return args


class SliceLayer(nn.Module):
    def __init__(self, idx):
        super().__init__()
        self.idx = idx

    def forward(self, arg):
        return arg[self.idx]


class SliceLayerSymbolicIdx(nn.Module):
    def forward(self, arg, idx):
        return arg[idx]


class MethodCall(nn.Module):
    def __init__(self, method_name, *args, **kwds):
        super().__init__()
        self.method_name = method_name
        self.args = args
        self.kwds = kwds

    def forward(self, tensor):
        return getattr(tensor, self.method_name)(*self.args, **self.kwds)


class GetAttr(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def forward(self, data):
        return getattr(data, self.name)
