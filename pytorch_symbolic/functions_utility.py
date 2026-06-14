#  Copyright (c) 2022 Szymon Mikler

from __future__ import annotations

from collections.abc import Callable, Hashable

from torch import nn

from . import useful_layers
from .symbolic_data import SymbolicData


def _replace_symbolic_with_value(container, extracted, navigation):
    """Search recursively for all occurences of Symbolic and replace them with their value.

    At the same time save navigation to know, how to do indexing to get to them.

    If navigation for container ends up as [..., [1, 2, 0, "TEST", 5], ...]
    this means that to get the element you should index container[1][2][0]["TEST"][5].
    """
    if isinstance(container, SymbolicData):
        navigation.append(navigation[-1].copy())
        extracted.append(container)
        return container.v

    if isinstance(container, (list, tuple)):
        new_list = []
        for i, x in enumerate(container):
            navigation[-1].append(i)
            new_list.append(_replace_symbolic_with_value(x, extracted, navigation))
            navigation[-1].pop()
        return new_list
    if isinstance(container, dict):
        new_dict = {}
        for k, v in container.items():
            navigation[-1].append(k)
            new_dict[k] = _replace_symbolic_with_value(v, extracted, navigation)
            navigation[-1].pop()
        return new_dict
    return container


def add_to_graph(func: Callable | nn.Module, *args, custom_name: str | None = None, **kwds):
    """Register a custom func or a module in the computation graph.

    This works will arbitrary functions and modules iff at least one Symbolic Data is among ``*args, **kwds``.

    This way of registering is flexible, but might add a small slowdown to the call,
    because it adds a wrapper for parsing arguments.
    If this is unacceptable, please create an torch.nn.Module that takes only Symbolic Data arguments.

    Here all arguments, including Symbolic Data, should be passed after the ``func`` argument.
    The arguments can be mixed and matched, even nested in lists, tuples and dictionaries.

    Convolution func example::

        inputs = Input(shape=(3, 32, 32))
        kernel = Input(batch_size=(16, 3, 3, 3))
        bias = Input(batch_size=(16,))
        output = add_to_graph(F.conv2d, input=inputs, weight=k, bias=bias, padding=1)
    """
    extracted_symbols: list[SymbolicData] = []

    real_call_args = []
    real_call_kwds = {}

    navigation: list[list[Hashable]] = [[]]
    real_call_args = _replace_symbolic_with_value(args, extracted_symbols, navigation)
    real_call_kwds = _replace_symbolic_with_value(kwds, extracted_symbols, navigation)
    navigation.pop()

    assert len(extracted_symbols) > 0, "No Symbolic Data detected in the input!"
    assert all((isinstance(symbol, SymbolicData) for symbol in extracted_symbols))
    assert len(extracted_symbols) == len(navigation)

    def wrapper_function(*args):
        assert len(args) == len(navigation), f"Expected {len(navigation)} inputs, not {len(args)}!"
        for arg, navi in zip(args, navigation, strict=True):
            obj = real_call_kwds if isinstance(navi[0], str) else real_call_args

            for idx in navi[:-1]:
                obj = obj[idx]

            obj[navi[-1]] = arg

        return func(*real_call_args, **real_call_kwds)

    if hasattr(func, "__name__"):
        name = func.__name__
    elif hasattr(func, "__class__"):
        name = func.__class__.__name__
    else:
        name = str(func)
    module = useful_layers.NamedLambdaOpLayer(op=wrapper_function, name=f"wrap({name})")
    # This might be a Symbolic Callable, so we use `apply_module` instead of `__call__`
    return extracted_symbols[0].apply_module(module, *extracted_symbols[1:], custom_name=custom_name)
