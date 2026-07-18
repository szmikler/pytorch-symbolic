#  Copyright (c) 2022 Szymon Mikler

"""
Enables Symbolic API 2, which allows for registering layers by calling
layer(*symbolic) instead of symbolic_1(layer, *other_symbolic).
"""

from __future__ import annotations

import logging

from torch import nn

from .symbolic_data import SymbolicData

__nn_module_old_new__ = nn.Module.__new__
__nn_module_old_call__ = nn.Module.__call__


def call_wrapper_for_api_2(self, *args, custom_name: str | None = None, **kwds):
    if not any(isinstance(x, SymbolicData) for x in args):
        return __nn_module_old_call__(self, *args, **kwds)
    elif len(kwds) == 0 and all(isinstance(x, SymbolicData) for x in args):
        node = args[0]
        return node(self, *args[1:], custom_name=custom_name)
    else:
        msg = "Only *args are allowed as arguments! If you need to use **kwds, try `functions_utility.add_to_graph`!"
        raise UserWarning(msg)


class SymbolicAPI2ContextManager:
    def __enter__(self):
        logging.debug("Symbolic API has been enabled!")
        nn.Module.__call__ = call_wrapper_for_api_2

    def __exit__(self, exit_type, value, traceback):
        logging.debug("Symbolic API has been disabled!")
        nn.Module.__call__ = __nn_module_old_call__


def optimize_module_calls():
    """Remove the call wrapper from `torch.nn.Module`."""
    msg = "Optimizing module calls! Reusing existing layers will not work with layer(*symbols) notation!"
    logging.warning(msg)
    SymbolicAPI2ContextManager().__exit__(None, None, None)


def is_symbolic_api_2_enabled():
    """Whether symbolic API 2 is enabled."""
    return nn.Module.__call__ is call_wrapper_for_api_2


def enable_symbolic_api_2_for_new_modules():
    """Add a __new__ wrapper for `torch.nn.Module`."""

    def wrapped_new(cls, *args, **kwds):
        SymbolicAPI2ContextManager().__enter__()
        obj = super(nn.Module, cls).__new__(cls)
        return obj

    nn.Module.__new__ = wrapped_new


def disable_symbolic_api_2_for_new_modules():
    """Remove the __new__ wrapper for `torch.nn.Module`."""

    nn.Module.__new__ = __nn_module_old_new__
