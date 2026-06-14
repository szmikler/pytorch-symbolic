#  Copyright (c) 2022 Szymon Mikler

from __future__ import annotations

import logging
from collections.abc import Callable
from types import MethodWrapperType
from typing import Any

import torch
from torch import nn

from . import useful_layers

_SYMBOLIC_DATA_COUNTER = 0


class SymbolicData:
    def __init__(
        self,
        value: Any,
        parents: tuple[SymbolicData, ...] = (),
        depth: int = 0,
        layer: nn.Module | None = None,
        batch_size_known: bool = False,
        custom_name: str | None = None,
    ):
        """Grandfather of all Symbolic datatypes.

        Underlying data is a normal Python object, for example a ``dict``.
        You can use methods and operators of the underlying object.
        You can also unpack or index it, if only the underlying data allows it.

        If the underlying data is ``torch.Tensor``, it should be created as ``SymbolicTensor`` instead.

        Parameters
        ----------
        value
        parents
        depth
        layer
        batch_size_known
        custom_name

        Attributes
        ----------
        v : Any
            Underlying data that is used during model tracing
        layer : nn.Module
            A torch.nn.Module that transforms parents' values into this value. Also it's the incoming edge.
        depth : int
            Maximum of parents' depths plus one
        batch_size_known : bool
            In case of Input, whether batch size was provided by the user.
            For non-Input nodes, batch size is known iff all parents' batch sizes are known.
        custom_name : str
            Instead of using the default name for the undelying layer, user can provide his own.
        """
        global _SYMBOLIC_DATA_COUNTER
        self._execution_order_idx = _SYMBOLIC_DATA_COUNTER
        _SYMBOLIC_DATA_COUNTER += 1

        # We use Symbolic Data for inheriting only
        assert self.__class__ is not SymbolicData, "Symbolic Data should not be created directly!"

        self._value = value
        self._underlying_type_name = type(value).__name__
        self._custom_provided_name = custom_name

        self.layer = layer
        self.depth = depth
        self.batch_size_known = batch_size_known

        self._children: list[SymbolicData] = []
        self._parents: tuple[SymbolicData, ...] = parents
        self._layer_full_siblings: tuple[SymbolicData, ...] = (self,)

        self._define_class_operators()

    @property
    def v(self):
        """Get the underlying value."""
        if self._value is None:
            self._recalculate_value()
        return self._value

    def _define_class_operators(self):
        """Define basic operators, e.g. +, -, *, ...

        This allows using operators for Symbolic Data if and only if the underlying data is compatible.
        """
        operators = [
            "__abs__",
            "__neg__",
            "__add__",
            "__radd__",
            "__sub__",
            "__rsub__",
            "__mul__",
            "__rmul__",
            "__pow__",
            "__rpow__",
            "__mod__",
            "__rmod__",
            "__truediv__",
            "__rtruediv__",
            "__and__",
            "__rand__",
            "__or__",
            "__ror__",
            "__xor__",
            "__rxor__",
            "__matmul__",
            "__rmatmul__",
        ]

        for operator in operators:
            if hasattr(self.v, operator) and (
                not hasattr(self.__class__, operator)
                or isinstance(getattr(self.__class__, operator), MethodWrapperType)
            ):
                logging.debug(f"Adding new operator to {self.__class__.__name__}: {operator}")

                def factory(op):
                    return lambda self, *args, **kwds: self._get_attr_node(op)(*args, **kwds)

                setattr(self.__class__, operator, factory(operator))

    @property
    def parents(self) -> tuple[SymbolicData, ...]:
        """Acces the tuple of parents of this node."""
        return tuple(self._parents)

    @property
    def children(self) -> tuple[SymbolicData, ...]:
        """Acces the tuple of children of this node."""
        return tuple(self._children)

    def apply_module(
        self,
        layer: nn.Module,
        *others: SymbolicData,
        custom_name: str | None = None,
    ) -> SymbolicData | tuple[SymbolicData, ...]:
        """Register a new layer in the graph. Layer must be nn.Module."""
        assert all([isinstance(other, SymbolicData) for other in others]), "Works with SymbolicData only!"

        parents = (self, *others)
        new_depth = max(parent.depth for parent in parents) + 1
        with torch.no_grad():
            new_value = layer.__call__(self.v, *(o.v for o in others))

        cls = _figure_out_symbolic_type(new_value)

        new_layer_node = cls(
            value=new_value,
            parents=parents,
            layer=layer,
            depth=new_depth,
            batch_size_known=all(parent.batch_size_known for parent in parents),
            custom_name=custom_name,
        )
        for parent in parents:
            parent._children.append(new_layer_node)
            logging.debug(f"Added {new_layer_node} as child of {parent}")
        return new_layer_node

    def _recalculate_value(self):
        """Recalulate ._value if it is None.

        Sometimes we remove ._value to reduce the memory footprint. This function restores it.
        """
        for parent in self._parents:
            if parent._value is None:
                parent._recalculate_value()  # recursive call to parents
        with torch.no_grad():
            outputs = self.layer(*(parent._value for parent in self._parents))
            if len(self._layer_full_siblings) == 1:
                outputs = (outputs,)
            for full_sibling, output in zip(self._layer_full_siblings, outputs, strict=True):
                full_sibling._value = output

    def _clear_value(self):
        """Clear the underlying value to save memory."""
        assert len(self._parents) > 0, "Cannot clear the underlying value of input nodes!"
        self._value = None

    def __iter__(self):
        """Creates the only layer that has multiple children: UnpackLayer.

        Suitable for unpacking results, even nested ones.
        """
        layer = useful_layers.UnpackLayer()
        new_outputs = layer.__call__(*self.v)

        new_layer_nodes = []
        for new_value in new_outputs:
            cls = _figure_out_symbolic_type(new_value)

            new_layer_nodes.append(
                cls(
                    value=new_value,
                    parents=(self,),
                    layer=layer,
                    depth=self.depth + 1,
                    batch_size_known=self.batch_size_known,
                )
            )
        for new_layer_node in new_layer_nodes:
            new_layer_node._layer_full_siblings = tuple(new_layer_nodes)

        self._children.extend(new_layer_nodes)
        for new_layer_node in new_layer_nodes:
            logging.debug(f"Added {new_layer_node} as child of {self}")
        for node in new_layer_nodes:
            yield node

    def _get_all_nodes_above(self) -> set[SymbolicData]:
        nodes_seen = {self}
        to_expand = [self]
        while to_expand:
            node = to_expand.pop()
            for parent in node._parents:
                if parent not in nodes_seen:
                    to_expand.append(parent)
                    nodes_seen.add(parent)
        return nodes_seen

    def _get_all_nodes_below(self) -> set[SymbolicData]:
        nodes_seen = {self}
        to_expand = [self]
        while to_expand:
            node = to_expand.pop()
            for child in node._children:
                if child not in nodes_seen:
                    to_expand.append(child)
                    nodes_seen.add(child)
        return nodes_seen

    def _launch_input(self, x):
        self._value = x

    def _launch(self):
        if len(self._layer_full_siblings) > 1:
            assert len(self._parents) == 1
            outputs = self.layer(*self._parents[0]._value)
            for node, output in zip(self._layer_full_siblings, outputs, strict=True):
                node._value = output
        else:
            self._value = self.layer(*(parent._value for parent in self._parents))

    def __len__(self) -> int:
        """Length of the symbolic data."""
        return len(self.v)

    def __getitem__(self, idx):
        if isinstance(idx, SymbolicData):
            layer = useful_layers.SliceLayerSymbolicIdx()
            return layer(self, idx)
        else:
            layer = useful_layers.SliceLayer(idx)
            return layer(self)

    def __call__(self, *args, custom_name: str | None = None):
        return self.apply_module(*args, custom_name=custom_name)

    def __repr__(self):
        addr = f"{self.__class__.__name__} at {hex(id(self))};"
        info = f"{len(self._parents)} parents; {len(self._children)} children"
        return "<" + addr + " " + info + ">"

    def __hash__(self):
        return id(self)

    def _get_attr_node(self, item):
        """Register a node that extracts an attribute of the underlying value."""
        return self(useful_layers.GetAttr(item))

    def __getattr__(self, item):
        if item.startswith("_"):
            # Private and dunder lookups must fail fast instead of being redirected to the
            # underlying value. Redirecting them breaks protocols that probe for optional
            # attributes before the instance is fully initialized: unpickling, for example,
            # probes for `__setstate__` while `_value` does not exist yet, which sent
            # `__getattr__` into infinite recursion through the `v` property.
            # This also covers `__torch_function__`, because pytorch wraps `+` and other operators.
            raise AttributeError(item)
        if hasattr(self.v, item):
            return self._get_attr_node(item)
        else:
            raise AttributeError(item)


class SymbolicCallable(SymbolicData):
    def __call__(self, *args, **kwds):
        assert isinstance(self.v, Callable)
        from . import add_to_graph

        def __func__(obj, *args, **kwds):
            return obj(*args, **kwds)

        __func__.__name__ = self.v.__name__

        returns = add_to_graph(__func__, self, *args, **kwds)
        if returns.v is NotImplemented:
            dtypes = [type(p.v).__name__ for p in returns.parents[1:]]
            raise NotImplementedError(f"Operation on {dtypes} returned NonImplemented object!")
        return returns


class SymbolicTensor(SymbolicData):
    def __init__(self, *args, **kwds):
        """Recommended to use Symbolic datatype. It mimics and extends ``torch.Tensor`` API.

        Treat it as a placeholder that will be replaced with real data after the model is created.
        For calculation purposes treat it as a normal ``torch.Tensor``: add, subtract, multiply,
        take absolute value of, index, slice, etc.
        """
        super().__init__(*args, **kwds)
        assert isinstance(self.v, torch.Tensor)
        self._shape = self.v.shape
        self._dynamic_dims: set[int] = set()

    @property
    def features(self) -> int | None:
        """Size of the last dimension."""
        return self.shape[-1]

    @property
    def C(self) -> int | None:
        """Number of channels in Image data."""
        assert len(self._shape) == 4, "The data is not of [C,H,W] form!"
        return self.shape[1]

    @property
    def channels(self) -> int:
        """Same as ``.C``"""
        return self.C

    @property
    def H(self) -> int | None:
        """Height in Image data."""
        assert len(self._shape) == 4, "The data is not of [C,H,W] form!"
        return self.shape[2]

    @property
    def W(self) -> int | None:
        """Width in Image data."""
        assert len(self._shape) == 4, "The data is not of [C,H,W] form!"
        return self.shape[3]

    @property
    def HW(self) -> tuple[int, int]:
        """Tuple of (height, width) in Image data."""
        return (self.H, self.W)

    @property
    def CHW(self) -> tuple[int, int, int]:
        """Tuple of (channels, height, width) in Image data."""
        return (self.C, self.H, self.W)

    @property
    def HWC(self) -> tuple[int, int, int]:
        """Tuple of (height, width, channels) in Image data."""
        return (self.H, self.W, self.C)

    @property
    def batch_size(self) -> int | None:
        """Batch size of the data. Will be default if was not provided."""
        return self.shape[0]

    @property
    def shape(self) -> tuple[int | None, ...]:
        """Shape of the underlying Symbolic Tensor, including batch size.

        Dimensions created with ``None`` in ``Input`` are dynamic and are returned as ``None``.
        """
        if self._dynamic_dims:
            return tuple(None if idx in self._dynamic_dims else int(size) for idx, size in enumerate(self._shape))
        return self._shape

    @property
    def numel(self) -> int:
        """Number of the values in underlying Symbolic Tensor. If batch size is known, it is used too.

        Dynamic dimensions are counted with the placeholder size they were traced with.
        """
        return self._shape.numel()

    # These methods do not need to be defined, because SymbolicData is redirecting __getattr__.
    # However, we define basic methods to ensure they will be used without overhead of __getattr__.

    def reshape(self, *shape) -> SymbolicTensor:
        reshape_layer = useful_layers.ReshapeLayer(shape, batch_size_included=True)
        return reshape_layer(self)

    def view(self, *shape) -> SymbolicTensor:
        view_copy_layer = useful_layers.ViewCopyLayer(shape, batch_size_included=True)
        return view_copy_layer(self)

    def t(self) -> SymbolicTensor:
        transpose_layer = useful_layers.LambdaOpLayer(op=lambda x: x.t())
        return transpose_layer(self)

    @property
    def T(self) -> SymbolicTensor:
        transpose_layer = useful_layers.LambdaOpLayer(op=lambda x: x.T)
        return transpose_layer(self)

    def mean(self, dim=None, keepdim=False) -> SymbolicTensor:
        layer = useful_layers.AggregateLayer(torch.mean, dim=dim, keepdim=keepdim)
        return layer(self)

    def sum(self, dim=None, keepdim=False) -> SymbolicTensor:
        layer = useful_layers.AggregateLayer(torch.sum, dim=dim, keepdim=keepdim)
        return layer(self)

    def median(self, dim=None, keepdim=False) -> SymbolicTensor:
        layer = useful_layers.AggregateLayer(torch.median, dim=dim, keepdim=keepdim)
        return layer(self)

    def argmax(self, dim=None, keepdim=False) -> SymbolicTensor:
        layer = useful_layers.AggregateLayer(torch.argmax, dim=dim, keepdim=keepdim)
        return layer(self)

    def argmin(self, dim=None, keepdim=False) -> SymbolicTensor:
        layer = useful_layers.AggregateLayer(torch.argmin, dim=dim, keepdim=keepdim)
        return layer(self)

    def flatten(self, start_dim=0, end_dim=-1) -> SymbolicTensor:
        return nn.Flatten(start_dim, end_dim)(self)

    # These operators do not need to be defined!
    # However, we define basic operators to ensure they will be used without overhead of __getattr__.

    def __abs__(self):
        return self(useful_layers.LambdaOpLayer(lambda x: abs(x)))

    def __neg__(self):
        return self(useful_layers.LambdaOpLayer(op=lambda x: -x))

    def __add__(self, other):
        if isinstance(other, SymbolicTensor):
            return self(useful_layers.AddOpLayer(), other)
        else:
            return self(useful_layers.LambdaOpLayer(op=lambda x: x + other))

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if isinstance(other, SymbolicTensor):
            return self(useful_layers.MulOpLayer(), other)
        else:
            return self(useful_layers.LambdaOpLayer(op=lambda x: x * other))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __mod__(self, other):
        if isinstance(other, SymbolicTensor):
            return self(useful_layers.ModOpLayer(), other)
        else:
            return self(useful_layers.LambdaOpLayer(op=lambda x: x % other))

    def __rmod__(self, other):
        return self(useful_layers.LambdaOpLayer(op=lambda x: other % x))

    def __pow__(self, other):
        if isinstance(other, SymbolicTensor):
            return self(useful_layers.LambdaOpLayer(op=lambda x, y: x**y), other)
        else:
            return self(useful_layers.LambdaOpLayer(op=lambda x: x**other))

    def __rpow__(self, other):
        return self(useful_layers.LambdaOpLayer(op=lambda x: other**x))

    def __sub__(self, other):
        if isinstance(other, SymbolicTensor):
            return self(useful_layers.SubOpLayer(), other)
        else:
            return self(useful_layers.LambdaOpLayer(op=lambda x: x - other))

    def __rsub__(self, other):
        return self(useful_layers.LambdaOpLayer(op=lambda x: other - x))

    def __truediv__(self, other):
        if isinstance(other, SymbolicTensor):
            return self(useful_layers.LambdaOpLayer(op=lambda x, y: x / y), other)
        else:
            return self(useful_layers.LambdaOpLayer(op=lambda x: x / other))

    def __rtruediv__(self, other):
        return self(useful_layers.LambdaOpLayer(op=lambda x: other / x))

    def __matmul__(self, other):
        if isinstance(other, SymbolicTensor):
            return self(useful_layers.MatmulOpLayer(), other)
        else:
            return self(useful_layers.LambdaOpLayer(op=lambda x: x @ other))

    def __rmatmul__(self, other):
        return self(useful_layers.LambdaOpLayer(op=lambda x: other @ x))


_SYMBOLIC_FACTORY_CACHE = {}


def SymbolicFactory(dtype):
    global _SYMBOLIC_FACTORY_CACHE

    # Cache by the type object itself: distinct types can share a name
    if dtype not in _SYMBOLIC_FACTORY_CACHE:
        logging.debug(f"New underlying data detected: {dtype.__name__}!")
        cls = type(f"SymbolicData({dtype.__name__})", (SymbolicData,), {})
        _SYMBOLIC_FACTORY_CACHE[dtype] = cls
    return _SYMBOLIC_FACTORY_CACHE[dtype]


def Input(
    shape: tuple | list = (),
    batch_size: int = 1,
    batch_shape: tuple | list | None = None,
    dtype=torch.float32,
    min_value: float = 0.0,
    max_value: float = 1.0,
    dynamic_size_hint: int = 16,
) -> SymbolicTensor:
    """Input to Symbolic Model. Create Symbolic Tensor as a root node in the graph.

    Symbolic Tensor returned by Input has no parents while every other Symbolic Tensor has at least one.

    Parameters
    ----------
    shape
        Shape of the real data NOT including the batch dimension.
        A dimension can be ``None`` to mark it as dynamic, like in Keras:
        it will be reported as ``None`` in ``.shape`` and in model summary,
        and the real data can have any size there.
    batch_size
        Optional batch size of the Tensor
    batch_shape
        Shape of the real data including the batch dimension.
        If both ``shape`` and ``batch_shape`` are given, ``batch_shape`` has higher priority.
        ``batch_shape[0]`` can be ``None`` to mark the batch size as unknown.
    dtype
        Dtype of the real data that will be the input of the network
    min_value
        In rare cases, if real world data is very specific and some values
        cannot work with the model, this should be used to set a
        reasonable minimal value that the model can take as an input.
    max_value
        As above, but the maximal value
    dynamic_size_hint
        Placeholder size used during tracing for the dimensions marked as ``None``.
        Increase it if the traced layers require larger inputs, e.g. for deep
        downsampling stacks.

    Returns
    -------
    SymbolicTensor
        Root node in the graph
    """
    assert dynamic_size_hint >= 1, "dynamic_size_hint must be a positive integer!"
    batch_size_known = True
    dynamic_dims = set()

    if batch_shape is not None:
        batch_size = batch_shape[0]
        shape = batch_shape[1:]
        if batch_size is None:
            batch_size = 1
            batch_size_known = False
            dynamic_dims.add(0)
    else:
        # By default, we use batch_size of 1 under the hood
        batch_size_known = False

    dynamic_dims.update(idx + 1 for idx, size in enumerate(shape) if size is None)
    shape = tuple(dynamic_size_hint if size is None else size for size in shape)

    value = torch.rand(batch_size, *shape) * (max_value - min_value) + min_value
    value = value.to(dtype)
    symbolic_tensor = SymbolicTensor(value=value, batch_size_known=batch_size_known)
    symbolic_tensor._dynamic_dims = dynamic_dims
    return symbolic_tensor


def CustomInput(data: Any) -> SymbolicData:
    """Input to Symbolic Model. Creates Symbolic Data as a root node in the graph.

    This should be used when Input won't work.

    Parameters
    ----------
    data
        Speficic data that will be used during the graph tracing.
        It can, but doesn't need to be a torch.Tensor.

    Returns
    -------
    SymbolicData
        Root node in the graph
    """
    cls = _figure_out_symbolic_type(data)
    return cls(value=data, batch_size_known=True)


def _figure_out_symbolic_type(v):
    if isinstance(v, torch.Tensor):
        cls = SymbolicTensor
    elif isinstance(v, Callable):
        cls = SymbolicCallable
    else:
        cls = SymbolicFactory(type(v))
    return cls
