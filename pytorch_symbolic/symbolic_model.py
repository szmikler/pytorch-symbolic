#  Copyright (c) 2022 Szymon Mikler

from __future__ import annotations

import copy
import logging
from types import MethodType
from typing import Any, Dict, List, Set, Tuple

import torch
from torch import nn

from . import code_generator, config
from .graph_algorithms import figure_out_nodes_between, sort_graph_and_check_DAG
from .model_tools import get_parameter_count
from .symbolic_data import SymbolicData, SymbolicTensor


class DetachedSymbolicModel(nn.Module):
    def __init__(self, names: List[str], layers: List[nn.Module], forward_src: str):
        """A tiny model detached from the SymbolicModel graph structure.

        It can live, even if the graph structure is removed!

        Parameters
        ----------
        names
            Names for all the layers. Must be equal in length to ``layers``
        layers
            Ordered list of layers to be executed
        forward_src
            String containing definition of a ``forward`` function.
            Must define ``def forward(self, ...)``.
            Layers are available under ``self._execution_order_layers`` in the function's body.
        """
        assert len(names) == len(layers)

        super().__init__()
        self._execution_order_layers = []
        for name, layer in zip(names, layers):
            try:
                layer = copy.deepcopy(layer)
            except Exception as e:
                logging.error(f"Deepcopy of {layer} failed!")
                raise e
            self.add_module(name, layer)
            self._execution_order_layers.append(layer)

        self._generated_forward_source = forward_src

        scope = {"self": self}
        exec(self._generated_forward_source, {}, scope)
        self.forward = MethodType(scope["forward"], self)

    def __getstate__(self):
        # The generated forward is created with `exec` and cannot be pickled.
        # It is recreated in `__setstate__` from `_generated_forward_source`.
        state = super().__getstate__()
        state.pop("forward", None)
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        scope = {"self": self}
        exec(self._generated_forward_source, {}, scope)
        self.forward = MethodType(scope["forward"], self)


class SymbolicModel(nn.Module):
    def __init__(
        self,
        inputs: Tuple[SymbolicData, ...] | List[SymbolicData] | SymbolicData,
        outputs: Tuple[SymbolicData, ...] | List[SymbolicData] | SymbolicData,
        enable_forward_codegen=None,
    ):
        """A PyTorch model that replays operations defined in the graph.

        All operations that were required to change ``inputs`` into ``outputs`` will be replayed
        in the same order, but on the real data provided as input to this model.

        Example::

            input1 = Input((10,))
            input2 = Input((10,))
            x = input1 + input2
            x = nn.Linear(x.features, 1)(x)
            model = SymbolicModel((input1, input2), x)

        Parameters
        ----------
        inputs
            A collection of SymbolicData that represent the input data used by the model.
            It is you who provide the specific data when the model is created.
            If you have mulitple inputs here, be prepared to pass multiple inputs during training/inference.
        outputs
            A collection of SymbolicTensors that will end the computations.
            These nodes return your final computation result.
            So if you have mulitple outputs, SymbolicModel will return a tuple of tensors.

        Attributes
        ----------
        inputs : tuple
            Non-modifiable tuple of input nodes
        outputs : tuple
            Non-modifiable tuple of output nodes
        """
        super().__init__()
        logging.debug("Creating a SymbolicModel...")

        if isinstance(inputs, SymbolicData):
            inputs = (inputs,)
        assert all(isinstance(x, SymbolicData) for x in inputs), "Only SymbolicData allowed in inputs!"
        self.inputs: Tuple[SymbolicData, ...] = tuple(inputs)

        if isinstance(outputs, SymbolicData):
            outputs = (outputs,)
        assert all(isinstance(x, SymbolicData) for x in outputs), "Only SymbolicData allowed in outputs!"
        self.outputs: Tuple[SymbolicData, ...] = tuple(outputs)

        # Initialize helper variables
        self._layer_type_counts: Dict[str, int] = {}
        self._node_to_layer_name: Dict[SymbolicData, str] = {}
        self._execution_order_nodes: List[SymbolicData] = []
        self._execution_order_layers: List[nn.Module] = []
        self._figure_out_execution_order()

        if enable_forward_codegen is None:
            enable_forward_codegen = config.CODEGEN_BY_DEFAULT
        self._enable_forward_codegen = enable_forward_codegen
        if self._enable_forward_codegen:
            self._replace_forward_with_codegen()

        self._clear_underlying_values()  # clear up after tracing to save memory

    def forward(self, *inputs: torch.Tensor) -> Any:
        """This function is executed by __call__. Do not use this directly, use __call__ instead.

        Warning!

        This function will be overwritten by `_replace_forward_with_codegen` if `enable_forward_codegen`
        is True. If this happened and you want to see your source, print `self._generated_forward_source`.
        """
        assert len(inputs) == len(self.inputs), "Number of inputs doesn't match!"
        for input_data, input_node in zip(inputs, self.inputs):
            input_node._launch_input(input_data)

        for node in self._execution_order_nodes:
            node._launch()

        if len(self.outputs) == 1:
            return self.outputs[0]._value
        else:
            return tuple(output_leaf._value for output_leaf in self.outputs)

    @property
    def input_shape(self):
        """Return shape of the input or in case of multiple inputs - a tuple of them."""
        shapes = [node.shape if isinstance(node, SymbolicTensor) else None for node in self.inputs]
        return tuple(shapes) if len(shapes) > 1 else shapes[0]

    @property
    def output_shape(self):
        """Return shape of the output or in case of multiple outputs - a tuple of them."""
        shapes = [node.shape if isinstance(node, SymbolicTensor) else None for node in self.outputs]
        return tuple(shapes) if len(shapes) > 1 else shapes[0]

    def add_output(self, node: SymbolicData):
        assert node not in self.inputs, "Node is an input of this SymbolicModel!"
        assert node in self._execution_order_nodes, "Node is out of reach for this SymbolicModel!"

        self.outputs = (*self.outputs, node)
        if self._enable_forward_codegen:
            self._replace_forward_with_codegen()

    def detach_from_graph(self) -> DetachedSymbolicModel:
        names = [self._node_to_layer_name[node] for node in self._execution_order_nodes]
        forward_src = code_generator.generate_forward_with_loops(
            self.inputs,
            self.outputs,
            execution_order=self._execution_order_nodes,
            nodes_in_subgraph=self._used_nodes(),
            min_loop_length=config.CODEGEN_MIN_LOOP_LENGTH,
        )
        return DetachedSymbolicModel(names, self._execution_order_layers, forward_src)

    def summary(self):
        """Print Keras-like model summary."""
        space_between_cols = 3

        data = [["", "Layer", "Output shape", "Params", "Parent"]]
        ncols = len(data[0])
        separators = ["="]
        node_to_idx = {}
        for node in self.inputs:
            node_to_idx[node] = len(data)
            if isinstance(node, SymbolicTensor):
                shape = list(node.shape)
                if not node.batch_size_known:
                    shape[0] = None
                shape = tuple(shape)
            else:
                shape = node._underlying_type_name
            data.append(
                [
                    f"{len(data)}" + ("*" if node in self.outputs else ""),
                    f"Input_{len(data)}",
                    str(shape),
                    "0",
                    "",
                ]
            )
            separators.append(None)

        for node in self._execution_order_nodes:
            node_to_idx[node] = len(data)
            layer = node.layer
            if isinstance(node, SymbolicTensor):
                shape = list(node.shape)
                if not node.batch_size_known:
                    shape[0] = None
                shape = tuple(shape)
            else:
                shape = node._underlying_type_name
            data.append(
                [
                    f"{len(data)}" + ("*" if node in self.outputs else ""),
                    f"{self._node_to_layer_name[node]}",
                    str(shape),
                    str(get_parameter_count(layer)),
                    ",".join(str(node_to_idx[parent]) for parent in node.parents),
                ]
            )
            separators.append(None)
        separators[-1] = "="

        maxcolwidth = [0 for _ in range(ncols)]
        for row in data:
            for idx, col in enumerate(row):
                if len(col) > maxcolwidth[idx]:
                    maxcolwidth[idx] = len(col)

        print("_" * (sum(maxcolwidth) + ncols * space_between_cols))
        for sep, row in zip(separators, data):
            for idx, col in enumerate(row):
                s = col.ljust(maxcolwidth[idx] + space_between_cols, " ")
                print(s, end="")
            print()
            if sep is not None:
                print(sep * (sum(maxcolwidth) + ncols * space_between_cols))

        parameter_count = get_parameter_count(self)
        trainable_count = get_parameter_count(self, only_trainable=True)
        print(f"Total params: {parameter_count}")
        print(f"Trainable params: {trainable_count}")
        print(f"Non-trainable params: {parameter_count - trainable_count}")
        print("_" * (sum(maxcolwidth) + ncols * space_between_cols))

    def _clear_underlying_values(self):
        """Clear values of the underlying nodes to save memory.

        Does not clear the values of input nodes, as they might be needed.
        """
        for node in self._used_nodes().difference(self.inputs):
            node._clear_value()

    def _replace_forward_with_codegen(self):
        self._generated_forward_source = code_generator.generate_forward_with_loops(
            self.inputs,
            self.outputs,
            execution_order=self._execution_order_nodes,
            nodes_in_subgraph=self._used_nodes(),
            min_loop_length=config.CODEGEN_MIN_LOOP_LENGTH,
        )
        scope = {"self": self}
        exec(self._generated_forward_source, {}, scope)
        self.forward = MethodType(scope["forward"], self)

    def __getstate__(self):
        # The codegen forward is created with `exec` and cannot be pickled.
        # It is recreated in `__setstate__` from `_generated_forward_source`.
        state = super().__getstate__()
        state.pop("forward", None)
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        if self._enable_forward_codegen:
            scope = {"self": self}
            exec(self._generated_forward_source, {}, scope)
            self.forward = MethodType(scope["forward"], self)

    def _used_nodes(self) -> Set[SymbolicData]:
        """Return a set of all nodes used in this model."""
        return figure_out_nodes_between(self.inputs, self.outputs)

    def _remove_repeated_execution(self, execution_order_nodes: List[SymbolicData]) -> List[SymbolicData]:
        """In case of multiple outputs, we need only one of the output node to launch the layer."""
        nodes_without_repeated_execution = []
        used_nodes = self._used_nodes()

        already_executed: Set[SymbolicData] = set()
        for node in execution_order_nodes:
            if node in already_executed:
                continue
            nodes_without_repeated_execution.append(node)
            already_executed.update(used_nodes.intersection(node._layer_full_siblings))

        assert len(already_executed) == len(execution_order_nodes)
        return nodes_without_repeated_execution

    def _figure_out_execution_order(self):
        used_nodes = self._used_nodes()
        sort_graph_and_check_DAG(used_nodes)
        execution_order_nodes = sorted(self._used_nodes(), key=lambda node: node._execution_order_idx)
        assert len(execution_order_nodes) == len(used_nodes)

        for input_node in used_nodes.intersection(self.inputs):  # Not all inputs must be in `used_nodes`
            execution_order_nodes.remove(input_node)  # Exclude inputs, as we don't execute any layers there

        # To avoid calling layers twice when they have multiple outputs
        # We remove all nodes with already executed layer from execution order
        execution_order_nodes = self._remove_repeated_execution(execution_order_nodes)

        self._execution_order_nodes = execution_order_nodes
        self._execution_order_layers = [node.layer for node in self._execution_order_nodes]

        for _idx, node in enumerate(self._execution_order_nodes):
            if node._custom_provided_name is not None:
                # Use layer name provided by the user
                layer_name = node._custom_provided_name
            else:
                # Extract the default name of the layer provided by pytorch
                layer_name = node.layer._get_name()

            # Check how many layers with this name already exist
            self._layer_type_counts.setdefault(layer_name, 0)
            self._layer_type_counts[layer_name] += 1

            if node._custom_provided_name is None or self._layer_type_counts[layer_name] > 1:
                full_layer_name = f"{layer_name}_{self._layer_type_counts[layer_name]}"
            else:
                # Skip first index only if it was provided by the user
                full_layer_name = layer_name

            self._node_to_layer_name[node] = full_layer_name
            self.add_module(name=full_layer_name, module=node.layer)

    def __deepcopy__(self, memo):
        """This copies a working Module, but the underlying graph structure is not copied!"""
        obj = self.detach_from_graph()
        memo[id(self)] = obj
        return obj
