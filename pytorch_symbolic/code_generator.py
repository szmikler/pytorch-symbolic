# Copyright (c) 2022 Szymon Mikler

from __future__ import annotations

from pytorch_symbolic.symbolic_data import SymbolicData


def generate_forward_with_loops(
    inputs: list[SymbolicData] | tuple[SymbolicData, ...],
    outputs: list[SymbolicData] | tuple[SymbolicData, ...],
    execution_order: list[SymbolicData] | tuple[SymbolicData, ...],
    nodes_in_subgraph: set[SymbolicData],
    min_loop_length: int | float = float("inf"),
) -> str:
    """Generate code for forward function of SymbolicModel.

    It assumes there is `self._execution_order_layers` available in the class.

    Parameters
    ----------
    inputs
        Inputs to the model
    outputs
        Outputs of the model
    execution_order
        Contains the exact order in which the nodes should be executed.
        If there are layers with multiple outputs, this will be a subset of `nodes_in_subgraph`.
        In such case, only one output of each layer needs to be in the execution_order.
    nodes_in_subgraph
        All nodes covered by the subgraph, including all nodes created by multiple-output layers.
    min_loop_length
        Minimal sequence length to replace sequential layers execution with a loop.

    Returns
    -------
    str
        Generated code.
    """
    assert min_loop_length >= 2, "Loop length cannot be smaller than 2!"

    str_length = len(str(max(len(inputs), len(outputs), len(execution_order))))
    node_to_name = {}
    for idx, node in enumerate(inputs):
        node_to_name[node] = f"i{str(idx).zfill(str_length)}"
    for idx, node in enumerate(execution_order):
        node_to_name[node] = f"x{str(idx).zfill(str_length)}"
    for idx, node in enumerate(nodes_in_subgraph.difference(execution_order)):
        node_to_name[node] = f"y{str(idx).zfill(str_length)}"
    for idx, node in enumerate(outputs):
        node_to_name[node] = f"o{str(idx).zfill(str_length)}"

    input_names = [node_to_name[node] for node in inputs]
    forward_definition = "def forward(self," + ", ".join(input_names) + "):"
    code_lines = [forward_definition]

    TAB = " " * 4
    code_lines.append(TAB + "l = self._execution_order_layers")

    nodes_looped_over = set()
    # All parents must be in the graph. Otherwise, forward is impossible.
    parents = {node: node.parents for node in execution_order}
    # We only count children in the graph. Thus the intersection.
    children = {node: list(nodes_in_subgraph.intersection(node.children)) for node in execution_order}

    siblings = {node: list(nodes_in_subgraph.intersection(node._layer_full_siblings)) for node in execution_order}

    for exec_id, node in enumerate(execution_order):
        if node in nodes_looped_over:
            continue

        input_names = [node_to_name[node] for node in node.parents]
        sequence = [node]
        last_node = node
        while (
            len(children[last_node]) == 1
            # stop iterating when need to unpack something!
            and len(last_node._layer_full_siblings) == 1
            and len(children[last_node][0]._layer_full_siblings) == 1  # needed, else tests fail
            # this should never be false, but just in case we make sure the child is next in execution order
            and children[last_node][0] is execution_order[exec_id + len(sequence)]
            and len(parents[last_node]) == 1
            and len(parents[children[last_node][0]]) == 1
        ):
            last_node = children[last_node][0]
            sequence.append(last_node)

        if len(sequence) >= min_loop_length:
            output_name = node_to_name[sequence[-1]]
            code_lines.append(TAB + f"{output_name} = {input_names[0]}")
            code_lines.append(TAB + f"for layer in l[{exec_id}:{exec_id + len(sequence)}]:")
            code_lines.append(TAB + TAB + f"{output_name} = layer({output_name})")
            nodes_looped_over.update(sequence)
        elif len(node._layer_full_siblings) > 1:  # Must unpack all siblings, even if not all are used
            output_names = []
            for n in node._layer_full_siblings:
                if n in siblings[node]:
                    output_names.append(node_to_name[n])
                else:
                    output_names.append("_")  # If sibling not used, we don't save it as a variable

            assert len(input_names) == 1, "Layer that has full siblings cannot have more than 1 input!"
            code_line = TAB + ", ".join(output_names) + f" = l[{exec_id}](" + "*" + input_names[0] + ")"
            code_lines.append(code_line)
        else:
            code_line = TAB + node_to_name[node] + f" = l[{exec_id}](" + ", ".join(input_names) + ")"
            code_lines.append(code_line)

    code_lines.append(TAB + "return " + ", ".join(node_to_name[node] for node in outputs))
    generated_forward = "\n".join(code_lines) + "\n"
    return generated_forward
