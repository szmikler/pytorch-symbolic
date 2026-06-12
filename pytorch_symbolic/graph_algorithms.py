#  Copyright (c) 2022 Szymon Mikler

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any

from .symbolic_data import SymbolicData, SymbolicTensor, useful_layers

if TYPE_CHECKING:
    from .symbolic_model import SymbolicModel

from collections import defaultdict

from torch import nn


def check_for_missing_inputs(
    used_nodes: set[SymbolicData],
    inputs: tuple[SymbolicData, ...] | list[SymbolicData],
):
    """Check if there exist nodes which require input from outside of the graph.

    It is forbidden, as it doesn't make sense.

    Example of such violation::

        x1 = Input(shape=(32,))
        x2 = Input(shape=(32,))
        x3 = x1 + x2
        model = SymbolicModel(inputs=x1, outputs=x3)

    Model cannot execute defined operations unless ``x2`` is given, because ``x3`` requires it.
    """
    for node in used_nodes:
        if node in inputs:
            continue
        for parent in node.parents:
            assert parent in used_nodes, (
                f"Node {node} depends on the output of a foreign node! Perhaps you set the wrong inputs?"
            )


def figure_out_nodes_between(
    inputs: tuple[SymbolicData, ...] | list[SymbolicData] | None = None,
    outputs: tuple[SymbolicData, ...] | list[SymbolicData] | None = None,
) -> set[SymbolicData]:
    """Returns intersection of predecessors tree of outputs and succesors tree of inputs."""

    all_nodes_above: list[SymbolicData] = []
    if outputs is not None:
        for output_leaf in outputs:
            nodes_above = output_leaf._get_all_nodes_above()
            all_nodes_above.extend(nodes_above)

    all_nodes_below: list[SymbolicData] = []
    if inputs is not None:
        for input_leaf in inputs:
            nodes_below = input_leaf._get_all_nodes_below()
            all_nodes_below.extend(nodes_below)

    # Get the intersection
    if inputs is None or outputs is None:
        used_nodes = set(all_nodes_above) | set(all_nodes_below)
    else:
        used_nodes = set(all_nodes_above) & set(all_nodes_below)
    if inputs is not None:
        check_for_missing_inputs(used_nodes, inputs)
    return used_nodes


def default_node_text(sym: SymbolicData) -> str:
    if isinstance(sym, SymbolicTensor):
        return str(tuple(sym.v.shape))
    # mypy hates the next line for some reason
    if isinstance(sym.v, Callable):  # type: ignore
        return "callable"
    if hasattr(sym.v, "__len__"):
        return f"{type(sym.v).__name__}({len(sym.v)})"
    return type(sym.v).__name__


def default_edge_text(layer: nn.Module) -> str:
    if isinstance(layer, useful_layers.GetAttr):
        return "." + layer.name
    else:
        return layer._get_name()


def _calc_sum_sq_distances(graph, positions_dict):
    sum_sq_distances = 0
    for node in positions_dict:
        related_nodes = list(graph.predecessors(node))

        for n2 in related_nodes:
            x1, y1 = positions_dict[node]
            x2, y2 = positions_dict[n2]
            sum_sq_distances += (x1 - x2) ** 2 + (y1 - y2) ** 2
    return sum_sq_distances


def _transpose_positions_dict(pos_dict):
    return {k: tuple(reversed(v)) for k, v in pos_dict.items()}


def _fix_positions_in_multipartite_layout(graph, orig_positions_dict, align: str = "vertical"):
    import networkx as nx
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    assert isinstance(graph, nx.DiGraph)

    if align == "vertical":
        orig_positions_dict = _transpose_positions_dict(orig_positions_dict)

    orig_sum = _calc_sum_sq_distances(graph, orig_positions_dict)

    positions = list(orig_positions_dict.values())
    selected_positions: dict[int, tuple[float, float]] = {}

    layers = defaultdict(list)
    for pos in positions:
        layers[pos[1]].append(pos[0])

    layer_to_nodes = defaultdict(list)
    for node, pos in orig_positions_dict.items():
        layer_to_nodes[pos[1]].append(node)

    # Layers are just unique Y positions on the plot
    # Go through the highest layer to the lowest one
    for layer, nodes_in_layer in reversed(sorted(layer_to_nodes.items())):
        n = len(nodes_in_layer)
        distances = np.zeros(shape=(n, n))

        # Extract available X positions in the layer
        avail_xs = layers[layer]

        for node_idx, node in enumerate(nodes_in_layer):
            related_nodes = list(graph.predecessors(node))
            related_nodes += list(graph.successors(node))

            for x_idx, x in enumerate(avail_xs):
                sum_distances = 0
                for related_node in related_nodes:
                    if related_node in selected_positions:
                        r_x, r_y = selected_positions[related_node]
                        sum_distances += (x - r_x) ** 2 + (layer - r_y) ** 2
                    else:
                        # If node doesn't have a position yet
                        # we use the original position but we weight it down
                        r_x, r_y = orig_positions_dict[related_node]
                        sum_distances += ((x - r_x) ** 2 + (layer - r_y) ** 2) / 2
                distances[node_idx, x_idx] = sum_distances

        # Minimize the sum of squared distances
        # This might work poorly when there are interconnections in the layer
        rows, cols = linear_sum_assignment(distances)
        for row, col in zip(rows, cols, strict=True):
            selected_positions[nodes_in_layer[row]] = (avail_xs[col], layer)

    new_sum = _calc_sum_sq_distances(graph, selected_positions)

    if new_sum <= orig_sum:
        r = selected_positions
    else:
        r = orig_positions_dict

    if align == "vertical":
        r = _transpose_positions_dict(r)

    return r


def variable_name_resolver(namespace):
    def resolver(x):
        names = [name for name in namespace if namespace[name] is x]
        return names[0] if names else "?"

    return resolver


def draw_graph(
    *,
    model: SymbolicModel | None = None,
    inputs: Iterable[SymbolicData] | SymbolicData | None = None,
    outputs: Iterable[SymbolicData] | SymbolicData | None = None,
    node_text_func: Callable[[SymbolicData], str] | None = None,
    edge_text_func: Callable[[nn.Module], str] | None = None,
    node_text_namespace: dict[str, Any] | None = None,
    rotate_graph: bool = False,
    rotate_labels: bool = False,
    show: bool = False,
    figsize=None,
):
    """Plot graph of the computations, nodes being placeholder variables and nn.Modules being edges.

    This is not suitable for large graphs or large neural networks. This is a simple tool that
    was designed to demonstrate that Pytorch Symbolic creates sensible graphs that are nice
    to visualize.

    Parameters
    ----------
    model
        A SymbolicModel to be plotted. This or ``inputs`` must be provided.
    inputs
        Input in the graph of Symbolic computations. This or ``model`` or/and ``inputs`` must be provided.
    outputs
        Output in the graph of Symbolic computations. This or ``model`` or/and ``outputs`` must be provided.
    node_text_func
        A function that returns text that will be written on Nodes.
    edge_text_func
        A function that returns text that will be written on Edges.
    node_text_namespace
        If used with ``globals()``, it will try to show variable name on each node.
        Has an effect only if `node_text_func` is None.
    rotate_labels
        If True, text on edges will be rotated in the direction of the arrow.
    rotate_graph
        If True, the consecutive layers will be shown to the right, instead of downwards.
    show
        Call matplotlib.pyplot.show
    """
    try:
        import matplotlib.patches
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError as e:
        print("To plot graphs, you need to install networkx, matplotlib and scipy. Run `pip install networkx`.")
        raise e

    from .symbolic_model import SymbolicModel

    if node_text_func is None:
        if node_text_namespace is not None:
            node_text_func = variable_name_resolver(node_text_namespace)
        else:
            node_text_func = default_node_text

    if edge_text_func is None:
        edge_text_func = default_edge_text

    if model is not None:
        assert isinstance(model, SymbolicModel)
        inputs = model.inputs
        outputs = model.outputs

    elif inputs is not None or outputs is not None:
        if inputs is not None:
            if isinstance(inputs, SymbolicData):
                inputs = (inputs,)
            assert all(isinstance(x, SymbolicData) for x in inputs)
            inputs = tuple(inputs)
        if outputs is not None:
            if isinstance(outputs, SymbolicData):
                outputs = (outputs,)
            assert all(isinstance(x, SymbolicData) for x in outputs)
            outputs = tuple(outputs)

    else:
        raise KeyError("Provide either `model` or `inputs` or/and `outputs`!")

    used_nodes = figure_out_nodes_between(inputs, outputs)

    graph = nx.DiGraph()
    assert isinstance(inputs, Iterable)
    to_visit = list(inputs)
    visited = set(inputs)

    node_labels = {}
    node_colors = {}
    edge_labels = {}

    INPUT_COLOR = (1.0, 0.3, 0.3)
    OUTPUT_COLOR = (0.3, 1.0, 0.3)
    OTHER_COLOR = (0.0, 1.0, 1.0)

    while to_visit:
        node = to_visit.pop()
        graph.add_node(node, depth=node.depth)

        if inputs and node in inputs:
            node_colors[node] = INPUT_COLOR
        elif outputs and node in outputs:
            node_colors[node] = OUTPUT_COLOR
        else:
            node_colors[node] = OTHER_COLOR

        node_labels[node] = node_text_func(node)

        for parent in node.parents:
            if parent not in used_nodes:
                continue
            assert node.layer is not None
            graph.add_edge(parent, node, layer=node.layer)
            edge_labels[parent, node] = edge_text_func(node.layer)

        for child in node.children:
            if child in used_nodes and child not in visited:
                to_visit.append(child)
                visited.add(child)

    scale = 1 if rotate_graph else -1
    align = "vertical" if rotate_graph else "horizontal"
    pos = nx.multipartite_layout(graph, subset_key="depth", align=align, scale=scale)
    pos = _fix_positions_in_multipartite_layout(graph, pos, align=align)

    nx.draw_networkx(
        graph,
        pos,
        with_labels=True,
        labels=node_labels,
        node_size=1000,
        arrowsize=20,
        node_shape="o",
        node_color=[node_colors[n] for n in graph.nodes],
        font_weight="bold",
        edge_color="grey",
    )

    nx.draw_networkx_edge_labels(
        graph,
        pos,
        edge_labels,
        label_pos=0.5,
        rotate=rotate_labels,
        clip_on=False,
    )
    handles = [
        matplotlib.patches.Patch(color=INPUT_COLOR, label="Input node"),
        matplotlib.patches.Patch(color=OUTPUT_COLOR, label="Output node"),
        matplotlib.patches.Patch(color=OTHER_COLOR, label="Hidden node"),
    ]
    plt.legend(handles=handles)

    fig: plt.Figure = plt.gcf()
    if figsize:
        fig.set_size_inches(*figsize)

    fig.tight_layout()
    if show:
        plt.show()
    return fig


def sort_graph_and_check_DAG(nodes: set[SymbolicData]) -> list[SymbolicData]:
    """Sort graph topologically.

    Wikipedia:
    In graph theory, a topological sort or topological ordering of a directed acyclic graph (DAG) is a
    linear ordering of its nodes in which each node comes before all nodes to which it has outbound edges.
    Every DAG has one or more topological sorts.
    """

    children = {node: set((c for c in node.children if c in nodes)) for node in nodes}
    parents = {node: set((p for p in node.parents if p in nodes)) for node in nodes}
    grandparents = [node for node in nodes if len(parents[node]) == 0]  # no incoming edges

    n_edges = sum(len(v) for v in children.values()) + sum(len(v) for v in parents.values())

    topologically_sorted = []

    while grandparents:
        node = grandparents.pop()
        topologically_sorted.append(node)

        for neighbor in tuple(children[node]):
            children[node].remove(neighbor)
            parents[neighbor].remove(node)

            if len(parents[neighbor]) == 0:
                grandparents.append(neighbor)

    n_edges = sum(len(v) for v in children.values()) + sum(len(v) for v in parents.values())

    assert n_edges == 0, "Graph is not a DAG (directed acyclic graph)!"
    return topologically_sorted
