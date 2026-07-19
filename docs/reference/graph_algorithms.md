# graph_algorithms

Utilities for inspecting, sorting, and drawing symbolic computation graphs.

## Public API

- `figure_out_nodes_between(inputs=None, outputs=None)` returns the graph nodes
  between the selected inputs and outputs.
- `draw_graph(...)` plots a graph from a model or from explicit inputs and
  outputs. Graph drawing requires the optional `networkx`, `matplotlib`, and
  `scipy` dependencies.
- `sort_graph_and_check_DAG(nodes)` returns nodes in execution order and checks
  that the graph is acyclic.
- `default_node_text(symbol)` and `default_edge_text(layer)` provide the default
  labels used by `draw_graph`.
- `variable_name_resolver(namespace)` creates a node-label function based on a
  namespace such as `globals()`.

[View signatures, implementation, and docstrings](../../pytorch_symbolic/graph_algorithms.py).
