# SymbolicModel

```text
SymbolicModel(inputs, outputs, enable_forward_codegen=None)
```

A `torch.nn.Module` that replays all operations needed to transform the symbolic
`inputs` into the symbolic `outputs`. A single node or a collection of nodes can
be supplied for either argument.

## Main attributes and methods

- `inputs` and `outputs` contain the graph boundary nodes.
- `input_shape` and `output_shape` report the corresponding tensor shapes.
- `add_output(node)` adds another output from the existing graph.
- `summary()` prints a Keras-like model summary.
- `detach_from_graph()` creates a smaller standalone PyTorch module that no
  longer needs the symbolic graph structure.

[View the implementation and full docstrings](../../pytorch_symbolic/symbolic_model.py).
