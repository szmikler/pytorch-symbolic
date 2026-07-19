# symbolic_data

This is a collection of symbolic data types. 

The main class and a grandfather for all other classes is Symbolic Data. 

Symbolic Tensor has similar API to ``torch.Tensor`` object,
but Symbolic Tensor is used only to define the graph, not to perform actual computations. 
You should use it to register new layers in your computation graph and later to 
create the model.
You can use methods of `torch.Tensor` when working with Symbolic Tensor.
For example `tensor.t()` or `tensor.T` will return another Symbolic Tensor
with transposition applied to the underlying data.

Symbolic Data supports slicing too, so you can do:
```python
from pytorch_symbolic import Input, SymbolicModel

x = Input(batch_shape=(3, 4, 5))

y = x[0]
for row in x[1:]:
	y += row
	
model = SymbolicModel(x, y)
```

But be careful! Each slice operation creates a new layer,
so if you do a lot of slicing, 
it is better enclose it in a custom module.
However, being able to do it directly on Symbolic Data is convenient for prototyping.


## `Input`

```text
Input(shape=(), batch_size=1, batch_shape=None, dtype=torch.float32,
      min_value=0.0, max_value=1.0)
```

Creates a root `SymbolicTensor`. `shape` excludes the batch dimension;
`batch_shape`, when provided, includes it and takes precedence. The value range
is used only for the sample tensor created while tracing the model.

## `CustomInput`

```text
CustomInput(data)
```

Creates a root symbolic value from arbitrary example data. Use it when the
tensor-oriented `Input` helper is not suitable.

## Symbolic types

- `SymbolicData` is the base class for symbolic values. Its `v`, `parents`, and
  `children` properties expose the tracing value and graph connections.
  `apply_module(layer, *others, custom_name=None)` registers an operation.
- `SymbolicTensor` represents tensor data and supplies tensor shape helpers such
  as `features`, `batch_size`, `C`, `H`, `W`, `CHW`, and `HWC`, as well as common
  tensor operations.
- `SymbolicCallable` represents callable values in a graph.

[View signatures, implementation, and full docstrings](../../pytorch_symbolic/symbolic_data.py).
