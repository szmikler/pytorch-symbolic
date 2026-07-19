# useful_layers

Collection of simple ``torch.nn.Modules``.

Some of them might be useful when building models using Pytorch Symbolic.

## Layers

- `LambdaOpLayer` and `NamedLambdaOpLayer` wrap a callable as a module.
- `CallbackLayer` runs a callback and returns its input unchanged.
- `AddOpLayer`, `SubOpLayer`, `MulOpLayer`, `ModOpLayer`, and `MatmulOpLayer`
  implement basic symbolic operators.
- `ConcatLayer` and `StackLayer` combine tensors along a selected dimension.
- `ReshapeLayer` and `ViewCopyLayer` change tensor shapes.
- `AggregateLayer` wraps reductions such as `torch.mean` and `torch.sum`.
- `UnpackLayer`, `SliceLayer`, and `SliceLayerSymbolicIdx` support unpacking and
  indexing graph values.
- `MethodCall` and `GetAttr` represent method calls and attribute access.

[View constructors and implementations](../../pytorch_symbolic/useful_layers.py).
