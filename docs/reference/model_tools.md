# model_tools

Small utilities for comparing and inspecting PyTorch models.

## Public API

- `get_parameter_count(model, only_trainable=False)` returns the number of model
  parameters.
- `get_parameter_shapes(model)` returns the shape of every parameter.
- `model_similar(a, b)` checks whether two models have matching parameter counts
  and shapes.
- `hash_torch_tensor(tensor)` hashes a tensor using its data, dtype, and shape.
- `models_have_corresponding_parameters(a, b)` checks whether two models contain
  parameters with identical hashes, regardless of parameter order.

[View signatures, implementation, and docstrings](../../pytorch_symbolic/model_tools.py).
