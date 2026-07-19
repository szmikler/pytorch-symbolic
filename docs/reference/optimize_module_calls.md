# optimize_module_calls

```text
optimize_module_calls()
```

Removes Pytorch Symbolic's wrapper from `torch.nn.Module.__call__`. This avoids
the wrapper overhead, but afterwards reusing existing layers with the
`layer(*symbols)` notation will not work.

[View the implementation and related API](../../pytorch_symbolic/symbolic_api_2.py).
