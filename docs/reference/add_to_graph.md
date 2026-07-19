# add_to_graph

This module provides a way to add custom functions to the graph.

To register a function in your graph, instead of doing:

- ``x = function(*args, **kwds)``

do this:

- ``x = add_to_graph(function, *args, **kwds)``.

For this to work, there must be at least one Symbolic Data among ``*args, **kwds``,
but other data types are allowed to be there as well.

Example for using ``torch.concat``:

```python
from pytorch_symbolic import Input
from pytorch_symbolic.functions_utility import add_to_graph
import torch

v1 = Input((10,))
v2 = Input((20,))
output = add_to_graph(torch.concat, tensors=(v1, v2), dim=1)
output
```

```
<SymbolicTensor at 0x7ffb77ba87f0; 2 parents; 0 children>
```

This will work for most of the user custom functions, even if Symbolic Tensors
are hidden in nested tuples, lists or dicts. You should also know that there is
a small time overhead for `__call__` during runtime for every function registered this way.
This overhead _should not_ be present when dealing with large models on GPU,
because then CPU does its work before GPU finishes previous kernel computation. 

Recommended, overhead-free way to use custom functions is to write yourself an ``nn.Module`` that does the same as the function of choice.
Then you can use the model without sacrificing performance.

## API

```text
add_to_graph(func, *args, custom_name=None, **kwargs)
```

Registers `func` or a `torch.nn.Module` in the computation graph. At least one
item in `args` or `kwargs` must be symbolic. Symbolic values may be nested in
lists, tuples, and dictionaries. `custom_name` sets the displayed graph-layer
name.

[View the implementation and full docstring](../../pytorch_symbolic/functions_utility.py).
