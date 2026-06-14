# Quick Start

## Features

* Easy to use:
	* Familiar for users of [Keras Functional API](https://keras.io/guides/functional_api/)
	* Symbolic Tensors have API similar to `torch.Tensor`
* Lots of flexibility and advanced control flow:
	* Reusing layers
	* Shared layers
	* Multiple inputs and outputs
	* Custom, user-defined modules and functions
	* No restrictions on module's signature

## Introduction to Symbolic Data

To register a new layer, e.g. ``torch.nn.Linear``, in your model, you have two options:

* `layer(symbolic_data)` (just like in [Keras Functional API](https://keras.io/guides/functional_api/))
* `symbolic_data(layer)` (like nowhere else)

There are no differences between these options.
Models produced with them will be identical.

To create a linear classifier without hidden layers, you can write the following:

```python
from torch import nn
from pytorch_symbolic import Input, SymbolicModel

inputs = Input(shape=(28 * 28,))
outputs = nn.Linear(in_features=inputs.features, out_features=10)(inputs)
model = SymbolicModel(inputs, outputs)
```

Or equivalently:

```python
from torch import nn
from pytorch_symbolic import Input, SymbolicModel

inputs = Input(shape=(28 * 28,))
outputs = inputs(nn.Linear(in_features=inputs.features, out_features=10))
model = SymbolicModel(inputs, outputs)
```

### Computation graph

Under the hood of Pytorch Symbolic, there lives a computation graph.

Every Symbolic Tensor is a node in it. When interacting with Symbolic Tensor:

* Think of it as a placeholder for your data
* Use it like `torch.Tensor` (e.g. slicing and iterating over it)

Let us play with Symbolic Tensor and see what we can do:

```python
from pytorch_symbolic import Input, SymbolicModel

inputs = Input((28 * 28,))  # create a symbolic batch of vectors
print(inputs)
outputs = inputs[0]  # access first vector from the batch
print(inputs)
```

```stdout
<SymbolicTensor at 0x7f55dc437130; 0 parents; 0 children>
<SymbolicTensor at 0x7f55dc437130; 0 parents; 1 children>
```

At first, `inputs` was the only node in the graph. It had 0 parents and 0 children.

But when we created `outputs`, a new node was registered as a child of `inputs`.

Symbolic Tensors have useful attributes.
Using them we can, for example, instantly obtain shapes of intermediate outputs,
instead of deriving them by hand. Let's see it:

```py
print(inputs.shape)
print(inputs.features)
```

```stdout
torch.Size([1, 784])
784
```

In some cases, calculating output shapes in your code adds unnecessary complexity,
for example when padding and convolutions are involved.
By calculating shapes automatically we write less code and write easier code.

After creating the graph,
you can replay all the defined operations by using a Symbolic Model.

```py
model = SymbolicModel(inputs=inputs, outputs=inputs + 1)
```

This is a model that adds 1 to the input tensor.
We defined it using Symbolic Tensors, but it will work on arbitrary tensors now!
Using Pytorch Symbolic, you can perform much more complicated operations.

## Examples step by step

### Model for RGB images

1. Get your symbolic inputs. Specifying batch size is optional. There are a few ways to do it:
	* `inputs = Input(shape=(C, H, W))`
	* `inputs = Input(shape=(C, H, W), batch_size=B)`
	* `inputs = Input(batch_shape=(B, C, H, W))`
	* `inputs = Input(shape=(C, None, None))` for dynamic height and width, like in Keras
2. Register new modules in the graph. There are a few ways:
	* `outputs = inputs(layer)`
	* `outputs = layer(inputs)`
	* `outputs = add_to_graph(layer, inputs)`
3. Use standard operations on Symbolic Tensors: `+, -, *, **, /, //, %` and `abs`:
	* For example, `x = 2 + inputs` or `x = inputs % y` will work as expected
4. Similarly, use all the **non in-place** methods of `torch.Tensor`, e.g. `inputs.reshape`
5. To concatenate (similarly for stacking) Symbolic Tensors:
	* use `useful_layers.ConcatOpLayer(dim=1)(x, y)`
	* add custom function to the model:  `add_to_graph(torch.concat, (x, y), dim=1)`
6. When working with Symbolic Tensors, use `.shape` property or one of the shortcuts:
	* `.C` and `.channels` equals `.shape[1]` for RGB data
	* `.H` equals `.shape[2]` for RGB data
	* `.W` equals `.shape[3]` for RGB data
	* `.HW` is (height, width) tuple for RGB data
7. Finally, create the model: `model = SymbolicModel(inputs, outputs)`
8. Use `model` as a normal PyTorch `nn.Module`. It's 100% compatible.
   When using the model,
   all the operations performed on Symbolic Data will be replayed on the real data.

### Dynamic dimensions

If a dimension of your data varies between batches, e.g. image size or sequence length,
mark it with `None`, like in Keras. It will be displayed as `None` in `.shape` and
in the model summary, and the model will accept real data of any size there:

```python
import torch
from torch import nn
from pytorch_symbolic import Input, SymbolicModel

inputs = Input(shape=(3, None, None))  # dynamic height and width
assert inputs.shape == (1, 3, None, None)

x = nn.Conv2d(in_channels=inputs.C, out_channels=16, kernel_size=3, padding=1)(inputs)
model = SymbolicModel(inputs, x)

outputs = model(torch.rand(2, 3, 32, 48))
assert outputs.shape == (2, 16, 32, 48)
```

Under the hood, dynamic dimensions are traced with a placeholder size of 16.
If the traced layers require larger inputs, increase it with
`Input(shape=(3, None, None), dynamic_size_hint=64)`.
Note that shapes of intermediate Symbolic Tensors are still reported with
the placeholder size - only the input nodes display `None`.

### Sequential topology example

In PyTorch, there's `torch.nn.Sequential` that allows creating simple sequential models.

In Pytorch Symbolic, you can create them too:

```python
from torch import nn
from pytorch_symbolic import Input, SymbolicModel

x = inputs = Input((3, 128, 128))

x = nn.Conv2d(in_channels=x.channels, out_channels=16, kernel_size=3)(x)
x = nn.MaxPool2d(kernel_size=2)(x)(nn.ReLU())

x = nn.Conv2d(in_channels=x.channels, out_channels=32, kernel_size=3)(x)
x = nn.MaxPool2d(kernel_size=2)(x)(nn.ReLU())

x = nn.Conv2d(in_channels=x.channels, out_channels=64, kernel_size=3)(x)
x = nn.MaxPool2d(kernel_size=2)(x)(nn.ReLU())

x = nn.Conv2d(in_channels=x.channels, out_channels=64, kernel_size=3)(x)
x = nn.MaxPool2d(kernel_size=2)(x)(nn.ReLU())(nn.Flatten())

outputs = nn.Linear(in_features=x.features, out_features=10)(x)
model = SymbolicModel(inputs=inputs, outputs=outputs)
model.summary()
```

```stdout
_____________________________________________________________
      Layer         Output shape           Params   Parent   
=============================================================
1     Input_1       (None, 3, 128, 128)    0                 
2     Conv2d_1      (None, 16, 126, 126)   448      1        
3     MaxPool2d_1   (None, 16, 63, 63)     0        2        
4     ReLU_1        (None, 16, 63, 63)     0        3        
5     Conv2d_2      (None, 32, 61, 61)     4640     4        
6     MaxPool2d_2   (None, 32, 30, 30)     0        5        
7     ReLU_2        (None, 32, 30, 30)     0        6        
8     Conv2d_3      (None, 64, 28, 28)     18496    7        
9     MaxPool2d_3   (None, 64, 14, 14)     0        8        
10    ReLU_3        (None, 64, 14, 14)     0        9        
11    Conv2d_4      (None, 64, 12, 12)     36928    10       
12    MaxPool2d_4   (None, 64, 6, 6)       0        11       
13    ReLU_4        (None, 64, 6, 6)       0        12       
14    Flatten_1     (None, 2304)           0        13       
15*   Linear_1      (None, 10)             23050    14       
=============================================================
Total params: 83562
Trainable params: 83562
Non-trainable params: 0
_____________________________________________________________
```

### Multiple inputs example

There's nothing stopping you from using multiple input and output nodes.

Just create multiple `SymbolicTensor` objects:

```python
from torch import nn
from pytorch_symbolic import Input, SymbolicModel

task1_input = x = Input(shape=(3, 32, 32))
task2_input = y = Input(shape=(64,))
```

Define the operations on them:

```py
x = x(nn.Conv2d(x.channels, 16, 3))
x = x(nn.MaxPool2d(3))(nn.ReLU())(nn.Flatten())
head1_out = x(nn.Linear(x.features, 200))

y = y(nn.Linear(y.features, 512))(nn.ReLU())
y = y(nn.Linear(y.features, 512))(nn.ReLU())
head2_out = y(nn.Linear(y.features, 200))

x = head1_out + head2_out  # elementwise sum
x = x(nn.Linear(x.features, 400))(nn.ReLU())
task1_out = x(nn.Linear(x.features, 10))
task2_out = x(nn.Linear(x.features, 1))
```

And create the model, passing tuples or lists as inputs and outputs:

```py
model = SymbolicModel([task1_input, task2_input], [task1_out, task2_out])
model.summary()
```

```stdout
____________________________________________________________
      Layer          Output shape         Params   Parent   
============================================================
1     Input_1        (None, 3, 32, 32)    0                 
2     Input_2        (None, 64)           0                 
3     Conv2d_1       (None, 16, 30, 30)   448      1        
4     MaxPool2d_1    (None, 16, 10, 10)   0        3        
5     ReLU_1         (None, 16, 10, 10)   0        4        
6     Flatten_1      (None, 1600)         0        5        
7     Linear_1       (None, 200)          320200   6        
8     Linear_2       (None, 512)          33280    2        
9     ReLU_2         (None, 512)          0        8        
10    Linear_3       (None, 512)          262656   9        
11    ReLU_3         (None, 512)          0        10       
12    Linear_4       (None, 200)          102600   11       
13    AddOpLayer_1   (None, 200)          0        7,12     
14    Linear_5       (None, 400)          80400    13       
15    ReLU_4         (None, 400)          0        14       
16*   Linear_6       (None, 10)           4010     15       
17*   Linear_7       (None, 1)            401      15       
============================================================
Total params: 803995
Trainable params: 803995
Non-trainable params: 0
____________________________________________________________
```

You can use this model in a following way:

```py
import torch

data1 = torch.rand(16, 3, 32, 32)
data2 = torch.rand(16, 64)

outs1, outs2 = model(data1, data2)
```

## Special cases

### Use custom functions on Symbolic Tensors

If you want to use a function from `torch.nn.functional` or basically
any custom function in your model, you have a few options.
The recommended way is to use it in a `torch.nn.Module`.

As an example, let us say this is the function you want to use:

```python
from torch import nn


def custom_func(*args):
    print("Arguments: ", *args)
    return args
```

This function just prints whatever is passed to it and returns it.

An equivalent `torch.nn.Module` can use this function directly:

```py
class CustomModule(nn.Module):
    def forward(self, *args):
        return custom_func(*args)
```

We already wrapped a few useful functions for you, e.g. `torch.concat` and `torch.stack`.

They are available in `pytorch_symbolic.useful_layers`.

#### Alternative for custom functions

If you really hate classes or are in a hurry, we got you covered.

You can add almost any function to your Symbolic Model using `add_to_graph`:

```python
import torch
from pytorch_symbolic import Input
from pytorch_symbolic.functions_utility import add_to_graph

x1 = Input(shape=(3, 3))
x2 = Input(shape=(5, 3))
x = add_to_graph(torch.concat, (x1, x2), dim=1)
x.shape  # (1, 8, 3)
```

Attempting to use a function without `add_to_graph`, e.g.  `x = torch.abs(x)` will most likely fail:

```
TypeError: abs(): argument 'input' (position 1) must be Tensor, not Input
```

So use `add_to_graph` instead, like `add_to_graph(torch.abs, x)`.

> If your `torch.nn.Module` requires named arguments, you can use `add_to_graph` to register it.

### Modules with multiple inputs

The best way to use modules with multiple inputs is to use `layer(*args)` notation.

Just pass multiple arguments to the layer:

```python
from pytorch_symbolic import Input, useful_layers

x1 = Input(shape=(1, 2, 3))
x2 = Input(shape=(5, 2, 3))
x = useful_layers.ConcatLayer(dim=1)(x1, x2)
x.shape  # (1, 6, 2, 3)
```

Alternatively, using the other notation, do it like this `arg0(layer, *other_args)`:

```py
x = x1(useful_layers.ConcatLayer(dim=1), x2)
x.shape  # (1, 6, 2, 3)
```
