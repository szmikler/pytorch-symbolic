#  Copyright (c) 2022 Szymon Mikler

"""
This is a flexible implementation of VGG architecture.
"""

from torch import nn

from pytorch_symbolic import Input, SymbolicModel

from .common import classifier

relu = nn.ReLU()


def VGG(
    input_shape,
    n_classes,
    version=None,
    group_sizes=(1, 1, 2, 2, 2),
    channels=(64, 128, 256, 512, 512),
    pools=(2, 2, 2, 2, 2),
    activation=relu,
    final_pooling="avgpool",
    **kwargs,
):
    if kwargs:
        print(f"VGG: unknown parameters: {kwargs.keys()}")
    if version:
        if version == 11:
            group_sizes = (1, 1, 2, 2, 2)
        elif version == 13:
            group_sizes = (2, 2, 2, 2, 2)
        elif version == 16:
            group_sizes = (2, 2, 3, 3, 3)
        elif version == 19:
            group_sizes = (2, 2, 4, 4, 4)
        else:
            raise NotImplementedError(f"Unkown version={version}!")

    inputs = Input(shape=input_shape)
    flow = inputs

    iteration = 0
    for group_size, width, pool in zip(group_sizes, channels, pools, strict=True):
        if iteration == 0:
            iteration = 1
        else:
            flow = flow(nn.MaxPool2d(pool))

        for _ in range(group_size):
            flow = flow(nn.Conv2d(flow.channels, width, 3, 1, 1, bias=False))
            flow = flow(nn.BatchNorm2d(flow.channels))(activation)

    outs = classifier(flow, n_classes, final_pooling)
    model = SymbolicModel(inputs=inputs, outputs=outs)
    return model
