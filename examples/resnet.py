#  Copyright (c) 2022 Szymon Mikler

"""
This is a flexible implementation of ResNet architecture.
It allows for creation of standard ResNet v2 or Wide ResNet variants.
"""

from torch import nn

from pytorch_symbolic import Input, SymbolicModel

from .common import classifier


def shortcut_func(x, channels, stride):
    if x.channels != channels or stride != 1:
        return x(nn.Conv2d(x.channels, channels, kernel_size=1, bias=False, stride=stride))
    else:
        return x


def ToyResNet(input_shape, n_classes):
    """A basic example of ResNet with low degree of customizability."""

    inputs = Input(input_shape)
    flow = inputs(nn.Conv2d(inputs.channels, 16, 3, 1, 1))

    for group_size, width, stride in zip((2, 2, 2), (16, 32, 64), (1, 2, 2), strict=True):
        for _ in range(group_size):
            shortcut = shortcut_func(flow, width, stride)
            flow = flow(nn.BatchNorm2d(flow.channels))(nn.ReLU())
            flow = flow(nn.Conv2d(flow.channels, width, 3, stride, 1))
            flow = flow(nn.BatchNorm2d(flow.channels))(nn.ReLU())
            flow = flow(nn.Conv2d(flow.channels, width, 3, 1, 1))

            flow = flow + shortcut
            stride = 1
    flow = flow(nn.BatchNorm2d(flow.channels))(nn.ReLU())
    outs = classifier(flow, n_classes, pooling="avgpool")
    model = SymbolicModel(inputs=inputs, outputs=outs)
    return model


relu = nn.ReLU()


def ResNet(
    input_shape,
    n_classes,
    version=None,
    bootleneck=False,
    strides=(1, 2, 2),
    group_sizes=(2, 2, 2),
    channels=(16, 32, 64),
    activation=relu,
    final_pooling="avgpool",
    dropout=0,
    bn_ends_block=False,
    **kwargs,
):
    if version:
        if version == 20:
            group_sizes = (3, 3, 3)
        elif version == 32:
            group_sizes = (5, 5, 5)
        elif version == 44:
            group_sizes = (7, 7, 7)
        elif version == 56:
            group_sizes = (9, 9, 9)
        elif version == 110:
            group_sizes = (18, 18, 18)
        elif version == 164:
            bootleneck = True
            channels = (64, 128, 256)
            group_sizes = (18, 18, 18)
        elif isinstance(version, tuple) and version[0] == "WRN":
            _, N, K = version
            assert (N - 4) % 6 == 0, "N-4 has to be divisible by 6"
            lpb = (N - 4) // 6  # layers per block
            group_sizes = (lpb, lpb, lpb)
            channels = tuple(c * K for c in channels)
        else:
            raise NotImplementedError(f"Unkown version={version}!")
    if kwargs:
        print(f"ResNet: unknown parameters: {kwargs.keys()}")

    def simple_block(flow, channels, stride):
        if preactivate_block:
            flow = flow(nn.BatchNorm2d(flow.channels))(activation)

        flow = flow(nn.Conv2d(flow.channels, channels, 3, stride, 1))
        flow = flow(nn.BatchNorm2d(flow.channels))(activation)

        if dropout:
            flow = flow(nn.Dropout(p=dropout))
        flow = flow(nn.Conv2d(flow.channels, channels, 3, 1, 1))

        if bn_ends_block:
            flow = flow(nn.BatchNorm2d(flow.channels))(activation)
        return flow

    def bootleneck_block(flow, channels, stride):
        if preactivate_block:
            flow = flow(nn.BatchNorm2d(flow.channels))(activation)

        flow = flow(nn.Conv2d(flow.channels, channels // 4, 1))
        flow = flow(nn.BatchNorm2d(flow.channels))(activation)

        flow = flow(nn.Conv2d(flow.channels, channels // 4, 3, stride=stride, padding=1))
        flow = flow(nn.BatchNorm2d(flow.channels))(activation)

        flow = flow(nn.Conv2d(flow.channels, channels, 1))
        if bn_ends_block:
            flow = flow(nn.BatchNorm2d(flow.channels))(activation)
        return flow

    if bootleneck:
        block = bootleneck_block
    else:
        block = simple_block

    inputs = Input(input_shape)

    # BUILDING HEAD OF THE NETWORK
    flow = inputs(nn.Conv2d(inputs.channels, 16, 3, 1, 1))

    # BUILD THE RESIDUAL BLOCKS
    for group_size, width, stride in zip(group_sizes, channels, strides, strict=True):
        flow = flow(nn.BatchNorm2d(flow.channels))(activation)
        preactivate_block = False

        for _ in range(group_size):
            residual = block(flow, width, stride)
            shortcut = shortcut_func(flow, width, stride)
            flow = residual + shortcut
            preactivate_block = True
            stride = 1

    # BUILDING THE CLASSIFIER
    flow = flow(nn.BatchNorm2d(flow.channels))(activation)
    outs = classifier(flow, n_classes, pooling=final_pooling)
    model = SymbolicModel(inputs=inputs, outputs=outs)
    return model
