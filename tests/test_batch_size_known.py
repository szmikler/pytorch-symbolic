#  Copyright (c) 2022 Szymon Mikler

from pytorch_symbolic import Input


def test_batch_size_known_order_independent():
    a = Input(batch_shape=(4, 8))
    b = Input(shape=(8,))
    assert a.batch_size_known
    assert not b.batch_size_known

    assert not (a + b).batch_size_known
    assert not (b + a).batch_size_known


def test_batch_size_known_propagation():
    a = Input(batch_shape=(4, 8))
    b = Input(batch_shape=(4, 8))
    assert (a + b).batch_size_known

    c = Input(shape=(8,))
    d = Input(shape=(8,))
    assert not (c + d).batch_size_known

    # A single unknown batch size anywhere poisons all downstream nodes
    x = (a + b) + c
    assert not x.batch_size_known
    assert not (x + a).batch_size_known
