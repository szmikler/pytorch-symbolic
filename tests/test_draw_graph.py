#  Copyright (c) 2022 Szymon Mikler

from torch import nn

from pytorch_symbolic import Input, SymbolicModel, graph_algorithms

try:
    import matplotlib
    import matplotlib.pyplot as plt
    import networkx  # noqa: F401
    import scipy  # noqa: F401

    matplotlib.use("Agg")
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False


def create_model():
    inputs = Input(shape=(8,))
    outputs = nn.Linear(8, 4)(inputs)
    return SymbolicModel(inputs=inputs, outputs=outputs)


def test_draw_graph_returns_figure():
    if not VISUALIZATION_AVAILABLE:
        return
    model = create_model()

    fig = graph_algorithms.draw_graph(model=model)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_draw_graph_show_calls_pyplot_show():
    if not VISUALIZATION_AVAILABLE:
        return
    model = create_model()

    show_calls = []
    original_show = plt.show
    plt.show = lambda *args, **kwds: show_calls.append(args)
    try:
        fig = graph_algorithms.draw_graph(model=model, show=True)
    finally:
        plt.show = original_show

    assert len(show_calls) == 1
    plt.close(fig)


def test_draw_graph_show_false_does_not_call_show():
    if not VISUALIZATION_AVAILABLE:
        return
    model = create_model()

    show_calls = []
    original_show = plt.show
    plt.show = lambda *args, **kwds: show_calls.append(args)
    try:
        fig = graph_algorithms.draw_graph(model=model, show=False)
    finally:
        plt.show = original_show

    assert len(show_calls) == 0
    plt.close(fig)
