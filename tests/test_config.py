#  Copyright (c) 2022 Szymon Mikler

import os

from pytorch_symbolic.config import read_from_env

TEST_ENTRY = "PYTORCH_SYMBOLIC_TEST_ENTRY"


def test_read_from_env_parses_literals():
    try:
        os.environ[TEST_ENTRY] = "False"
        assert read_from_env(TEST_ENTRY, True) is False

        os.environ[TEST_ENTRY] = "True"
        assert read_from_env(TEST_ENTRY, False) is True

        os.environ[TEST_ENTRY] = "50"
        assert read_from_env(TEST_ENTRY, 0) == 50
    finally:
        del os.environ[TEST_ENTRY]


def test_read_from_env_returns_default():
    assert TEST_ENTRY not in os.environ
    assert read_from_env(TEST_ENTRY, 123) == 123


def test_read_from_env_rejects_expressions():
    raised = False
    try:
        os.environ[TEST_ENTRY] = "__import__('os').getcwd()"
        read_from_env(TEST_ENTRY, None)
    except ValueError:
        raised = True
    finally:
        del os.environ[TEST_ENTRY]
    assert raised
