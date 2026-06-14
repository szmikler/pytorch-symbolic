#  Copyright (c) 2022 Szymon Mikler

import ast
import os

from .symbolic_api_2 import enable_symbolic_api_2_for_new_modules


def read_from_env(name, default):
    if name in os.environ:
        value = ast.literal_eval(os.environ[name])
    else:
        value = default
    return value


# Constants
CODEGEN_BY_DEFAULT = read_from_env("PYTORCH_SYMBOLIC_CODEGEN_BY_DEFAULT", True)
CODEGEN_MIN_LOOP_LENGTH = read_from_env("PYTORCH_SYMBOLIC_CODEGEN_MIN_LOOP_LENGTH", 50)
API_2_ENABLED_BY_DEFAULT = read_from_env("PYTORCH_SYMBOLIC_API_2_ENABLED_BY_DEFAULT", True)

if API_2_ENABLED_BY_DEFAULT:
    enable_symbolic_api_2_for_new_modules()
