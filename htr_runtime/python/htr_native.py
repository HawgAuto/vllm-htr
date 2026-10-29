"""Load packaged native extensions without invoking a build tool."""
import importlib.util
import os
from pathlib import Path


def load_native(name):
    path = Path(os.environ['HTR_NATIVE_DIR']) / (name + '.so')
    if not path.is_file():
        raise FileNotFoundError(f'Missing packaged native extension: {path}')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
