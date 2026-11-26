import importlib.util
import os

_ENABLED = os.environ.get("HTR_DECODE_META", "0") == "1"
_SO_PATH = os.environ.get(
    "HTR_DECODE_META_SO",
    "/opt/vllm-htr/native-bin/htr_decode_meta_v27.so",
)
_MOD = None


def get_decode_meta_mod():
    global _MOD
    if not _ENABLED:
        return None
    if _MOD is None:
        spec = importlib.util.spec_from_file_location("htr_decode_meta", _SO_PATH)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"HTR_DECODE_META: cannot load {_SO_PATH}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _MOD = mod
    return _MOD
