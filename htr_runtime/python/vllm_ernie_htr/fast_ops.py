# SPDX-License-Identifier: Apache-2.0
"""Loader for native gfx1151 Wave32 HIP fast ops (ernie_fast_ops.so)."""

from __future__ import annotations

import os
import sys

_USE_FAST_OPS = os.environ.get("ERNIE_FAST_OPS", "1") == "1"
_USE_DSA_ATTN = os.environ.get("ERNIE_DSA_ATTN_HIP", "1") == "1"
_USE_HCA_ATTN = os.environ.get("ERNIE_HCA_ATTN_HIP", "1") == "1"
_USE_K5_SP = os.environ.get("ERNIE_K5_SP_EXT", "1") == "1"
_EXT = None
_TRIED = False
_DSA_EXT = None
_DSA_TRIED = False
_HCA_EXT = None
_HCA_TRIED = False
_K5SP_PREF_EXT = None
_K5SP_PREF_TRIED = False


def _add_search_paths(so_name: str) -> None:
    so_dir = os.environ.get("ERNIE_HTR_MOE_SO_DIR")
    if so_dir and so_dir not in sys.path:
        sys.path.insert(0, so_dir)
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    cands = [
        pkg_dir,
        os.path.join(os.path.dirname(pkg_dir), "build"),
        os.path.join(os.path.dirname(os.path.dirname(pkg_dir)), "build"),
    ]
    native_dir = os.environ.get("HTR_NATIVE_DIR")
    if native_dir:
        cands.append(native_dir)
    for cand in cands:
        if os.path.isfile(os.path.join(cand, so_name)) and cand not in sys.path:
            sys.path.insert(0, cand)


def get_fast_ops():
    global _EXT, _TRIED
    if not _USE_FAST_OPS:
        return None
    if not _TRIED:
        _TRIED = True
        _add_search_paths("ernie_fast_ops.so")
        try:
            import ernie_fast_ops
            _EXT = ernie_fast_ops
            print(f"ERNIE_FAST_OPS_ACTIVE {ernie_fast_ops.__file__}", flush=True)
        except Exception as exc:
            print(f"ERNIE_FAST_OPS_FAILED {exc!r}", flush=True)
            _EXT = None
    return _EXT


def get_dsa_attn():
    global _DSA_EXT, _DSA_TRIED
    if not _USE_DSA_ATTN:
        return None
    if not _DSA_TRIED:
        _DSA_TRIED = True
        _add_search_paths("ernie_dsa_attn.so")
        try:
            import ernie_dsa_attn
            _DSA_EXT = ernie_dsa_attn
            print(f"ERNIE_DSA_ATTN_ACTIVE {ernie_dsa_attn.__file__}", flush=True)
        except Exception as exc:
            print(f"ERNIE_DSA_ATTN_FAILED {exc!r}", flush=True)
            _DSA_EXT = None
    return _DSA_EXT


_P7_ATTN = None
_P7_ATTN_TRIED = False


def get_p7_attn():
    """Opt-in (ERNIE_P7_ATTN=1) HCA / DSA sparse-prefill kernels from ernie_p7_attn.so; bit-identical to the
    ernie_hca_attn T > 16 path and the ernie_dsa_attn fp8_ds_mla single-pass path. None when off or unavailable."""
    global _P7_ATTN, _P7_ATTN_TRIED
    if not _P7_ATTN_TRIED:
        _P7_ATTN_TRIED = True
        if os.environ.get("ERNIE_P7_ATTN", "0") == "1":
            _add_search_paths("ernie_p7_attn.so")
            try:
                import ernie_p7_attn
                _P7_ATTN = ernie_p7_attn
                print(f"ERNIE_P7_ATTN_ACTIVE {ernie_p7_attn.__file__}", flush=True)
            except Exception as exc:
                print(f"ERNIE_P7_ATTN_FAILED {exc!r}", flush=True)
    return _P7_ATTN


def get_hca_attn():
    global _HCA_EXT, _HCA_TRIED
    if not _USE_HCA_ATTN:
        return None
    if not _HCA_TRIED:
        _HCA_TRIED = True
        _add_search_paths("ernie_hca_attn.so")
        try:
            import ernie_hca_attn
            _HCA_EXT = ernie_hca_attn
            print(f"ERNIE_HCA_ATTN_ACTIVE {ernie_hca_attn.__file__}", flush=True)
        except Exception as exc:
            print(f"ERNIE_HCA_ATTN_FAILED {exc!r}", flush=True)
            _HCA_EXT = None
    return _HCA_EXT


def get_k5_sp_prefill():
    global _K5SP_PREF_EXT, _K5SP_PREF_TRIED
    if not _USE_K5_SP:
        return None
    if not _K5SP_PREF_TRIED:
        _K5SP_PREF_TRIED = True
        _add_search_paths("ernie_k5_sp_prefill.so")
        try:
            import ernie_k5_sp_prefill
            _K5SP_PREF_EXT = ernie_k5_sp_prefill
            print(f"ERNIE_K5_SP_PREFILL_ACTIVE {ernie_k5_sp_prefill.__file__}", flush=True)
        except Exception as exc:
            print(f"ERNIE_K5_SP_PREFILL_FAILED {exc!r}", flush=True)
            _K5SP_PREF_EXT = None
    return _K5SP_PREF_EXT


# Single shared Split-KV scratch (HCA decode + HCA short prefill + DSA decode run sequentially).
# Pre-sized to cap>=32 (32 MiB) so HIP graph capture sizes [1..32] NEVER reallocate; any grown
# predecessor is retained in _PINNED_SCRATCH so previously captured graphs stay memory-safe.
_SPLIT_KV_SCRATCH: dict[str, tuple] = {}
_PINNED_SCRATCH: list = []


def get_split_kv_scratch(dev, T: int, H: int = 64, S: int = 8, D: int = 512):
    import torch
    key = f"{dev}_{H}_{S}_{D}"
    buf = _SPLIT_KV_SCRATCH.get(key)
    cap = max(32, T)
    if buf is None or buf[0].shape[0] < cap:
        if buf is not None:
            _PINNED_SCRATCH.append(buf)
        po = torch.empty((cap, H, S, D), dtype=torch.float32, device=dev)
        pm = torch.empty((cap, H, S), dtype=torch.float32, device=dev)
        pl = torch.empty((cap, H, S), dtype=torch.float32, device=dev)
        buf = _SPLIT_KV_SCRATCH[key] = (po, pm, pl)
    return buf[0][:T], buf[1][:T], buf[2][:T]
