#!/usr/bin/env python3
"""Additive PLE opt-in adapter; the installed release launcher stays unchanged."""
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent


def budget_bytes(gib):
    if not math.isfinite(gib) or not 0 <= gib <= 64:
        raise ValueError('PLE row cache must be a finite size in [0, 64] GiB; 0 disables it')
    n = int(gib * (1 << 30))
    if gib != 0 and n < 65536:
        raise ValueError('nonzero PLE row-cache budget must be at least 64 KiB')
    return n


def add_options(parser):
    parser.add_argument('--ple-row-cache-gib', type=float, default=0,
                        help='Opt-in CPU RAM row cache in GiB (default 0=OFF); budget includes index/LRU metadata. '
                             'Initial qualification sizes: 2, 4, 8. Cache scope: one TP1 engine process.')
    parser.add_argument('--ple-cache-observe', action='store_true',
                        help='Research only: count native gather I/O with a zero-size cache')
    return parser


def configure_cache(args):
    budget = budget_bytes(args.ple_row_cache_gib)
    if not budget and not args.ple_cache_observe:
        return
    so = ROOT / 'libple_row_cache.so'
    if not so.is_file():
        raise FileNotFoundError(so)
    os.environ['DIRECT_PLE_NATIVE_LIB'] = str(so)
    os.environ['HTR_PLE_ROW_CACHE_BYTES'] = str(budget)
    os.environ['HTR_PLE_ROW_CACHE_STATS'] = '/workspace/ple-cache-stats.bin'
    print('HTR_PLE_ROW_CACHE_CONFIG ' + json.dumps({
        'enabled': budget > 0, 'budget_bytes': budget,
        'library': str(so), 'bookkeeping_included': True,
        'policy': 'byte-preserving global LRU; native gather for misses',
        'scope': 'per-engine-process, TP1', 'observer_only': budget == 0}), flush=True)


def main():
    path = Path('/opt/vllm-htr/launch_htr.py')
    spec = importlib.util.spec_from_file_location('htr_release_launch', path)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    parser, check, configure = native.parser, native.check, native.configure
    native.parser = lambda: add_options(parser())

    def checked(ap, args):
        check(ap, args)
        try:
            budget_bytes(args.ple_row_cache_gib)
        except ValueError as exc:
            ap.error(str(exc))

    def configured(args, name):
        result = configure(args, name)
        configure_cache(args)  # AFTER the release installs runtime-env.json
        return result

    native.check, native.configure = checked, configured
    native.main()


if __name__ == '__main__':
    main()
