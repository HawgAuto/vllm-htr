#!/usr/bin/env python3
"""Build the opt-in cache and independent native CPU oracle; no GPU required."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parent


def main():
    qualified = json.loads((ROOT / "QUALIFICATION.json").read_text())
    for name, expected in qualified["native_source_sha256"].items():
        actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Qualified source identity changed: {name}")
    compiler = shlex.split(os.environ.get("CC", "cc"))
    if not compiler:
        raise ValueError("CC must name a C compiler")
    common = ["-O3", "-std=c11", "-fPIC", "-shared", "-pthread"]
    for source, output in [("vendor/ple_gather_v2.c", "vendor/native.so"),
                           ("ple_row_cache.c", "libple_row_cache.so")]:
        command = compiler + common + [str(ROOT / source), "-o", str(ROOT / output)]
        print(shlex.join(command), flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
