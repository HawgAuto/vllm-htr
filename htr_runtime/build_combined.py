"""Reproducible combined HTR image builder; no GPU access or service changes.

Run from a clean committed checkout. Output must be outside the checkout and
must not exist. A pinned dependency snapshot can be supplied for offline builds.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time

BASE = 'ghcr.io/hawgauto/vllm-htr@sha256:dde8c3ea80e5d0a56e9143894a12622d0c236bab1f6d3d8a10e4ab911c836afb'
TRITON_COMMIT = '669b31acc1dd1b3fd93286afd5db67f65d9f7557'
TRITON_TREE = '528405b13ae2d62d658e28fe53f1dfba63b6871a216869d0e5ecd4c211a38948'
REPO = Path(__file__).resolve().parents[1]


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def tree_manifest(root):
    result = {}
    for p in sorted(root.rglob('*')):
        if '__pycache__' in p.parts:
            continue
        if p.is_symlink():
            raise ValueError('symlink in pinned input: ' + str(p))
        if p.is_file():
            result[str(p.relative_to(root))] = sha(p)
    return result


def git(*args):
    return subprocess.check_output(['git', '-C', str(REPO), *args], text=True).strip()


def write(path, obj):
    with path.open('x') as f:
        json.dump(obj, f, indent=2)
        f.write('\n')


def main():
    # Retain the public builder entry point; current source uses the optimized
    # recipe. Historical v0.2.2 builds remain available at their immutable tag.
    from build_optimized import main as build_current
    return build_current()


if __name__ == '__main__':
    main()
