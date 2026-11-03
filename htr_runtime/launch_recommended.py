#!/usr/bin/env python3
"""Recommended Qwen Flash HTR sampling over the unchanged v0.2.0 runtime.

Accepts every flag of launch_htr.py, including --print-command, --cpu-check
and --build-seed. Checkpoint and installed-runtime integrity checks still run.
This file can be mounted separately into the immutable released image.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

LAUNCH_SHA256 = '65aa8d9358d81dcf28eba63c378db3d95983d236c25fd07b49d1933168a0695e'
SAMPLING = {
    'temperature': 1.0, 'top_p': 0.95, 'top_k': 20,
    'min_p': 0.0, 'presence_penalty': 0.0, 'repetition_penalty': 1.0,
}


def load_release(root):
    path = root / 'launch_htr.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != LAUNCH_SHA256:
        raise ValueError('v0.2.0 launcher identity mismatch')
    spec = importlib.util.spec_from_file_location('htr_release', path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load HTR launcher: {path}')
    release = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(release)
    return release


def apply_profile(release):
    original_command = release.command
    original_verify = release.verify_runtime

    def command(args):
        argv = original_command(args)
        if ('--override-generation-config' in argv or
                argv[argv.index('--generation-config') + 1] != 'auto'):
            raise ValueError('Unexpected released generation-config arguments')
        return argv + ['--override-generation-config',
                       json.dumps(SAMPLING, sort_keys=True)]

    def verify_runtime(args):
        # Validate the original checkpoint profile first, then report the
        # effective server defaults rather than the frozen checkpoint values.
        count, report, _ = original_verify(args)
        return count, report, dict(SAMPLING)

    release.command = command
    release.verify_runtime = verify_runtime


if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    if not (root / 'launch_htr.py').is_file():
        root = Path('/opt/vllm-htr')
    release = load_release(root)
    apply_profile(release)
    release.main()
