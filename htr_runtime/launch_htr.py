#!/usr/bin/env python3
"""Launch vLLM HTR for Qwen3.8-Flash-Next-HTR on gfx1151.

Serving levers (all optional; the defaults are the release profile):
  --mtp {0,3}                     MTP speculative decoding off / 3 draft tokens (default 3)
  --max-num-seqs N                concurrent sequences (default 1; MTP3 up to 4, MTP0 up to 32)
  --context N                     max model length (default 262,144 = native; up to 1,048,576 with YaRN)
  --yarn-factor F                 static YaRN factor (default for --context > 262,144: 2 or 4)
  --kv-cache-gib G                BF16 KV pool (default 8)
  --max-num-batched-tokens N      prefill chunk / scheduler token budget (default 12,288)
  --cudagraph-sizes a,b,...       decode graph capture sizes (default chosen from --max-num-seqs and --mtp)

Quick load: every distinct set of compile-relevant levers gets its own compile/autotune cache under
/workspace/cache/<profile>. The first start compiles and tunes; later starts with the same levers load that cache.
--build-seed does the first start ahead of time: it starts the server, sends warm-up requests across prefill
lengths, stops the server and writes <cache>/SEED.json. Keep /workspace on a volume to reuse the seed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
WORK = Path('/workspace')
NATIVE_CONTEXT = 262144
MAX_CONTEXT = 1048576
DEFAULT_KV_GIB = 8.0
DEFAULT_BATCHED_TOKENS = 12288
LADDER = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 128]
# Capture sizes of the measured geometries: (max_num_seqs, mtp) -> sizes.
QUALIFIED_SIZES = {
    (1, 3): [1, 2, 3, 4, 5],
    (4, 3): [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20],
    (4, 0): [1, 2, 3, 4],
    (32, 0): [1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32],
}
CACHE_KEYS = ['XDG_CACHE_HOME', 'VLLM_CACHE_ROOT', 'TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR', 'AITER_JIT_DIR',
              'TVM_FFI_CACHE_DIR']
# Base-layer settings of the retired QSA union path; nothing in this runtime reads them.
RETIRED_BASE_ENV = ['DIRECT_QSA_UNION_AUDIT', 'DIRECT_QSA_UNION_HSACO', 'DIRECT_QSA_UNION_MIN_CAP',
                    'DIRECT_QSA_NATIVE_FN', 'DIRECT_QSA_NATIVE_HSACO', 'DIRECT_QSA_NATIVE_MIN_TOKENS']


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--host', default='0.0.0.0')
    p.add_argument('--port', type=int, default=8000)
    p.add_argument('--served-model-name', default='qwen38-htr')
    p.add_argument('--mtp', type=int, choices=[0, 3], default=3)
    p.add_argument('--max-num-seqs', type=int, default=1)
    p.add_argument('--context', type=int, default=NATIVE_CONTEXT)
    p.add_argument('--yarn-factor', type=float, default=None)
    p.add_argument('--kv-cache-gib', type=float, default=DEFAULT_KV_GIB)
    p.add_argument('--max-num-batched-tokens', type=int, default=DEFAULT_BATCHED_TOKENS)
    p.add_argument('--cudagraph-sizes', default=None, help='comma-separated decode graph capture sizes')
    p.add_argument('--no-prefix-caching', action='store_true')
    p.add_argument('--build-seed', action='store_true',
                   help='start, warm up, stop, and write SEED.json for this lever set (quick-load seed)')
    p.add_argument('--cpu-check', action='store_true')
    p.add_argument('--print-command', action='store_true')
    return p


def default_sizes(seqs, mtp):
    if (seqs, mtp) in QUALIFIED_SIZES:
        return QUALIFIED_SIZES[(seqs, mtp)]
    top = seqs * (1 + mtp)
    return sorted({s for s in LADDER if s <= top} | {top})


def check(ap, args):
    if not 1 <= args.port <= 65535:
        ap.error('port out of range')
    limit = 4 if args.mtp else 32
    if not 1 <= args.max_num_seqs <= limit:
        ap.error(f'--max-num-seqs must be 1..{limit} with --mtp {args.mtp}')
    if not 1 <= args.context <= MAX_CONTEXT:
        ap.error(f'--context must be 1..{MAX_CONTEXT}')
    if args.context > NATIVE_CONTEXT and args.yarn_factor is None:
        args.yarn_factor = 2.0 if args.context <= 2 * NATIVE_CONTEXT else 4.0
    if args.yarn_factor is not None:
        if not 1.0 < args.yarn_factor <= 4.0:
            ap.error('--yarn-factor must be in (1, 4]')
        if args.context > args.yarn_factor * NATIVE_CONTEXT:
            ap.error('--context exceeds yarn-factor x 262144')
    if not 0 < args.kv_cache_gib <= 96:
        ap.error('--kv-cache-gib must be in (0, 96]')
    if args.max_num_batched_tokens < 1:
        ap.error('--max-num-batched-tokens must be positive')
    if args.cudagraph_sizes:
        try:
            sizes = sorted({int(x) for x in args.cudagraph_sizes.split(',')})
        except ValueError:
            ap.error('--cudagraph-sizes must be comma-separated integers')
        if sizes[0] < 1:
            ap.error('--cudagraph-sizes must be positive')
        args.sizes = sizes
    else:
        args.sizes = default_sizes(args.max_num_seqs, args.mtp)


def profile(args):
    """Compile-relevant levers -> cache profile name. KV pool size and host/port do not change compiled code."""
    levers = {'mtp': args.mtp, 'max_num_seqs': args.max_num_seqs, 'context': args.context,
              'yarn_factor': args.yarn_factor, 'max_num_batched_tokens': args.max_num_batched_tokens,
              'cudagraph_sizes': args.sizes, 'prefix_caching': not args.no_prefix_caching}
    runtime = hashlib.sha256((ROOT / 'runtime-env.json').read_bytes() + (ROOT / 'server-argv.json').read_bytes())
    # Old images/seeds must not bypass a changed numerical or launcher policy.
    members = [ROOT / 'VERSION', ROOT / 'NATIVE-BUILD.json', ROOT / 'launch_htr.py',
               *sorted((ROOT / 'python').rglob('*.py')),
               *sorted((ROOT / 'python').rglob('*.pth'))]
    for member in members:
        if member.is_file():
            runtime.update(str(member.relative_to(ROOT)).encode() + b'\0' + member.read_bytes())
    key = hashlib.sha256((json.dumps(levers, sort_keys=True) + runtime.hexdigest()).encode()).hexdigest()[:12]
    name = f'htr-m{args.mtp}-s{args.max_num_seqs}-c{args.context}'
    if args.yarn_factor:
        name += f'-y{args.yarn_factor:g}'
    return name + '-' + key, levers


def yarn_overrides(factor):
    """Qwen's static YaRN rope override (text_config.rope_parameters)."""
    return {'text_config': {'rope_parameters': {
        'mrope_interleaved': True, 'mrope_section': [11, 11, 10], 'rope_type': 'yarn',
        'rope_theta': 10000000, 'partial_rotary_factor': 0.25, 'factor': float(factor),
        'original_max_position_embeddings': NATIVE_CONTEXT}}}


def command(args):
    argv = json.loads((ROOT / 'server-argv.json').read_text())
    argv[0] = sys.executable

    def put(flag, value):
        argv[argv.index(flag) + 1] = value

    put('--host', args.host)
    put('--port', str(args.port))
    put('--served-model-name', args.served_model_name)
    put('--max-num-seqs', str(args.max_num_seqs))
    put('--max-model-len', str(args.context))
    put('--kv-cache-memory-bytes', str(int(round(args.kv_cache_gib * (1 << 30)))))
    put('--max-num-batched-tokens', str(args.max_num_batched_tokens))
    cc = json.loads(argv[argv.index('--compilation-config') + 1])
    cc['cudagraph_capture_sizes'] = args.sizes
    cc['max_cudagraph_capture_size'] = max(args.sizes)
    put('--compilation-config', json.dumps(cc))
    if args.mtp == 0:
        i = argv.index('--speculative-config')
        del argv[i:i + 2]
    if args.no_prefix_caching:
        argv[argv.index('--enable-prefix-caching')] = '--no-enable-prefix-caching'
    if args.yarn_factor:
        argv += ['--hf-overrides', json.dumps(yarn_overrides(args.yarn_factor))]
    argv += ['--default-chat-template-kwargs', '{"enable_thinking":true}']
    return argv


def tree(root):
    files = {}
    for p in sorted(root.rglob('*')):
        if p.is_file() and not p.is_symlink() and p.name != 'SEED.json':
            with p.open('rb') as f:
                files[str(p.relative_to(root))] = hashlib.file_digest(f, 'sha256').hexdigest()
    return files


def configure(args, name):
    """Base runtime + HTR runtime environment, with every cache directory under this profile's cache root."""
    sys.path.insert(0, '/opt/haloloom/qwen38')
    import launch as base
    base.configure(argparse.Namespace(variant='bf16', mtp=args.mtp, yarn_factor=args.yarn_factor,
                                      context=args.context))
    os.environ.update(json.loads((ROOT / 'runtime-env.json').read_text()))
    os.environ['HTR_GDN_MTP'] = str(args.mtp)
    for key in RETIRED_BASE_ENV:
        os.environ.pop(key, None)
    cache_root = WORK / 'cache' / name
    for key in CACHE_KEYS:
        value = os.environ.get(key, '')
        if value.startswith('/workspace/cache/'):
            rest = value[len('/workspace/cache/'):].split('/', 1)
            os.environ[key] = str(cache_root / rest[1]) if len(rest) == 2 else str(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    return cache_root


def verify_runtime(args):
    manifest = json.loads((ROOT / 'installed-files.json').read_text())
    for relative, expected in manifest.items():
        with (ROOT / relative).open('rb') as f:
            if hashlib.file_digest(f, 'sha256').hexdigest() != expected:
                raise ValueError(f'HTR runtime integrity mismatch: {relative}')
    generation = json.loads(Path('/model/generation_config.json').read_text())
    expected = json.loads((ROOT / 'sampling.json').read_text())
    for k, v in expected.items():
        if generation.get(k) != v:
            raise ValueError(f'model sampling profile mismatch: {k}')
    from validate import verify
    return len(manifest), verify(Path('/model'), 'bf16', args.mtp), expected


def warmup_prompts(args):
    """Token-id prompts across prefill lengths; a unique first line per prompt avoids prefix-cache reuse."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('/model')
    text = ('The quick brown fox jumps over the lazy dog while the committee reviews the quarterly report, '
            'compares regional sales figures, and drafts a short summary for the board. ')
    base = tok.encode(text * 40, add_special_tokens=False)
    top = min(args.context - 64, 32768)
    lengths = sorted({n for n in (64, 512, 2048, 8192, args.max_num_batched_tokens,
                                  args.max_num_batched_tokens + 4096, 32768) if n <= top})
    prompts = []
    for i, n in enumerate(lengths):
        ids = tok.encode(f'Warm-up {i}:', add_special_tokens=False)
        while len(ids) < n:
            ids += base
        prompts.append(ids[:n])
    return prompts


def post(port, body, timeout=1800):
    req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/completions', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def build_seed(args, argv, cache_root, levers):
    before = tree(cache_root)
    t0 = time.time()
    server = subprocess.Popen(argv)
    rows = []
    try:
        while True:
            if server.poll() is not None:
                raise RuntimeError(f'server exited during seed build: {server.returncode}')
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{args.port}/health', timeout=5) as r:
                    if r.status == 200:
                        break
            except OSError:
                pass
            time.sleep(2)
        ready = time.time() - t0
        for ids in warmup_prompts(args):
            t = time.time()
            post(args.port, {'model': args.served_model_name, 'prompt': ids, 'max_tokens': 16, 'temperature': 0.0})
            rows.append({'prompt_tokens': len(ids), 'seconds': round(time.time() - t, 3)})
            print('HTR_SEED_WARMUP ' + json.dumps(rows[-1]), flush=True)
        if args.max_num_seqs > 1:
            import concurrent.futures as cf
            n = args.max_num_seqs
            with cf.ThreadPoolExecutor(n) as ex:
                list(ex.map(lambda i: post(args.port, {'model': args.served_model_name, 'prompt': f'Count to {i + 5}.',
                                                       'max_tokens': 64, 'temperature': 0.0}), range(n)))
            rows.append({'concurrent': n})
            print('HTR_SEED_WARMUP ' + json.dumps(rows[-1]), flush=True)
    finally:
        server.terminate()
        try:
            server.wait(timeout=180)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
    after = tree(cache_root)
    seed = {'schema': 'vllm-htr.seed.v1', 'profile': cache_root.name, 'levers': levers, 'created_unix': int(time.time()),
            'ready_seconds': round(ready, 1), 'warmup': rows, 'files': len(after),
            'added': len([k for k in after if k not in before]),
            'tree_sha256': hashlib.sha256(json.dumps(after, sort_keys=True).encode()).hexdigest()}
    (cache_root / 'SEED.json').write_text(json.dumps(seed, indent=1) + '\n')
    print('HTR_SEED_BUILT ' + json.dumps({k: v for k, v in seed.items() if k != 'warmup'}), flush=True)


def main():
    ap = parser()
    args = ap.parse_args()
    check(ap, args)
    name, levers = profile(args)
    argv = command(args)
    if args.print_command:
        print(json.dumps({'argv': argv, 'profile': name, 'levers': levers, 'kv_cache_gib': args.kv_cache_gib,
                          'cache_root': str(WORK / 'cache' / name)}, indent=2))
        return
    cache_root = configure(args, name)
    n_files, report, sampling = verify_runtime(args)
    seed_file = cache_root / 'SEED.json'
    seed = json.loads(seed_file.read_text()) if seed_file.is_file() else None
    print(json.dumps({'htr_version': (ROOT / 'VERSION').read_text().strip(),
                      'htr_runtime_files': n_files, 'base_and_model': report, 'sampling': sampling, 'thinking': True,
                      'profile': name, 'levers': levers, 'kv_cache_gib': args.kv_cache_gib,
                      'cache_root': str(cache_root),
                      'seed': {'present': seed is not None, 'tree_sha256': seed and seed['tree_sha256'],
                               'files': seed and seed['files']},
                      'argv': argv}), flush=True)
    if args.cpu_check:
        return
    import torch
    if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).gcnArchName.split(':')[0] != 'gfx1151':
        raise RuntimeError('This release requires a gfx1151 ROCm GPU')
    if args.build_seed:
        build_seed(args, argv, cache_root, levers)
        return
    os.execv(sys.executable, argv)


if __name__ == '__main__':
    main()
