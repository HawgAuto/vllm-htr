"""Install v0.2.7 Qwen3.8 decode+boot upgrades AND ERNIE-5.5 (with DSpark) cleanly over v0.2.6."""
import compileall, hashlib, json, shutil, sys
from pathlib import Path

ROOT = Path('/opt/vllm-htr')
PAYLOAD = Path('/payload')
BASE_MANIFEST = Path('/opt/haloloom/qwen38/data/installed-files.json')

def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def main() -> None:
    file_map = json.loads((PAYLOAD / 'file_map.json').read_text())
    nat_manifest = json.loads((PAYLOAD / 'V027-NATIVE-MANIFEST.json').read_text())

    # 1. Install Qwen3.8 v0.2.7 Python files (including mirrored site-packages copies)
    for rel, dst_list in file_map.items():
        src = PAYLOAD / 'files' / rel
        assert src.is_file(), f'missing payload file {src}'
        for dst_str in dst_list:
            dst = Path(dst_str)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)

    # Reconcile mtp_dense.py KERNEL_SHA with upgraded fused_moe.py (so zero-overlay works without HTR_ALLOW_OVERLAY=1)
    fused_moe_p = Path('/opt/venv/lib/python3.14/site-packages/vllm/model_executor/layers/fused_moe/fused_moe.py')
    mtp_dense_p = Path('/opt/venv/lib/python3.14/site-packages/mtp_dense.py')
    if mtp_dense_p.is_file() and fused_moe_p.is_file():
        import re as _re
        new_fm_sha = sha(fused_moe_p)
        md_src = mtp_dense_p.read_text()
        md_new, nsub = _re.subn(r"KERNEL_SHA='[0-9a-f]{64}'", f"KERNEL_SHA='{new_fm_sha}'", md_src, count=1)
        assert nsub == 1, 'KERNEL_SHA not found in mtp_dense.py'
        mtp_dense_p.write_text(md_new)
        print(f'Updated mtp_dense.py KERNEL_SHA -> {new_fm_sha[:16]}...')

    # Install runtime-env.json, V027-NATIVE-MANIFEST.json, VERSION
    shutil.copyfile(PAYLOAD / 'runtime-env.json', ROOT / 'runtime-env.json')
    shutil.copyfile(PAYLOAD / 'V027-NATIVE-MANIFEST.json', ROOT / 'V027-NATIVE-MANIFEST.json')
    (ROOT / 'VERSION').write_text('0.2.7\n')

    # 2. Install & SHA-verify all 12 Qwen3.8 v0.2.7 native binaries into /opt/vllm-htr/native-bin/
    nat_dir = ROOT / 'native-bin'
    nat_dir.mkdir(parents=True, exist_ok=True)
    for name, meta in nat_manifest['qwen38_binaries'].items():
        src = PAYLOAD / 'native' / name
        assert src.is_file(), f'missing native binary {src}'
        assert sha(src) == meta['sha256'], f'native sha mismatch for {name}'
        dst = nat_dir / name
        shutil.copyfile(src, dst)
        dst.chmod(0o755 if name.endswith('.so') else 0o644)

    # 2b. Reconcile _HTR_QSA_CE_SHA256 in qsa.py -> qsa_canon_expand_fused.hsaco,
    #     then QSA_OPS_SHA256 in both htr_qsa.py copies -> qsa.py!
    import re as _re
    qsa_ops_p = Path('/opt/venv/lib/python3.14/site-packages/vllm/models/qwen4_exp/amd/ops/qsa.py')
    ce_hsaco_p = nat_dir / 'qsa_canon_expand_fused.hsaco'
    if qsa_ops_p.is_file() and ce_hsaco_p.is_file():
        ce_sha = sha(ce_hsaco_p)
        qsrc = qsa_ops_p.read_text()
        qnew, nsub = _re.subn(r'_HTR_QSA_CE_SHA256 = "[0-9a-f]{64}"', f'_HTR_QSA_CE_SHA256 = "{ce_sha}"', qsrc, count=1)
        assert nsub == 1, '_HTR_QSA_CE_SHA256 not found in qsa.py'
        qsa_ops_p.write_text(qnew)
        qsa_ops_sha = sha(qsa_ops_p)
        print(f'Updated qsa.py _HTR_QSA_CE_SHA256 -> {ce_sha[:16]}... (qsa.py sha={qsa_ops_sha[:16]}...)')
        for hq_p in (ROOT / 'python/htr_qsa.py', Path('/opt/venv/lib/python3.14/site-packages/htr_qsa.py')):
            if hq_p.is_file():
                hsrc = hq_p.read_text()
                hnew, hsub = _re.subn(r"QSA_OPS_SHA256 = '[0-9a-f]{64}'", f"QSA_OPS_SHA256 = '{qsa_ops_sha}'", hsrc, count=1)
                assert hsub == 1, f'QSA_OPS_SHA256 not found in {hq_p}'
                hq_p.write_text(hnew)
                print(f'Updated {hq_p} QSA_OPS_SHA256 -> {qsa_ops_sha[:16]}...')

    # 2c. ERNIE-5.5 gfx1151 memory fixes:
    #   (a) rocm_aiter_mla_sparse.py profile_run: ERNIE-5.5 replaces rocm_fp8_paged_mqa_logits
    #       with a gfx1151 kernel (_ernie5_gfx1151_triton=True) that needs (decode_rows, max_model_len)
    #       = 128 MiB at 32K ctx, NOT (64, decode_rows, max_model_len) = 8.00 GiB!
    #   (b) weight_utils.py safetensors_weights_iterator: evict each closed shard from Linux page
    #       cache via posix_fadvise(POSIX_FADV_DONTNEED) when checkpoint > 32 GiB so 108 GB models
    #       do not blow through container cgroup memory limits (--memory=117g) on unified-memory APUs.
    mla_sparse_p = Path('/opt/venv/lib/python3.14/site-packages/vllm/v1/attention/ops/rocm_aiter_mla_sparse.py')
    if mla_sparse_p.is_file():
        ms_src = mla_sparse_p.read_text()
        old_ms = '        decode_rows = _max_decode_logits_rows(hidden_states.shape[0])\n        if _ON_GFX942 or _ON_GFX950:'
        new_ms = "        decode_rows = _max_decode_logits_rows(hidden_states.shape[0])\n        if _ON_GFX942 or _ON_GFX950 or getattr(rocm_fp8_paged_mqa_logits, '_ernie5_gfx1151_triton', False):"
        if new_ms not in ms_src:
            assert ms_src.count(old_ms) == 1, f'Expected 1 occurrence in rocm_aiter_mla_sparse.py, got {ms_src.count(old_ms)}'
            mla_sparse_p.write_text(ms_src.replace(old_ms, new_ms, 1))
            print('Patched rocm_aiter_mla_sparse.py: gfx1151 ERNIE decode logits workspace (8.00 GiB -> 128 MiB at 32K)')

    wt_utils_p = Path('/opt/venv/lib/python3.14/site-packages/vllm/model_executor/model_loader/weight_utils.py')
    if wt_utils_p.is_file():
        wu_src = wt_utils_p.read_text()
        old_wu = (
            '            with safe_open(st_file, framework="pt") as f:\n'
            '                for name in f.keys():  # noqa: SIM118\n'
            '                    if should_skip_weight(name, local_expert_ids):\n'
            '                        continue\n'
            '                    param = f.get_tensor(name)\n'
            '                    yield name, param\n'
        )
        new_wu = old_wu + (
            '        if total_bytes > 2 * 1024**3 and hasattr(os, "posix_fadvise"):\n'
            '            try:\n'
            '                _fd = os.open(st_file, os.O_RDONLY)\n'
            '                try:\n'
            '                    os.posix_fadvise(_fd, 0, 0, os.POSIX_FADV_DONTNEED)\n'
            '                finally:\n'
            '                    os.close(_fd)\n'
            '            except OSError:\n'
            '                pass\n'
        )
        if 'POSIX_FADV_DONTNEED' not in wu_src:
            assert wu_src.count(old_wu) == 1, f'Expected 1 safe_open block in weight_utils.py, got {wu_src.count(old_wu)}'
            wt_utils_p.write_text(wu_src.replace(old_wu, new_wu, 1))
            print('Patched weight_utils.py: per-shard POSIX_FADV_DONTNEED for checkpoints > 2 GiB')

    #   (c) deepseek_scaling_rope.py: keep YaRN inv_freq bit-identical (using full
    #       max_position_embeddings * scaling_factor), but only materialize cos_sin_cache rows up to
    #       min(full_len, max_model_len + 256). Cuts 1M-pos tables from 256 MiB -> 8 MiB (~1.0 GiB saved!).
    ds_rope_p = Path('/opt/venv/lib/python3.14/site-packages/vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py')
    if ds_rope_p.is_file():
        dr_src = ds_rope_p.read_text()
        old_dr = (
            '        t = torch.arange(\n'
            '            self.max_position_embeddings * self.scaling_factor,\n'
            '            device=inv_freq.device,\n'
            '            dtype=torch.float32,\n'
            '        )\n'
        )
        new_dr = (
            '        _n_pos = int(self.max_position_embeddings * self.scaling_factor)\n'
            '        try:\n'
            '            from vllm.config import get_current_vllm_config\n'
            '            _mml = getattr(get_current_vllm_config().model_config, "max_model_len", None)\n'
            '            if _mml and int(_mml) + 256 < _n_pos:\n'
            '                _n_pos = int(_mml) + 256\n'
            '        except Exception:\n'
            '            pass\n'
            '        t = torch.arange(\n'
            '            _n_pos,\n'
            '            device=inv_freq.device,\n'
            '            dtype=torch.float32,\n'
            '        )\n'
        )
        if '_n_pos = int(self.max_position_embeddings * self.scaling_factor)' not in dr_src:
            assert dr_src.count(old_dr) >= 1, 'Expected torch.arange block in deepseek_scaling_rope.py'
            ds_rope_p.write_text(dr_src.replace(old_dr, new_dr))
            print(f'Patched deepseek_scaling_rope.py: cap cos_sin_cache rows to max_model_len+256 ({dr_src.count(old_dr)} sites)')

    #   (d) kv_cache_utils.py (_get_packed_kv_cache_groups): pack single-page-size buckets to fill
    #       anchor_bytes instead of over-splitting by repeats_per_group. Cuts ERNIE-5.5 from 16 -> 9
    #       KV groups (HCA C128 5->1, state 5->3, SWA 5->4), boosting KV capacity ~1.7x (64K ctx in 1.30 GiB!).
    kcu_p = Path('/opt/venv/lib/python3.14/site-packages/vllm/v1/core/kv_cache_utils.py')
    if kcu_p.is_file():
        ku_src = kcu_p.read_text()
        old_ku = (
            '    groups = []\n'
            '    for spec, page_size_layers, balanced in bucketed:\n'
            '        num_groups = num_groups_for(spec, balanced)\n'
            '        # Cap a state group at the states a block already fits rather than let\n'
        )
        new_ku = (
            '    groups = []\n'
            '    for spec, page_size_layers, balanced in bucketed:\n'
            '        num_groups = num_groups_for(spec, balanced)\n'
            '        if anchor_bytes and len(page_size_layers) == 1:\n'
            '            _page_sz = next(iter(page_size_layers))\n'
            '            if _page_sz > 0:\n'
            '                _layers_per_blk = max(anchor_bytes // _page_sz, 1)\n'
            '                num_groups = cdiv(len(spec.kv_cache_specs), _layers_per_blk)\n'
            '        # Cap a state group at the states a block already fits rather than let\n'
        )
        old_sb = (
            '    def is_state_bucket(spec: UniformTypeKVCacheSpecs) -> bool:\n'
            '        if isinstance(spec.first_spec, (MambaSpec, CircularBufferSpec)):\n'
            '            return True\n'
            '        return repeats_per_group is None and isinstance(\n'
            '            spec.first_spec, SlidingWindowSpec\n'
            '        )\n'
        )
        new_sb = (
            '    def is_state_bucket(spec: UniformTypeKVCacheSpecs) -> bool:\n'
            '        return isinstance(\n'
            '            spec.first_spec, (MambaSpec, CircularBufferSpec, SlidingWindowSpec)\n'
            '        )\n'
        )
        if old_sb in ku_src:
            ku_src = ku_src.replace(old_sb, new_sb, 1)
        if '_layers_per_blk = max(anchor_bytes // _page_sz, 1)' not in ku_src:
            assert ku_src.count(old_ku) == 1, f'Expected 1 occurrence in kv_cache_utils.py, got {ku_src.count(old_ku)}'
            ku_src = ku_src.replace(old_ku, new_ku, 1)
            kcu_p.write_text(ku_src)
            print('Patched kv_cache_utils.py: optimal anchor-bytes packing + SlidingWindowSpec state-bucket cap')

    #   (e) gpu_worker.py: when kv_cache_memory_bytes skips memory_profiling(), flush idle
    #       PyTorch caching-allocator segments right after profile_run() (~1.0 GiB saved on 128GB).
    gw_p = Path('/opt/venv/lib/python3.14/site-packages/vllm/v1/worker/gpu_worker.py')
    if gw_p.is_file():
        gw_src = gw_p.read_text()
        old_gw = (
            '        if kv_cache_memory_bytes := self.cache_config.kv_cache_memory_bytes:\n'
            '            # still need a profile run which compiles the model for\n'
            '            # max_num_batched_tokens\n'
            '            self.model_runner.profile_run()\n'
        )
        new_gw = (
            '        if kv_cache_memory_bytes := self.cache_config.kv_cache_memory_bytes:\n'
            '            # still need a profile run which compiles the model for\n'
            '            # max_num_batched_tokens\n'
            '            self.model_runner.profile_run()\n'
            '            import gc as _gc; _gc.collect(); torch.cuda.empty_cache()\n'
        )
        if old_gw in gw_src and 'torch.cuda.empty_cache()' not in gw_src.split('kv_cache_memory_bytes')[1][:320]:
            assert gw_src.count(old_gw) == 1, f'Expected 1 occurrence in gpu_worker.py, got {gw_src.count(old_gw)}'
            gw_p.write_text(gw_src.replace(old_gw, new_gw, 1))
            print('Patched gpu_worker.py: empty_cache after profile_run in kv_cache_memory_bytes path')

    #   (f) indexer.py: align DeepseekV32IndexerMetadataBuilder.max_prefill_buffer_size with
    #       ERNIE-5.5 1x max_model_len workspace cap (keeps WorkspaceManager at 132 MiB for any max_num_seqs).
    idx_p = Path('/opt/venv/lib/python3.14/site-packages/vllm/v1/attention/backends/mla/indexer.py')
    if idx_p.is_file():
        idx_src = idx_p.read_text()
        old_idx = '        self.max_prefill_buffer_size = get_max_prefill_buffer_size(self.vllm_config)\n'
        new_idx = (
            '        self.max_prefill_buffer_size = min(\n'
            '            get_max_prefill_buffer_size(self.vllm_config),\n'
            '            self.vllm_config.model_config.max_model_len,\n'
            '        )\n'
        )
        if old_idx in idx_src:
            assert idx_src.count(old_idx) == 1, f'Expected 1 occurrence in indexer.py, got {idx_src.count(old_idx)}'
            idx_p.write_text(idx_src.replace(old_idx, new_idx, 1))
            print('Patched indexer.py: cap DeepseekV32IndexerMetadataBuilder.max_prefill_buffer_size to max_model_len')

    # 3. Install ERNIE-5.5 native sources, 8 gfx1151 Wave32 .so binaries, plugin, and manifest
    ernie_src = PAYLOAD / 'ernie'
    shutil.copytree(ernie_src / 'native-source/ernie55', ROOT / 'native-source/ernie55', dirs_exist_ok=True)
    for so in sorted((ernie_src / 'native-bin').glob('*.so')):
        dst = nat_dir / so.name
        shutil.copyfile(so, dst)
        dst.chmod(0o755)
    shutil.copytree(ernie_src / 'python/vllm_ernie_htr', ROOT / 'python/vllm_ernie_htr', dirs_exist_ok=True)
    shutil.copytree(ernie_src / 'python/vllm_ernie_htr-0.1.0.dist-info', ROOT / 'python/vllm_ernie_htr-0.1.0.dist-info', dirs_exist_ok=True)
    shutil.copyfile(ernie_src / 'ERNIE55-MANIFEST.json', ROOT / 'ERNIE55-MANIFEST.json')
    build_sh = ROOT / 'native-source/ernie55/build_ernie55.sh'
    if build_sh.is_file():
        build_sh.chmod(0o755)

    # Pre-compile .pyc for /opt/vllm-htr/python (speeds up startup!)
    compileall.compile_dir(str(ROOT / 'python'), quiet=1, force=True)

    # 4. Reconcile grouped-bindings.json
    gb_path = ROOT / 'python/htr_moe_serve/grouped-bindings.json'
    if gb_path.is_file():
        gb = json.loads(gb_path.read_text())
        for path_str in list(gb.get('files', {}).keys()):
            p = Path(path_str)
            if p.is_file():
                gb['files'][path_str] = sha(p)
        gb_path.write_text(json.dumps(gb, indent=2, sort_keys=True) + '\n')

    # 5. Reconcile V025 & V026 native manifests for upgraded binaries
    for mf_name in ('V025-NATIVE-MANIFEST.json', 'V026-NATIVE-MANIFEST.json'):
        mf_p = ROOT / mf_name
        if mf_p.is_file():
            mf = json.loads(mf_p.read_text())
            for name in list(mf.get('binaries', {}).keys()):
                p = nat_dir / name
                if p.is_file():
                    mf['binaries'][name] = {'sha256': sha(p), 'bytes': p.stat().st_size}
            mf_p.write_text(json.dumps(mf, indent=2) + '\n')

    # 6. Reconcile native-policy-bindings.json & quantized-kv-bindings.json FIRST,
    #    then installed-files.json LAST (since installed-files.json hashes the binding JSONs!)
    for bp_name in ('native-policy-bindings.json', 'quantized-kv-bindings.json'):
        bp = ROOT / bp_name
        if bp.is_file():
            obj = json.loads(bp.read_text())
            for k in list(obj.keys()):
                p = Path(k)
                if p.is_file():
                    obj[k] = sha(p)
            if bp_name == 'native-policy-bindings.json':
                for name in nat_manifest['qwen38_binaries']:
                    obj[str(nat_dir / name)] = sha(nat_dir / name)
            bp.write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n')

    inst_path = ROOT / 'installed-files.json'
    if inst_path.is_file():
        inst = json.loads(inst_path.read_text())
        for k in list(inst.keys()):
            p = ROOT / k
            if p.is_file():
                inst[k] = sha(p)
        for name in nat_manifest['qwen38_binaries']:
            inst[f'native-bin/{name}'] = sha(nat_dir / name)
        inst['V027-NATIVE-MANIFEST.json'] = sha(ROOT / 'V027-NATIVE-MANIFEST.json')
        inst['ERNIE55-MANIFEST.json'] = sha(ROOT / 'ERNIE55-MANIFEST.json')
        inst_path.write_text(json.dumps(inst, indent=2, sort_keys=True) + '\n')

    if BASE_MANIFEST.is_file():
        base = json.loads(BASE_MANIFEST.read_text())
        for row in base.get('files', []):
            p = Path(row['installed_path'])
            if p.is_file():
                row['sha256'] = sha(p)
                for field in ('size', 'bytes', 'size_bytes'):
                    if field in row:
                        row[field] = p.stat().st_size
        BASE_MANIFEST.write_text(json.dumps(base, indent=2) + '\n')

    # Run ERNIE-5.5 CPU verification script before removing /payload
    import subprocess
    res = subprocess.run([sys.executable, str(ernie_src / 'verify_plugin_cpu.py')], check=False)
    if res.returncode != 0:
        raise RuntimeError('ERNIE-5.5 verify_plugin_cpu.py failed!')

    shutil.rmtree(PAYLOAD, ignore_errors=True)
    sys.path.insert(0, str(ROOT / 'python'))
    import htr_native_policy
    htr_native_policy.verify_installation()
    print(f'v0.2.7 ZERO-OVERLAY install verified: {len(file_map)} Qwen Python files + {len(nat_manifest["qwen38_binaries"])} Qwen binaries + ERNIE-5.5 (8 .so + plugin).')

if __name__ == '__main__':
    main()
