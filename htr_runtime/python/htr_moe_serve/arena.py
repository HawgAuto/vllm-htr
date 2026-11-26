"""HTR routed-expert arena loader for serving (e15, 2026-10-23).

Reads the parked P2 packed chunks (format HTR-W1-P2-PACKED-1: per layer 32 files x 16 experts, packed gate/up/down
{qweight int32 [mt, nt, 8K], scales f16 [m], su u8 [n/8], sv u8 [m/8]}), verifies every file's sha256 against the park
manifest while reading it (single read, fail-closed), and builds the e12 v2 serving arena layout:
  qw     int32 1-D: per expert e (0..511), per role (gate, up, down): [mt strips x (nt*8K + spad_K words)]  (e12 moe.Arena order)
  qoff   int64 [512, 3] word offsets;   scales f16 [512*3840] (gate 640 | up 640 | down 2560)
  sv     u8 [512*480] (80 | 80 | 320);  su u8 [512*400] (gate/up shared 320 | down 80);  kexp int32 [512]
Byte layout must equal e12 moe.Arena(L, widths, 'cpu', spad=DEFAULT_SPADK) for the same packed experts (CPU test T1).
A background prefetcher (one daemon thread pool) loads layers in order straight to the device while vLLM loads the trunk."""
import hashlib, io, json, os, threading, time
from concurrent.futures import ThreadPoolExecutor
import torch

FORMAT = 'HTR-W1-P2-PACKED-1'
H, I, NE = 2560, 640, 512
ROLES = ('gate', 'up', 'down')
GEOM = {'gate': (I, H), 'up': (I, H), 'down': (H, I)}          # (m, n)
SC_OFF, SC_ES = {'gate': 0, 'up': I, 'down': 2 * I}, 2 * I + H                 # scales per expert: 3840
SV_OFF, SV_ES = {'gate': 0, 'up': I // 8, 'down': 2 * I // 8}, (2 * I + H) // 8  # sv per expert: 480
SU_OFF, SU_ES = {'gate': 0, 'up': 0, 'down': H // 8}, (H + I) // 8              # su per expert: 400 (gate/up share)
DEFAULT_SPADK = {3: 0, 4: 64, 5: 0}                                              # e12 v2 (bench_layout_L3.json)


def read_manifest(path, want_sha=None):
    raw = open(path, 'rb').read()
    if want_sha is not None and hashlib.sha256(raw).hexdigest() != want_sha:
        raise ValueError(f'HTR MoE manifest {path}: sha256 mismatch')
    man = {}
    for line in raw.decode().splitlines():
        s, p = line.split(None, 1)
        man[p.strip()] = s
    return man



_SHA_VERIFIED_DIRS = {}
_SHA_LOCK = threading.Lock()

def _shard_fingerprint(d):
    parts = []
    for name in sorted(os.listdir(d)):
        if name.endswith(('.htr', '.sha256', '.json')):
            st = os.stat(os.path.join(d, name))
            parts.append((name, st.st_size, st.st_mtime_ns))
    return hashlib.sha256(json.dumps(parts).encode()).hexdigest()

def _sha_marker_path(d):
    root = os.environ.get('HTR_CACHE_HOME', '/workspace/cache')
    return os.path.join(root, 'htr-moe-verified.json')

def _should_verify_sha(d):
    if os.environ.get('HTR_MOE_FORCE_SHA', '0') == '1':
        return True
    if os.environ.get('HTR_MOE_SKIP_SHA', '0') == '1':
        return False
    if os.environ.get('HTR_MOE_FAST_LOAD', '1') != '1':
        return True
    cached = _SHA_VERIFIED_DIRS.get(d)
    if cached is not None:
        return not cached
    with _SHA_LOCK:
        cached = _SHA_VERIFIED_DIRS.get(d)
        if cached is not None:
            return not cached
        ok = False
        try:
            mp = _sha_marker_path(d)
            if os.path.isfile(mp):
                rec = json.load(open(mp))
                if rec.get('dir') == os.path.realpath(d) and rec.get('fp') == _shard_fingerprint(d):
                    ok = True
        except Exception:
            ok = False
        _SHA_VERIFIED_DIRS[d] = ok
        return not ok

def _mark_sha_verified(d):
    try:
        mp = _sha_marker_path(d)
        os.makedirs(os.path.dirname(mp), exist_ok=True)
        tmp = mp + f'.tmp.{os.getpid()}'
        with open(tmp, 'w') as f:
            json.dump({'dir': os.path.realpath(d), 'fp': _shard_fingerprint(d)}, f)
        os.replace(tmp, mp)
        _SHA_VERIFIED_DIRS[d] = True
    except Exception:
        pass

def iter_p2_experts(d, L, man, info=None):
    """Yield (e, packed) for layer L from the P2 chunk dir; every file sha-checked against the manifest as it is read."""
    sharded = os.path.isfile(os.path.join(d, 'chunks.index.json'))
    if sharded:
        fs = sorted(p.split('/')[1] for p in man if p.startswith(f'L{L}/') and p.endswith('.pt'))
    else:
        fs = sorted(f for f in os.listdir(os.path.join(d, f'L{L}')) if f.startswith('c') and f.endswith('.pt'))
    if len(fs) != 32:
        raise ValueError(f'HTR MoE L{L}: {len(fs)} chunk files, expected 32')
    # v0.2.7 boot optimization: avoid hashing all 62 GB TWICE (read_chunk already verifies!).
    # Plus optional verified-once stat cache (HTR_MOE_FAST_LOAD=1, default 1) skips redundant
    # 25s SHA-256 re-verification on warm boots once the shard files (inode/size/mtime) passed full SHA-256.
    verify_sha = _should_verify_sha(d)
    for f in fs:
        rel = f'L{L}/{f}'
        want_h = man[rel]
        if sharded:
            from release_shards import read_chunk
            b = read_chunk(d, rel, want_h, verify_sha=verify_sha)
            h = want_h
        else:
            with open(os.path.join(d, rel), 'rb') as source:
                b = source.read()
            if verify_sha:
                h = hashlib.sha256(b).hexdigest()
                if want_h != h:
                    raise ValueError(f'HTR MoE {rel}: sha256 {h} != manifest {want_h}')
            else:
                h = want_h
        ck = torch.load(io.BytesIO(b), weights_only=False, map_location='cpu')
        del b
        if ck.get('format') != FORMAT or ck.get('L') != L or len(ck['experts']) != 16:
            raise ValueError(f'HTR MoE {rel}: format {ck.get("format")} L {ck.get("L")} n {len(ck.get("experts", []))}')
        if info is not None:
            info.setdefault('K', set()).add(int(ck['K'])); info.setdefault('files', []).append([rel, h])
        for x in ck['experts']:
            yield int(x['e']), int(ck['K']), x['packed']


def build_arena_cpu(experts, spadk=DEFAULT_SPADK):
    """experts: iterable of (e, K, packed{gate,up,down}) covering 0..511 exactly once -> dict of CPU tensors (e12 v2 layout)."""
    per, kk = {}, {}
    for e, k, p in experts:
        if e in per:
            raise ValueError(f'duplicate expert {e}')
        per[e], kk[e] = p, k
    if sorted(per) != list(range(NE)):
        raise ValueError(f'expert coverage {len(per)} != {NE}')
    # Pass 1: validate + compute exact total words so we allocate ONE contiguous qw/scales/sv/su (zero double-copy!)
    qoff = []
    off = 0
    for e in range(NE):
        k, p = kk[e], per[e]
        if not torch.equal(p['gate']['su'], p['up']['su']):
            raise ValueError(f'expert {e}: gate/up su differ')
        pk = spadk.get(k, 0)
        o = []
        for r in ROLES:
            m, n = GEOM[r]
            q = p[r]['qweight']
            if q.dtype != torch.int32 or tuple(q.shape) != (m // 16, n // 16, 8 * k):
                raise ValueError(f'expert {e} {r}: qweight {q.dtype} {tuple(q.shape)}')
            if (p[r]['scales'].dtype, p[r]['scales'].numel()) != (torch.float16, m) or p[r]['sv'].numel() != m // 8 \
                    or p[r]['su'].numel() != n // 8 or p[r]['sv'].dtype != torch.uint8 or p[r]['su'].dtype != torch.uint8:
                raise ValueError(f'expert {e} {r}: scales/su/sv geometry or dtype')
            rows_r = m // 16
            cols_r = (n // 16) * (8 * k) + pk
            o.append(off)
            off += rows_r * cols_r
        qoff.append(o)
    sh_qw_off = None
    total_words = off
    if os.environ.get('HTR_MOE_K6_SHARED') == '1':
        spad5 = int(spadk.get(5, 0))
        sh_words = (40 + 40) * (160 * 40 + spad5) + 160 * (40 * 40 + spad5)
        sh_qw_off = off
        total_words += sh_words
    qw = torch.empty(total_words, dtype=torch.int32)
    if sh_qw_off is not None:
        qw[sh_qw_off:].zero_()
    scales = torch.empty(NE * SC_ES, dtype=torch.float16)
    sv = torch.empty(NE * SV_ES, dtype=torch.uint8)
    su = torch.empty(NE * SU_ES, dtype=torch.uint8)
    # Pass 2: direct in-place copy into pre-allocated buffers (saves ~2.36 GiB transient memcpy/layer)
    for e in range(NE):
        k, p = kk[e], per[e]
        pk = spadk.get(k, 0)
        for ri, r in enumerate(ROLES):
            m, n = GEOM[r]
            rows_r = m // 16
            raw_cols = (n // 16) * (8 * k)
            cols_r = raw_cols + pk
            o0 = qoff[e][ri]
            q = p[r]['qweight'].contiguous().view(rows_r, raw_cols)
            if pk:
                dst_v = qw[o0:o0 + rows_r * cols_r].view(rows_r, cols_r)
                dst_v[:, :raw_cols].copy_(q)
                dst_v[:, raw_cols:].zero_()
            else:
                qw[o0:o0 + rows_r * raw_cols].copy_(q.reshape(-1))
        sb = e * SC_ES
        scales[sb:sb + I].copy_(p['gate']['scales'])
        scales[sb + I:sb + 2 * I].copy_(p['up']['scales'])
        scales[sb + 2 * I:sb + SC_ES].copy_(p['down']['scales'])
        vb = e * SV_ES
        sv[vb:vb + I // 8].copy_(p['gate']['sv'])
        sv[vb + I // 8:vb + 2 * I // 8].copy_(p['up']['sv'])
        sv[vb + 2 * I // 8:vb + SV_ES].copy_(p['down']['sv'])
        ub = e * SU_ES
        su[ub:ub + H // 8].copy_(p['gate']['su'])
        su[ub + H // 8:ub + SU_ES].copy_(p['down']['su'])
    out = dict(
        qw=qw,
        qoff=torch.tensor(qoff, dtype=torch.int64),
        scales=scales,
        sv=sv,
        su=su,
        kexp=torch.tensor([kk[e] for e in range(NE)], dtype=torch.int32))
    assert out['scales'].numel() == NE * SC_ES and out['sv'].numel() == NE * SV_ES and out['su'].numel() == NE * SU_ES
    if sh_qw_off is not None:
        out['_sh_qw_off'] = sh_qw_off
    return out



_FAST_ARENA_MOD = False
_FAST_ARENA_LOCK = threading.Lock()

def _get_fast_arena():
    global _FAST_ARENA_MOD
    if _FAST_ARENA_MOD is not False:
        return _FAST_ARENA_MOD
    with _FAST_ARENA_LOCK:
        if _FAST_ARENA_MOD is not False:
            return _FAST_ARENA_MOD
        if os.environ.get('HTR_MOE_FAST_ARENA', '1') != '1':
            _FAST_ARENA_MOD = None
            return None
        so_path = os.environ.get('HTR_FAST_ARENA_SO', '/opt/vllm-htr/native-bin/htr_fast_arena_v27.so')
        if not os.path.isfile(so_path):
            _FAST_ARENA_MOD = None
            return None
        try:
            import importlib.util
            sp = importlib.util.spec_from_file_location('htr_fast_arena', so_path)
            mod = importlib.util.module_from_spec(sp)
            sp.loader.exec_module(mod)
            _FAST_ARENA_MOD = mod
        except Exception:
            _FAST_ARENA_MOD = None
        return _FAST_ARENA_MOD


def _load_layer_fast_cpu(d, L, man, K, spadk, info):
    fa = _get_fast_arena()
    if fa is None or _should_verify_sha(d):
        return None
    if not os.path.isfile(os.path.join(d, 'chunks.index.json')):
        return None
    from release_shards import _get_index_and_fd
    fs = sorted(p.split('/')[1] for p in man if p.startswith(f'L{L}/') and p.endswith('.pt'))
    if len(fs) != 32:
        return None
    fds, offs, szs, ebases, files_info = [], [], [], [], []
    for f in fs:
        if not (f.startswith('c') and f.endswith('.pt') and f[1:-3].isdigit()):
            return None
        rel = f'L{L}/{f}'
        want_h = man.get(rel)
        r, (fd, fsize) = _get_index_and_fd(d, rel)
        if r['sha256'] != want_h:
            raise ValueError(f'HTR MoE {rel}: chunk index hash mismatch')
        off, sz = r['offset'], r['size']
        if off < 0 or sz <= 0 or off + sz > fsize:
            raise ValueError(f'HTR MoE {rel}: invalid chunk bounds')
        fds.append(fd); offs.append(off); szs.append(sz)
        ebases.append(int(f[1:-3]))
        files_info.append([rel, want_h])

    pk = int(spadk.get(K, 0))
    w_gu = (I // 16) * ((H // 16) * (8 * K) + pk)
    w_dn = (H // 16) * ((I // 16) * (8 * K) + pk)
    w_e = 2 * w_gu + w_dn
    e_idx = torch.arange(NE, dtype=torch.int64)
    qoff = torch.empty((NE, 3), dtype=torch.int64)
    qoff[:, 0] = e_idx * w_e
    qoff[:, 1] = e_idx * w_e + w_gu
    qoff[:, 2] = e_idx * w_e + 2 * w_gu

    total_words = NE * w_e
    sh_qw_off = None
    if os.environ.get('HTR_MOE_K6_SHARED') == '1':
        spad5 = int(spadk.get(5, 0))
        sh_words = (40 + 40) * (160 * 40 + spad5) + 160 * (40 * 40 + spad5)
        sh_qw_off = total_words
        total_words += sh_words

    qw = torch.empty(total_words, dtype=torch.int32)
    scales = torch.empty(NE * SC_ES, dtype=torch.float16)
    sv = torch.empty(NE * SV_ES, dtype=torch.uint8)
    su = torch.empty(NE * SU_ES, dtype=torch.uint8)
    ok = fa.load_layer_fast(fds, offs, szs, ebases, int(K), pk, qw, qoff, scales, sv, su)
    if not ok:
        return None
    if sh_qw_off is not None:
        qw[sh_qw_off:].zero_()
    info['K'] = {int(K)}
    info['files'] = files_info
    out = dict(
        qw=qw,
        qoff=qoff,
        scales=scales,
        sv=sv,
        su=su,
        kexp=torch.full((NE,), int(K), dtype=torch.int32),
    )
    if sh_qw_off is not None:
        out['_sh_qw_off'] = sh_qw_off
    return out


def load_layer(spec, L, man, dev=None):
    t0 = time.time(); info = {}
    spadk = {int(k): int(v) for k, v in spec['spad'].items()}
    a = _load_layer_fast_cpu(spec['dir'], L, man, int(spec['K']), spadk, info)
    if a is None:
        a = build_arena_cpu(iter_p2_experts(spec['dir'], L, man, info), spadk)
    if info['K'] != {spec['K']}:
        raise ValueError(f'HTR MoE L{L}: chunk widths {info["K"]} != spec K {spec["K"]}')
    t1 = time.time()
    if dev is not None:
        a = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in a.items()}
    # uniform layer -> fixed-K kernel launch (e12: mixed vs fixed within 1 us for uniform recipes); spad per K (e12 v2)
    a['_K'] = a['_kf'] = int(spec['K'])
    a['_spad'] = {int(k): int(v) for k, v in spec['spad'].items()}
    a['_meta'] = dict(L=L, K=spec['K'], files=len(info['files']), files_sha256=hashlib.sha256(
        json.dumps(info['files']).encode()).hexdigest(), read_build_s=round(t1 - t0, 2), to_dev_s=round(time.time() - t1, 2),
        bytes=int(sum(v.numel() * v.element_size() for v in a.values() if isinstance(v, torch.Tensor))))
    return a


class _PinnedSlotPipeline:
    """2-slot reusable pinned-memory sequential NVMe reader -> async GPU H2D pipeline.

    DRAM-less NVMe SSDs (e.g. Lexar NQ790) achieve ~3.0 GB/s on 1 sequential reader vs
    ~1.95 GB/s with 8 concurrent random readers, and reusing 2 pinned staging buffers
    eliminates 48x 1.3 GB pageable allocations (16M page faults) and ROCm bounce copies.
    """

    def __init__(self, spec, dev, man):
        import queue
        from concurrent.futures import Future
        self.spec, self.dev, self.man = spec, dev, man
        self.t0 = time.time()
        self.layers = list(spec['layers'])
        self.futs = {L: Future() for L in self.layers}
        self._fa = _get_fast_arena()
        self._spadk = {int(k): int(v) for k, v in spec['spad'].items()}
        self._K = int(spec['K'])
        pk = int(self._spadk.get(self._K, 0))
        w_gu = (I // 16) * ((H // 16) * (8 * self._K) + pk)
        w_dn = (H // 16) * ((I // 16) * (8 * self._K) + pk)
        w_e = 2 * w_gu + w_dn
        e_idx = torch.arange(NE, dtype=torch.int64)
        self._qoff = torch.empty((NE, 3), dtype=torch.int64)
        self._qoff[:, 0] = e_idx * w_e
        self._qoff[:, 1] = e_idx * w_e + w_gu
        self._qoff[:, 2] = e_idx * w_e + 2 * w_gu
        self._kexp = torch.full((NE,), self._K, dtype=torch.int32)
        self._total_words = NE * w_e
        self._sh_qw_off = None
        if os.environ.get('HTR_MOE_K6_SHARED') == '1':
            spad5 = int(self._spadk.get(5, 0))
            sh_words = (40 + 40) * (160 * 40 + spad5) + 160 * (40 * 40 + spad5)
            self._sh_qw_off = self._total_words
            self._total_words += sh_words
        self._pk = pk
        self._free_q = queue.Queue(maxsize=2)
        self._h2d_q = queue.Queue(maxsize=2)
        self._io_thread = threading.Thread(target=self._io_loop, name='htr-moe-io', daemon=True)
        self._h2d_thread = threading.Thread(target=self._h2d_loop, name='htr-moe-h2d', daemon=True)
        self._io_thread.start()
        self._h2d_thread.start()

    def _alloc_slot(self):
        qw = torch.empty(self._total_words, dtype=torch.int32, pin_memory=False)
        if self._sh_qw_off is not None:
            qw[self._sh_qw_off:].zero_()
        scales = torch.empty(NE * SC_ES, dtype=torch.float16, pin_memory=False)
        sv = torch.empty(NE * SV_ES, dtype=torch.uint8, pin_memory=False)
        su = torch.empty(NE * SU_ES, dtype=torch.uint8, pin_memory=False)
        return (qw, scales, sv, su)

    def _io_loop(self):
        try:
            torch.cuda.set_device(self.dev)
            slots = [self._alloc_slot(), self._alloc_slot()]
            for s in slots:
                self._free_q.put(s)
            from release_shards import _get_index_and_fd
            d = self.spec['dir']
            for L in self.layers:
                slot = self._free_q.get()
                t0 = time.time()
                fs = sorted(p.split('/')[1] for p in self.man if p.startswith(f'L{L}/') and p.endswith('.pt'))
                fds, offs, szs, ebases, files_info = [], [], [], [], []
                for f in fs:
                    rel = f'L{L}/{f}'
                    want_h = self.man.get(rel)
                    r, (fd, fsize) = _get_index_and_fd(d, rel)
                    if r['sha256'] != want_h:
                        raise ValueError(f'HTR MoE {rel}: chunk index hash mismatch')
                    off, sz = r['offset'], r['size']
                    if off < 0 or sz <= 0 or off + sz > fsize:
                        raise ValueError(f'HTR MoE {rel}: invalid chunk bounds')
                    fds.append(fd); offs.append(off); szs.append(sz)
                    ebases.append(int(f[1:-3]))
                    files_info.append([rel, want_h])
                qw, scales, sv, su = slot
                ok = self._fa.load_layer_fast(fds, offs, szs, ebases, self._K, self._pk, qw, self._qoff, scales, sv, su)
                if not ok:
                    raise RuntimeError(f'HTR MoE L{L}: load_layer_fast failed')
                read_s = time.time() - t0
                self._h2d_q.put((L, slot, files_info, read_s))
            self._h2d_q.put(None)
        except BaseException as exc:
            self._h2d_q.put(('ERR', exc))

    def _h2d_loop(self):
        try:
            torch.cuda.set_device(self.dev)
            stream = torch.cuda.Stream(self.dev)
            qoff_gpu_proto = self._qoff.to(self.dev)
            kexp_gpu_proto = self._kexp.to(self.dev)
            while True:
                item = self._h2d_q.get()
                if item is None:
                    break
                if isinstance(item, tuple) and len(item) == 2 and item[0] == 'ERR':
                    exc = item[1]
                    for fut in self.futs.values():
                        if not fut.done():
                            fut.set_exception(exc)
                    return
                L, slot, files_info, read_s = item
                t1 = time.time()
                qw, scales, sv, su = slot
                with torch.cuda.stream(stream):
                    out = dict(
                        qw=torch.empty_like(qw, device=self.dev).copy_(qw, non_blocking=True),
                        qoff=qoff_gpu_proto.clone(),
                        scales=torch.empty_like(scales, device=self.dev).copy_(scales, non_blocking=True),
                        sv=torch.empty_like(sv, device=self.dev).copy_(sv, non_blocking=True),
                        su=torch.empty_like(su, device=self.dev).copy_(su, non_blocking=True),
                        kexp=kexp_gpu_proto.clone(),
                    )
                    if self._sh_qw_off is not None:
                        out['_sh_qw_off'] = self._sh_qw_off
                stream.synchronize()
                to_dev_s = time.time() - t1
                self._free_q.put(slot)
                out['_K'] = out['_kf'] = self._K
                out['_spad'] = dict(self._spadk)
                out['_meta'] = dict(
                    L=L, K=self._K, files=len(files_info),
                    files_sha256=hashlib.sha256(json.dumps(files_info).encode()).hexdigest(),
                    read_build_s=round(read_s, 2), to_dev_s=round(to_dev_s, 2),
                    ready_at_s=round(time.time() - self.t0, 1),
                    bytes=int(sum(v.numel() * v.element_size() for v in out.values() if isinstance(v, torch.Tensor))),
                )
                self.futs[L].set_result(out)
        except BaseException as exc:
            for fut in self.futs.values():
                if not fut.done():
                    fut.set_exception(exc)
        finally:
            # Release reusable pinned buffers once all 48 layers are on GPU!
            while not self._free_q.empty():
                try:
                    self._free_q.get_nowait()
                except Exception:
                    break

    def get(self, L):
        return self.futs[L].result()


class Prefetcher:
    """Loads every spec layer in order on daemon threads straight to `dev` on a private stream.
    Uses the fast 2-slot pinned sequential NVMe->GPU pipeline on verified sharded models,
    falling back to ThreadPoolExecutor when SHA-256 verification or unsharded loading is needed."""

    def __init__(self, spec, dev):
        self.spec, self.dev = spec, dev
        self.man = read_manifest(spec['manifest'], spec['manifest_sha256'])
        use_pipe = (
            os.environ.get('HTR_MOE_PINNED_PIPE', '0') == '1'
            and _get_fast_arena() is not None
            and not _should_verify_sha(spec['dir'])
            and os.path.isfile(os.path.join(spec['dir'], 'chunks.index.json'))
            and dev is not None
            and torch.device(dev).type == 'cuda'
        )
        if use_pipe:
            self._pipe = _PinnedSlotPipeline(spec, dev, self.man)
            self.pool = None
            return
        self._pipe = None
        n = int(os.environ.get('HTR_MOE_LOAD_WORKERS', '4'))
        self.pool = ThreadPoolExecutor(max(1, n), thread_name_prefix='htr-moe-load')
        self.tls = threading.local()
        self.t0 = time.time()
        self._need_mark = _should_verify_sha(spec['dir'])
        self._done_cnt = 0
        self._total_layers = len(spec['layers'])
        self._done_lock = threading.Lock()
        self.futs = {L: self.pool.submit(self._one, L) for L in spec['layers']}

    def _one(self, L):
        torch.cuda.set_device(self.dev)
        s = getattr(self.tls, 's', None)
        if s is None:
            s = self.tls.s = torch.cuda.Stream(self.dev)
        with torch.cuda.stream(s):
            a = load_layer(self.spec, L, self.man, self.dev)
        s.synchronize()
        a['_meta']['ready_at_s'] = round(time.time() - self.t0, 1)
        if self._need_mark:
            with self._done_lock:
                self._done_cnt += 1
                if self._done_cnt == self._total_layers:
                    _mark_sha_verified(self.spec['dir'])
        return a

    def get(self, L):
        if self._pipe is not None:
            return self._pipe.get(L)
        return self.futs[L].result()

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=False)


_PREFETCH = {}
_PLOCK = threading.Lock()


def prefetcher(spec, dev):
    key = (spec['dir'], spec['manifest_sha256'], str(dev))
    with _PLOCK:
        if key not in _PREFETCH:
            _PREFETCH[key] = Prefetcher(spec, dev)
        return _PREFETCH[key]
