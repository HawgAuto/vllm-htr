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


def iter_p2_experts(d, L, man, info=None):
    """Yield (e, packed) for layer L from the P2 chunk dir; every file sha-checked against the manifest as it is read."""
    sharded = os.path.isfile(os.path.join(d, 'chunks.index.json'))
    if sharded:
        fs = sorted(p.split('/')[1] for p in man if p.startswith(f'L{L}/') and p.endswith('.pt'))
    else:
        fs = sorted(f for f in os.listdir(os.path.join(d, f'L{L}')) if f.startswith('c') and f.endswith('.pt'))
    if len(fs) != 32:
        raise ValueError(f'HTR MoE L{L}: {len(fs)} chunk files, expected 32')
    for f in fs:
        rel = f'L{L}/{f}'
        if sharded:
            from release_shards import read_chunk
            b = read_chunk(d, rel, man[rel])
        else:
            with open(os.path.join(d, rel), 'rb') as source:
                b = source.read()
        h = hashlib.sha256(b).hexdigest()
        if man.get(rel) != h:
            raise ValueError(f'HTR MoE {rel}: sha256 {h} != manifest {man.get(rel)}')
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
    words, qoff, off = [], [], 0
    for e in range(NE):
        k, p = kk[e], per[e]
        if not torch.equal(p['gate']['su'], p['up']['su']):
            raise ValueError(f'expert {e}: gate/up su differ')
        o = []
        for r in ROLES:
            m, n = GEOM[r]
            q = p[r]['qweight']
            if q.dtype != torch.int32 or tuple(q.shape) != (m // 16, n // 16, 8 * k):
                raise ValueError(f'expert {e} {r}: qweight {q.dtype} {tuple(q.shape)}')
            if (p[r]['scales'].dtype, p[r]['scales'].numel()) != (torch.float16, m) or p[r]['sv'].numel() != m // 8 \
                    or p[r]['su'].numel() != n // 8 or p[r]['sv'].dtype != torch.uint8 or p[r]['su'].dtype != torch.uint8:
                raise ValueError(f'expert {e} {r}: scales/su/sv geometry or dtype')
            q = q.contiguous().view(q.shape[0], -1)
            pk = spadk.get(k, 0)
            if pk:
                q = torch.cat([q, torch.zeros(q.shape[0], pk, dtype=q.dtype)], 1)
            q = q.reshape(-1)
            o.append(off); words.append(q); off += q.numel()
        qoff.append(o)
    out = dict(
        qw=torch.cat(words),
        qoff=torch.tensor(qoff, dtype=torch.int64),
        scales=torch.cat([torch.cat([per[e]['gate']['scales'], per[e]['up']['scales'], per[e]['down']['scales']]) for e in range(NE)]),
        sv=torch.cat([torch.cat([per[e]['gate']['sv'], per[e]['up']['sv'], per[e]['down']['sv']]) for e in range(NE)]),
        su=torch.cat([torch.cat([per[e]['gate']['su'], per[e]['down']['su']]) for e in range(NE)]),
        kexp=torch.tensor([kk[e] for e in range(NE)], dtype=torch.int32))
    assert out['scales'].numel() == NE * SC_ES and out['sv'].numel() == NE * SV_ES and out['su'].numel() == NE * SU_ES
    return out


def load_layer(spec, L, man, dev=None):
    t0 = time.time(); info = {}
    a = build_arena_cpu(iter_p2_experts(spec['dir'], L, man, info), {int(k): int(v) for k, v in spec['spad'].items()})
    if info['K'] != {spec['K']}:
        raise ValueError(f'HTR MoE L{L}: chunk widths {info["K"]} != spec K {spec["K"]}')
    t1 = time.time()
    if dev is not None:
        a = {k: v.to(dev) for k, v in a.items()}
    # uniform layer -> fixed-K kernel launch (e12: mixed vs fixed within 1 us for uniform recipes); spad per K (e12 v2)
    a['_K'] = a['_kf'] = int(spec['K'])
    a['_spad'] = {int(k): int(v) for k, v in spec['spad'].items()}
    a['_meta'] = dict(L=L, K=spec['K'], files=len(info['files']), files_sha256=hashlib.sha256(
        json.dumps(info['files']).encode()).hexdigest(), read_build_s=round(t1 - t0, 2), to_dev_s=round(time.time() - t1, 2),
        bytes=int(sum(v.numel() * v.element_size() for v in a.values() if isinstance(v, torch.Tensor))))
    return a


class Prefetcher:
    """Loads every spec layer in order on daemon threads (HTR_MOE_LOAD_WORKERS, default 2) straight to `dev` on a private
    stream, started at model construction so expert I/O (Seagate) overlaps vLLM's trunk load (NVMe). get(L) blocks."""

    def __init__(self, spec, dev):
        self.spec, self.dev = spec, dev
        self.man = read_manifest(spec['manifest'], spec['manifest_sha256'])
        n = int(os.environ.get('HTR_MOE_LOAD_WORKERS', '2'))
        self.pool = ThreadPoolExecutor(max(1, n), thread_name_prefix='htr-moe-load')
        self.tls = threading.local()
        self.t0 = time.time()
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
        return a

    def get(self, L):
        return self.futs[L].result()


_PREFETCH = {}
_PLOCK = threading.Lock()


def prefetcher(spec, dev):
    key = (spec['dir'], spec['manifest_sha256'], str(dev))
    with _PLOCK:
        if key not in _PREFETCH:
            _PREFETCH[key] = Prefetcher(spec, dev)
        return _PREFETCH[key]
