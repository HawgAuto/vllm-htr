"""HTR routed-MoE forward routes on a loaded arena (no vLLM imports; unit-testable).

  decode_route  T <= 8 tokens: e12 v2 kernels (k_moe_plan -> k_moe gate|up -> k_moe down -> k_moe_combine), exactly the
                e12 Moe.forward call sequence (fixed-K launch for uniform layers, bps gu5/dn1, fence_ra 1, per-K strip padding).
                Static shared scratch sized for T=8 -> graph-capturable (vLLM captures T=1..4).
  chunk_route   8 < T <= HTR_MOE_PREFILL_CUT (default 512): decode_route over consecutive 8-token chunks (eager only).
  v2_route      HTR_MOE_V2_MIN <= T (set from the tests/test_gpu_sweep.py crossover): csrc/htr_moe_prefill_v2.cu (moe_align 64-row
                blocks -> on-the-fly HTR dequant inside a WMMA GEMM, gate|up -> SiLU*up -> down -> weighted reduce); G1/G2/SAT passed
                2026-10-23 (rel <= 4.3e-3 vs fp64). Takes priority over chunk/dequant when enabled (HTR_MOE_V2_MIN > 0).
  dequant_route T > cut: whole-layer fp32-exact dequant (dq_arena) into ONE shared BF16 scratch (w13 [512,1280,2560], w2
                [512,2560,640]) + stock vLLM BF16 fused_experts (Triton). Padding ids (-1) get weight 0 and id 0.
All routes return bf16 [T, 2560] = sum_s w[t,s] * expert(x[t]) (router weights applied once, shared expert NOT included)."""
import os
import torch
from . import arena as A
from . import wire
from .kernels import moe_ext, dq_ext
from .kernels import moe2_ext  # E29
from .kernels import hpf3_ext  # E31

TOPK, MAXT = 10, 8
BPS_GU, BPS_DN, FENCE_RA = 5, 1, 1
CUT = int(os.environ.get('HTR_MOE_PREFILL_CUT', '512'))
V2_MIN = int(os.environ.get('HTR_MOE_V2_MIN', '0'))      # 0 = v2 route disabled (pre-sweep behaviour, byte-identical routing)
FUSE = os.environ.get('HTR_MOE_FUSE_COMBINE') == '1'   # E29: fused down+combine decode route (bitwise == decode_route)
HPF3 = os.environ.get('HTR_MOE_HPF3') == '1'           # E31: v2 route through hpf3_forward (bitwise == hpf2_forward)
_SCR, _BF, _CB, _V2 = {}, {}, {}, {}


def bucket(T):
    return 1 if T == 1 else 2 if T == 2 else 4 if T <= 4 else 8


def _scratch(dev):
    s = _SCR.get(str(dev))
    if s is None:
        a, m = MAXT * TOPK, 8
        ns_gu, ns_dn = -(-(A.H // 128) // BPS_GU), -(-(A.I // 128) // BPS_DN)
        f = dict(dtype=torch.float32, device=dev)
        s = _SCR[str(dev)] = dict(
            gu=torch.empty(a * m * 2 * A.I, **f), dn=torch.empty(a * m * A.H, **f),
            part=torch.empty(max(a * ns_gu * m * 2 * A.I, a * ns_dn * m * A.H), **f),
            cnt_gu=torch.zeros(a * 10, dtype=torch.int32, device=dev), cnt_dn=torch.zeros(a * 20, dtype=torch.int32, device=dev),
            cnt_c2=torch.zeros(20, dtype=torch.int32, device=dev))
    return s


def cbscale():
    if 'cb' not in _CB:
        _CB['cb'] = wire.cbscale()
    return _CB['cb']


def spadk(ar):
    return tuple(int(ar['_spad'].get(k, 0)) for k in (3, 4, 5))


def decode_route(ar, x, ids32, w32):
    """x bf16 [T, 2560] contiguous, ids32 int32 [T, 10] contiguous (-1 = no expert), w32 f32 [T, 10] contiguous; T <= 8."""
    T = x.shape[0]
    assert 1 <= T <= MAXT and ids32.shape == (T, TOPK) and w32.shape == (T, TOPK)
    if FUSE:
        return decode_route_fused(ar, x, ids32, w32)
    E, s, M = moe_ext(), _scratch(x.device), bucket(T)
    ge, gr, gn, slot = E.moe_plan(ids32, M)
    kf = ar['_kf']
    sp = spadk(ar)
    E.moe_gemv(0, kf, M, ar['qw'], ar['qoff'], ar['kexp'], ar['scales'], ar['su'], ar['sv'], ge, gr, gn, x, s['part'], s['cnt_gu'],
               s['gu'], BPS_GU, cbscale(), *sp, FENCE_RA)
    E.moe_gemv(1, kf, M, ar['qw'], ar['qoff'], ar['kexp'], ar['scales'], ar['su'], ar['sv'], ge, gr, gn, s['gu'], s['part'], s['cnt_dn'],
               s['dn'], BPS_DN, cbscale(), *sp, FENCE_RA)
    return E.moe_combine(s['dn'], slot, w32)


def decode_route_fused(ar, x, ids32, w32):
    """E29: same kernels/arguments as decode_route; plan2 also counts groups; the down GEMV's last arrivers do the combine."""
    T = x.shape[0]
    E, s, M = moe2_ext(), _scratch(x.device), bucket(T)
    ge, gr, gn, slot, ng = E.moe_plan2(ids32, M)
    kf = ar['_kf']
    sp = spadk(ar)
    E.moe_gemv(0, kf, M, ar['qw'], ar['qoff'], ar['kexp'], ar['scales'], ar['su'], ar['sv'], ge, gr, gn, x, s['part'], s['cnt_gu'],
               s['gu'], BPS_GU, cbscale(), *sp, FENCE_RA)
    return E.moe_gemv_dn_fused(kf, M, ar['qw'], ar['qoff'], ar['kexp'], ar['scales'], ar['su'], ar['sv'], ge, gr, gn, s['gu'], s['part'],
                               s['cnt_dn'], s['dn'], BPS_DN, cbscale(), *sp, FENCE_RA, slot, w32, ng, s['cnt_c2'])


def chunk_route(ar, x, ids32, w32):
    out = torch.empty_like(x)
    for i in range(0, x.shape[0], MAXT):
        j = min(i + MAXT, x.shape[0])
        out[i:j] = decode_route(ar, x[i:j], ids32[i:j], w32[i:j])
    return out


def bf16_scratch(dev):
    b = _BF.get(str(dev))
    if b is None:
        b = _BF[str(dev)] = (torch.empty(A.NE, 2 * A.I, A.H, dtype=torch.bfloat16, device=dev),
                             torch.empty(A.NE, A.H, A.I, dtype=torch.bfloat16, device=dev))
    return b


def codebook(dev):
    k = 'tab:' + str(dev)
    if k not in _CB:
        _CB[k] = wire.codebook_f32(dev)
    return _CB[k]


def dequant_layer(ar, w13, w2):
    """All 512 experts of the arena -> w13 [512, 1280, 2560] (gate rows 0..639, up 640..1279), w2 [512, 2560, 640] bf16."""
    D, cb, K = dq_ext(), codebook(ar['qw'].device), ar['_K']
    pad = int(ar['_spad'].get(K, 0))
    for role, (dst, row) in enumerate(((w13, 0), (w13, A.I), (w2, 0))):
        r = A.ROLES[role]
        m, n = A.GEOM[r]
        D.dq_arena(ar['qw'], ar['qoff'], ar['scales'], ar['su'], ar['sv'], cb, K, role, pad, m, n, A.SC_OFF[r], A.SC_ES,
                   A.SU_OFF[r], A.SU_ES, A.SV_OFF[r], A.SV_ES, dst, dst.stride(0), row)
    return w13, w2


def dequant_route(ar, x, ids, w):
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
    w13, w2 = dequant_layer(ar, *bf16_scratch(x.device))
    valid = ids >= 0
    return fused_experts(x, w13, w2, torch.where(valid, w, torch.zeros_like(w)).to(torch.float32),
                         torch.where(valid, ids, torch.zeros_like(ids)), global_num_experts=A.NE)


def v2_route(ar, x, ids, w):
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
    from .prefill2 import hpf2_ext
    valid = ids >= 0
    idc = torch.where(valid, ids, torch.zeros_like(ids)).to(torch.int32)
    wc = torch.where(valid, w, torch.zeros_like(w)).to(torch.float32).contiguous()
    sid, eid, ntpp = moe_align_block_size(idc, 64, A.NE, pad_sorted_ids=True)
    dev, Pn = x.device, sid.numel()
    sc = _V2.setdefault(str(dev), {})
    for k, (n, dt) in dict(A1=(Pn * A.H, torch.float16), Y1=(Pn * 2 * A.I, torch.bfloat16), A2=(Pn * A.I, torch.float16),
                           Y2=(Pn * A.H, torch.bfloat16), S=(Pn, torch.float32), inv=(idc.numel(), torch.int32)).items():
        if k not in sc or sc[k].numel() < n:
            sc[k] = torch.empty(n, dtype=dt, device=dev)
    nsat = sc.setdefault('nsat', torch.zeros(1, dtype=torch.int32, device=dev))
    K = ar['_K']
    if HPF3:
        return hpf3_ext().hpf3_forward(x.contiguous(), sid.to(torch.int32), eid.to(torch.int32), ntpp.to(torch.int32), wc, ar['qw'], ar['qoff'],
                                       ar['scales'], ar['su'], ar['sv'], cbscale(), K, int(ar['_spad'].get(K, 0)), sc['A1'], sc['Y1'],
                                       sc['A2'], sc['Y2'], sc['S'], sc['inv'], nsat, 31)
    return hpf2_ext().hpf2_forward(x.contiguous(), sid.to(torch.int32), eid.to(torch.int32), ntpp.to(torch.int32), wc, ar['qw'], ar['qoff'],
                                   ar['scales'], ar['su'], ar['sv'], cbscale(), K, int(ar['_spad'].get(K, 0)), sc['A1'], sc['Y1'],
                                   sc['A2'], sc['Y2'], sc['S'], sc['inv'], nsat, 31)


def route_name(T):
    if T <= MAXT:
        return 'decode'
    if V2_MIN and T >= V2_MIN:
        return 'v2'
    return 'chunk' if T <= CUT else 'dequant'


def forward(ar, x, ids, w):
    T = x.shape[0]
    if T > MAXT and V2_MIN and T >= V2_MIN:
        return v2_route(ar, x, ids, w)
    if T <= CUT:
        ids32 = ids.to(torch.int32).contiguous()
        w32 = w.to(torch.float32).contiguous()
        return decode_route(ar, x, ids32, w32) if T <= MAXT else chunk_route(ar, x, ids32, w32)
    return dequant_route(ar, x, ids, w)
