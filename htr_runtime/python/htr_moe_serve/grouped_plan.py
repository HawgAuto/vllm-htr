"""Whole-step expert-major grouped plan. Same native htr_moe2 kernels, M=4 rows/group, but groups span the whole
step: every expert gets ceil(max(even,odd)/2) groups, each row placed at a slot whose parity equals the row's native
8-token-chunk rank parity (the invariant partition4 preserves). Pure torch integer ops, static shapes, no host sync."""
import torch

TOPK = 10


def make_plan(ids):
    """Integer-only C07 successor: -1 slots use an unconsumed sink group.

    Preserve every valid row's native eight-token-chunk parity, M=4 slots,
    expert order and original unpadded metadata. No device-to-host decision.
    Fully padded steps use the release's one-unused-group/slot=-1 convention.
    Actual Torch/device and changed-input graph qualification remain required.
    """
    T = ids.shape[0]
    A = T * TOPK
    dev = ids.device
    e = ids.reshape(-1).to(torch.int64)
    idx = torch.arange(A, device=dev)
    tok = idx // TOPK
    ch = tok // 8
    valid = e >= 0
    safe_e = torch.where(valid, e, 0)
    same = (e[:, None] == e[None, :]) & (ch[:, None] == ch[None, :]) & (idx[None, :] < idx[:, None])
    par = same.sum(1) & 1
    key = torch.where(valid, e * 2 + par, 1024)
    order = torch.argsort(key * A + idx)
    ks = key[order]
    cnt = torch.zeros(1025, dtype=torch.int64, device=dev).scatter_add_(0, key, torch.ones_like(key))
    start = torch.cumsum(cnt, 0) - cnt
    k = torch.empty_like(idx)
    k[order] = torch.arange(A, device=dev) - start[ks]
    c2 = cnt[:1024].view(512, 2)
    ge_n = torch.maximum((c2[:, 0] + 1) // 2, (c2[:, 1] + 1) // 2)
    goff = torch.cumsum(ge_n, 0) - ge_n
    g = torch.where(valid, goff[safe_e] + k // 2, A)
    pos = torch.where(valid, 2 * (k % 2) + par, 0)
    i32 = dict(dtype=torch.int32, device=dev)
    # The extra group is a write sink only; it is not passed to native kernels.
    ge = torch.full((A + 1,), -1, **i32)
    ge[g] = safe_e.to(torch.int32)
    gn = torch.zeros(A + 1, dtype=torch.int64, device=dev).scatter_reduce_(0, g, pos + 1, 'amax').to(torch.int32)
    gr = torch.zeros((A + 1) * 4, **i32)
    gr[g * 4 + pos] = torch.where(valid, tok, 0).to(torch.int32)
    slot = torch.where(valid, g * 4 + pos, -1).to(torch.int32)
    groups = ge_n.sum()
    ge[0] = torch.where(groups == 0, 0, ge[0])
    gn[0] = torch.where(groups == 0, 1, gn[0])
    ng = groups.clamp(min=1).to(torch.int32).reshape(1)
    return ge[:A], gr.view(A + 1, 4)[:A], gn[:A], slot, ng


_S = {}


def scratch(dev, A, H=2560, I=640, bps_gu=5, bps_dn=1):
    s = _S.get((str(dev), A))
    if s is None:
        m = 4
        ns_gu, ns_dn = -(-(H // 128) // bps_gu), -(-(I // 128) // bps_dn)
        f = dict(dtype=torch.float32, device=dev)
        s = _S[(str(dev), A)] = dict(
            gu=torch.empty(A * m * 2 * I, **f), dn=torch.empty(A * m * H, **f),
            part=torch.empty(max(A * ns_gu * m * 2 * I, A * ns_dn * m * H), **f),
            cnt_gu=torch.zeros(A * 10, dtype=torch.int32, device=dev), cnt_dn=torch.zeros(A * 20, dtype=torch.int32, device=dev),
            cnt_c2=torch.zeros(20, dtype=torch.int32, device=dev))
    return s


def apply(R, ar, x, w, plan):
    E = R.moe2_ext()
    s = scratch(x.device, plan[0].numel())
    ge, gr, gn, slot, ng = plan
    sp = R.spadk(ar); kf = ar['_kf']
    E.moe_gemv(0, kf, 4, ar['qw'], ar['qoff'], ar['kexp'], ar['scales'], ar['su'], ar['sv'], ge, gr, gn, x,
               s['part'], s['cnt_gu'], s['gu'], R.BPS_GU, R.cbscale(), *sp, R.FENCE_RA)
    return E.moe_gemv_dn_fused(kf, 4, ar['qw'], ar['qoff'], ar['kexp'], ar['scales'], ar['su'], ar['sv'], ge, gr, gn,
                               s['gu'], s['part'], s['cnt_dn'], s['dn'], R.BPS_DN, R.cbscale(), *sp, R.FENCE_RA, slot, w, ng,
                               s['cnt_c2'])


def forward(R, ar, x, ids, w):
    ids = ids.to(torch.int32).contiguous(); w = w.to(torch.float32).contiguous()
    return apply(R, ar, x, w, make_plan(ids))
