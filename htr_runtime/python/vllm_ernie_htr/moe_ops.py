# SPDX-License-Identifier: Apache-2.0
"""Packed HTR-W1 routed MoE module for ERNIE-5.5 on Strix Halo (gfx1151).

Routes:
  - Decode (T <= 8, and chunked 9..PREFILL_CUT): native K6+SAD_HI+b128 gfx1151 GEMV
    (htr_moe_ernie.so) with fused SiTU-GLU(4.0, 25.0) + fused combine.
  - Prefill (T > PREFILL_CUT): native HPF5-ERNIE fused WMMA + SAD_HI + Register-FWHT128
    kernel (hpf5_ernie.so) with ZERO BF16 weight materialization!
"""

from __future__ import annotations

import os
import sys
import torch
from torch import nn
import torch.nn.functional as F

from htr import wire

MOE_H = 2048
MOE_I = 2048
TOPK = 10
# Cap M bucket at 4 by default (93 VGPRs / 8.5 KB LDS vs 145 VGPRs / 16.9 KB LDS at M=8 -> 2.3x CU occupancy!)
# Also allows single-launch MAXT=16 (A=160<=256) with identical 36.7 MB scratch (0 extra VRAM, 2x fewer launches at T=9..16)!
MAX_M = int(os.environ.get("ERNIE_HTR_MOE_MAX_M", "4"))
if MAX_M not in (1, 2, 4, 8):
    MAX_M = 4
MAXT = 16 if MAX_M <= 4 else 8
SC_STRIDE = 6144   # w1 2048 | w3 2048 | w2 2048
SV_STRIDE = 768    # w1 256  | w3 256  | w2 256
SU_STRIDE = 512    # w1/w3 256 | w2 256

BPS_GU = int(os.environ.get("ERNIE_HTR_MOE_BPS_GU", "8"))
BPS_DN = int(os.environ.get("ERNIE_HTR_MOE_BPS_DN", "8"))
SPAD3 = int(os.environ.get("ERNIE_HTR_MOE_SPAD3", "0"))
PREFILL_CUT = int(os.environ.get("ERNIE_HTR_MOE_PREFILL_CUT", "32"))
PREFILL_TILE_E = int(os.environ.get("ERNIE_HTR_MOE_TILE_E", "16"))
USE_HPF5 = os.environ.get("ERNIE_HPF5_PREFILL", "1") == "1"
HPF5_BMW = int(os.environ.get("ERNIE_HPF5_BMW", "64"))
HPF5_NT = int(os.environ.get("ERNIE_HPF5_NT", "1"))

_EXT = None
_HPF5 = None
_HPF5_TRIED = False
_SCRATCH: dict[str, dict[str, torch.Tensor]] = {}
_HPF5_SCRATCH: dict[str, dict[str, torch.Tensor]] = {}
_PREFILL_SCRATCH: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
_CBTAB: dict[str, torch.Tensor] = {}
_CBSCALE: float | None = None


def _search_so(name: str) -> None:
    so_dir = os.environ.get("ERNIE_HTR_MOE_SO_DIR")
    if so_dir and so_dir not in sys.path:
        sys.path.insert(0, so_dir)
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    cands = [
        pkg_dir,
        os.path.join(os.path.dirname(pkg_dir), "build"),
        os.path.join(os.path.dirname(os.path.dirname(pkg_dir)), "build"),
    ]
    native_dir = os.environ.get("HTR_NATIVE_DIR")
    if native_dir:
        cands.append(native_dir)
    for cand in cands:
        if os.path.isfile(os.path.join(cand, name)) and cand not in sys.path:
            sys.path.insert(0, cand)


def _ext():
    global _EXT
    if _EXT is None:
        _search_so("htr_moe_ernie.so")
        import htr_moe_ernie
        _EXT = htr_moe_ernie
    return _EXT


def _hpf5():
    global _HPF5, _HPF5_TRIED
    if not USE_HPF5:
        return None
    if not _HPF5_TRIED:
        _HPF5_TRIED = True
        _search_so("hpf5_ernie.so")
        try:
            import hpf5_ernie
            _HPF5 = hpf5_ernie
            print(f"ERNIE_HPF5_PREFILL_ACTIVE {hpf5_ernie.__file__} bmw={HPF5_BMW}", flush=True)
        except Exception as exc:
            print(f"ERNIE_HPF5_PREFILL_FAILED {exc!r}", flush=True)
            _HPF5 = None
    return _HPF5


def _cbscale() -> float:
    global _CBSCALE
    if _CBSCALE is None:
        _CBSCALE = float(wire.cbscale())
    return _CBSCALE


def _cbtab(dev: torch.device) -> torch.Tensor:
    key = str(dev)
    tab = _CBTAB.get(key)
    if tab is None:
        tab = wire.codebook_f32(dev)
        _CBTAB[key] = tab
    return tab


def _decode_scratch(dev: torch.device) -> dict[str, torch.Tensor]:
    key = str(dev)
    s = _SCRATCH.get(key)
    if s is None:
        a = MAXT * TOPK
        m = MAX_M
        ns_gu = (MOE_H // 128 + BPS_GU - 1) // BPS_GU
        ns_dn = (MOE_I // 128 + BPS_DN - 1) // BPS_DN
        f = dict(dtype=torch.float32, device=dev)
        i32 = dict(dtype=torch.int32, device=dev)
        s = _SCRATCH[key] = dict(
            gu=torch.empty(a * m * 2 * MOE_I, **f),
            dn=torch.empty(a * m * MOE_H, **f),
            part=torch.empty(max(a * ns_gu * m * 2 * MOE_I, a * ns_dn * m * MOE_H), **f),
            cnt_gu=torch.zeros(a * (2 * MOE_I // 128), **i32),
            cnt_dn=torch.zeros(a * (MOE_H // 128), **i32),
            cnt_c2=torch.zeros(MOE_H // 128, **i32),
            grp_e=torch.empty(a, **i32),
            grp_rows=torch.empty(a * m, **i32),
            grp_n=torch.empty(a, **i32),
            slot=torch.empty(a, **i32),
            ng=torch.empty(1, **i32),
        )
    return s


def _hpf5_scratch(dev: torch.device, Pmax: int, nv: int) -> dict[str, torch.Tensor]:
    """One shared reusable HPF5 scratch across all layers (~110-330 MiB for T<=2048)."""
    key = str(dev)
    s = _HPF5_SCRATCH.get(key)
    if s is None or s["Pmax"] < Pmax or s["nv_cap"] < nv:
        Pcap = max(Pmax, 8192)
        Pcap = ((Pcap + 15) // 16) * 16
        nv_cap = max(nv, 8192)
        s = _HPF5_SCRATCH[key] = dict(
            Pmax=Pcap,
            nv_cap=nv_cap,
            sorted_ids=torch.empty(Pcap, dtype=torch.int32, device=dev),
            expert_ids=torch.empty(Pcap // 16, dtype=torch.int32, device=dev),
            ntpp=torch.empty(1, dtype=torch.int32, device=dev),
            grp_y=torch.empty(Pcap // 16, dtype=torch.int32, device=dev),
            grp_nblk=torch.empty(Pcap // 16, dtype=torch.int32, device=dev),
            grp_cnt=torch.empty(1, dtype=torch.int32, device=dev),
            A_buf=torch.empty(Pcap, MOE_H, dtype=torch.float16, device=dev),
            Y_buf=torch.empty(Pcap, 2 * MOE_I, dtype=torch.bfloat16, device=dev),
            S_buf=torch.empty(Pcap, dtype=torch.float32, device=dev),
            inv_buf=torch.empty(nv_cap, dtype=torch.int32, device=dev),
            nsat=torch.zeros(1, dtype=torch.int32, device=dev),
        )
    return s


def _prefill_scratch(dev: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Fallback BF16 dequant scratch (only allocated if ERNIE_HPF5_PREFILL=0)."""
    key = str(dev)
    b = _PREFILL_SCRATCH.get(key)
    if b is None:
        b = (
            torch.empty(PREFILL_TILE_E, 2 * MOE_I, MOE_H, dtype=torch.bfloat16, device=dev),
            torch.empty(PREFILL_TILE_E, MOE_H, MOE_I, dtype=torch.bfloat16, device=dev),
        )
        _PREFILL_SCRATCH[key] = b
    return b


def _bucket(T: int) -> int:
    b = 1 if T == 1 else 2 if T == 2 else 4 if T <= 4 else 8
    return b if b <= MAX_M else MAX_M


def _situ_glu_bf16(gu: torch.Tensor) -> torch.Tensor:
    g, u = gu.chunk(2, dim=-1)
    g32 = g.float()
    u32 = u.float()
    y = 4.0 * torch.tanh(g32 * 0.25) * torch.sigmoid(g32) * (25.0 * torch.tanh(u32 * 0.04))
    return y.to(gu.dtype)


class Ernie5_5HTRRoutedExperts(nn.Module):
    """Per-layer packed HTR-W1 routed-expert arena (512 experts x [2048, 2048])."""

    def __init__(self, num_experts: int = 512, K: int = 3) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.K = K
        mt, nt = MOE_I // 16, MOE_H // 16
        self.mt = mt
        self.nt = nt
        self.spad = SPAD3 if K == 3 else 0
        self.pitch_raw = nt * 8 * K
        self.pitch = self.pitch_raw + self.spad
        self.mat_words = mt * self.pitch

        self.register_buffer("qw", torch.empty(num_experts * 3 * self.mat_words, dtype=torch.int32))
        qoff = torch.arange(num_experts * 3, dtype=torch.int64) * self.mat_words
        self.register_buffer("qoff", qoff)
        self.register_buffer("kexp", torch.full((num_experts,), K, dtype=torch.int32))
        self.register_buffer("scales", torch.empty(num_experts * SC_STRIDE, dtype=torch.float16))
        self.register_buffer("su", torch.empty(num_experts * SU_STRIDE, dtype=torch.uint8))
        self.register_buffer("sv", torch.empty(num_experts * SV_STRIDE, dtype=torch.uint8))
        self._su13_loaded: set[int] = set()

    def load_expert_tensor(self, expert_id: int, role_name: str, leaf: str, tensor: torch.Tensor) -> None:
        role = {"w1": 0, "w3": 1, "w2": 2}[role_name]
        if leaf == "htr_qweight":
            off = (expert_id * 3 + role) * self.mat_words
            if self.spad == 0:
                flat = tensor.reshape(-1)
                assert flat.numel() == self.mat_words, (expert_id, role_name, flat.shape, self.mat_words)
                self.qw[off : off + self.mat_words].copy_(flat)
            else:
                src = tensor.reshape(self.mt, self.pitch_raw)
                dst = self.qw[off : off + self.mat_words].view(self.mt, self.pitch)
                dst[:, : self.pitch_raw].copy_(src)
        elif leaf == "htr_scales":
            off = expert_id * SC_STRIDE + role * MOE_I
            self.scales[off : off + MOE_I].copy_(tensor)
        elif leaf == "htr_sv":
            off = expert_id * SV_STRIDE + role * (MOE_I // 8)
            self.sv[off : off + (MOE_I // 8)].copy_(tensor)
        elif leaf == "htr_su":
            if role in (0, 1):
                off = expert_id * SU_STRIDE
                dst = self.su[off : off + (MOE_H // 8)]
                if expert_id not in self._su13_loaded:
                    dst.copy_(tensor)
                    self._su13_loaded.add(expert_id)
            else:
                off = expert_id * SU_STRIDE + (MOE_H // 8)
                self.su[off : off + (MOE_I // 8)].copy_(tensor)
        else:
            raise ValueError(f"Unknown HTR expert leaf: {leaf}")

    def _decode_chunk(
        self, x: torch.Tensor, ids32: torch.Tensor, w32: torch.Tensor
    ) -> torch.Tensor:
        T = x.shape[0]
        E = _ext()
        s = _decode_scratch(x.device)
        M = _bucket(T)
        cb = _cbscale()
        A = T * ids32.shape[1]
        if hasattr(E, "moe_plan2_into") and A <= s["grp_e"].numel() and A * M <= s["grp_rows"].numel():
            ge = s["grp_e"][:A]
            gr = s["grp_rows"][: A * M].view(A, M)
            gn = s["grp_n"][:A]
            slot = s["slot"][:A]
            ng = s["ng"]
            E.moe_plan2_into(ids32, M, ge, gr, gn, slot, ng)
        else:
            ge, gr, gn, slot, ng = E.moe_plan2(ids32, M)
        sp3 = self.spad if self.K == 3 else 0
        E.moe_gemv(
            0, self.K, M,
            self.qw, self.qoff, self.kexp, self.scales, self.su, self.sv,
            ge, gr, gn, x, s["part"], s["cnt_gu"], s["gu"],
            BPS_GU, cb, 0, sp3, 0, 0, 1,
        )
        return E.moe_gemv_dn_fused(
            self.K, M,
            self.qw, self.qoff, self.kexp, self.scales, self.su, self.sv,
            ge, gr, gn, s["gu"], s["part"], s["cnt_dn"], s["dn"],
            BPS_DN, cb, 0, sp3, 0, 0, 1,
            slot, w32, ng, s["cnt_c2"],
        )

    def _prefill_hpf5(
        self, x: torch.Tensor, ids32: torch.Tensor, w32: torch.Tensor
    ) -> torch.Tensor:
        T = x.shape[0]
        topk = ids32.shape[1]
        nv = T * topk
        Pmax = ((nv + self.num_experts * 15 + 15) // 16) * 16
        s = _hpf5_scratch(x.device, Pmax, nv)
        # Dynamic bmw: 32 for small prefill (T<=128, ~2.5 toks/expert), 64/128 for larger
        bmw = 32 if T <= 128 else HPF5_BMW
        return _hpf5().hpf5_ernie_forward(
            x, ids32, w32,
            self.qw, self.qoff, self.scales, self.su, self.sv,
            _cbscale(), self.K, self.spad, self.num_experts,
            s["sorted_ids"][:Pmax], s["expert_ids"][: Pmax // 16], s["ntpp"],
            s["grp_y"][: Pmax // 16], s["grp_nblk"][: Pmax // 16], s["grp_cnt"],
            s["A_buf"][:Pmax], s["Y_buf"][:Pmax], s["S_buf"][:Pmax],
            s["inv_buf"][:nv], s["nsat"],
            bmw, HPF5_NT,
        )

    def _prefill_tiled(
        self, x: torch.Tensor, ids32: torch.Tensor, w32: torch.Tensor
    ) -> torch.Tensor:
        T, H = x.shape
        topk = ids32.shape[1]
        dev = x.device
        flat_ids = ids32.reshape(-1)
        flat_w = w32.reshape(-1)
        flat_tok = torch.arange(T, device=dev, dtype=torch.int64).unsqueeze(1).expand(T, topk).reshape(-1)

        valid = flat_ids >= 0
        if not bool(valid.all()):
            flat_ids = flat_ids[valid]
            flat_w = flat_w[valid]
            flat_tok = flat_tok[valid]
        if flat_ids.numel() == 0:
            return torch.zeros_like(x)

        order = torch.argsort(flat_ids)
        sorted_ids = flat_ids[order]
        sorted_tok = flat_tok[order]
        sorted_w = flat_w[order]

        unique_e, counts = torch.unique_consecutive(sorted_ids, return_counts=True)
        unique_e_i32 = unique_e.to(torch.int32).contiguous()
        counts_cpu = counts.cpu().tolist()
        starts_cpu = [0]
        for c in counts_cpu[:-1]:
            starts_cpu.append(starts_cpu[-1] + c)

        E = _ext()
        cbtab = _cbtab(dev)
        w13_buf, w2_buf = _prefill_scratch(dev)
        out = torch.zeros(T, H, dtype=torch.float32, device=dev)

        n_active = unique_e_i32.numel()
        for t0 in range(0, n_active, PREFILL_TILE_E):
            t1 = min(n_active, t0 + PREFILL_TILE_E)
            tile_ids = unique_e_i32[t0:t1].contiguous()
            E.dq_arena_subset(
                self.qw, self.qoff, self.scales, self.su, self.sv, cbtab, tile_ids, self.K,
                0, self.spad, MOE_I, MOE_H, 0, SC_STRIDE, 0, SU_STRIDE, 0, SV_STRIDE,
                w13_buf, 2 * MOE_I * MOE_H, 0,
            )
            E.dq_arena_subset(
                self.qw, self.qoff, self.scales, self.su, self.sv, cbtab, tile_ids, self.K,
                1, self.spad, MOE_I, MOE_H, MOE_I, SC_STRIDE, 0, SU_STRIDE, MOE_I // 8, SV_STRIDE,
                w13_buf, 2 * MOE_I * MOE_H, MOE_I,
            )
            E.dq_arena_subset(
                self.qw, self.qoff, self.scales, self.su, self.sv, cbtab, tile_ids, self.K,
                2, self.spad, MOE_H, MOE_I, 2 * MOE_I, SC_STRIDE, MOE_H // 8, SU_STRIDE, 2 * MOE_I // 8, SV_STRIDE,
                w2_buf, MOE_H * MOE_I, 0,
            )
            for local_i, global_i in enumerate(range(t0, t1)):
                s0 = starts_cpu[global_i]
                s1 = s0 + counts_cpu[global_i]
                tok_idx = sorted_tok[s0:s1]
                xe = x.index_select(0, tok_idx)
                gu = F.linear(xe, w13_buf[local_i])
                h = _situ_glu_bf16(gu)
                ye = F.linear(h, w2_buf[local_i]).float()
                ye.mul_(sorted_w[s0:s1].unsqueeze(-1))
                out.index_add_(0, tok_idx, ye)

        return out.to(x.dtype)

    def forward(
        self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor
    ) -> torch.Tensor:
        T = x.shape[0]
        if T == 0:
            return torch.empty_like(x)
        x = x.contiguous()
        ids32 = topk_ids.to(torch.int32).contiguous()
        w32 = topk_weights.to(torch.float32).contiguous()
        if T <= MAXT:
            return self._decode_chunk(x, ids32, w32)
        if T > PREFILL_CUT and self.K in (2, 3) and _hpf5() is not None:
            return self._prefill_hpf5(x, ids32, w32)
        if T <= PREFILL_CUT:
            out = torch.empty_like(x)
            for i in range(0, T, MAXT):
                j = min(i + MAXT, T)
                out[i:j] = self._decode_chunk(x[i:j], ids32[i:j], w32[i:j])
            return out
        return self._prefill_tiled(x, ids32, w32)

