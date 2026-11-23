"""E22 HIP routes for packed GPTQ8-HC connections (fork gp8hc_e22p of gp8hc_e22).

K1c decode (GP8HC_HIP=1; 1 <= M <= 8): hc_combine_norm -> hiphc.down_silu -> hiphc.up_gatemix   (unchanged from gp8hc_e22).
K1d prefill (GP8HC_HIPP=1; GP8HC_HIPP_MIN <= M <= 12288): the image's HCV3 v3g pipeline with GPTQ8 weights dequantised IN-KERNEL:
  direct_hc_v3.combine_norm_pack(ly=1) (unchanged Triton, = BF16-lane route) -> hiphcp.dn5q (v3g hc_dn5 port, U8+sz B operand)
  -> hiphcp.ugmqq (v3g hc_ugmq port, U8+sz B operand).  Per element w = bf16_rne((q - z) * s): bitwise == v3g on reconstructed
  weights (bench_hiphcp.py).  No BF16 weight copy.  Both routes hook direct_nb1's module globals _hcv3_ok/_hcv3_run, looked up at
  call time inside the opaque qwen4_exp_nb2_hc_combine_and_mix op.  Everything else (9 <= M < MIN, BF16 layers, draft, first/final
  mixers) is untouched.  Defaults off.
"""
import json, os, sys
import torch

ENABLED = os.environ.get('GP8HC_HIP', '0') == '1'
DCFG = {int(k): int(v) for k, v in json.loads(os.environ.get('GP8HC_HIP_DCFG', '{"1":4,"2":2,"3":2,"4":2,"5":2,"6":2,"7":2,"8":4}')).items()}
UVAR = int(os.environ.get('GP8HC_HIP_UVAR', '1'))
PENABLED = os.environ.get('GP8HC_HIPP', '0') == '1'
PMIN = int(os.environ.get('GP8HC_HIPP_MIN', '256'))
PMAX = 12288
# [[lo, hi, RG, KS], ...] for dn5q; default = the image HCV3 table, (4,4) below 1200
PTAB = json.loads(os.environ.get('GP8HC_HIPP_TABLE', '[[256,1200,4,4],[1200,1800,4,4],[1800,2900,8,2],[2900,3850,4,4],[3850,5200,8,1],'
                                                     '[5200,7168,4,2],[7168,12289,8,2]]'))
PLYB = int(os.environ.get('GP8HC_HIPP_LYB', '0'))
PUP = json.loads(os.environ.get('GP8HC_HIPP_UP', '[8,0]'))  # ugmqq MW, RF
# V5-A4: opt-in closing of the M=9..255 HC dead zone ('off' | 'hipp' | 'chunk8')
M9_MODE = os.environ.get('GP8HC_HIP_M9_MODE', 'off')
if M9_MODE not in ('off', 'hipp', 'chunk8'):
    raise ValueError(f'GP8HC_HIP_M9_MODE must be off/hipp/chunk8, got {M9_MODE!r}')
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BUILD = os.path.join(_HERE, 'hiphc_build')
_PBUILD = os.path.join(_HERE, 'hiphcp_build')
_mod = [None]
_pmod = [None]
_seen = set()
_pseen = set()
_rep = {}


def pack_sz(s, z):
    return ((s.view(torch.int16).to(torch.int32) & 0xFFFF) | (z.to(torch.int32) << 16)).contiguous()


def _hip():
    if _mod[0] is None:
        sys.path.insert(0, _BUILD)
        import hiphc
        _mod[0] = hiphc
        print('GP8HC_HIP_LOAD ' + json.dumps(dict(build=json.load(open(os.path.join(_BUILD, 'BUILD.json'))), dcfg=DCFG, uvar=UVAR)), flush=True)
    return _mod[0]


def _hipp():
    if _pmod[0] is None:
        sys.path.insert(0, _PBUILD)
        import hiphcp
        _pmod[0] = hiphcp
        print('GP8HC_HIPP_LOAD ' + json.dumps(dict(build=json.load(open(os.path.join(_PBUILD, 'BUILD.json'))), pmin=PMIN, table=PTAB,
                                                   lyb=PLYB, up=PUP)), flush=True)
    return _pmod[0]


def _pcfg(M):
    for lo, hi, rg, ks in PTAB:
        if lo <= M < hi:
            return rg, ks
    return None


def _m9_ok(M, K, merged_w, up_w, hidden, hc):
    if M9_MODE == 'off' or not (9 <= M < PMIN):
        return False
    if M9_MODE == 'chunk8' and M > 64:
        return False
    return (hc == 4 and merged_w.dtype == torch.uint8 and up_w.dtype == torch.uint8
            and tuple(merged_w.shape) == (336, K) and tuple(up_w.shape) == (K, 320)
            and hidden.dtype == torch.bfloat16 and hidden.is_contiguous() and K % 256 == 0)


def _p_ok(M, K, merged_w, up_w, hidden, hc):
    return (PENABLED and PMIN <= M <= PMAX and _pcfg(M) is not None and hc == 4 and merged_w.dtype == torch.uint8
            and up_w.dtype == torch.uint8 and tuple(merged_w.shape) == (336, K) and tuple(up_w.shape) == (K, 320)
            and hidden.dtype == torch.bfloat16 and hidden.is_contiguous() and K % 256 == 0)


def install():
    if not (ENABLED or PENABLED):
        return
    import direct_nb1 as nb
    if getattr(nb._hcv3_ok, '_gp8hip', False):
        return
    old_ok, old_run = nb._hcv3_ok, nb._hcv3_run

    def ok(hidden, blk, inj, norm_w, merged_w, up_w, hc):
        M, K = hidden.shape
        if (ENABLED and 1 <= M <= 8 and hc == 4 and merged_w.dtype == torch.uint8 and up_w.dtype == torch.uint8
                and tuple(merged_w.shape) == (336, K) and tuple(up_w.shape) == (K, 320)
                and hidden.dtype == torch.bfloat16 and hidden.is_contiguous()):
            return True
        if _m9_ok(M, K, merged_w, up_w, hidden, hc) or _p_ok(M, K, merged_w, up_w, hidden, hc):
            return True
        return old_ok(hidden, blk, inj, norm_w, merged_w, up_w, hc)

    def run(hidden, blk, inj, norm_w, merged_w, up_w, eps, hc, layer_id):
        M, K = hidden.shape
        if merged_w.dtype != torch.uint8:
            return old_run(hidden, blk, inj, norm_w, merged_w, up_w, eps, hc, layer_id)
        mod = nb._HC_LAYERS[layer_id]
        dn, up = mod.input_mix_weight_down_block_inject, mod.input_mix_weight_up
        if not (hasattr(dn, 'gp8_sz') and hasattr(up, 'gp8_sz')):
            raise RuntimeError(f'GP8HC_HIP: layer {layer_id} has uint8 carriers but no packed sz buffers')
        if ENABLED and 1 <= M <= 8:
            from vllm.models.qwen4_exp.amd.ops.hc import hc_combine_norm
            h = _hip()
            out, xn = hc_combine_norm(hidden, blk, inj, norm_w, eps, hc)
            lora, d = h.down_silu(xn, dn.gp8_qweight, dn.gp8_sz, 336, mod.lora_rank, hc, DCFG[M])
            y = h.up_gatemix(lora, up.gp8_qweight, up.gp8_sz, xn, hc, UVAR)
            if M not in _seen:
                _seen.add(M)
                print('GP8HC_HIP_ROUTE ' + json.dumps(dict(m=M, layer_id=layer_id, dcfg=DCFG[M], uvar=UVAR)), flush=True)
            return out, y, d[:, mod.lora_rank:mod.lora_rank + hc]
        if _m9_ok(M, K, merged_w, up_w, hidden, hc):
            if M9_MODE == 'chunk8':
                from vllm.models.qwen4_exp.amd.ops.hc import hc_combine_norm
                h = _hip()
                out, xn = hc_combine_norm(hidden, blk, inj, norm_w, eps, hc)
                ys, ds = [], []
                for r0 in range(0, M, 8):
                    r1 = min(M, r0 + 8)
                    ms = r1 - r0
                    xn_s = xn[r0:r1].contiguous()
                    l_s, d_s = h.down_silu(xn_s, dn.gp8_qweight, dn.gp8_sz, 336, mod.lora_rank, hc, DCFG[ms])
                    ys.append(h.up_gatemix(l_s, up.gp8_qweight, up.gp8_sz, xn_s, hc, UVAR))
                    ds.append(d_s[:, mod.lora_rank:mod.lora_rank + hc])
                if ('m9_chunk8', M) not in _seen:
                    _seen.add(('m9_chunk8', M))
                    print('GP8HC_HIP_M9_ROUTE ' + json.dumps(dict(mode='chunk8', m=M, layer_id=layer_id)), flush=True)
                return out, torch.cat(ys, dim=0), torch.cat(ds, dim=0)
            # M9_MODE == 'hipp': single-pass WMMA via hiphcp (RG=4, KS=4, MW=4)
            hp = _hipp()
            H = K // hc; R = mod.lora_rank; dev = hidden.device
            out, xb = nb._hv().combine_norm_pack(hidden, blk, inj, norm_w, eps, hc, ly=1)
            if dev not in nb._DUMMY:
                nb._DUMMY[dev] = (torch.empty(nb.HCV3_MAX_KS * nb.HCV3_MAX_M * 336, dtype=torch.float32, device=dev),
                                  torch.zeros(4096, dtype=torch.int32, device=dev))
            part, cnt = nb._DUMMY[dev]
            RG, KS = 4, 4
            q, z = (dn.gp8_qweight, dn.gp8_sz)
            lora = torch.empty((M, R), dtype=torch.bfloat16, device=dev)
            injb = torch.empty((M, 336), dtype=torch.bfloat16, device=dev)
            hp.dn5q(xb, q, z, dn.gp8_qweight.shape[0], lora, injb, part, cnt, M, K, R, hc, RG, KS, 0)
            y = torch.empty((M, H), dtype=torch.bfloat16, device=dev)
            hp.ugmqq(lora, up.gp8_qweight, up.gp8_sz, xb, y, M, H, hc, R, 4, 0, 1)
            if ('m9_hipp', M) not in _pseen:
                _pseen.add(('m9_hipp', M))
                print('GP8HC_HIP_M9_ROUTE ' + json.dumps(dict(mode='hipp', m=M, layer_id=layer_id, rg=RG, ks=KS, mw=4)), flush=True)
            return out, y, injb[:, :hc]
        if not _p_ok(M, K, merged_w, up_w, hidden, hc):
            return old_run(hidden, blk, inj, norm_w, merged_w, up_w, eps, hc, layer_id)
        hp = _hipp()
        H = K // hc; R = mod.lora_rank; dev = hidden.device
        out, xb = nb._hv().combine_norm_pack(hidden, blk, inj, norm_w, eps, hc, ly=1)
        if dev not in nb._DUMMY:  # shared with the BF16 HCV3 route: fp32 split-K partials + self-resetting counters
            nb._DUMMY[dev] = (torch.empty(nb.HCV3_MAX_KS * nb.HCV3_MAX_M * 336, dtype=torch.float32, device=dev),
                              torch.zeros(4096, dtype=torch.int32, device=dev))
        part, cnt = nb._DUMMY[dev]
        RG, KS = _pcfg(M)
        if PLYB == 1:
            if layer_id not in _rep:
                _rep[layer_id] = hp.repack_dn(dn.gp8_qweight, dn.gp8_sz)
            q, z = _rep[layer_id]
        else:
            q, z = dn.gp8_qweight, dn.gp8_sz
        lora = torch.empty((M, R), dtype=torch.bfloat16, device=dev)
        injb = torch.empty((M, 336), dtype=torch.bfloat16, device=dev)
        hp.dn5q(xb, q, z, dn.gp8_qweight.shape[0], lora, injb, part, cnt, M, K, R, hc, RG, KS, PLYB)
        y = torch.empty((M, H), dtype=torch.bfloat16, device=dev)
        hp.ugmqq(lora, up.gp8_qweight, up.gp8_sz, xb, y, M, H, hc, R, PUP[0], PUP[1], 1)
        key = (RG, KS)
        if key not in _pseen:
            _pseen.add(key)
            print('GP8HC_HIPP_ROUTE ' + json.dumps(dict(m=M, layer_id=layer_id, rg=RG, ks=KS, lyb=PLYB, up=PUP)), flush=True)
        return out, y, injb[:, :hc]

    ok._gp8hip = True
    nb._hcv3_ok, nb._hcv3_run = ok, run
    print('GP8HC_HIP_INSTALL ' + json.dumps(dict(decode=ENABLED, prefill=PENABLED, dcfg=DCFG, uvar=UVAR, pmin=PMIN)), flush=True)
