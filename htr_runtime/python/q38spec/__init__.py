"""E21 q38spec: env-gated speculative-decoding draft-policy patches for vLLM (general plugin; inert unless an env flag is set).

Q38SPEC_DRAFT_FILTER=1
    Stock vLLM probabilistic drafting (draft_sample_method="probabilistic") applies ONLY temperature to the draft logits and ignores
    top_k/top_p (upstream NOTE(woosuk) "may slightly degrade the acceptance rate"). The target, in rejection sampling, uses
    temperature + min_p + top_k + top_p, so every unit of draft mass outside the target's filtered support is guaranteed rejection.
    This patch applies the request's top_k/top_p to the draft logits, in the target order (temperature -> top-k -> top-p on the
    post-top-k softmax). Lossless: the draft token is drawn from, and the rejection sampler is given, exactly the same filtered
    distribution. Only draft_sample_method="probabilistic" is affected (greedy drafting never reaches the patched branch).

    v2 (2026-10-25): two targets.
      (a) Model runner V2 (vllm.v1.worker.gpu, the path our serving stack uses): Speculator.sample_draft caches PRE-temperature
          draft logits (logits_cache) which the rejection kernels later scale by 1/T. We mask the raw logits with -inf outside the
          filtered set BEFORE gumbel_sample, so the cache holds the masked logits and q = softmax(masked/T) exactly.
          CUDA-graph safe: static shapes (top-KC candidates, KC=64), no host syncs during capture; per-request top_k/top_p copied
          into static speculator buffers in _copy_request_inputs (eager, like temperature; UvaBackedTensor.gpu rotates buffers so
          it must not be read from inside a graph). Rows with top_k > KC (or top_k disabled) are left unfiltered (still lossless).
      (b) Legacy V1 llm_base_proposer.compute_probs_and_sample_next_token (kept from v1; unused by MRV2).
    Not mirrored: logits processors/penalties, min_p (target-only; lossless either way).
"""
import os
import sys

_PATCHED = False
KC = 64
_STATE = {"sampling_states": None}


def _log(msg):
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------- shared mask (pure torch, graph-capturable)
def q38_filter_logits(logits, idx_mapping, temperature, top_k, top_p, kc=KC):
    """Return logits with -inf outside each row's top-k/top-p set (computed on logits/T, vLLM order and tie rule).

    logits [rows, V] (any float dtype; returned in the same dtype); idx_mapping [rows] request-state index (-1 = pad row);
    temperature/top_k/top_p [max_num_reqs] indexed by request-state index. Rows with idx<0 or top_k>kc are returned unchanged.
    """
    import torch
    rows, V = logits.shape
    kc = min(kc, V)
    safe = idx_mapping.clamp_min(0).long()
    k = top_k[safe].long()
    p = top_p[safe].float()
    t = temperature[safe].float()
    t = torch.where(t < 1e-5, torch.ones_like(t), t)          # greedy rows: argmax is always kept, scale irrelevant
    vals, ids = torch.topk(logits, kc, dim=-1)                # sorted descending
    v = vals.float() / t[:, None]
    kk = k.clamp(1, kc)
    kth = v.gather(1, (kk - 1)[:, None])
    keep = v >= kth                                           # vLLM top-k keeps ties with the k-th value
    vm = v.masked_fill(~keep, float("-inf"))
    pr = vm.softmax(dim=-1)
    tail = pr.flip(-1).cumsum(-1).flip(-1)                    # = vLLM ascending cumsum at each position
    rm = tail <= (1.0 - p)[:, None]
    rm[:, 0] = False                                          # keep at least the top-1
    keep = keep & ~rm
    neg = torch.full_like(vals, float("-inf"))
    out = torch.full_like(logits, float("-inf"))
    out.scatter_(1, ids, torch.where(keep, vals, neg))
    filt = (k <= kc) & (idx_mapping >= 0)
    return torch.where(filt[:, None], out, logits)


# ---------------------------------------------------------------- (a) model runner V2
def _patch_mrv2():
    import torch
    from vllm.v1.worker.gpu.sample import states as st_mod
    from vllm.v1.worker.gpu.spec_decode import speculator as sp_mod

    orig_states_init = st_mod.SamplingStates.__init__

    def states_init(self, *a, **kw):
        orig_states_init(self, *a, **kw)
        _STATE["sampling_states"] = self

    st_mod.SamplingStates.__init__ = states_init

    Spec = sp_mod.DraftModelSpeculator
    orig_copy = Spec._copy_request_inputs
    orig_sample = Spec.sample_draft
    calls = {"n": 0, "eager": 0}

    def _bufs(self):
        b = getattr(self, "_q38_topk", None)
        if b is None:
            n = self.temperature.shape[0]
            self._q38_topk = torch.full((n,), 1 << 30, dtype=torch.int32, device=self.temperature.device)
            self._q38_topp = torch.ones((n,), dtype=torch.float32, device=self.temperature.device)
        return self._q38_topk, self._q38_topp

    def _copy_request_inputs(self, num_reqs, idx_mapping, temperature, seeds, dummy_run=False):
        orig_copy(self, num_reqs, idx_mapping, temperature, seeds, dummy_run=dummy_run)
        tk, tp = _bufs(self)
        ss = _STATE["sampling_states"]
        if ss is not None and not dummy_run:
            tk.copy_(ss.top_k.gpu[: tk.shape[0]], non_blocking=True)
            tp.copy_(ss.top_p.gpu[: tp.shape[0]], non_blocking=True)

    def sample_draft(self, hidden_states, sample_src_positions, idx_mapping, temperature, seeds, draft_step, draft_logits):
        if draft_logits is None:
            return orig_sample(self, hidden_states, sample_src_positions, idx_mapping, temperature, seeds, draft_step,
                               draft_logits)
        tk, tp = _bufs(self)
        logits = self.model.compute_logits(hidden_states)
        logits = q38_filter_logits(logits, idx_mapping, self.temperature, tk, tp)
        sampled = sp_mod.gumbel_sample(
            logits, idx_mapping, temperature, seeds, sample_src_positions,
            apply_temperature=True, is_drafting=True, logits_cache=draft_logits, logits_cache_col=draft_step,
            use_fp64=self.use_fp64_gumbel)
        if self.draft_watermarker is not None:
            sampled = self.draft_watermarker.sample(logits, sampled, idx_mapping, temperature)
        self._maybe_predict_acceptance(logits, idx_mapping, draft_step)
        capturing = torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
        if calls["n"] < 4:
            calls["n"] += 1
            _log(f"Q38SPEC draft_filter call mrv2 rows={logits.shape[0]} capturing={capturing} "
                 f"top_k={'set' if _STATE['sampling_states'] is not None else None} cls={type(self).__name__}")
        elif not capturing and calls["eager"] < 6 and int(idx_mapping[0]) >= 0:
            calls["eager"] += 1
            r = int(idx_mapping[0])
            _log(f"Q38SPEC draft_filter eager row0 req={r} top_k={int(tk[r])} top_p={float(tp[r]):.3f} "
                 f"T={float(self.temperature[r]):.3f} kept={int(torch.isfinite(logits[0]).sum())}")
        return sampled

    Spec._copy_request_inputs = _copy_request_inputs
    Spec.sample_draft = sample_draft
    _log("Q38SPEC draft_filter PATCHED mrv2 DraftModelSpeculator.sample_draft/_copy_request_inputs + SamplingStates")


# ---------------------------------------------------------------- (b) legacy V1 proposer (unchanged from v1)
def _expand(t, factor):
    return None if t is None or factor == 1 else t.repeat_interleave(factor, dim=0)


def make_filtered_sampler(mod, apply_top_k_top_p):
    """Return a drop-in replacement for llm_base_proposer.compute_probs_and_sample_next_token."""
    import torch
    _SAMPLING_EPS = 1e-5
    empty_noise = mod.empty_exponential_noise_like
    sample_noise = mod.sample_with_exponential_noise
    calls = {"n": 0}

    def compute_probs_and_sample_next_token(logits, sampling_metadata, use_fp64_gumbel=False):
        if sampling_metadata.all_greedy:
            return logits.argmax(dim=-1), logits
        assert sampling_metadata.temperature is not None
        temperature = sampling_metadata.temperature
        rows = logits.shape[0]
        factor = rows // sampling_metadata.top_k.shape[0] if sampling_metadata.top_k is not None else (
            rows // sampling_metadata.top_p.shape[0] if sampling_metadata.top_p is not None else 1)
        top_k = sampling_metadata.top_k if factor == 1 else _expand(sampling_metadata.top_k, factor)
        top_p = sampling_metadata.top_p if factor == 1 else _expand(sampling_metadata.top_p, factor)
        is_greedy = None
        if not sampling_metadata.all_random:
            is_greedy = temperature < _SAMPLING_EPS
            temperature = torch.where(is_greedy, 1.0, temperature)
        logits.div_(temperature.view(-1, 1))
        logits = apply_top_k_top_p(logits, top_k, top_p)
        probs = logits.softmax(dim=-1, dtype=torch.float32)
        q = empty_noise(probs, use_fp64_gumbel)
        q.exponential_()
        next_token_ids = sample_noise(probs.clone(), q)
        if is_greedy is not None:
            next_token_ids = torch.where(is_greedy, probs.argmax(dim=-1), next_token_ids)
        if calls["n"] < 3:
            calls["n"] += 1
            _log(f"Q38SPEC draft_filter call v1 rows={rows} factor={factor} top_k={'set' if top_k is not None else None} "
                 f"top_p={'set' if top_p is not None else None} support={int((probs[0] > 0).sum())}")
        return next_token_ids, probs

    return compute_probs_and_sample_next_token


def register():
    global _PATCHED
    if _PATCHED or os.environ.get("Q38SPEC_DRAFT_FILTER") != "1":
        return
    try:
        _patch_mrv2()
    except Exception as e:  # fail loudly in the log; the Q2 gate then fails the arm
        _log(f"Q38SPEC draft_filter MRV2 PATCH FAILED: {e!r}")
    from vllm.v1.spec_decode import llm_base_proposer as mod
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
    mod.compute_probs_and_sample_next_token = make_filtered_sampler(mod, apply_top_k_top_p)
    _PATCHED = True
    _log("Q38SPEC draft_filter PATCHED llm_base_proposer.compute_probs_and_sample_next_token")
