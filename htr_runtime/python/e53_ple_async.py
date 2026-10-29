# SPDX-License-Identifier: Apache-2.0
"""E53 (2026-10-26) asynchronous hoisted PLE gather for prefill-sized steps.

The E15 hoist gathers the PLE rows of a step in prepare_inputs (blocking D2H ids -> pread -> H2D) before any GPU work of the
forward is queued, so the GPU idles for the whole host gather (warm ~7 ms @2K, ~28 ms @8K; cold 0.2-1 s). The PLE layer is
layer 2, so layers 0-1 do not need the rows. For steps with >= E53_PLE_ASYNC_MIN tokens:

  prepare_inputs: ngram ids on GPU (unchanged) -> non_blocking D2H into a pinned buffer + event -> worker thread:
                  event.synchronize(); native gather (same C library / reader / shard map as the sync path) into a pinned rows
                  buffer. The device rows buffer is handed to the forward unfilled.
  PLE layer:      torch.ops.e53.ple_wait(rows, hidden_states) waits for the worker, then H2D-copies the pinned rows into the
                  device buffer on the current stream. hidden_states (layer-1 output) is a data dependency, so the wait is
                  ordered after layers 0-1 have been queued: their kernels run while the host gathers.

Bytes are identical to the sync path (same lookup); the first async step per process is self-checked against the sync gather
(torch.equal) and aborts on mismatch. Decode-sized steps (< min tokens; graph replay) keep the synchronous hoist unchanged.
"""
import json
import os
import threading
import time
import concurrent.futures as _cf

import numpy as np
import torch

ASYNC_MIN = int(os.environ.get("E53_PLE_ASYNC_MIN", "0"))      # 0 = disabled (module inert)
_POOL = None
_PENDING = []     # single slot: [(future, n, pinned rows view, emb, ngram_ids)] for the step in flight (max_num_seqs=1 lane)
_STATE = dict(steps=0, selfchecked=False)
_LOCK = threading.Lock()


def enabled(n):
    return ASYNC_MIN > 0 and n >= ASYNC_MIN


def _pool():
    global _POOL
    if _POOL is None:
        _POOL = _cf.ThreadPoolExecutor(max_workers=1, thread_name_prefix="e53-ple")
    return _POOL


class Buffers:
    def __init__(self, max_rows, dim):
        self.ids = torch.empty(max_rows, dtype=torch.int64, pin_memory=True)
        self.rows = torch.empty((max_rows // 16, dim * 16), dtype=torch.bfloat16, pin_memory=True)


def submit(emb, ngram_ids, rows_dev, bufs):
    """Queue the async gather for one step. ngram_ids [n, 16] int64 on GPU; rows_dev [n, 2560] device buffer slice."""
    import direct_ple_ab_v1 as P
    arm, case = P.read_arm()
    if arm != "native":          # async path replicates only the native exact-byte gather; anything else stays synchronous
        return False
    reader = emb.ngram_embedding.reader
    flat = ngram_ids.detach().reshape(-1)
    m = int(flat.numel())
    n = m // 16
    ids_h = bufs.ids[:m]
    ids_h.copy_(flat, non_blocking=True)
    ev = torch.cuda.Event()
    ev.record()
    rows_h = bufs.rows[:n]
    dim = emb.ngram_embedding.embedding_dim

    def job():
        t0 = time.perf_counter()
        ev.synchronize()
        t1 = time.perf_counter()
        ids_np = ids_h.numpy()
        out = rows_h.view(torch.int16).numpy().reshape(m, dim)
        mb = getattr(reader, "max_batch_rows", 65536)
        for s in range(0, m, mb):
            out[s:s + mb] = P.gather_rows_native(reader, ids_np[s:s + mb]).view(np.int16)
        t2 = time.perf_counter()
        P.record("native", case, m, (t2 - t0) * 1e3)
        return dict(wait_ms=round((t1 - t0) * 1e3, 3), gather_ms=round((t2 - t1) * 1e3, 3), ids=m)

    with _LOCK:
        if _PENDING:
            raise RuntimeError("E53: previous async PLE step was never consumed by the PLE layer")
        _PENDING.append((_pool().submit(job), n, rows_h, emb, ngram_ids))
    _STATE["steps"] += 1
    return True


@torch.library.custom_op("e53::ple_wait", mutates_args=("rows",))
def ple_wait(rows: torch.Tensor, dep: torch.Tensor) -> None:
    # keyed by step, not data_ptr: functionalization may hand the op a clone of the rows buffer (copied back afterwards)
    with _LOCK:
        ent = _PENDING.pop() if _PENDING else None
    if ent is None:
        return
    fut, n, rows_h, emb, ngram_ids = ent
    if rows.shape[0] < n:
        raise RuntimeError(f"E53: PLE rows buffer has {rows.shape[0]} rows, async step gathered {n}")
    t0 = time.perf_counter()
    info = fut.result()
    blocked = (time.perf_counter() - t0) * 1e3
    rows[:n].copy_(rows_h, non_blocking=True)
    if not _STATE["selfchecked"]:
        ref = emb.ngram_embedding(ngram_ids).flatten(-2)          # synchronous reference (same reader path as the E15 hoist)
        ok = bool(torch.equal(ref.view(torch.int16), rows[:n].view(torch.int16)))
        print("E53_PLE_ASYNC_SELFCHECK " + json.dumps(dict(tokens=n, equal=ok, **info)), flush=True)
        if not ok:
            raise RuntimeError("E53 async PLE rows differ from the synchronous hoist gather")
        _STATE["selfchecked"] = True
    if _STATE["steps"] <= 3 or _STATE["steps"] % 256 == 0:
        print("E53_PLE_ASYNC_STEP " + json.dumps(dict(step=_STATE["steps"], tokens=n, host_blocked_ms=round(blocked, 3), **info)),
              flush=True)


@ple_wait.register_fake
def _(rows, dep):
    return None
