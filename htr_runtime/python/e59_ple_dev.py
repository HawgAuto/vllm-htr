# SPDX-License-Identifier: Apache-2.0
"""E59 (2026-10-26) asynchronous hoisted PLE gather for DECODE-sized steps (graph replay), GPU-side wait.

E15 hoist (decode): prepare_inputs gathers the step's PLE rows synchronously (blocking D2H of the ngram ids -> pread -> H2D) before
the graph replay is launched, so the GPU idles for the whole host round trip every token (E57: -0.56 ms ITL when skipped).
E59 for steps with n <= E59_PLE_DEV_MAXN tokens (native arm only):
  prepare_inputs: ngram ids (GPU, unchanged) -> non_blocking D2H into pinned ids + event; k_set_exp(seq, n16) on the stream;
                  native C++ worker thread (no GIL): hipEventSynchronize(event); E27 ple_gather_ids (the function the Python
                  native arm calls, same table/threads arguments) straight into a COHERENT pinned rows buffer; ctrl[0] = seq.
  PLE layer:      torch.ops.e59.ple_dev_wait(rows, hidden_states) launches k_wait_copy (captured in the decode graph): spins on
                  ctrl[0] == exp[0] (system-scope acquire), then copies exp[1] 16-byte chunks into the device rows buffer.
Every other step calls mark_sync() (stream-ordered n16 = 0 -> the kernel neither waits nor copies). Only async producers publish
ctrl[0]; a host-side sync publication could overwrite readiness before the preceding GPU consumer finishes. Bytes identical to the sync path
(same lookup); the first E59_PLE_DEV_CHECKS async steps are self-checked (torch.equal vs the synchronous reference) and abort on
mismatch. A 2 s GPU timeout sets ctrl[1]; the host raises on the next step.
"""
import json
import os
import concurrent.futures as _cf

import numpy as np
import torch

ENABLED = os.environ.get("E59_PLE_DEV", "0") == "1"
MAXN = int(os.environ.get("E59_PLE_DEV_MAXN", "4"))
CHECKS = int(os.environ.get("E59_PLE_DEV_CHECKS", "3"))
SO = os.environ.get("E59_PLE_DEV_SO", "/opt/e59/e59_pledev_ext.so")
_ext = [None]
_pool = [None]
_B = {}
_S = dict(seq=0, steps=0, syncs=0, checks=0, pending=None, check=None)


def _x():
    if _ext[0] is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location("e59_pledev_ext", SO)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        _ext[0] = m
    return _ext[0]


def _buffers():
    if not _B:
        x = _x()
        ctrl = x.alloc_coherent(64)                       # int32[16]: [0] ready seq, [1] GPU timeout flag
        rows_h = x.alloc_coherent(MAXN * 2560 * 2)        # bf16 rows of up to MAXN tokens (16 n-gram rows x 160 each)
        _B.update(ctrl=ctrl, ctrl_d=x.device_ptr(ctrl), rows_h=rows_h, rows_d=x.device_ptr(rows_h),
                  ids_h=torch.empty(MAXN * 16, dtype=torch.int64, pin_memory=True),
                  exp=torch.zeros(2, dtype=torch.int32, device="cuda"))
        _B["err_d"] = _B["ctrl_d"] + 4
        print("E59_PLE_DEV_INIT " + json.dumps(dict(maxn=MAXN, checks=CHECKS, so=SO)), flush=True)
    return _B


def _drain():
    if _S["pending"] is not None:
        _x().worker_wait_idle()
        _S["pending"] = None
    b = _B
    if b:
        e = _x().host_load(b["ctrl"], 1)
        if e:
            raise RuntimeError(f"E59: decode PLE failure ctrl[1]={e} (1 = GPU wait timeout, 2 = gather rc {_x().host_load(b['ctrl'], 2)})")


def _start_worker(reader):
    import direct_ple_ab_v1 as P
    import ctypes
    lib = P._lib()
    if not getattr(lib, "_e27_ids", False):
        raise RuntimeError("E59 needs the E27 v2 native library (ple_gather_ids)")
    locs = reader._locations
    tab = dict(fds=np.array([loc.fd for loc in locs], np.int32), base=np.array([loc.offset for loc in locs], np.int64),
               rows=np.array([loc.rows for loc in locs], np.int64),
               shard=(reader._num_embeddings + reader._parts - 1) // reader._parts)
    fn = ctypes.cast(lib.ple_gather_ids, ctypes.c_void_p).value
    b = _B
    _x().worker_start(fn, tab["fds"].ctypes.data, tab["base"].ctypes.data, tab["rows"].ctypes.data, len(tab["fds"]), tab["shard"],
                      reader._row_bytes, reader._num_embeddings, b["rows_h"], b["ctrl"], P._threads(), P.MIN_RUNS_PER_THREAD)
    b["tab"] = tab
    b["reader"] = reader
    print("E59_PLE_DEV_WORKER " + json.dumps(dict(parts=len(tab["fds"]), shard=int(tab["shard"]), row_bytes=int(reader._row_bytes),
                                                  threads=int(P._threads()), min_runs=int(P.MIN_RUNS_PER_THREAD))), flush=True)


def _fstat_check(reader):
    for loc in reader._locations:  # same identity guard as the native arm (every DIRECT_PLE_FSTAT_EVERY calls)
        st = os.fstat(loc.fd)
        if st.st_dev != loc.device or st.st_ino != loc.inode or st.st_size != loc.file_size:
            raise OSError(f"safetensors shard identity changed: {loc.path}")


def _selfcheck():
    ck = _S["check"]
    if ck is None:
        return
    _S["check"] = None
    rows, emb, ngram_ids, n = ck
    torch.cuda.synchronize()
    ref = emb.ngram_embedding(ngram_ids).flatten(-2)
    ok = bool(torch.equal(ref.view(torch.int16), rows[:n].view(torch.int16)))
    _S["checks"] += 1
    print("E59_PLE_DEV_SELFCHECK " + json.dumps(dict(n=n, equal=ok, check=_S["checks"], step=_S["steps"])), flush=True)
    if not ok:
        raise RuntimeError("E59 async decode PLE rows differ from the synchronous hoist gather")


def mark_sync():
    """Every non-E59 step (prefill, E53 async, sync fallback, dummy runs): the GPU kernel must neither wait nor copy."""
    if not ENABLED:
        return
    _drain()
    _selfcheck()
    b = _buffers()
    x = _x()
    _S["seq"] += 1
    s = _S["seq"]
    # Stream-order the no-copy marker; do NOT publish host readiness here.
    # The preceding GPU consumer may still be waiting on the last async seq.
    # Only its producer may publish ctrl[0]; n16==0 consumers bypass the wait.
    x.set_exp(b["exp"], s, 0)
    _S["syncs"] += 1


def submit_decode(emb, ngram_ids, rows, n):
    if not ENABLED or n > MAXN or n <= 0:
        return False
    import direct_ple_ab_v1 as P
    arm, case = P.read_arm()
    if arm != "native":
        return False
    _drain()
    _selfcheck()
    b = _buffers()
    x = _x()
    reader = emb.ngram_embedding.reader
    flat = ngram_ids.detach().reshape(-1)
    m = int(flat.numel())
    dim = emb.ngram_embedding.embedding_dim
    if m != n * 16 or dim * 16 != rows.shape[-1] or rows.shape[-1] * 2 * n > MAXN * 2560 * 2:
        return False
    ids_h = b["ids_h"][:m]
    ids_h.copy_(flat, non_blocking=True)
    ev = torch.cuda.Event()
    ev.record()
    _S["seq"] += 1
    s = _S["seq"]
    x.set_exp(b["exp"], s, n * rows.shape[-1] * 2 // 16)
    if "reader" not in b:
        _start_worker(reader)
    if b["reader"] is not reader:
        raise RuntimeError("E59: PLE reader object changed")
    if _S["steps"] % P.FSTAT_EVERY == 0:
        _fstat_check(reader)
    x.worker_submit(s, ev.cuda_event, ids_h.data_ptr(), m)
    _S["pending"] = ev            # keeps the event alive until the worker finished
    _S["steps"] += 1
    if _S["checks"] < CHECKS:
        _S["check"] = (rows, emb, ngram_ids.detach().clone(), n)
    if _S["steps"] <= 3 or _S["steps"] % 1024 == 0:
        print("E59_PLE_DEV_STEP " + json.dumps(dict(step=_S["steps"], syncs=_S["syncs"], n=n, seq=s, worker=x.worker_stats())), flush=True)
    return True


@torch.library.custom_op("e59::ple_dev_wait", mutates_args=("rows",))
def ple_dev_wait(rows: torch.Tensor, dep: torch.Tensor) -> None:
    if not ENABLED:
        return
    b = _buffers()
    _x().wait_copy(rows, b["ctrl_d"], b["exp"], b["rows_d"], b["err_d"])


@ple_dev_wait.register_fake
def _(rows, dep):
    return None
