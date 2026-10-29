# SPDX-License-Identifier: Apache-2.0
"""Direct-session PLE SSD gather A/B. Private sequential benchmark interface.

The native path returns exactly the bytes PLESafetensorsReader.gather_bytes
would return for the same row IDs: identical shard/offset mapping, buffered
pread, no row cache. Arm selection comes from an atomically replaced phase
file written by the isolated bench between requests. Missing env or missing
phase file means the installed reader path (no behaviour change).
"""
import collections, ctypes, json, os, threading, time
from pathlib import Path

import numpy as np

ARMS = ('installed', 'native', 'shadow')
PHASE_ENV = 'DIRECT_PLE_AB_PHASE'
LIB_ENV = 'DIRECT_PLE_NATIVE_LIB'
THREADS_ENV = 'DIRECT_PLE_NATIVE_THREADS'
COUNTS = collections.Counter()
_LIB = None
_LOCK = threading.Lock()
_LAST_CASE = None


_ARM_TTL = float(os.environ.get('DIRECT_PLE_ARM_TTL', '0'))  # E27: >0 caches read_arm() for that many seconds
_ARM_CACHE = [0.0, None]


def read_arm():
    if _ARM_TTL > 0:
        now = time.monotonic()
        if _ARM_CACHE[1] is not None and now - _ARM_CACHE[0] < _ARM_TTL:
            return _ARM_CACHE[1]
        r = _read_arm()
        _ARM_CACHE[0], _ARM_CACHE[1] = now, r
        return r
    return _read_arm()


def _read_arm():
    path = os.environ.get(PHASE_ENV)
    if not path:
        return 'installed', None
    try:
        data = json.loads(Path(path).read_text())
    except FileNotFoundError:
        return 'installed', None
    if not isinstance(data, dict) or set(data) != {'arm', 'case'}:
        raise ValueError('invalid PLE A/B phase schema')
    if data['arm'] not in ARMS or not isinstance(data['case'], str) or not 0 < len(data['case']) <= 128:
        raise ValueError('invalid PLE A/B phase value')
    return data['arm'], data['case']


def _lib():
    global _LIB
    with _LOCK:
        if _LIB is None:
            path = os.environ.get(LIB_ENV)
            if not path:
                raise RuntimeError(f'{LIB_ENV} is required for native PLE arm')
            lib = ctypes.CDLL(path)  # CDLL releases the GIL for the call
            lib.ple_gather.restype = ctypes.c_int
            lib.ple_gather.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int64, ctypes.c_int]
            try:  # E27 v2 library: whole gather in C
                f = lib.ple_gather_ids
                f.restype = ctypes.c_int
                f.argtypes = ([ctypes.c_void_p, ctypes.c_int64] + [ctypes.c_void_p] * 3 + [ctypes.c_int64] * 4
                              + [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p])
                lib._e27_ids = True
            except AttributeError:
                lib._e27_ids = False
            _LIB = lib
        return _LIB


MIN_RUNS_PER_THREAD = int(os.environ.get('DIRECT_PLE_MIN_RUNS', '512'))  # E25 knob; 512 = image default (v2)


def _threads(runs=None):
    value = int(os.environ.get(THREADS_ENV, '16'))
    if not 1 <= value <= 256:
        raise ValueError('native PLE thread count out of range')
    if runs is not None:
        value = max(1, min(value, int(runs) // MIN_RUNS_PER_THREAD))
    return value


IDS_FAST = os.environ.get('DIRECT_PLE_IDS_FAST') == '1'  # E27: one C call per lookup (v2 library)
FSTAT_EVERY = int(os.environ.get('DIRECT_PLE_FSTAT_EVERY', '1'))  # E27: shard identity guard every N lookups (1 = every call)
_TAB = {}


def _gather_ids(reader, ids, lib):
    ids = np.ascontiguousarray(ids, dtype=np.int64)
    dim = reader._embedding_dim
    if ids.ndim != 1:
        raise ValueError('row IDs must be one-dimensional')
    if ids.size > reader._max_batch_rows:
        raise ValueError('row_ids exceeds max_batch_rows')
    if ids.size == 0:
        return np.empty((0, dim), np.uint16)
    with reader._lifecycle_lock:
        if reader._closed:
            raise ValueError('reader is closed')
        t = _TAB.get(id(reader))
        if t is None or t['reader'] is not reader:
            locs = reader._locations
            t = dict(reader=reader, calls=0, fds=np.array([loc.fd for loc in locs], np.int32),
                     base=np.array([loc.offset for loc in locs], np.int64), rows=np.array([loc.rows for loc in locs], np.int64),
                     shard=(reader._num_embeddings + reader._parts - 1) // reader._parts)
            _TAB[id(reader)] = t
        if t['calls'] % FSTAT_EVERY == 0:
            for loc in reader._locations:  # same identity guard as _read_range
                st = os.fstat(loc.fd)
                if st.st_dev != loc.device or st.st_ino != loc.inode or st.st_size != loc.file_size:
                    raise OSError(f'safetensors shard identity changed: {loc.path}')
        t['calls'] += 1
        out = np.empty((ids.size, dim), np.uint16)
        st3 = np.zeros(3, np.int64)
        t0 = time.perf_counter()
        rc = lib.ple_gather_ids(ids.ctypes.data, ids.size, t['fds'].ctypes.data, t['base'].ctypes.data, t['rows'].ctypes.data,
                                len(t['fds']), t['shard'], reader._row_bytes, reader._num_embeddings, out.ctypes.data,
                                _threads(), MIN_RUNS_PER_THREAD, st3.ctypes.data)
        elapsed = time.perf_counter() - t0
        if rc == -3:
            raise ValueError('row ID out of range')
        if rc == -4:
            raise ValueError('row beyond shard rows')
        if rc != 0:
            raise OSError(f'native PLE gather failed rc={rc}')
        with reader._stats_lock:
            reader._requested_rows += int(ids.size)
            reader._unique_rows += int(st3[0])
            reader._read_calls += int(st3[1])
            reader._read_bytes += int(st3[2])
            reader._gather_wall_seconds += elapsed
    return out


def gather_rows_native(reader, ids):
    """Return uint16 [len(ids), dim] rows, byte-identical to reader.gather_bytes."""
    if IDS_FAST:
        lib = _lib()
        if lib._e27_ids:
            return _gather_ids(reader, ids, lib)
    ids = np.ascontiguousarray(ids, dtype=np.int64)
    dim = reader._embedding_dim
    rb = reader._row_bytes
    if ids.ndim != 1:
        raise ValueError('row IDs must be one-dimensional')
    if ids.size > reader._max_batch_rows:
        raise ValueError('row_ids exceeds max_batch_rows')
    if ids.size == 0:
        return np.empty((0, dim), np.uint16)
    if int(ids.min()) < 0 or int(ids.max()) >= reader._num_embeddings:
        raise ValueError('row ID out of range')
    with reader._lifecycle_lock:
        if reader._closed:
            raise ValueError('reader is closed')
        locs = reader._locations
        for loc in locs:  # same identity guard as _read_range, once per call
            st = os.fstat(loc.fd)
            if st.st_dev != loc.device or st.st_ino != loc.inode or st.st_size != loc.file_size:
                raise OSError(f'safetensors shard identity changed: {loc.path}')
        fds = np.array([loc.fd for loc in locs], np.int32)
        base = np.array([loc.offset for loc in locs], np.int64)
        rows = np.array([loc.rows for loc in locs], np.int64)
        shard = (reader._num_embeddings + reader._parts - 1) // reader._parts
        uniq, inv = np.unique(ids, return_inverse=True)
        sh = uniq // shard
        rin = uniq - sh * shard
        if np.any(rin >= rows[sh]):
            raise ValueError('row beyond shard rows')
        off = base[sh] + rin * rb
        brk = np.ones(uniq.size, bool)
        brk[1:] = (sh[1:] != sh[:-1]) | (off[1:] != off[:-1] + rb)
        starts = np.flatnonzero(brk)
        counts = np.diff(np.append(starts, uniq.size))
        rfd = np.ascontiguousarray(fds[sh[starts]])
        roff = np.ascontiguousarray(off[starts])
        rlen = np.ascontiguousarray(counts * rb)
        rout = np.ascontiguousarray(starts.astype(np.int64) * rb)
        buf = np.empty((uniq.size, dim), np.uint16)
        t0 = time.perf_counter()
        rc = _lib().ple_gather(rfd.ctypes.data, roff.ctypes.data, rlen.ctypes.data, rout.ctypes.data,
                               buf.ctypes.data, ctypes.c_int64(starts.size), ctypes.c_int(_threads(starts.size)))
        elapsed = time.perf_counter() - t0
        if rc != 0:
            raise OSError(f'native PLE gather failed rc={rc}')
        with reader._stats_lock:
            reader._requested_rows += int(ids.size)
            reader._unique_rows += int(uniq.size)
            reader._read_calls += int(starts.size)
            reader._read_bytes += int(rlen.sum())
            reader._gather_wall_seconds += elapsed
    return buf[inv]


def record(arm, case, rows, ms):
    global _LAST_CASE
    COUNTS[(case, arm, 'calls')] += 1
    COUNTS[(case, arm, 'rows')] += rows
    if rows >= 1024:
        print('DIRECT_PLE_AB_GATHER ' + json.dumps({'case': case, 'arm': arm, 'rows': rows, 'host_ms': ms,
                                                   'scope': 'host lookup wall incl. D2H ids/H2D rows; not GPU timing'},
                                                  sort_keys=True), flush=True)
    if case != _LAST_CASE and _LAST_CASE is not None:
        flush_counts(_LAST_CASE)
    _LAST_CASE = case


def flush_counts(case):
    summary = {f'{a}:{k}': v for (c, a, k), v in COUNTS.items() if c == case}
    print('DIRECT_PLE_AB_CASE_TOTALS ' + json.dumps({'case': case, 'totals': summary}, sort_keys=True), flush=True)
