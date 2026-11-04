"""Native CPU contract: synthetic faults and real release bytes, no GPU required."""
import concurrent.futures
import ctypes as C
import json
import os
from pathlib import Path
import random
import struct
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent
FIELDS = ('budget_bytes allocated_bytes capacity resident_rows lookups hits misses evictions '
          'read_runs read_bytes calls errors invalidations pread_calls pread_bytes '
          'nowait_calls nowait_bytes requested_rows').split()
ARGS = [C.c_void_p, C.c_int64, C.c_void_p, C.c_void_p, C.c_void_p,
        C.c_int64, C.c_int64, C.c_int64, C.c_int64, C.c_void_p, C.c_int, C.c_int, C.c_void_p]
LIBS = {}


def load(path):
    if str(path) not in LIBS:
        lib = C.CDLL(str(path))
        lib.ple_gather_ids.argtypes = ARGS
        lib.ple_gather_ids.restype = C.c_int
        if hasattr(lib, 'ple_cache_configure'):
            lib.ple_cache_configure.argtypes = [C.c_uint64]
            lib.ple_cache_configure.restype = C.c_int
            lib.ple_cache_stats.argtypes = [C.c_void_p, C.c_uint64]
            lib.ple_cache_stats.restype = C.c_int
        LIBS[str(path)] = lib
    return LIBS[str(path)]


def stats(lib):
    a = (C.c_uint64 * len(FIELDS))()
    assert lib.ple_cache_stats(a, len(a)) == 0
    return dict(zip(FIELDS, a))


class Table:
    def __init__(self, directory, rb=320, count=2048):
        self.rb, self.shard, self.total = rb, count, count * 2
        self.paths, self.handles, self.blobs = [], [], []
        for k in range(2):
            p = Path(directory) / f'table-{k}'
            data = random.Random(100 + k).randbytes(rb * count)
            p.write_bytes(b'H' * 16 + data)
            self.paths.append(p)
            self.handles.append(os.open(p, os.O_RDONLY))
            self.blobs.append(data)
        self.fds = (C.c_int32 * 2)(*self.handles)
        self.base = (C.c_int64 * 2)(16, 16)
        self.rows = (C.c_int64 * 2)(count, count)

    def close(self):
        for fd in self.handles:
            os.close(fd)

    def expected(self, ids):
        return b''.join(self.blobs[x // self.shard][(x % self.shard)*self.rb:(x % self.shard+1)*self.rb] for x in ids)

    def gather(self, lib, ids):
        a = (C.c_int64 * len(ids))(*ids)
        out = C.create_string_buffer(b'\xa5' * (len(ids)*self.rb))
        s = (C.c_int64 * 3)(-99, -99, -99)
        rc = lib.ple_gather_ids(a, len(ids), self.fds, self.base, self.rows,
                                len(self.handles), self.shard, self.rb, self.total, out, 16, 1, s)
        return rc, out.raw[:len(ids)*self.rb], list(s)


class RowCacheTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((ROOT / 'libple_row_cache.so').is_file(), 'bounded native row cache is not implemented')
        self.lib = load(ROOT / 'libple_row_cache.so')
        self.native = load(ROOT / 'vendor/native.so')
        self.tmp = tempfile.TemporaryDirectory()
        self.table = Table(self.tmp.name)
        self.assertEqual(self.lib.ple_cache_configure(0), 0)

    def tearDown(self):
        self.lib.ple_cache_configure(0)
        self.table.close()
        self.tmp.cleanup()

    def check(self, ids):
        rc, out, s = self.table.gather(self.lib, ids)
        self.assertEqual(rc, 0)
        self.assertEqual(out, self.table.expected(ids))
        return s

    def test_off_allocates_no_cache_and_matches_original(self):
        ids = [0, 1, 1, 2047, 2048, 4095, 35, 13]
        self.assertEqual(self.table.gather(self.lib, ids), self.table.gather(self.native, ids))
        self.check(ids)
        s = stats(self.lib)
        self.assertEqual(s['allocated_bytes'], 0)
        self.assertEqual(s['resident_rows'], 0)
        self.assertEqual(s['hits'], 0)

    def test_repeat_is_exact_hit_without_reads(self):
        self.assertEqual(self.lib.ple_cache_configure(2**20), 0)
        ids = [2049, 3, 3, 10, 11, 4095, 0]
        self.check(ids)
        s = stats(self.lib)
        self.check(ids)
        t = stats(self.lib)
        self.assertEqual(t['hits']-s['hits'], len(set(ids)))
        self.assertEqual(t['pread_calls'], s['pread_calls'])
        self.assertEqual(t['nowait_calls'], s['nowait_calls'])
        self.assertEqual(t['read_bytes'], len(set(ids))*320)

    def test_budget_counts_metadata_and_eviction_is_exact(self):
        budget = 65536
        self.assertEqual(self.lib.ple_cache_configure(budget), 0)
        self.check([0])
        s = stats(self.lib)
        self.assertGreater(s['capacity'], 0)
        self.assertLess(s['capacity']*320, s['allocated_bytes'])
        self.assertLessEqual(s['allocated_bytes'], budget)
        self.check(list(range(s['capacity']*3)))
        self.check(list(reversed(range(s['capacity']*3))))
        t = stats(self.lib)
        self.assertGreater(t['evictions'], 0)
        self.assertLessEqual(t['resident_rows'], t['capacity'])
        self.assertEqual(t['allocated_bytes'], s['allocated_bytes'])

    def test_lru_keeps_recent_row(self):
        self.lib.ple_cache_configure(65536)
        self.check([0])
        cap = stats(self.lib)['capacity']
        self.check(list(range(cap)))
        self.check([0])
        self.check([cap])
        before = stats(self.lib)
        self.check([0])
        self.assertEqual(stats(self.lib)['misses'], before['misses'])
        self.check([1])
        self.assertEqual(stats(self.lib)['misses'], before['misses'] + 1)

    def test_invalid_id_does_not_modify_output_or_cache(self):
        self.lib.ple_cache_configure(65536)
        self.check([0])
        for ids in ([-1], [4096], [0, 4096]):
            before = stats(self.lib)
            rc, out, _ = self.table.gather(self.lib, ids)
            self.assertNotEqual(rc, 0)
            self.assertEqual(out, b'\xa5' * (len(ids)*320))
            self.assertEqual(stats(self.lib)['resident_rows'], before['resident_rows'])

    def test_bad_shard_range_rejected(self):
        self.lib.ple_cache_configure(65536)
        self.table.rows[1] = 1
        rc, out, _ = self.table.gather(self.lib, [2049])
        self.assertNotEqual(rc, 0)
        self.assertEqual(out, b'\xa5' * 320)

    def test_same_size_mutation_is_rejected_even_for_hit(self):
        self.lib.ple_cache_configure(65536)
        self.check([1])
        with self.table.paths[0].open('r+b') as f:
            f.seek(16+320)
            f.write(b'changed!')
            f.flush()
            os.fsync(f.fileno())
        rc, out, _ = self.table.gather(self.lib, [1])
        self.assertNotEqual(rc, 0)
        self.assertEqual(out, b'\xa5'*320)

    def test_truncated_cached_shard_rejected(self):
        self.lib.ple_cache_configure(65536)
        self.check([1])
        os.truncate(self.table.paths[0], 18)
        self.assertNotEqual(self.table.gather(self.lib, [1])[0], 0)

    def test_closed_fd_returns_error_and_never_hit(self):
        self.lib.ple_cache_configure(65536)
        self.check([1])
        self.table.fds[0] = -1
        self.assertNotEqual(self.table.gather(self.lib, [1])[0], 0)

    def test_changed_table_namespace_flushes_not_stale_hit(self):
        self.lib.ple_cache_configure(65536)
        self.check([0])
        self.table.base[0] += 320
        rc, out, _ = self.table.gather(self.lib, [0])
        # Declared table now extends past backing file: fail closed.
        self.assertNotEqual(rc, 0)
        self.table.rows[0] -= 1
        rc, out, _ = self.table.gather(self.lib, [0])
        self.assertEqual(rc, 0)
        self.assertEqual(out, self.table.expected([1]))
        self.assertGreater(stats(self.lib)['invalidations'], 0)

    def test_empty_input(self):
        self.lib.ple_cache_configure(65536)
        self.assertEqual(self.table.gather(self.lib, []), (0, b'', [0, 0, 0]))

    def test_concurrent_callers_are_byte_exact_and_bounded(self):
        self.lib.ple_cache_configure(65536)
        ids = [[random.Random(k+i*197).randrange(4096) for k in range(80)] for i in range(40)]
        with concurrent.futures.ThreadPoolExecutor(8) as pool:
            out = list(pool.map(lambda a:self.table.gather(self.lib, a), ids))
        for a, (rc, data, _) in zip(ids, out):
            self.assertEqual(rc, 0)
            self.assertEqual(data, self.table.expected(a))
        s = stats(self.lib)
        self.assertLessEqual(s['allocated_bytes'], 65536)
        self.assertLessEqual(s['resident_rows'], s['capacity'])

    def test_config_rejects_tiny_and_unbounded_budgets(self):
        for budget in (1, 4096, 2**64-1):
            self.assertNotEqual(self.lib.ple_cache_configure(budget), 0)
        self.assertEqual(stats(self.lib)['budget_bytes'], 0)

    def test_config_budget_choices_and_arbitrary_size(self):
        for gib in (2, 3, 4, 8):
            budget = gib * 2**30
            self.assertEqual(self.lib.ple_cache_configure(budget), 0)
            self.check([0, 200, 2049, 4095])
            s = stats(self.lib)
            self.assertEqual(s['budget_bytes'], budget)
            self.assertLessEqual(s['allocated_bytes'], budget)
            self.assertGreater(s['capacity']*320, budget*0.8)
            self.assertEqual(s['resident_rows'], 4)

    def test_real_release_rows_match_original_library_and_pread(self):
        model_dir = os.environ.get('HTR_TEST_MODEL_DIR')
        if not model_dir:
            self.skipTest('Set HTR_TEST_MODEL_DIR for the real-model row oracle')
        root = Path(model_dir)
        index = json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
        parts = []
        for i in range(128):
            key = f'model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{i}.weight'
            p = root/index[key]
            with p.open('rb') as f:
                n = struct.unpack('<Q',f.read(8))[0]
                h = json.loads(f.read(n))[key]
            parts.append((os.open(p,os.O_RDONLY),8+n+h['data_offsets'][0],h['shape'][0]))
        try:
            rng = random.Random(7159)
            rows = parts[0][2]
            ids = [0,rows-1,rows,128*rows-1]+[rng.randrange(128*rows) for _ in range(512)]
            ids += ids[:50]
            fds = (C.c_int32*128)(*[x[0] for x in parts])
            base = (C.c_int64*128)(*[x[1] for x in parts])
            sizes = (C.c_int64*128)(*[x[2] for x in parts])
            a = (C.c_int64*len(ids))(*ids)
            self.lib.ple_cache_configure(2**20)
            expect = b''.join(os.pread(parts[x//rows][0],320,parts[x//rows][1]+x%rows*320) for x in ids)
            for lib in (self.native,self.lib,self.lib):
                out = C.create_string_buffer(len(ids)*320)
                s = (C.c_int64*3)()
                rc = lib.ple_gather_ids(a,len(ids),fds,base,sizes,128,rows,320,128*rows,out,16,1,s)
                self.assertEqual(rc,0)
                self.assertEqual(out.raw,expect)
            self.assertGreater(stats(self.lib)['hits'], 0)
        finally:
            for fd,_,_ in parts:os.close(fd)


if __name__ == '__main__':
    unittest.main(verbosity=2)
