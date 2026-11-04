"""CPU contract for additive rotated INT4 main-KV storage.

The public helper owns the packed wire and view validation, not a serving
fallback. Guard round/zero-point conventions and accidental alias/padding writes
at the cheapest CPU level; GPU arithmetic has a separate qualification.
"""
import importlib
import importlib.util
import json
import hashlib
import pathlib
import sys
import unittest

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))


class KV4Contract(unittest.TestCase):
    def codec(self):
        self.assertIsNotNone(importlib.util.find_spec('htr_kv4'),
                             'rotated INT4 main-KV helper is not implemented')
        return importlib.import_module('htr_kv4')

    def test_metadata_inclusive_content_and_bad_head(self):
        m = self.codec()
        self.assertEqual(m.content_bytes(256), 264)
        for d in [0, 3, 12, 255, 257]:
            with self.assertRaises(ValueError):
                m.content_bytes(d)

    def allocate(self, layout='NHD', offset=16):
        page, heads, width = 16, 2, 264
        pitch = page * heads * width + 64
        base = torch.full((3 * pitch + 64,), 61, dtype=torch.uint8)
        strides = ((pitch, width, heads * width, 1) if layout == 'NHD'
                   else (pitch, page * width, width, 1))
        return base, base.as_strided((3, heads, page, width), strides, offset)

    def test_views_preserve_aliases_offset_and_padding_without_writes(self):
        m = self.codec()
        for layout in ['NHD', 'HND']:
            base, cache = self.allocate(layout)
            before = base.clone()
            k, v, ks, vs = m.cache_views(cache, 256)
            self.assertEqual(tuple(k.shape), (3, 16, 2, 128))
            self.assertEqual(tuple(v.shape), tuple(k.shape))
            self.assertEqual(tuple(ks.shape), (3, 16, 2))
            self.assertEqual(ks.dtype, torch.float32)
            self.assertEqual(k.data_ptr(), cache.data_ptr())
            self.assertEqual(v.data_ptr() - k.data_ptr(), 132)
            self.assertEqual(ks.data_ptr() - k.data_ptr(), 128)
            self.assertEqual(vs.data_ptr() - k.data_ptr(), 260)
            self.assertTrue(torch.equal(base, before))

    def test_overlap_misalignment_and_wrong_storage_rejected(self):
        m = self.codec()
        base, cache = self.allocate()
        bad = [cache.to(torch.int8), cache[..., :-1],
               base.as_strided(cache.shape, (cache.stride(0), 0, 528, 1), 16),
               base.as_strided(cache.shape, cache.stride(), 17)]
        for x in bad:
            with self.assertRaises(ValueError):
                m.cache_views(x, 256)

    def test_rotation_inverse_and_dot_normalization(self):
        m = self.codec()
        torch.manual_seed(972)
        x = torch.randn(3, 2, 256, dtype=torch.float64)
        y = torch.randn_like(x)
        rx, ry = m.single_rht_cpu(x), m.single_rht_cpu(y)
        torch.testing.assert_close(m.single_rht_cpu(rx, inverse=True) / 256,
                                   x, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close((rx * ry).sum(-1) / 256,
                                   (x * y).sum(-1), rtol=1e-12, atol=1e-12)
        self.assertTrue(torch.equal(rx, m.single_rht_cpu(x)))
        self.assertFalse(torch.equal(rx, x))

    def test_exact_nibbles_and_embedded_zero_point(self):
        m = self.codec()
        x = torch.arange(-8, 8, dtype=torch.float32).repeat(16).view(1, 256)
        packed, scales = m.pack_rotated_cpu(x)
        expected = torch.tensor([0x10, 0x32, 0x54, 0x76,
                                 0x98, 0xba, 0xdc, 0xfe],
                                dtype=torch.uint8).repeat(16).view(1, 128)
        self.assertTrue(torch.equal(packed, expected))
        self.assertEqual(scales.view(torch.int32).item(), 0x3f800008)
        self.assertTrue(torch.equal(m.dequantize_cpu(packed, scales), x.bfloat16()))

    def test_half_away_floor_and_zero(self):
        m = self.codec()
        x = torch.zeros(1, 256)
        x[0, :5] = torch.tensor([0, 15, .5, 1.5, 2.5])
        codes, scales = m.pack_rotated_cpu(x)
        values = m.dequantize_cpu(codes, scales)
        self.assertTrue(torch.equal(values[0, :5],
                                   torch.tensor([0, 15, 1, 2, 3]).bfloat16()))
        z, s = m.pack_rotated_cpu(torch.zeros(2, 256))
        self.assertTrue(torch.equal(z, torch.zeros_like(z)))
        self.assertTrue(torch.isfinite(s).all())
        self.assertTrue((s.view(torch.int32) & 15 == 0).all())
        self.assertTrue(torch.equal(m.dequantize_cpu(z, s),
                                   torch.zeros(2, 256, dtype=torch.bfloat16)))

    def test_writer_active_rows_reuse_and_failure_atomicity(self):
        m = self.codec()
        torch.manual_seed(731)
        key = torch.randn(5, 2, 256, dtype=torch.bfloat16)
        value = torch.randn_like(key)
        key[-1].fill_(float('nan'))
        value[-1].fill_(float('nan'))
        slots = torch.tensor([0, 15, 16, 47, -1])
        for layout in ['NHD', 'HND']:
            base, cache = self.allocate(layout)
            m.write_cpu(cache, key, value, slots)
            k, v, ks, vs = m.cache_views(cache, 256)
            expected_k, expected_ks = m.quantize_cpu(key[:4])
            expected_v, expected_vs = m.quantize_cpu(value[:4])
            page, tok = slots[:4] // 16, slots[:4] % 16
            for a, b in [(k[page, tok], expected_k),
                         (v[page, tok], expected_v),
                         (ks[page, tok], expected_ks),
                         (vs[page, tok], expected_vs)]:
                self.assertTrue(torch.equal(a, b))
            before = base.clone()
            for bad_slots in [torch.tensor([0, 0, 1, 2, -1]),
                              torch.tensor([0, 15, 16, 48, -1])]:
                with self.assertRaises(ValueError):
                    m.write_cpu(cache, key, value, bad_slots)
                self.assertTrue(torch.equal(base, before))
            with self.assertRaises(ValueError):
                m.write_cpu(cache, key, value, torch.arange(5))
            self.assertTrue(torch.equal(base, before))
            m.write_cpu(cache, key[:1] * 2, value[:1] * -3, torch.tensor([16]))
            self.assertFalse(torch.equal(base, before))
            self.assertTrue(torch.equal(base[:16], before[:16]))

    def test_rebinding_never_returns_cached_scale_pointers(self):
        m = self.codec()
        b1, c1 = self.allocate()
        b2, c2 = self.allocate()
        one, two = m.cache_views(c1, 256), m.cache_views(c2, 256)
        self.assertNotEqual(one[2].data_ptr(), two[2].data_ptr())
        one[2].fill_(1)
        self.assertTrue((b2 == 61).all())




if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
