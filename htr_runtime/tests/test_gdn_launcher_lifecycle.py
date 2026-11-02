"""No-device launcher-lifecycle regressions with the real Torch Python path.

Only the driver leaf is a CPU double. Fixture bytes/handles are NOT native GPU
artifacts. The real load_kernel clears its path/raw bytes and the real
StaticTritonCompileResult generates the ordinary launcher.
"""
import contextlib
import copy
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


class CpuDriver:
    def __init__(self):
        self.loads = []
        self.unloads = []
        self.launches = []

    def _load_kernel(self, path, name, shared, device):
        self.loads.append((path, name, shared, device))
        return 11, 12, 13, 0

    def _unload_kernel(self, module):
        self.unloads.append(module)

    def _launch_kernel(self, *args):
        self.launches.append(args)


class GdnLauncherLifecycle(unittest.TestCase):
    def exercise(self, form, repeats=1, wrong_geometry=False):
        import torch
        import triton
        from torch._inductor.runtime import triton_heuristics as th
        from torch._inductor.runtime.static_triton_launcher import StaticallyLaunchedCudaKernel
        m = importlib.import_module('htr_gdn_pin')
        raw = b'CPU fixture only: not an executable code object'
        driver = CpuDriver()
        with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as stack:
            binary = Path(tmp) / th.triton_hash_to_path_key('0' * 64) / (m.NAME + '.hsaco')
            binary.parent.mkdir()
            if form != 'serialized-raw':
                binary.write_bytes(raw)
            k = StaticallyLaunchedCudaKernel.__new__(StaticallyLaunchedCudaKernel)
            k.__dict__.update(name=m.NAME, hash='0' * 64,
                cubin_path=str(binary) if form == 'fresh' else None,
                cubin_raw=raw if form != 'bundle-file' else None,
                arg_names=list(m.SIGNATURE), declared_constexprs=[7], full_constexprs=[7],
                num_warps=2, shared=0, arg_tys='OOOOlii', function=None, module=None,
                has_global_scratch=False, has_profile_scratch=False, C_impl=driver)
            result = th.StaticTritonCompileResult(k,
                triton.Config({'XBLOCK': 16 if wrong_geometry else 1}, num_warps=2, num_stages=1),
                {'signature': copy.deepcopy(m.SIGNATURE), 'constants': {'XBLOCK': 1},
                 'device': 0, 'device_type': 'hip'},
                {'grid_type': 'Grid1D', 'kernel_name': m.NAME})
            stack.enter_context(patch.object(th, 'triton_cache_dir', return_value=tmp))
            stack.enter_context(patch.object(th, 'cached_autotune', th.cached_autotune))
            stack.enter_context(patch.object(th.StaticTritonCompileResult, 'make_launcher',
                                            th.StaticTritonCompileResult.make_launcher))
            stack.enter_context(patch.object(th.TritonCompileResult, 'make_launcher',
                                            th.TritonCompileResult.make_launcher))
            stack.enter_context(patch.dict(os.environ, {'HTR_GDN_XBLOCK1': '1'}))
            m.install()
            output = io.StringIO()
            try:
                if wrong_geometry:
                    with self.assertRaisesRegex(RuntimeError, 'geometry'):
                        result.make_launcher()
                    self.assertEqual(driver.loads, [])
                    return
                with contextlib.redirect_stderr(output):
                    for _ in range(repeats):
                        launcher = result.make_launcher()
                        self.assertTrue(launcher._is_static)
                        self.assertIs(launcher.__globals__['runner'].__self__, k)
                        launcher(None, None, None, None, 1, 48, 128, stream=0)
                self.assertEqual(len(driver.loads), 1)
                self.assertEqual(len(driver.launches), repeats)
                self.assertIsNone(k.cubin_raw)
                rows = [json.loads(line.split(' ', 1)[1]) for line in output.getvalue().splitlines()
                        if line.startswith('HTR_GDN_PIN_LOADED ')]
                self.assertEqual(len(rows), repeats)
                self.assertTrue(all(r['binary_sha256'] == hashlib.sha256(raw).hexdigest() for r in rows))
                self.assertFalse(torch.cuda.is_initialized())
            finally:
                k.close()

    def test_fresh_launcher_hashes_before_native_loader_consumes_bytes(self):
        self.exercise('fresh')

    def test_serialized_raw_is_restored_before_native_loader_consumes_bytes(self):
        self.exercise('serialized-raw')

    def test_restored_bundle_file_survives_native_loader_consumption(self):
        self.exercise('bundle-file')

    def test_repeated_launcher_construction_preserves_native_module(self):
        self.exercise('fresh', repeats=2)

    def test_bad_geometry_still_fails_before_driver_loading(self):
        self.exercise('fresh', wrong_geometry=True)


if __name__ == '__main__':
    unittest.main()
