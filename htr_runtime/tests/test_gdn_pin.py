"""CPU regressions for the source-bound GDN compile/cache numerical policy.

Selection, dispatch and fail-closed cache validation only; not a GPU oracle.
"""
import copy
import importlib
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

NAME = "triton_per_fused__to_copy_add_linear_mean_mul_pow_rsqrt_sigmoid_view_0"
SIGNATURE = dict(in_ptr0="*bf16", in_ptr1="*bf16", in_ptr2="*bf16",
                 out_ptr1="*bf16", ks0="i64", xnumel="i32",
                 r0_numel="i32", XBLOCK="constexpr")


class GdnPin(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("htr_gdn_pin"),
                             "shipping runtime has no source-bound GDN tile policy")
        self.m = importlib.import_module("htr_gdn_pin")
        import triton
        self.triton = triton
        path = Path(__file__).parent / "fixtures/gdn_kernel.py"
        spec = importlib.util.spec_from_file_location("gdn_pin_fixture", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.fn = getattr(mod, NAME)
        self.meta = {"signature": copy.deepcopy(SIGNATURE)}
        self.imeta = {"kernel_name": NAME}
        self.configs = [triton.Config({"XBLOCK": 16}, num_warps=2, num_stages=1)]

    def test_exact_source_gets_single_reference_tile_without_mutation(self):
        before = copy.deepcopy(self.meta)
        c = self.m.select(self.fn, self.configs, self.meta, self.imeta)
        self.assertEqual(len(c), 1)
        self.assertEqual((c[0].kwargs, c[0].num_warps, c[0].num_stages),
                         ({"XBLOCK": 1}, 2, 1))
        self.assertEqual(self.configs[0].kwargs, {"XBLOCK": 16})
        self.assertEqual(self.meta, before)

    def test_unrelated_kernel_is_exact_passthrough(self):
        self.assertIs(self.m.select(object(), self.configs, {},
                                   {"kernel_name": "unrelated"}), self.configs)

    def test_source_change_with_same_name_is_rejected(self):
        changed = SimpleNamespace(src=self.fn.src.replace("1e-06", "1e-05"))
        with self.assertRaisesRegex(RuntimeError, "source"):
            self.m.select(changed, self.configs, self.meta, self.imeta)

    def test_scalar_width_change_is_rejected(self):
        self.meta["signature"]["ks0"] = "i32"
        with self.assertRaisesRegex(RuntimeError, "signature"):
            self.m.select(self.fn, self.configs, self.meta, self.imeta)

    def result(self, xblock):
        return SimpleNamespace(kernel=SimpleNamespace(name=NAME),
            config=self.triton.Config({"XBLOCK": xblock}, num_warps=2, num_stages=1),
            compile_meta={"signature": copy.deepcopy(SIGNATURE),
                          "constants": {"XBLOCK": xblock}})

    def test_serialized_old_tile_is_rejected_before_loading(self):
        with self.assertRaisesRegex(RuntimeError, "geometry"):
            self.m.validate_result(self.result(16))
        self.assertTrue(self.m.validate_result(self.result(1)))
        self.assertFalse(self.m.validate_result(
            SimpleNamespace(kernel=SimpleNamespace(name="unrelated"))))

    def test_serialized_abi_and_constants_are_checked(self):
        r = self.result(1)
        r.compile_meta["constants"]["XBLOCK"] = 16
        with self.assertRaisesRegex(RuntimeError, "constants"):
            self.m.validate_result(r)
        r = self.result(1)
        r.compile_meta["signature"]["ks0"] = "i32"
        with self.assertRaisesRegex(RuntimeError, "signature"):
            self.m.validate_result(r)

    def test_disabled_bootstrap_does_not_patch_framework(self):
        import torch._inductor.runtime.triton_heuristics as th
        before = th.cached_autotune
        with patch.dict(os.environ, {"HTR_GDN_XBLOCK1": "0"}):
            self.assertFalse(self.m.install())
        self.assertIs(th.cached_autotune, before)

    def test_enabled_bootstrap_pins_before_native_cache_selection(self):
        import torch._inductor.runtime.triton_heuristics as th
        seen = []
        def original(*args, **kwargs):
            def decorate(fn):
                seen.append((fn, args, kwargs))
                return SimpleNamespace(configs=args[1])
            return decorate
        with patch.object(th, "cached_autotune", original), \
             patch.object(th.StaticTritonCompileResult, "make_launcher",
                          th.StaticTritonCompileResult.make_launcher), \
             patch.object(th.TritonCompileResult, "make_launcher",
                          th.TritonCompileResult.make_launcher), \
             patch.dict(os.environ, {"HTR_GDN_XBLOCK1": "1"}):
            self.assertTrue(self.m.install())
            wrap = th.cached_autotune
            self.assertTrue(self.m.install())
            self.assertIs(th.cached_autotune, wrap)
            decorator = th.cached_autotune({"x": 64, "r0_": 128}, self.configs,
                self.meta, th.HeuristicType.PERSISTENT_REDUCTION,
                filename="fixture.py", inductor_meta=self.imeta)
            self.assertEqual(seen, [])
            result = decorator(self.fn)
            self.assertEqual(result.configs[0].kwargs, {"XBLOCK": 1})
            self.assertEqual(seen[0][1][1][0].kwargs, {"XBLOCK": 1})


if __name__ == "__main__":
    unittest.main()
