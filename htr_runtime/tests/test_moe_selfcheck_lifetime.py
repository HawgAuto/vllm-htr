"""CPU lifetime tests for the real self-check and scratch owners; GPU math is replaced only at its boundary."""
import contextlib
import importlib.util
import sys
import types
import unittest
import weakref
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
PKG = '_htr_selfcheck_lifetime_test'
package = types.ModuleType(PKG)
package.__path__ = [str(ROOT / 'python/htr_moe_serve')]
sys.modules[PKG] = package

def load(name):
    spec = importlib.util.spec_from_file_location(PKG + '.' + name, Path(package.__path__[0]) / (name + '.py'))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    setattr(package, name, mod)
    return mod

R = load('routes')
P = load('plugin')

class ScratchLifetime(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.multiple(R.A, NE=2, I=3, H=4))
        R._BF.clear()
        self.addCleanup(R._BF.clear)
        self.refs = []
        self.reference_calls = []
        self.native_calls = []
        self.logs = []
        self.reference_delta = 0.0
        self.reference_fail = False
        self.capture = False
        self.empty_calls = []
        self.x = torch.arange(32, dtype=torch.float32).reshape(8, 4) + 1
        self.ids = torch.zeros((8, 10), dtype=torch.int32)
        self.w = torch.full((8, 10), 0.1)
        def dq(ar, w13, w2):
            self.refs.extend((weakref.ref(w13), weakref.ref(w2)))
            w13.fill_(1); w2.fill_(2)
            return w13, w2
        def fused(x, w13, w2, weights, ids, **kwargs):
            self.reference_calls.append((x.clone(), weights.clone(), ids.clone(), kwargs))
            if self.reference_fail:
                raise RuntimeError('injected reference failure')
            return x.clone() + self.reference_delta
        def native(ar, x, ids, w):
            self.native_calls.append(len(x))
            return x.clone()
        fused_module = types.ModuleType('vllm.model_executor.layers.fused_moe.fused_moe')
        fused_module.fused_experts = fused
        quark = types.ModuleType('quark_mixed_moe')
        quark.QuarkMixedMoEMethod = type('Base', (), {})
        quark.PoolPlan = object
        quark.validate_outer_shared_owner = lambda *a: None
        self.stack.enter_context(patch.dict(sys.modules, {fused_module.__name__: fused_module, 'quark_mixed_moe': quark}))
        self.stack.enter_context(patch.object(R, 'dequant_layer', dq))
        self.stack.enter_context(patch.object(R, 'forward', native))
        self.stack.enter_context(patch.object(R, 'v2_route', native))
        self.stack.enter_context(patch.object(P, '_log', lambda tag, **kw: self.logs.append((tag, kw))))
        self.stack.enter_context(patch.object(torch.cuda, 'is_current_stream_capturing', lambda: self.capture))
        self.stack.enter_context(patch.object(torch.cuda, 'device', lambda dev: contextlib.nullcontext()))
        self.stack.enter_context(patch.object(torch.cuda, 'empty_cache', lambda: self.empty_calls.append(True)))
        cls = P._method_class()
        self.method = cls.__new__(cls)
        self.method.L, self.method.ar, self.method._checked = 0, {}, set()

    def check(self, route='decode'):
        self.method._selfcheck(self.x, self.ids, self.w, route)

    def test_successful_selfcheck_does_not_retain_expanded_weights(self):
        self.check()
        self.assertEqual(len(self.reference_calls), 1)
        self.assertEqual(R._BF, {}, 'self-check must not leave its reference weights in process-global scratch')
        self.assertTrue(all(ref() is None for ref in self.refs))
        self.assertTrue(self.empty_calls, 'temporary scratch must also be released from the caching allocator')
        self.assertEqual(self.logs[-1][0], 'HTR_MOE_SELFCHECK')
        self.assertTrue(self.logs[-1][1]['ok'])
        self.assertEqual(self.logs[-1][1]['rel'], 0.0)

    def test_checks_remain_once_per_layer_and_route(self):
        self.check(); self.check()
        self.check('v2'); self.check('v2')
        self.check('chunk'); self.check('chunk')
        self.assertEqual(len(self.reference_calls), 3)
        self.assertEqual(self.method._checked, {'decode', 'v2', 'chunk'})
        self.assertEqual(R._BF, {})

    def test_capture_never_runs_or_releases_reference_scratch(self):
        self.capture = True
        self.check()
        self.assertEqual(self.reference_calls, [])
        self.assertEqual(self.empty_calls, [])
        self.assertEqual(self.method._checked, set())

    def test_numerical_failure_still_raises_and_is_not_marked_checked(self):
        self.reference_delta = 1000.0
        for _ in range(2):
            with self.assertRaisesRegex(RuntimeError, 'HTR MoE selfcheck'):
                self.check()
            self.assertNotIn('decode', self.method._checked)
            self.assertEqual(R._BF, {})
        self.assertEqual(len(self.reference_calls), 2)

    def test_reference_exception_does_not_leave_global_scratch(self):
        self.reference_fail = True
        with self.assertRaisesRegex(RuntimeError, 'injected reference failure'):
            self.check()
        self.assertEqual(R._BF, {})
        self.assertEqual(self.method._checked, set())
        self.assertTrue(self.empty_calls)

    def test_default_fallback_still_reuses_its_persistent_scratch(self):
        a = R.dequant_route({}, self.x, self.ids, self.w)
        pair = R._BF['cpu']
        b = R.dequant_route({}, self.x, self.ids, self.w)
        self.assertIs(pair, R._BF['cpu'])
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(a, self.x))

    def test_selfcheck_does_not_delete_a_fallback_owners_buffers(self):
        R.dequant_route({}, self.x, self.ids, self.w)
        pair = R._BF['cpu']
        count = len(self.refs)
        self.check()
        self.assertIs(pair, R._BF['cpu'])
        self.assertTrue(all(ref() is None for ref in self.refs[count:]))

if __name__ == '__main__':
    unittest.main()
