"""Host-only control-flow regression; GPU arithmetic remains separately gated."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

class GroupedStartup(unittest.TestCase):
    def load_bridge(self, capturing=False):
        package = '_htr_grouped_startup_test'
        torch: Any = types.ModuleType('torch')
        torch.cuda = types.SimpleNamespace(is_current_stream_capturing=lambda: capturing)
        dispatch: Any = types.ModuleType(package+'.dispatch')
        dispatch.CANDIDATE_ROWS = {12, 16, 24, 32, 40}
        modules = {'torch': torch, package: types.ModuleType(package),
                   package+'.grouped_plan': types.ModuleType(package+'.grouped_plan'),
                   package+'.dispatch': dispatch}
        modules[package].__path__ = []
        previous = {key: sys.modules.get(key) for key in modules}
        try:
            sys.modules.update(modules)
            spec = importlib.util.spec_from_file_location(package+'.grouped', ROOT/'python/htr_moe_serve/grouped.py')
            assert spec is not None and spec.loader is not None
            module: Any = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        finally:
            for key,value in previous.items():
                if value is None: sys.modules.pop(key,None)
                else: sys.modules[key]=value
        module.verify_sources = lambda: {}
        module.log = lambda *args,**kwargs: None
        return module

    def invoke(self, rows, capturing=False, k=4):
        module = self.load_bridge(capturing)
        trace=[]; answer=object()
        def original(*args): trace.append(('original',args)); return answer
        route = types.SimpleNamespace(forward=original, _htr_partition4=True,
            BPS_GU=5, BPS_DN=1, FENCE_RA=1, FUSE=True, V2_MIN=48)
        module.metadata = lambda *args: trace.append(('metadata',args))
        module.prime_smaller = lambda *args: trace.append(('prime',args))
        x=types.SimpleNamespace(shape=(rows,2560),device='cuda:0')
        ar={'_kf':k,'_meta':{'L':0}}; ids=object(); weights=object()
        result=module.install(route)(ar,x,ids,weights)
        self.assertIs(result,answer)
        self.assertEqual(trace[-1],('original',(ar,x,ids,weights)))
        return trace

    def test_large_live_eager_call_primes_before_original(self):
        trace=self.invoke(2048)
        self.assertEqual([x[0] for x in trace],['metadata','prime','original'])

    def test_larger_captured_fallback_is_untouched(self):
        self.assertEqual([x[0] for x in self.invoke(2048,capturing=True)],['original'])

    def test_other_k_and_short_unselected_rows_are_untouched(self):
        self.assertEqual([x[0] for x in self.invoke(2048,k=3)],['original'])
        self.assertEqual([x[0] for x in self.invoke(33)],['original'])

    def test_unchecked_selected_capture_still_refuses(self):
        module=self.load_bridge(capturing=True)
        module.metadata=lambda *args:None
        route=types.SimpleNamespace(forward=lambda *args:None,_htr_partition4=True,
            BPS_GU=5,BPS_DN=1,FENCE_RA=1,FUSE=True,V2_MIN=48)
        x=types.SimpleNamespace(shape=(32,2560),device='cuda:0')
        with self.assertRaisesRegex(RuntimeError,'unchecked capture'):
            module.install(route)({'_kf':4,'_meta':{'L':0}},x,object(),object())

if __name__=='__main__':unittest.main()
