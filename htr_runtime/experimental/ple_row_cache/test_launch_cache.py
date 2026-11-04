import argparse
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

R=Path(__file__).resolve().parent

class LaunchTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((R/'launch_cache.py').is_file(), 'opt-in launch adapter missing')
        spec=importlib.util.spec_from_file_location('ple_launch_test', R/'launch_cache.py')
        self.m=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.m)

    def test_budget_off_default_and_configurable(self):
        p=self.m.add_options(argparse.ArgumentParser())
        self.assertEqual(p.parse_args([]).ple_row_cache_gib,0)
        for text in ('2','3.5','4','8'):
            a=p.parse_args(['--ple-row-cache-gib',text])
            self.assertEqual(self.m.budget_bytes(a.ple_row_cache_gib),int(float(text)*2**30))

    def test_bad_budgets_fail_instead_of_fallback(self):
        for n in (-1,float('nan'),float('inf'),10000):
            with self.assertRaises(ValueError):self.m.budget_bytes(n)

    def test_default_leaves_native_library_unchanged(self):
        args=argparse.Namespace(ple_row_cache_gib=0,ple_cache_observe=False)
        with patch.dict(os.environ,{'DIRECT_PLE_NATIVE_LIB':'/native/library.so'},clear=True):
            self.m.configure_cache(args)
            self.assertEqual(os.environ['DIRECT_PLE_NATIVE_LIB'],'/native/library.so')
            self.assertNotIn('HTR_PLE_ROW_CACHE_STATS',os.environ)

    def test_enabled_configuration_is_applied_after_native(self):
        args=argparse.Namespace(ple_row_cache_gib=4,ple_cache_observe=False)
        with patch.dict(os.environ,{'DIRECT_PLE_NATIVE_LIB':'/native/library.so'},clear=True):
            self.m.configure_cache(args)
            self.assertEqual(os.environ['DIRECT_PLE_NATIVE_LIB'],str(R/'libple_row_cache.so'))
            self.assertEqual(os.environ['HTR_PLE_ROW_CACHE_BYTES'],str(4*2**30))
            self.assertEqual(os.environ['HTR_PLE_ROW_CACHE_STATS'],'/workspace/ple-cache-stats.bin')

    def test_native_observer_does_not_enable_cache(self):
        args=argparse.Namespace(ple_row_cache_gib=0,ple_cache_observe=True)
        with patch.dict(os.environ,{},clear=True):
            self.m.configure_cache(args)
            self.assertEqual(os.environ['HTR_PLE_ROW_CACHE_BYTES'],'0')
            self.assertEqual(os.environ['DIRECT_PLE_NATIVE_LIB'],str(R/'libple_row_cache.so'))

if __name__=='__main__':unittest.main(verbosity=2)
