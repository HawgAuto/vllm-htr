"""Installed v0.2.1 CPU suite; retain the immutable predecessor version test."""
import json,os,pathlib,sys,unittest
ROOT=pathlib.Path(__file__).resolve().parents[1]
os.environ.update(json.loads((ROOT/'runtime-env.json').read_text()))
sys.path.insert(0,str(ROOT/'python'));sys.path.insert(0,str(ROOT/'tests'))
import test_release_seed_policy
old=test_release_seed_policy.ReleaseSeedPolicy
name='test_release_version_is_separate_from_upstream_vllm'
setattr(old,name,unittest.skip('Frozen v0.2.0 literal; v0.2.1 replacement is test_current_distribution_version')(getattr(old,name)))
suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'))
result=unittest.TextTestRunner(verbosity=2).run(suite)
print(json.dumps({'run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),'scope':'CPU/source and installed prebuilt extensions, not GPU execution'}))
raise SystemExit(0 if result.wasSuccessful() else 1)
