"""CPU binding/selection tests for the MTP-only wide RMS reduction."""
import copy,importlib,importlib.util,os,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
NAME='triton_red_fused__to_copy_abs_add_all_eq_mul_ne_rms_norm_view_2'
SIG=dict(in_ptr0='*bf16',in_ptr1='*bf16',out_ptr1='*i1',out_ptr2='*bf16',xnumel='i32',r0_numel='i32',XBLOCK='constexpr',R0_BLOCK='constexpr')
class MtpNorm(unittest.TestCase):
 def setUp(self):
  self.assertIsNotNone(importlib.util.find_spec('htr_mtp_norm_pin'),'MTP reference reduction policy missing')
  self.m=importlib.import_module('htr_mtp_norm_pin');import triton;self.triton=triton
  p=Path(__file__).parent/'fixtures/mtp_norm_kernel.py';s=importlib.util.spec_from_file_location('mtp_norm_fixture',p);assert s is not None and s.loader is not None;m=importlib.util.module_from_spec(s);s.loader.exec_module(m);self.fn=getattr(m,NAME)
  self.meta={'signature':copy.deepcopy(SIG)};self.imeta={'kernel_name':NAME};self.configs=[triton.Config({'XBLOCK':1,'R0_BLOCK':1024},num_warps=8,num_stages=1)]
 def result(self,rblock=16384,warps=32):
  return SimpleNamespace(kernel=SimpleNamespace(name=NAME),config=self.triton.Config({'XBLOCK':1,'R0_BLOCK':rblock},num_warps=warps,num_stages=1),compile_meta={'signature':copy.deepcopy(SIG),'constants':{'XBLOCK':1,'R0_BLOCK':rblock}})
 def test_exact_source_selects_reference_without_mutating_metadata(self):
  before=copy.deepcopy(self.meta);c=self.m.select(self.fn,self.configs,self.meta,self.imeta)
  self.assertEqual([(x.kwargs,x.num_warps,x.num_stages) for x in c],[({'XBLOCK':1,'R0_BLOCK':16384},32,1)])
  self.assertEqual(self.meta,before);self.assertEqual(self.configs[0].kwargs['R0_BLOCK'],1024)
 def test_other_kernel_is_passthrough(self):self.assertIs(self.m.select(object(),self.configs,{},{}),self.configs)
 def test_source_and_abi_drift_rejected(self):
  with self.assertRaisesRegex(RuntimeError,'source'):self.m.select(SimpleNamespace(src=self.fn.src.replace('1e-06','1e-05')),self.configs,self.meta,self.imeta)
  self.meta['signature']['xnumel']='i64'
  with self.assertRaisesRegex(RuntimeError,'signature'):self.m.select(self.fn,self.configs,self.meta,self.imeta)
 def test_serialized_old_geometry_rejected(self):
  with self.assertRaisesRegex(RuntimeError,'geometry'):self.m.validate_result(self.result(1024,8))
  self.assertTrue(self.m.validate_result(self.result()))
 def test_both_serialized_constants_checked(self):
  for key in ('XBLOCK','R0_BLOCK'):
   r=self.result();r.compile_meta['constants'][key]=2
   with self.subTest(key=key),self.assertRaisesRegex(RuntimeError,'constants'):self.m.validate_result(r)
 def test_mode0_and_disabled_are_unpatched(self):
  import torch._inductor.runtime.triton_heuristics as th
  before=th.cached_autotune
  for flag,mode in [('1','0'),('0','3')]:
   with patch.dict(os.environ,{'HTR_GDN_XBLOCK1':flag,'HTR_GDN_MTP':mode}):self.assertFalse(self.m.install())
  self.assertIs(th.cached_autotune,before)
 def test_two_policies_compose_before_native_cache(self):
  import torch._inductor.runtime.triton_heuristics as th
  import test_gdn_pin as g
  seen=[]
  def native(*a,**kw):
   def decorate(fn):seen.append(a[1]);return SimpleNamespace(configs=a[1])
   return decorate
  with patch.object(th,'cached_autotune',native),patch.object(th.StaticTritonCompileResult,'make_launcher',th.StaticTritonCompileResult.make_launcher),patch.object(th.TritonCompileResult,'make_launcher',th.TritonCompileResult.make_launcher),patch.dict(os.environ,{'HTR_GDN_XBLOCK1':'1','HTR_GDN_MTP':'3'}):
   gd=g.GdnPin();gd.setUp();self.assertTrue(gd.m.install());self.assertTrue(self.m.install());saved=th.cached_autotune
   self.assertTrue(gd.m.install());self.assertTrue(self.m.install());self.assertIs(th.cached_autotune,saved)
   for fn,c,meta,imeta in [(self.fn,self.configs,self.meta,self.imeta),(gd.fn,gd.configs,gd.meta,gd.imeta)]:th.cached_autotune({'x':16,'r0_':16384},c,meta,th.HeuristicType.REDUCTION,filename='fixture',inductor_meta=imeta)(fn)
   self.assertEqual(seen[0][0].kwargs,{'XBLOCK':1,'R0_BLOCK':16384});self.assertEqual(seen[1][0].kwargs,{'XBLOCK':4})
if __name__=='__main__':unittest.main()
