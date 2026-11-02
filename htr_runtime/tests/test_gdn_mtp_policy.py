"""Mode-specific reference geometry; CPU policy only, not model equality."""
import importlib.util,os,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import test_gdn_pin as existing

class MtpReferenceTile(unittest.TestCase):
 def setUp(self):
  self.f=existing.GdnPin();self.f.setUp()
 def test_mtp3_selects_reference_four_row_tile(self):
  with patch.dict(os.environ,{'HTR_GDN_MTP':'3'}):
   c=self.f.m.select(self.f.fn,self.f.configs,self.f.meta,self.f.imeta)
   self.assertEqual((c[0].kwargs,c[0].num_warps,c[0].num_stages),({'XBLOCK':4},2,1))
 def test_mtp0_preserves_qualified_one_row_tile(self):
  with patch.dict(os.environ,{'HTR_GDN_MTP':'0'}):
   self.assertEqual(self.f.m.select(self.f.fn,self.f.configs,self.f.meta,self.f.imeta)[0].kwargs,{'XBLOCK':1})
 def test_mtp3_rejects_old_one_row_cache(self):
  with patch.dict(os.environ,{'HTR_GDN_MTP':'3'}):
   with self.assertRaisesRegex(RuntimeError,'geometry'):self.f.m.validate_result(self.f.result(1))
   self.assertTrue(self.f.m.validate_result(self.f.result(4)))
 def test_mtp0_rejects_mtp3_cache(self):
  with patch.dict(os.environ,{'HTR_GDN_MTP':'0'}):
   with self.assertRaisesRegex(RuntimeError,'geometry'):self.f.m.validate_result(self.f.result(4))
 def test_invalid_mode_fails_closed(self):
  for value in ('','1','4','03','True'):
   with self.subTest(value=value),patch.dict(os.environ,{'HTR_GDN_MTP':value}):
    with self.assertRaisesRegex(RuntimeError,'HTR_GDN_MTP'):self.f.m.select(self.f.fn,self.f.configs,self.f.meta,self.f.imeta)
 def test_launcher_binds_actual_mode_not_ambient_value(self):
  path=Path(__file__).resolve().parents[1]/'launch_htr.py'
  spec=importlib.util.spec_from_file_location('mode_launcher',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  for mode in (0,3):
   with self.subTest(mode=mode),tempfile.TemporaryDirectory() as d,patch.dict(os.environ,{'HTR_GDN_MTP':'wrong'}),patch.dict('sys.modules',{'launch':SimpleNamespace(configure=lambda _:None)}),patch.object(m,'WORK',Path(d)):
    args=m.parser().parse_args(['--mtp',str(mode)]);m.check(m.parser(),args);m.configure(args,'test-profile')
    self.assertEqual(os.environ['HTR_GDN_MTP'],str(mode))
if __name__=='__main__':unittest.main()
