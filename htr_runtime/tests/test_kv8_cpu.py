"""CPU-only contract tests. Never represent these as GPU/model qualification."""
import importlib.util
from pathlib import Path
import unittest
import torch

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / 'python/htr_kv8.py'


def load_codec():
    assert MODULE.exists(), 'INT8 main-KV layout/codec module has not been implemented'
    spec = importlib.util.spec_from_file_location('htr_kv8_under_test', MODULE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class KV8Contract(unittest.TestCase):
    def test_01_inline_layout_nhd_and_hnd_with_offset(self):
        m = load_codec()
        for nhd in [False, True]:
            raw = torch.full((16 + 2*3*5*520 + 64,), 91, dtype=torch.int8)
            strides = (3*5*520, 520, 3*520, 1) if nhd else (3*5*520, 5*520, 520, 1)
            cache = raw.as_strided((2,3,5,520), strides, 16)
            before = raw.clone()
            k,v,ks,vs = m.cache_views(cache, 256)
            self.assertTrue(torch.equal(raw,before), 'view creation must not initialize or overwrite cache')
            self.assertEqual(k.shape,(2,5,3,256)); self.assertEqual(ks.shape,(2,5,3))
            self.assertEqual(v.dtype,torch.int8); self.assertEqual(ks.dtype,torch.float32)
            k[1,4,2].fill_(-7); v[1,4,2].fill_(12)
            ks[1,4,2]=0.125; vs[1,4,2]=0.25
            record=cache[1,2,4]
            self.assertTrue(torch.equal(record[:256],torch.full((256,),-7,dtype=torch.int8)))
            self.assertEqual(record[256:260].view(torch.float32).item(),0.125)
            self.assertEqual(record[516:520].view(torch.float32).item(),0.25)
            self.assertTrue(torch.equal(raw[:16],before[:16])); self.assertTrue(torch.equal(raw[-64:],before[-64:]))

    def test_02_padded_page_stride(self):
        m=load_codec(); raw=torch.zeros(2*4096,dtype=torch.int8)
        c=raw.as_strided((2,2,3,520),(4096,520,1040,1))
        k,v,ks,vs=m.cache_views(c,256)
        self.assertEqual(ks.stride(),(1024,260,130)); self.assertEqual(v.storage_offset(),260)

    def test_03_bf16_views_are_the_original_split(self):
        m=load_codec(); c=torch.arange(2*2*3*512,dtype=torch.float32).bfloat16().reshape(2,3,2,512).transpose(1,2)
        before=c.clone(); k,v,ks,vs=m.cache_views(c,256); ek,ev=c.transpose(1,2).split(256,dim=-1)
        self.assertIsNone(ks); self.assertIsNone(vs)
        self.assertEqual(k.data_ptr(),ek.data_ptr()); self.assertEqual(v.stride(),ev.stride())
        self.assertTrue(torch.equal(c,before)); self.assertTrue(torch.equal(v,ev))

    def test_04_round_half_away_from_zero_not_ties_even(self):
        m=load_codec(); x=torch.tensor([[127.,.5,1.5,2.5,-.5,-1.5,-2.5,-127.]])
        q,s=m.quantize_cpu(x)
        self.assertTrue(torch.equal(q,torch.tensor([[127,1,2,3,-1,-2,-3,-127]],dtype=torch.int8)))
        self.assertEqual(s.item(),1.)

    def test_05_zero_scale_floor_and_finite_rejection(self):
        m=load_codec(); q,s=m.quantize_cpu(torch.zeros(2,3,256,dtype=torch.bfloat16))
        self.assertEqual(torch.count_nonzero(q).item(),0)
        self.assertTrue(torch.equal(s,torch.full((2,3),1e-6,dtype=torch.float32)))
        for value in [float('inf'),float('-inf'),float('nan')]:
            with self.assertRaisesRegex(ValueError,'finite'):
                m.quantize_cpu(torch.tensor([[value]]))

    def test_06_page_writes_masks_and_reuse(self):
        m=load_codec(); c=torch.full((3,4,2,520),37,dtype=torch.int8).transpose(1,2)
        k=torch.arange(3*2*256,dtype=torch.float32).reshape(3,2,256).bfloat16(); v=-k
        before=c.clone(); slots=torch.tensor([7,-1,8],dtype=torch.int64)
        m.write_cpu(c,k,v,slots); kd,vd,ks,vs=m.cache_views(c,256)
        q,s=m.quantize_cpu(k)
        self.assertTrue(torch.equal(kd[1,3],q[0])); self.assertTrue(torch.equal(ks[2,0],s[2]))
        self.assertTrue(torch.equal(c[0],before[0])); self.assertTrue(torch.equal(c[1,:,0],before[1,:,0]))
        m.write_cpu(c,k[0:1]*0,v[0:1]*0,torch.tensor([7]))
        self.assertEqual(kd[1,3].count_nonzero().item(),0)
        self.assertTrue(torch.equal(kd[2,0],q[2]))

    def test_07_invalid_write_is_failure_atomic(self):
        m=load_codec(); c=torch.full((2,2,3,520),17,dtype=torch.int8); x=torch.ones(2,2,256,dtype=torch.bfloat16)
        for slots in [torch.tensor([0,6]),torch.tensor([1,1])]:
            before=c.clone()
            with self.assertRaises(ValueError): m.write_cpu(c,x,x,slots)
            self.assertTrue(torch.equal(c,before))

    def test_08_reject_invalid_or_overlapping_layout(self):
        m=load_codec()
        for c in [torch.zeros(2,2,3,512,dtype=torch.int8),torch.zeros(2,2,3,520,dtype=torch.uint8),
                  torch.zeros(520,dtype=torch.int8).as_strided((2,2,3,520),(0,0,0,1))]:
            with self.assertRaises(ValueError): m.cache_views(c,256)
        raw=torch.zeros(2*2*3*520+1,dtype=torch.int8)
        with self.assertRaises(ValueError): m.cache_views(raw[1:].reshape(2,2,3,520),256)

    def test_09_reconstruction_rounds_before_attention(self):
        m=load_codec(); x=torch.linspace(-4,5,2*3*256).reshape(2,3,256).bfloat16()
        q,s=m.quantize_cpu(x); y=m.dequantize_cpu(q,s)
        self.assertEqual(y.dtype,torch.bfloat16)
        self.assertTrue(torch.equal(y,(q.float()*s[...,None]).bfloat16()))

    def test_10_supported_dtype_policy_is_main_cache_only(self):
        m=load_codec()
        self.assertEqual(m.content_bytes(256,'int8_per_token_head'),520)
        self.assertEqual(m.content_bytes(256,'bfloat16'),1024)
        self.assertEqual(m.content_bytes(256,'auto'),1024)
        for name in ['fp8','int4_per_token_head','float16']:
            with self.assertRaises(ValueError): m.content_bytes(256,name)

    def test_11_fp8_comparison_is_cpu_only_not_a_serving_dtype(self):
        m=load_codec(); x=torch.tensor([[0.,-1.,3.,448.]])
        q,s=m.quantize_cpu(x,mode='fp8_e4m3fn')
        self.assertEqual(q.dtype,torch.float8_e4m3fn)
        self.assertTrue(torch.equal(m.dequantize_cpu(q,s),x.bfloat16()))
        with self.assertRaises(ValueError): m.content_bytes(256,'fp8_e4m3fn')

    def test_13_attention_preserves_duplicates_and_masks(self):
        m=load_codec(); self.assertTrue(hasattr(m,'sparse_attention_cpu'),'CPU attention oracle missing')
        q=torch.zeros(2,2,16,dtype=torch.bfloat16); k=torch.zeros(2,1,16,dtype=torch.bfloat16)
        v=torch.stack([torch.ones(1,16),torch.full((1,16),5.)]).bfloat16()
        ids=torch.tensor([[0,0,1,-1,999],[-1,-1,-1,-1,-1]])
        out,probs=m.sparse_attention_cpu(q,k,v,ids)
        expected=(v[0].double()*2+v[1].double())/3
        torch.testing.assert_close(out[0],expected.expand(2,16),rtol=1e-14,atol=1e-14)
        self.assertEqual(out[1].count_nonzero().item(),0); self.assertEqual(probs[1].numel(),0)
        self.assertEqual(probs[0].shape,(1,2,3))

    def test_14_capture_page_remap_is_not_assumed_contiguous(self):
        m=load_codec(); self.assertTrue(hasattr(m,'compact_capture_cpu'),'capture remap missing')
        k=torch.arange(2*2*1*16,dtype=torch.float32).reshape(2,2,1,16).bfloat16()
        b={'q':torch.zeros(1,2,16,dtype=torch.bfloat16),'idx':torch.tensor([[0,3,0,-1]]),
           'bt0':torch.tensor([3,7]),'pages':torch.tensor([7,3]),'t2r':torch.tensor([0]),
           'k_pages':k,'v_pages':-k,'page':2,'context_len':4}
        q,ck,cv,ids=m.compact_capture_cpu(b,[0])
        self.assertTrue(torch.equal(ck[0],k[1,0])); self.assertTrue(torch.equal(ck[1],k[0,1]))
        self.assertTrue(torch.equal(ids,torch.tensor([[0,1,0,-1]])))
        self.assertTrue(torch.equal(cv,-ck))
        b['bt0'][0]=99
        with self.assertRaisesRegex(ValueError,'captured page'): m.compact_capture_cpu(b,[0])

    def test_15_attention_identity_is_zero_error(self):
        m=load_codec(); self.assertTrue(hasattr(m,'sparse_attention_cpu'),'CPU attention oracle missing')
        g=torch.Generator().manual_seed(123)
        q=torch.randn(2,4,16,generator=g).bfloat16(); k=torch.randn(5,2,16,generator=g).bfloat16(); v=k.clone()
        ids=torch.tensor([[0,3,2],[1,4,-1]])
        a,p=m.sparse_attention_cpu(q,k,v,ids); z,pz=m.sparse_attention_cpu(q,k.clone(),v.clone(),ids)
        self.assertTrue(torch.equal(a,z)); self.assertTrue(all(torch.equal(x,y) for x,y in zip(p,pz)))

    def test_12_unchanged_control_and_protected_owners(self):
        import hashlib
        root=ROOT.parent
        expected={
          'vllm/models/qwen4_exp/amd/ops/qsa.py':'6ab20eb2778759530ab9fbb7c4a9b435a169d68da85b2b02a75b158301407990',
          'htr_runtime/python/htr_qsa.py':'38699ece0a5e3bb0f8d041979264ce01d2d4ce228ee6e2af68311d2d4aa958a1'}
        for path,digest in expected.items(): self.assertEqual(hashlib.sha256((root/path).read_bytes()).hexdigest(),digest)


if __name__=='__main__': unittest.main(verbosity=2)
