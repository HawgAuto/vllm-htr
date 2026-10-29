"""CPU-only release boundary tests; no device access or kernel execution."""
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from release_shards import pack_chunks


class RuntimePackaging(unittest.TestCase):
    def test_prebuilt_extensions_load_without_jit_or_writes(self):
        with patch('torch.utils.cpp_extension.load', side_effect=AssertionError('release must not JIT')):
            from htr.ops import ext
            from htr_moe_serve.kernels import moe_ext,dq_ext,moe2_ext,hpf3_ext
            from htr_moe_serve.prefill import hpf_ext
            from htr_moe_serve.prefill2 import hpf2_ext
            for name,fn in [('htr_kernels',ext),('htr_moe_serve_k',moe_ext),('htr_dq_arena',dq_ext),('htr_moe2',moe2_ext),('htr_moe_prefill_v3',hpf3_ext),('htr_moe_prefill',hpf_ext),('htr_moe_prefill_v2',hpf2_ext)]:
                with self.subTest(name=name):
                    mod=fn()
                    self.assertEqual(Path(mod.__file__).resolve(),Path(os.environ['HTR_NATIVE_DIR'])/(name+'.so'))

    def test_sharded_expert_iterator_preserves_all_512_entries(self):
        from htr_moe_serve.arena import iter_p2_experts
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);src=root/'original';(src/'L0').mkdir(parents=True);man={}
            for start in range(0,512,16):
                rel=f'L0/c{start:03}.pt'
                ck={'format':'HTR-W1-P2-PACKED-1','L':0,'K':4,'experts':[{'e':i,'packed':{'sentinel':torch.tensor([i],dtype=torch.int32)}} for i in range(start,start+16)]}
                stream=io.BytesIO();torch.save(ck,stream);raw=stream.getvalue()
                (src/rel).write_bytes(raw);man[rel]=hashlib.sha256(raw).hexdigest()
            pack_chunks(src,root/'packed',man,200_000)
            original=list(iter_p2_experts(str(src),0,man));packed=list(iter_p2_experts(str(root/'packed'),0,man))
            self.assertEqual(len(packed),512)
            for a,b in zip(original,packed):
                self.assertEqual(a[:2],b[:2]);self.assertTrue(torch.equal(a[2]['sentinel'],b[2]['sentinel']))

if __name__=='__main__':unittest.main()
