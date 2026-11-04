"""CPU development gates for additive INT8 consumers (not GPU qualification)."""
import ast,hashlib,pathlib,unittest
R=pathlib.Path(__file__).resolve().parents[1]

class Consumers(unittest.TestCase):
    def test_sparse_reader_reconstructs_before_dot(self):
        p=R/'python/htr_qsa_kv8_ops.py'
        self.assertTrue(p.exists(),'INT8 sparse reader is missing')
        s=p.read_text(); ast.parse(s)
        self.assertIn('tl.pointer_type(tl.float32)',s)
        self.assertIn('keys.to(tl.float32) * k_scale[None, :]',s)
        self.assertIn('values.to(tl.float32) * v_scale[:, None]',s)
        self.assertLess(s.index('keys = (keys.to(tl.float32)'),s.index('scores = tl.dot(query, keys)'))
        self.assertIn('triton_reshape_and_cache_flash_per_token_head_quant',s)
        self.assertNotIn('.cpu()',s)

    def test_native_tau_int8_addition_preserves_parent(self):
        p=R/'native-source/qsa_tau/qsa_tau_kv8.hip'
        self.assertTrue(p.exists(),'INT8 TAU reader is missing')
        s=p.read_text()
        self.assertIn('qsa_tau_kv8(',s)
        self.assertIn('const int8_t* __restrict__ K',s)
        self.assertIn('const int8_t* __restrict__ V',s)
        self.assertIn('load_kv8_bf16',s)
        self.assertIn('bf16_rne',s)
        self.assertIn('qsa_tau_v2(', (R/'native-source/qsa_tau/qsa_tau_v2.hip').read_text())

    def test_no_automatic_serving_admission(self):
        root=R.parent
        p=root/'vllm/models/qwen4_exp/amd/qsa.py'
        s=p.read_text()
        self.assertIn('_KV8_ENABLED = _gl_os.environ.get("HTR_KV8_ENABLE", "0") == "1"',s)
        self.assertIn('_KV4_ENABLED = _gl_os.environ.get("HTR_KV4_ENABLE", "0") == "1"',s)
        self.assertIn('(["int8_per_token_head"] if _KV8_ENABLED else [])',s)

if __name__=='__main__': unittest.main(verbosity=2)
