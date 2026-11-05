"""Dependency-free source regression gates; not GPU/numerical qualification."""
import ast
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
QSA = ROOT / 'vllm/models/qwen4_exp/amd/qsa.py'


class OctavePort(unittest.TestCase):
    def test_parent_dtype_swap(self):
        tree = ast.parse(QSA.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                   and n.name == 'Qwen4ExpQSAFlashAttentionImpl')
        init = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                    and n.name == '__init__')
        src = ast.unparse(init)
        self.assertIn('wanted in _OCTAVE_CACHE_DTYPES', src)
        self.assertIn("parent_kwargs['kv_cache_dtype'] = 'bfloat16'", src)
        self.assertIn('self.kv_cache_dtype = wanted', src)
        self.assertIn('wanted not in _OCTAVE_CACHE_DTYPES', src)

    def test_single_writer_preserves_all_formats(self):
        tree = ast.parse(QSA.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                   and n.name == 'Qwen4ExpQSAFlashAttentionImpl')
        writers = [n for n in cls.body if isinstance(n, ast.FunctionDef)
                   and n.name == 'do_kv_cache_update']
        self.assertEqual(len(writers), 1)
        src = ast.unparse(writers[0])
        for needle in ('octave_cache_store', 'octave_window_store',
                       'htr_qsa_kv4', 'htr_qsa_kv8', 'reshape_and_cache_flash'):
            self.assertIn(needle, src)

    def test_no_dtype_override(self):
        tree = ast.parse(QSA.read_text())
        for name in ('_QSA_CACHE_DTYPES', '_OCTAVE_CACHE_DTYPES', '_PACKED_CACHE_DTYPES'):
            definitions = [n for n in tree.body if isinstance(n, ast.AnnAssign)
                           and isinstance(n.target, ast.Name) and n.target.id == name]
            self.assertEqual(len(definitions), 1, name)
        self.assertIn('and self.kv_cache_dtype not in _PACKED_CACHE_DTYPES', QSA.read_text())


if __name__ == '__main__':
    unittest.main()
