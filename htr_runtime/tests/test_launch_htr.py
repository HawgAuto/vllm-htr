"""CPU-only launcher and runtime-environment contract tests (repo checkout or /opt/vllm-htr in the image)."""
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent


def launcher():
    spec = importlib.util.spec_from_file_location('launch_htr', ROOT / 'launch_htr.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build(module, *flags):
    ap = module.parser()
    args = ap.parse_args(list(flags))
    module.check(ap, args)
    return args, module.command(args)


def value(argv, k):
    return argv[argv.index(k) + 1]


class LaunchContract(unittest.TestCase):
    def test_release_defaults_preserve_geometry_and_enable_thinking(self):
        m = launcher()
        args, argv = build(m)
        self.assertEqual(value(argv, '--max-model-len'), '262144')
        self.assertEqual(value(argv, '--kv-cache-memory-bytes'), str(8 * 1024 ** 3))
        self.assertEqual(value(argv, '--max-num-seqs'), '1')
        self.assertEqual(value(argv, '--max-num-batched-tokens'), '12288')
        self.assertEqual(value(argv, '--generation-config'), 'auto')
        self.assertEqual(json.loads(value(argv, '--default-chat-template-kwargs')), {'enable_thinking': True})
        self.assertEqual(json.loads(value(argv, '--speculative-config')),
                         {'method': 'mtp', 'model': '/model/draft', 'num_speculative_tokens': 3,
                          'draft_sample_method': 'probabilistic'})
        self.assertEqual(json.loads(value(argv, '--compilation-config')),
                         {'cudagraph_capture_sizes': [1, 2, 3, 4, 5], 'cudagraph_mode': 'FULL_AND_PIECEWISE',
                          'max_cudagraph_capture_size': 5,
                          'inductor_compile_config': {'combo_kernels': False, 'benchmark_combo_kernel': False}})
        self.assertIn('--enable-prefix-caching', argv)
        self.assertNotIn('--hf-overrides', argv)
        self.assertNotIn('--enforce-eager', argv)

    def test_combo_kernel_benchmark_off_under_deterministic_inductor(self):
        # htr_autotune_pin keeps Inductor deterministic mode on; torch then bans non-vetted benchmarking, and vLLM's
        # default inductor_compile_config (combo_kernels + benchmark_combo_kernel) would abort the first compile.
        # Combo kernels are disabled explicitly (both keys, so vLLM does not inject its defaults); the reference
        # compiles produced zero combo kernels, so the kernel set is unchanged.
        m = launcher()
        for extra in ((), ('--mtp', '0', '--max-num-seqs', '4', '--kv-cache-gib', '6', '--context', '196608')):
            args, argv = build(m, *extra)
            ic = json.loads(value(argv, '--compilation-config'))['inductor_compile_config']
            self.assertIs(ic['combo_kernels'], False)
            self.assertIs(ic['benchmark_combo_kernel'], False)

    def test_levers(self):
        m = launcher()
        args, argv = build(m, '--mtp', '0', '--max-num-seqs', '4', '--kv-cache-gib', '6', '--context', '196608')
        self.assertNotIn('--speculative-config', argv)
        self.assertEqual(value(argv, '--max-num-seqs'), '4')
        self.assertEqual(value(argv, '--max-model-len'), '196608')
        self.assertEqual(value(argv, '--kv-cache-memory-bytes'), str(6 * 1024 ** 3))
        self.assertEqual(json.loads(value(argv, '--compilation-config'))['cudagraph_capture_sizes'], [1, 2, 3, 4])
        args, argv = build(m, '--mtp', '3', '--max-num-seqs', '4')
        self.assertEqual(json.loads(value(argv, '--compilation-config'))['cudagraph_capture_sizes'],
                         [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20])
        args, argv = build(m, '--context', '1048576', '--mtp', '0')
        self.assertEqual(args.yarn_factor, 4.0)
        rope = json.loads(value(argv, '--hf-overrides'))['text_config']['rope_parameters']
        self.assertEqual((rope['rope_type'], rope['factor'], rope['original_max_position_embeddings']),
                         ('yarn', 4.0, 262144))
        args, argv = build(m, '--mtp', '0', '--max-num-seqs', '8', '--cudagraph-sizes', '8,1,2,4')
        self.assertEqual(json.loads(value(argv, '--compilation-config'))['cudagraph_capture_sizes'], [1, 2, 4, 8])
        for bad in (['--mtp', '3', '--max-num-seqs', '5'], ['--context', '600000', '--yarn-factor', '2'],
                    ['--mtp', '0', '--max-num-seqs', '33']):
            with self.assertRaises(SystemExit):
                build(m, *bad)

    def test_cache_profiles_separate_compile_relevant_levers_only(self):
        m = launcher()
        a, _ = build(m)
        b, _ = build(m, '--kv-cache-gib', '20', '--port', '9000')
        c, _ = build(m, '--mtp', '0')
        d, _ = build(m, '--context', '524288')
        self.assertEqual(m.profile(a)[0], m.profile(b)[0])
        self.assertEqual(len({m.profile(x)[0] for x in (a, c, d)}), 3)

    def test_runtime_env_lists_only_shipped_features(self):
        env = json.loads((ROOT / 'runtime-env.json').read_text())
        for k in ['HTR_QSA_TAU', 'HTR_AUTOTUNE_PIN', 'HTR_MOE_PARTITION4', 'E59_PLE_DEV', 'GP8HC_HIP', 'GP8HC_HIPP',
                  'HTR_PLE_HOIST', 'Q38SPEC_DRAFT_FILTER']:
            self.assertEqual(env[k], '1')
        for k in ['E95_QSA_MTP1600', 'E26_QSA_W168', 'DIRECT_QSA_UNION_NATIVE', 'DIRECT_QSA_CAP', 'DIRECT_QSA_NATIVE',
                  'E82_GRAPH_PROOF_DIR', 'HTR_DQ_CACHE']:
            self.assertNotIn(k, env)
        self.assertNotIn('/home/', json.dumps(env))

    def test_retired_research_paths_are_gone(self):
        python = ROOT / 'python'
        for name in ['direct_qsa_union_adapter_v1.py', 'e95_qsa_1600.py', 'e82_graphproof.py', 'e94_geometry.py']:
            self.assertFalse((python / name).exists(), name)
        self.assertNotIn('DQ_CACHE', (python / 'htr/ops.py').read_text())
        plugins = sorted(p.name for p in python.glob('*.dist-info'))
        self.assertEqual(plugins, ['gp8hc-0.1.0.dist-info', 'htr_autotune_pin-0.1.0.dist-info',
                                   'htr_moe_serve-0.1.0.dist-info', 'htr_toolbox-0.1.0.dist-info',
                                   'q38spec-0.1.0.dist-info'])

    @unittest.skipUnless((REPO / 'vllm/models/qwen4_exp/amd/qsa.py').is_file(), 'repo checkout only')
    def test_qsa_owner_calls_htr_dispatch(self):
        qsa = (REPO / 'vllm/models/qwen4_exp/amd/qsa.py').read_text()
        self.assertNotIn('direct_qsa_union', qsa)
        self.assertIn('from htr_qsa import attention', qsa)
        self.assertIn('from htr_qsa import gate_apply_', qsa)
        self.assertNotIn('e82_graphproof', (REPO / 'vllm/models/qwen4_exp/amd/model.py').read_text())

    def test_tau_kernel_is_source_built_and_pinned(self):
        manifest = json.loads((ROOT / 'NATIVE-BUILD.json').read_text())
        row, = manifest['hsaco']
        self.assertIn('-save-temps', row['command'])
        self.assertIn(row['required_sha256'], (ROOT / 'python/htr_qsa.py').read_text())
        self.assertIn(row['source'], manifest['files'])


if __name__ == '__main__':
    unittest.main()
