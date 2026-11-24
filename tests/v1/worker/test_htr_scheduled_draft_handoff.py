"""Device-free regression of the production execute->sample control flow.

Exercise the real per-step state constructor and entire sample_tokens body;
replace device / model collaborators only. No vLLM installation required.
"""
from __future__ import annotations

import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any, NamedTuple
import unittest
from unittest.mock import MagicMock

GPU = Path(__file__).resolve().parents[3] / 'vllm/v1/worker/gpu'


def compile_nodes(nodes, env):
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<runner-test>', 'exec'), env)


class Draft:
    supports_mm_inputs = False

    def __init__(self):
        self.calls = []

    def propose(self, *args, **kwargs):
        self.calls.append(kwargs.get('num_spec_tokens', 'default'))
        return MagicMock()


class AR(Draft):
    pass


class Handoff(unittest.TestCase):
    def test_execute_then_sample(self):
        tree = ast.parse((GPU / 'model_runner.py').read_text())
        classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
        methods = {n.name: n for n in classes['GPUModelRunner'].body if isinstance(n, ast.FunctionDef)}
        save, = [n for n in ast.walk(methods['execute_model']) if isinstance(n, ast.Assign)
                 and isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == 'ExecuteModelState']
        fn = methods['sample_tokens']
        fn.decorator_list = []
        env: dict[str, Any] = dict(NamedTuple=NamedTuple, DraftModelSpeculator=type('OtherDraft', (), {}),
                   GPUWatermarkSampler=type('Watermark', (), {}), ModelRunnerOutput=NS,
                   AsyncOutput=NS, use_workspace_lane=lambda _: nullcontext())
        # Do not inject a missing production import; that would mask a second NameError.
        imports = [(n.module, a.name) for n in tree.body if isinstance(n, ast.ImportFrom) for a in n.names]
        if ('vllm.v1.worker.gpu.spec_decode.autoregressive.speculator', 'AutoRegressiveSpeculator') in imports:
            env['AutoRegressiveSpeculator'] = AR
        compile_nodes([classes['ExecuteModelState'], fn], env)
        for k, spec_type, expected, width in [(3, AR, [3], 3), (2, AR, [2], 2),
                                               (1, AR, [1], 1), (0, AR, [], 0),
                                               (2, Draft, ['default'], 3)]:
            with self.subTest(k=k, spec=spec_type.__name__):
                runner = MagicMock()
                runner.is_last_pp_rank = True
                runner.pp_handler = runner.pcp_manager = runner.aux_output_connector = None
                runner.adaptive_verification = None
                runner.model = NS(compute_logits=None)
                runner.num_speculative_steps = 3
                runner.speculator = spec_type()
                runner.sample.return_value = (NS(sampled_token_ids=[1]), 1, 0)
                drafts = MagicMock()
                drafts.shape = (1, 3)
                runner.req_states.draft_tokens.__getitem__.return_value = drafts
                assert isinstance(save.value, ast.Call)
                values: dict[str, Any] = {kw.arg: MagicMock() for kw in save.value.keywords if kw.arg}
                values.update(self=runner, scheduler_output=NS(num_spec_tokens_to_schedule=k),
                              input_batch=NS(idx_mapping=0, query_start_loc=None, req_ids=['test']))
                compile_nodes([save], {**env, **values})
                env['sample_tokens'](runner, None)
                self.assertIsNone(runner.execute_model_state)
                self.assertEqual(runner.speculator.calls, expected)
                seen = runner.draft_tokens_handler.set_draft_tokens.call_args.args[1]
                if width == 3:
                    self.assertIs(seen, drafts)
                else:
                    drafts.__getitem__.assert_called_once_with((slice(None), slice(None, width)))
                    self.assertIs(seen, drafts.__getitem__.return_value)

    def test_synthetic_warmup_requests_set_next_depth(self):
        tree = ast.parse((GPU / 'warmup.py').read_text())
        for fn, variables in [('_warmup_kernels', ['prefill_output', 'decode_output']),
                              ('run_mixed_prefill_decode_warmup', ['decode_prefill_output', 'mixed_output'])]:
            method, = [n for n in tree.body if getattr(n, 'name', None) == fn]
            for var in variables:
                with self.subTest(fn=fn, output=var):
                    obj = NS(num_spec_tokens_to_schedule=0)
                    writes = [n for n in ast.walk(method) if isinstance(n, ast.Assign)
                              and ast.unparse(n.targets[0]) == var + '.num_spec_tokens_to_schedule']
                    compile_nodes(writes, {var: obj, 'num_spec_steps': 3, 'model_runner': NS(num_speculative_steps=3)})
                    self.assertEqual(obj.num_spec_tokens_to_schedule, 3)


if __name__ == '__main__':
    unittest.main()
