import importlib.util
import json
from pathlib import Path
import unittest

class LaunchContract(unittest.TestCase):
    def test_release_defaults_preserve_geometry_and_enable_thinking(self):
        path=Path(__file__).resolve().parents[1]/'launch_htr.py'
        self.assertTrue(path.is_file(),'release launcher missing')
        spec=importlib.util.spec_from_file_location('launch_htr',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        args=module.parser().parse_args([])
        argv=module.command(args)
        def value(k):return argv[argv.index(k)+1]
        self.assertEqual(value('--max-model-len'),'262144')
        self.assertEqual(value('--kv-cache-memory-bytes'),str(8*1024**3))
        self.assertEqual(value('--max-num-seqs'),'1')
        self.assertEqual(value('--generation-config'),'auto')
        self.assertEqual(json.loads(value('--default-chat-template-kwargs')),{'enable_thinking':True})
        self.assertEqual(json.loads(value('--speculative-config')),{'method':'mtp','model':'/model/draft','num_speculative_tokens':3,'draft_sample_method':'probabilistic'})
        self.assertEqual(json.loads(value('--compilation-config')),{'cudagraph_capture_sizes':[1,2,3,4,5],'cudagraph_mode':'FULL_AND_PIECEWISE','max_cudagraph_capture_size':5})
        self.assertNotIn('--enforce-eager',argv)
        env=json.loads((path.parent/'runtime-env.json').read_text())
        for k in ['E95_QSA_MTP1600','E59_PLE_DEV','GP8HC_HIP','GP8HC_HIPP','HTR_PLE_HOIST','Q38SPEC_DRAFT_FILTER']:
            self.assertEqual(env[k],'1')
        self.assertNotIn('/home/',json.dumps(env))

if __name__=='__main__':unittest.main()
