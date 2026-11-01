"""CPU-only: HTR autotune pin keeps Inductor deterministic mode on across Dynamo frames."""
import unittest

import torch


class AutotunePin(unittest.TestCase):
    def test_deterministic_survives_dynamo_restore(self):
        import torch._inductor.config as ic
        import htr_autotune_pin
        htr_autotune_pin.register()
        self.assertTrue(ic.deterministic)
        # Dynamo restores the eager flag after every traced frame with exactly this call; torch>=2.13 mirrors it into
        # inductor config.deterministic.
        torch.use_deterministic_algorithms(False, warn_only=False)
        self.assertTrue(ic.deterministic)
        self.assertFalse(torch.are_deterministic_algorithms_enabled())

    def test_deterministic_survives_compiled_frames(self):
        import torch._inductor.config as ic
        import htr_autotune_pin
        htr_autotune_pin.register()

        def f(x):
            return x.sin() + 1

        def g(x):
            return x.cos() * 2

        x = torch.randn(8)
        torch.compile(f, backend='eager')(x)
        self.assertTrue(ic.deterministic)
        torch.compile(g, backend='eager')(x)
        self.assertTrue(ic.deterministic)


if __name__ == '__main__':
    unittest.main()
