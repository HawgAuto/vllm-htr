"""Installed-package CPU preflight (real loader/entrypoint; no fake GPU arithmetic)."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import torch
import q38spec
import htr_q38f

orig = q38spec.q38_filter_logits
os.environ['HTR_Q38F'] = 'off'
htr_q38f.register()
assert q38spec.q38_filter_logits is orig
assert any(e.name == 'htr_q38f' for e in importlib.metadata.entry_points(group='vllm.general_plugins'))
os.environ['HTR_Q38F'] = 'on'
htr_q38f.register()
wrapped = q38spec.q38_filter_logits
assert wrapped is not orig
htr_q38f.register()
assert q38spec.q38_filter_logits is wrapped
# CPU/nonserving input falls back to the actual original implementation.
x = torch.tensor([[1., 3., 2., -1.]], dtype=torch.float32)
args = (torch.tensor([0], dtype=torch.int32), torch.ones(1), torch.tensor([2], dtype=torch.int32), torch.tensor([0.95]))
a = orig(x, *args); b = wrapped(x, *args)
assert torch.equal(a.view(torch.int32), b.view(torch.int32))
manifest = htr_q38f.validate_payload()
assert manifest['library_sha256'] == hashlib.sha256(Path(htr_q38f.LIB).read_bytes()).hexdigest()
print(json.dumps(dict(status='PASS', file=htr_q38f.__file__, entries=True, inert_off=True, idempotent=True,
                      cpu_fallback_exact=True, manifest=manifest)), flush=True)
