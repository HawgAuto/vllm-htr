"""CPU-only verification of vllm-htr:v0.2.6-ernie55 plugin discovery, binaries, and Qwen3.8 isolation."""
import importlib.metadata
import json
import os
from pathlib import Path

# 1. Verify vLLM general plugin auto-discovery
eps = {ep.name: ep.value for ep in importlib.metadata.entry_points(group="vllm.general_plugins")}
print("Discovered vllm.general_plugins:", sorted(eps.keys()))
assert "ernie_htr" in eps and eps["ernie_htr"] == "vllm_ernie_htr:register", eps

# 2. Verify all 8 .so binaries exist in HTR_NATIVE_DIR and match MANIFEST
ndir = Path(os.environ["HTR_NATIVE_DIR"])
manifest = json.load(open("/opt/vllm-htr/ERNIE55-MANIFEST.json"))
nbins = manifest["native_bins"]
tk = manifest["total_kernels"]
for so, meta in nbins.items():
    p = ndir / so
    assert p.is_file(), f"Missing {p}"
    assert meta["vgpr_spills"] == 0 and meta["sgpr_spills"] == 0 and meta["div_ops"] == 0, (so, meta)
print(f"All {len(nbins)} .so binaries verified in HTR_NATIVE_DIR={ndir} ({tk} kernels, 0 spills, 0 divOps)")

# 3. Verify registration, ModelRegistry entries, and Qwen3.8 isolation (_serving_ernie5 == False)
import vllm_ernie_htr
vllm_ernie_htr.register()
from vllm import ModelRegistry
import vllm_ernie_htr.patches as e_patches

archs = ModelRegistry.get_supported_archs()
assert "Ernie5_5_MoeForCausalLM" in archs, "Missing Ernie5_5_MoeForCausalLM!"
assert "Ernie5_5_MoeDSparkDraftModel" in archs, "Missing Ernie5_5_MoeDSparkDraftModel!"
assert e_patches._serving_ernie5 is False, "Qwen3.8 isolation broken: _serving_ernie5 must start False!"
from vllm.reasoning import ReasoningParserManager
from vllm.tool_parsers import ToolParserManager
from vllm_ernie_htr.reasoning_parser import Ernie5ParserReasoningAdapter
from vllm_ernie_htr.tool_parser import Ernie5ToolParser

for pname in ("ernie5", "ernie55"):
    assert pname in ReasoningParserManager.list_registered(), f"Missing reasoning parser {pname}"
    assert pname in ToolParserManager.list_registered(), f"Missing tool parser {pname}"
    assert ReasoningParserManager.get_reasoning_parser(pname) is Ernie5ParserReasoningAdapter
    assert ToolParserManager.get_tool_parser(pname) is Ernie5ToolParser

print("ModelRegistry (target + DSpark draft), parsers (ernie5/ernie55), & Qwen3.8 isolation (_serving_ernie5=False) VERIFIED!")
print("ALL CPU-ONLY RELEASE VERIFICATION CHECKS PASSED!")
