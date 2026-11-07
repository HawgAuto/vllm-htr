#!/bin/bash
# Offline native core build; no GPU devices or host ROCm are required.
set -euo pipefail
export PATH=/opt/venv/bin:/opt/rocm/core-10.0/bin:/usr/local/bin:/usr/bin:/bin
export PYTORCH_ROCM_ARCH=gfx1151 VLLM_TARGET_DEVICE=rocm MAX_JOBS=4
export TRITON_KERNELS_SRC_DIR=/deps/triton_kernels
unset HSA_OVERRIDE_GFX_VERSION
mkdir -p /build/tmp
export TMPDIR=/build/tmp
cmake -S /source -B /build/cmake -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DVLLM_TARGET_DEVICE=rocm -DVLLM_PYTHON_EXECUTABLE=/opt/venv/bin/python3 -DCMAKE_HIP_ARCHITECTURES=gfx1151 -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DFETCHCONTENT_SOURCE_DIR_TRITON_KERNELS=/deps/triton_kernels
cmake --build /build/cmake --target _C --parallel 4
/opt/venv/bin/python3 -c 'import torch; from pathlib import Path; p=next(Path("/build/cmake").glob("_C*.so")); torch.ops.load_library(str(p)); names={str(s.name) for s in torch._C._jit_get_all_schemas()}; assert "_C::octave_decode_sparse" in names; print("HTR_NATIVE_CORE_IMPORT_OK",p)'
