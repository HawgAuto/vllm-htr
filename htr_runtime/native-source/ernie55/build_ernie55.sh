#!/usr/bin/env bash
set -euo pipefail
SRC_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${1:-/opt/vllm-htr/native-bin}"
mkdir -p "$OUT_DIR"
export TMPDIR="${TMPDIR:-/tmp}"

PYINC=$(python3 -c 'import sysconfig; print(sysconfig.get_path("include"))')
TORCH_DIR=$(python3 -c 'import torch, os; print(os.path.dirname(torch.__file__))')
COMMON="-O3 --offload-arch=gfx1151 -shared -fPIC -std=c++20 -DUSE_ROCM -D__HIP_PLATFORM_AMD__ -I${PYINC} -I${TORCH_DIR}/include -I${TORCH_DIR}/include/torch/csrc/api/include -L${TORCH_DIR}/lib -Wl,-rpath,${TORCH_DIR}/lib -ltorch -ltorch_cpu -ltorch_hip -lc10 -lc10_hip -ltorch_python"

echo "Building 8 ERNIE-5.5 gfx1151 Wave32 HIP extensions into ${OUT_DIR}..."
hipcc $COMMON -DTORCH_EXTENSION_NAME=htr_moe_ernie "$SRC_DIR/htr_moe_ernie.hip" "$SRC_DIR/htr_moe_ernie.cpp" -o "$OUT_DIR/htr_moe_ernie.so" &
hipcc $COMMON -DTORCH_EXTENSION_NAME=htr_wo_a_ernie "$SRC_DIR/htr_wo_a_ernie.hip" "$SRC_DIR/htr_wo_a_ernie.cpp" -o "$OUT_DIR/htr_wo_a_ernie.so" &
hipcc $COMMON -DTORCH_EXTENSION_NAME=mhc_ernie "$SRC_DIR/mhc_ernie.hip" "$SRC_DIR/mhc_ernie.cpp" -o "$OUT_DIR/mhc_ernie.so" &
hipcc $COMMON "$SRC_DIR/hpf5_ernie.hip" -o "$OUT_DIR/hpf5_ernie.so" &
hipcc $COMMON "$SRC_DIR/ernie_fast_ops.hip" -o "$OUT_DIR/ernie_fast_ops.so" &
hipcc $COMMON "$SRC_DIR/ernie_hca_attn.hip" -o "$OUT_DIR/ernie_hca_attn.so" &
hipcc $COMMON "$SRC_DIR/ernie_dsa_attn.hip" -o "$OUT_DIR/ernie_dsa_attn.so" &
hipcc $COMMON "$SRC_DIR/ernie_k5_sp_prefill.hip" -o "$OUT_DIR/ernie_k5_sp_prefill.so" &
if [ -f "$SRC_DIR/../spec_sample/htr_spec_sample_v28.hip" ]; then
  hipcc $COMMON "$SRC_DIR/../spec_sample/htr_spec_sample_v28.hip" -o "$OUT_DIR/htr_spec_sample.so" &
fi
wait

echo "Auditing gfx1151 ISA (VGPRs, spills, divOps)..."
python3 - "$OUT_DIR" <<'AUDIT'
import os, re, subprocess, sys
out_dir = sys.argv[1]
libs = [
    "htr_moe_ernie.so", "hpf5_ernie.so", "htr_wo_a_ernie.so", "mhc_ernie.so",
    "ernie_fast_ops.so", "ernie_hca_attn.so", "ernie_dsa_attn.so", "ernie_k5_sp_prefill.so",
]
# htr_spec_sample.so uses exact FP32 probability division (checked separately for vSpill==0, scrOps==0)
total_k = 0
for so in libs:
    p = os.path.join(out_dir, so)
    fat, hsaco = f"/tmp/{so}.hipfb", f"/tmp/{so}.hsaco"
    subprocess.check_call(["/opt/rocm/llvm/bin/llvm-objcopy", "--dump-section", f".hip_fatbin={fat}", p])
    subprocess.check_call(["/opt/rocm/llvm/bin/clang-offload-bundler", "-type=o", "-targets=hipv4-amdgcn-amd-amdhsa--gfx1151", f"-input={fat}", f"-output={hsaco}", "-unbundle"])
    note = subprocess.check_output(["/opt/rocm/llvm/bin/llvm-readobj", "--notes", hsaco], text=True, errors="ignore")
    dis = subprocess.check_output(["/opt/rocm/llvm/bin/llvm-objdump", "-d", "--mcpu=gfx1151", hsaco], text=True, errors="ignore")
    vgprs = [int(x) for x in re.findall(r"\.vgpr_count:\s*(\d+)", note)]
    vspills = sum(int(x) for x in re.findall(r"\.vgpr_spill_count:\s*(\d+)", note))
    sspills = sum(int(x) for x in re.findall(r"\.sgpr_spill_count:\s*(\d+)", note))
    divs = len(re.findall(r"v_div_scale|v_div_fmas|v_div_fixup", dis))
    scrs = len(re.findall(r"scratch_load|scratch_store", dis))
    total_k += len(vgprs)
    print(f"  {so:24s}: kernels={len(vgprs):3d}, maxV={max(vgprs):3d}, vSpill={vspills}, sSpill={sspills}, divOps={divs}, scrOps={scrs}")
    assert len(vgprs) > 0 and vspills == 0 and sspills == 0 and divs == 0 and scrs == 0, f"ISA AUDIT FAILED on {so}!"
    os.remove(fat); os.remove(hsaco)
spec_so = os.path.join(out_dir, "htr_spec_sample.so")
if os.path.isfile(spec_so):
    fat, hsaco = "/tmp/spec.hipfb", "/tmp/spec.hsaco"
    subprocess.check_call(["/opt/rocm/llvm/bin/llvm-objcopy", "--dump-section", f".hip_fatbin={fat}", spec_so])
    subprocess.check_call(["/opt/rocm/llvm/bin/clang-offload-bundler", "-type=o", "-targets=hipv4-amdgcn-amd-amdhsa--gfx1151", f"-input={fat}", f"-output={hsaco}", "-unbundle"])
    note = subprocess.check_output(["/opt/rocm/llvm/bin/llvm-readobj", "--notes", hsaco], text=True, errors="ignore")
    dis = subprocess.check_output(["/opt/rocm/llvm/bin/llvm-objdump", "-d", "--mcpu=gfx1151", hsaco], text=True, errors="ignore")
    vgprs = [int(x) for x in re.findall(r"\.vgpr_count:\s*(\d+)", note)]
    vsp = sum(int(x) for x in re.findall(r"\.vgpr_spill_count:\s*(\d+)", note))
    scrs = len(re.findall(r"scratch_load|scratch_store", dis))
    print(f"  {'htr_spec_sample.so':24s}: kernels={len(vgprs):3d}, maxV={max(vgprs):3d}, vSpill={vsp}, scrOps={scrs}")
    assert len(vgprs) > 0 and vsp == 0 and scrs == 0
    os.remove(fat); os.remove(hsaco)
print(f"ALL {len(libs)} ERNIE-5.5 LIBRARIES PASSED gfx1151 ISA AUDIT ({total_k} kernels, 0 spills, 0 divOps, 0 scrOps)!")
AUDIT
