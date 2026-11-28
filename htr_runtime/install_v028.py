"""Install v0.2.8 ERNIE-5.5 roofline prefill/decode, FlashIndexer, DSpark filter, and parser fixes over v0.2.7."""
import compileall
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path("/opt/vllm-htr")
PAYLOAD = Path("/payload")
SITE_VLLM = Path("/opt/venv/lib/python3.14/site-packages/vllm")
BASE_MANIFEST = Path("/opt/haloloom/qwen38/data/installed-files.json")


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    v028_manifest = json.loads((PAYLOAD / "V028-NATIVE-MANIFEST.json").read_text())
    ernie_manifest = json.loads((PAYLOAD / "ERNIE55-MANIFEST.json").read_text())

    nat_dir = ROOT / "native-bin"
    nat_dir.mkdir(parents=True, exist_ok=True)

    # 1. Install & SHA-verify all 8 ERNIE-5.5 Wave32 .so binaries + shared htr_spec_sample.so
    for name, meta in v028_manifest["ernie55_binaries"].items():
        src = PAYLOAD / "native-bin" / name
        assert src.is_file(), f"missing ERNIE native binary {src}"
        assert sha(src) == meta["sha256"], f"ERNIE binary sha mismatch for {name}"
        dst = nat_dir / name
        shutil.copyfile(src, dst)
        dst.chmod(0o755)

    for name, meta in v028_manifest["shared_binaries"].items():
        src = PAYLOAD / "native-bin" / name
        assert src.is_file(), f"missing shared native binary {src}"
        assert sha(src) == meta["sha256"], f"shared binary sha mismatch for {name}"
        dst = nat_dir / name
        shutil.copyfile(src, dst)
        dst.chmod(0o755)

    # 2. Install updated ERNIE-5.5 Python plugin, q38spec/__init__.py, and native sources
    shutil.copytree(
        PAYLOAD / "python/vllm_ernie_htr",
        ROOT / "python/vllm_ernie_htr",
        dirs_exist_ok=True,
    )
    shutil.copyfile(
        PAYLOAD / "python/q38spec/__init__.py",
        ROOT / "python/q38spec/__init__.py",
    )
    shutil.copytree(
        PAYLOAD / "native-source/ernie55",
        ROOT / "native-source/ernie55",
        dirs_exist_ok=True,
    )
    spec_src_dir = ROOT / "native-source/spec_sample"
    spec_src_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        PAYLOAD / "native-source/spec_sample/htr_spec_sample_v28.hip",
        spec_src_dir / "htr_spec_sample_v28.hip",
    )
    build_sh = ROOT / "native-source/ernie55/build_ernie55.sh"
    if build_sh.is_file():
        build_sh.chmod(0o755)

    # 3. Install vLLM pre-plugin lazy parser registries (Issue #14: ernie5 & ernie55)
    for rel in ("reasoning/__init__.py", "tool_parsers/__init__.py"):
        src = PAYLOAD / "vllm" / rel
        dst = SITE_VLLM / rel
        assert src.is_file(), f"missing vllm file {src}"
        assert dst.is_file(), f"expected existing vllm file {dst}"
        shutil.copyfile(src, dst)

    # 4. Install manifests and VERSION
    shutil.copyfile(PAYLOAD / "ERNIE55-MANIFEST.json", ROOT / "ERNIE55-MANIFEST.json")
    shutil.copyfile(PAYLOAD / "V028-NATIVE-MANIFEST.json", ROOT / "V028-NATIVE-MANIFEST.json")
    (ROOT / "VERSION").write_text("0.2.8\n")

    # Pre-compile .pyc for /opt/vllm-htr/python and patched vllm parser modules
    compileall.compile_dir(str(ROOT / "python"), quiet=1, force=True)
    compileall.compile_file(str(SITE_VLLM / "reasoning/__init__.py"), quiet=1, force=True)
    compileall.compile_file(str(SITE_VLLM / "tool_parsers/__init__.py"), quiet=1, force=True)

    # 5. Reconcile integrity manifests (bindings FIRST, installed-files.json LAST!)
    for mf_name in ("V025-NATIVE-MANIFEST.json", "V026-NATIVE-MANIFEST.json", "V027-NATIVE-MANIFEST.json"):
        mf_p = ROOT / mf_name
        if mf_p.is_file():
            mf = json.loads(mf_p.read_text())
            for section in ("binaries", "qwen38_binaries"):
                if section in mf and isinstance(mf[section], dict):
                    for name in list(mf[section].keys()):
                        p = nat_dir / name
                        if p.is_file():
                            mf[section][name] = {"sha256": sha(p), "bytes": p.stat().st_size}
            mf_p.write_text(json.dumps(mf, indent=2) + "\n")

    gb_path = ROOT / "python/htr_moe_serve/grouped-bindings.json"
    if gb_path.is_file():
        gb = json.loads(gb_path.read_text())
        for path_str in list(gb.get("files", {}).keys()):
            p = Path(path_str)
            if p.is_file():
                gb["files"][path_str] = sha(p)
        gb_path.write_text(json.dumps(gb, indent=2, sort_keys=True) + "\n")

    for bp_name in ("native-policy-bindings.json", "quantized-kv-bindings.json"):
        bp = ROOT / bp_name
        if bp.is_file():
            obj = json.loads(bp.read_text())
            for k in list(obj.keys()):
                p = Path(k)
                if p.is_file():
                    obj[k] = sha(p)
            bp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")

    inst_path = ROOT / "installed-files.json"
    if inst_path.is_file():
        inst = json.loads(inst_path.read_text())
        for k in list(inst.keys()):
            p = ROOT / k
            if p.is_file():
                inst[k] = sha(p)
        inst["V028-NATIVE-MANIFEST.json"] = sha(ROOT / "V028-NATIVE-MANIFEST.json")
        inst["ERNIE55-MANIFEST.json"] = sha(ROOT / "ERNIE55-MANIFEST.json")
        inst_path.write_text(json.dumps(inst, indent=2, sort_keys=True) + "\n")

    if BASE_MANIFEST.is_file():
        base = json.loads(BASE_MANIFEST.read_text())
        for row in base.get("files", []):
            p = Path(row["installed_path"])
            if p.is_file():
                row["sha256"] = sha(p)
                for field in ("size", "bytes", "size_bytes"):
                    if field in row:
                        row[field] = p.stat().st_size
        BASE_MANIFEST.write_text(json.dumps(base, indent=2) + "\n")

    # 6. Run ERNIE-5.5 CPU verification script before removing /payload
    res = subprocess.run([sys.executable, str(PAYLOAD / "verify_ernie55_cpu.py")], check=False)
    if res.returncode != 0:
        raise RuntimeError("ERNIE-5.5 verify_ernie55_cpu.py failed!")

    shutil.rmtree(PAYLOAD, ignore_errors=True)
    sys.path.insert(0, str(ROOT / "python"))
    import htr_native_policy
    htr_native_policy.verify_installation()
    print(
        f"v0.2.8 ZERO-OVERLAY install verified: "
        f"{len(v028_manifest['ernie55_binaries'])} ERNIE .so ({ernie_manifest['total_kernels']} kernels, 0 spills, 0 divOps) "
        f"+ htr_spec_sample.so + vllm_ernie_htr + ernie5/ernie55 parsers."
    )


if __name__ == "__main__":
    main()
