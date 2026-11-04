# Optional bounded PLE row cache

Experimental, byte-preserving native row cache for HTR's SSD-backed PLE table.
The default is **OFF**. This source commit does not change the production entrypoint,
publish an image, or claim that an SSD slowdown has been fixed.

## Build and test

On Linux with a C11 compiler and Python 3:

```sh
python3 htr_runtime/experimental/ple_row_cache/build.py
python3 -m unittest discover -s htr_runtime/experimental/ple_row_cache -p 'test_*.py' -v
```

The tests use deterministic generated table bytes and an independently built copy
of the unchanged native gather as the oracle. Set `HTR_TEST_MODEL_DIR` to a local
HTR checkpoint to additionally run the actual 128-shard PLE byte oracle; without
it, that one test is explicitly skipped. No GPU is used. `capacity_stress.py` is a
separate opt-in full-capacity CPU stress, not part of the normal small test run;
running that script fills the 2/4/8 GiB budgets in sequence (plus its sparse-memfd
backing and scratch). Ensure sufficient free RAM before explicitly running it.

`QUALIFICATION.json` binds the original cache, launcher and vendor source hashes
and retains non-private summary measurements. The original research tests and raw
model evidence remain unchanged outside this package. Only the public test copy's
checkpoint locator was made configurable. No binaries, model files, private
prompts, generated answers, compile caches or local machine paths are committed.

## Use in a compatible HTR image

Build the library, bind this directory read-only at `/research`, and select:

```text
/opt/venv/bin/python3 /research/launch_cache.py [normal HTR options] --ple-row-cache-gib 2
```

The adapter calls the image's unchanged `/opt/vllm-htr/launch_htr.py`. With no new
option, or `--ple-row-cache-gib 0`, it leaves the original native library selected.
Positive values opt in; 2, 4 and 8 GiB have real MTP0/MTP3 evidence. Fractions are
accepted, but other sizes are not serving-qualified by this evidence. The observer
flag is for research only. This adapter is not yet part of a published image and
does not implicitly compose the quantized-KV or recommended-sampler entrypoints.

Use a fresh writable `/workspace` for each isolated process: the fixed stats file
is intentionally created exclusively and stale files are rejected, not deleted.
This package is a research adapter, not a qualified restart/multi-worker manager.

## Contract

- Same `ple_gather_ids` C ABI beneath both synchronous and E59 asynchronous callers.
- Original row bytes; no PLE quantization, model mutation or floating-point changes.
- Fixed-capacity LRU; the per-engine-process budget includes rows, index/LRU
  metadata, fixed cache bookkeeping and allocation rounding. TP1 was tested.
- Capacity is reserved; row pages are populated on demand. Eviction reuses slots.
- The ceiling is not a total-server RAM limit. Gather scratch, framework/model/KV
  allocations, OS page cache and other processes are separate.
- Source identity and geometry are checked, including cached hits. Mutated,
  truncated or invalid sources fail rather than returning stale data.
- Shared native counters expose hits, misses, errors, allocation and file reads.
  File-read calls/bytes are not physical disk I/O.

## Evidence and limits

MTP0 and MTP3 passed at 2/4/8 GiB in the pinned incumbent image: all generated token
sequences, streamed text and teacher choice/logprob structures matched their
same-mode native control. Seeded compile/tuning artifacts were unchanged. Native
cache errors were zero and every captured allocation respected its budget.

The timing profile was one sequence, 32768 context, a 2 GiB BF16 KV pool, 2048-token
prefill chunks and 512/2048/8192-token excerpts with 128 forced output tokens. The
latter is timing stress, not a natural-termination quality benchmark. Ordinary
arithmetic chat and teacher checks were separate. The smallest cache already held
the working set: row payload was about 24–25 MiB, excluding metadata, with no
serving evictions. Separate full-capacity CPU stress covered 2/4/8 GiB eviction.

Warmed cache-on requests issued zero PLE file reads, but overall speed changes were
small and mixed. There is no demonstrated general speedup, larger-budget benefit,
or matched slow-storage-mode benefit. Cache-arena RSS was not isolated from
whole-process memory. Do not transfer this qualification to the vLLM v0.33 rebase,
full-context/concurrent serving, or other KV combinations without their own checks.
