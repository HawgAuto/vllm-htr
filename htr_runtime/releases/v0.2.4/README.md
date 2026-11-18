# HTR v0.2.4 — faster single-stream decode and safer PLE startup

The selected K6 routed-MoE decode kernel (2–4 rows), dense WMMA extension to
eligible 9–15-row projections, and asynchronous-PLE zero-row repair are now
native runtime defaults. Ordinary `vllm serve` receives them without overlays.
Weights, tokenizer/configuration, KV formats and sampling defaults are unchanged.
The empty-workspace cold-start fix from v0.2.3-coldstart.1 is included.

## What changed versus v0.2.3

These are the completed selected-stack screens, not a new performance benchmark
of the packaged image. Radeon 8060S/gfx1151, Htr5Core, MTP3, maximum four sequences,
40,960 configured context; two mirrored visits per arm.

| Workload / metric | v0.2.3 | Selected v0.2.4 stack | Change |
|---|---:|---:|---:|
| Steady 1-stream aggregate output rate | 48.05 tok/s | 55.35 tok/s | +15.19% |
| Steady 2-stream aggregate output rate | 60.34 tok/s | 56.96 tok/s | −5.60% |
| Steady 3-stream aggregate output rate | 61.31 tok/s | 62.80 tok/s | +2.44% |
| Steady 4-stream aggregate output rate | 68.29 tok/s | 71.11 tok/s | +4.14% |
| Mixed real prompts, 1 stream, output / full wall | 34.08 tok/s | 36.41 tok/s | +6.82% |
| Mixed real prompts, 1 stream, per-request decode p50 | 48.10 tok/s | 53.83 tok/s | +11.91% |
| Mixed real prompts, 2 streams, output / full wall | 46.52 tok/s | 47.35 tok/s | +1.77% |
| Mixed real prompts, 3 streams, output / full wall | 47.61 tok/s | 49.74 tok/s | +4.46% |
| Mixed real prompts, 4 streams, output / full wall | 51.46 tok/s | 51.95 tok/s | +0.96% |

The steady test uses approximately 100 input tokens and a 512-token output cap;
the mixed test uses 24 prompts at each concurrency with a 256-token output cap.
Rates above are medians over the two visits, not summed isolated kernel gains.
There is no established 2–4-stream improvement: observed changes vary by workload,
including the −5.60% steady two-stream result. Two visits do not establish a
statistical noise bound. No new prefill gain or maximum-context claim is made.
The historical throughput targets remain unmet.

## Task-quality screen

262,144 configured context, MTP3, INT8 per-token/head KV, maximum four sequences,
2 GiB optional PLE hot-row cache, thinking on, xhigh reasoning effort, temperature
1.0 / top-p 0.95 / top-k 20 / min-p 0 / presence 0 / repetition 1.0. Evaluation
seed 42 is a harness setting, not a shipped launcher default.

| Suite | v0.2.3 | Selected v0.2.4 stack |
|---|---:|---:|
| HumanEval+ | 159 / 164 | 162 / 164 |
| ToolEval (69 cases, partial-credit points) | 125 / 138 | 125 / 138 |
| Hermes20 (20 cases, local semantic edition) | 1850 / 2000 | 1950 / 2000 |

The complete 164-item native HumanEval results were retained even though the
outer bridge later stopped on its duplicate-generation safeguard. ToolEval and
Hermes20 completed in the successor harness. These are finite sampled screens,
not proof of general quality improvement or whole-model bitwise equivalence to
v0.2.3. The PLE hot-row cache remains off in shipped defaults.

Known limitation: HumanEval/157 entered repetitive reasoning on the selected
stack with seed 42 (reproduced twice under a 1200-second deadline), but not with
seeds 1, 2, 3 or 101–140. Across 44 distinct tested seeds, that is one loop-triggering
seed for the selected stack and none for v0.2.3; it is not an estimated population
loop rate. The 40 fresh-seed follow-up completed all requests: candidate 40/40
function tests passed, predecessor 39/40. A forced-prefix diagnostic reproduced
the repetitive attractor in both stacks. The release does not claim to fix
model-level reasoning loops.

[Machine-readable selected-stack rows and input hashes](SELECTED-STACK.json)
keep every visit and the K6-only intermediate arm separate.

## Runtime changes and controls

- `HTR_MOE_K6=1`, `HTR_MOE_K6_ROWS=2-4`: K6 is the outer routed-MoE wrapper;
  the existing partition/grouped parity checks and predecessor native library
  are retained. Set `HTR_MOE_K6=0` to disable it.
- `HTR_DENSE_WMMA_MIN9=1`: widen eligible dense projections from 16 to 9 rows.
  Set it to 0 to restore the 16-row floor. Existing shape-specific exceptions
  still apply; this is not an all-shapes WMMA path.
- `HTR_PLE_ASYNC_MAXN=0`: disable asynchronous PLE instead of allocating a
  zero-byte coherent buffer. The asynchronous self-check wait is bounded.
- K6 is compiled from pinned source, verified at installation and included in
  the installed native integrity bindings. Both K6 and dense WMMA rebuild
  byte-identically through the public source builder.
- No shipped sampling seed is added. Upstream vLLM's default seed remains 0;
  clients retain native sampling and seed controls.

## Qualification and scope

[Final-image qualification](QUALIFICATION.json) records the exact image digest,
source commit, actual startup times, matched greedy requests in MTP3 and MTP0,
route markers, tool-call delivery and CPU tests. Short output identity checks
are packaging evidence, not whole-model mathematical equivalence.

| Fresh process | Workspace at launch | Ready time |
|---|---|---:|
| Selected-stack reference, MTP3/INT8 | Copy of campaign cache | 165.5 s |
| Packaged image, MTP3/INT8 | Independent copy of same cache | 391.2 s |
| Packaged image, documented MTP3/BF16 command | Empty | 436.1 s |
| Selected-stack reference, MTP0/BF16 | Copy of completed card cache | 290.9 s |
| Packaged image, MTP0/BF16 | Independent copy of completed card cache | 290.8 s |

The packaged image used a different bound cache identity and compiled graphs
where the MTP3 reference loaded AOT cache entries. These are actual ready times,
not a controlled cold-start speed comparison or a warm-throughput regression.
All three greedy cases in both modes matched numeric token IDs and logprobs.
Four concurrent requests completed on each MTP3 task-geometry visit; the
packaged image also completed the tool/result round trip in both MTP3 geometries.
Speculative activity was observed in server metrics, including35 accepted of36
drafted tokens in one documented-command interval (a smoke, not a benchmark).
The reference MTP3 and packaged MTP0 logs contain an AsyncLLM output-handler
`EngineDeadError` after commanded shutdown and engine-manager stop; there were
no serving-time ERROR/Traceback lines. Cleanup removed the owned test container
and freed the test listener and GPU locks; this release does not activate a
production service.

The documented start uses Htr5Core, BF16 KV, MTP3, one sequence, configured 32,768
context and a 2 GiB KV pool. Optional formats and earlier million-context records
retain their original release scope; they were not exhaustively rerun for this
update. No prebuilt compiler cache is shipped. First start and new shapes may
compile. Bf16Core and multimodal serving remain separately unqualified.

[Current launch and source-build instructions](../../docs/optimized-serving.md).
