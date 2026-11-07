# HTR v0.2.2 — completed local qualification

All 18 bounded model configurations passed: 16 loads on the fixed combined image and two explicitly inherited, unchanged accepted controls. The reducer verified 324 requests (234 nonstreaming, 90 raw SSE streams). New-image requests: 288. Re-running the reducer produced byte-identical output.

Image: `sha256:034898ebd182a240fa403ad5ea24bea3f647d933571e43420a27daf9af079d1c`  
Source: `9a4917f41101926c8cb23d630018b8e5e4aec44c`  
Source tree: `e35ec4798c8e03dde85334ef0f200547cbf092e6`

## Gates

- Original CPU source suite: 110 reported, 106 executed successfully, four explicitly superseded historical checks retained/skipped.
- Installed parser: 6; real native PLE including model rows: 15; PLE launch/cache: 5; installed filter off/on/fallback/integrity: PASS.
- Octave GPU: 166 passed, zero failures/errors/skips, including 48 new masked-slot regressions. Accepted HIP filter: 40 eager cases plus 20 changed-input captured-graph checks.
- Repeatability, legacy identities, nonempty default-sampling responses and tools round trips passed for every model configuration. All raw token-bearing streams were independently reconstructed.
- PLE-on execution has bounded allocation, lookups/hits and zero error counters in the retained native statistics.
- Each MTP3 process logged the accepted HIP eager and captured paths; no fallback was logged. These are activation records, not per-request counters.

## Matched cross-process output equality

- control-bf16-m0 versus combined-bf16-m0: exact streamed/legacy tokens, text and finish reasons; teacher-probe logprobs exact. Legacy requests did not request logprobs.
- control-bf16-m3 versus combined-bf16-m3: exact streamed/legacy tokens, text and finish reasons; teacher-probe logprobs exact. Legacy requests did not request logprobs.
- combined-bf16-m0 versus ple16m-bf16-m0: exact streamed/legacy tokens, text and finish reasons; teacher-probe logprobs exact. Legacy requests did not request logprobs.
- combined-bf16-m3 versus ple16m-bf16-m3: exact streamed/legacy tokens, text and finish reasons; teacher-probe logprobs exact. Legacy requests did not request logprobs.
- native-default-m3 versus native-seeded-m3: exact streamed/legacy tokens, text and finish reasons; teacher-probe logprobs exact. Legacy requests did not request logprobs.

Parity refers to matched configurations, not quantized-KV versus BF16 equality. Sampling is held equal inside each comparison. Existing accepted-control tuning was copied to matched BF16/PLE workspaces. Native/quantized configurations have separately identified compile caches. The repaired compact/MTP3 row reused a partial same-image startup cache after the unrelated foreground timeout; it is not a cold-load timing claim.

## Raw timing summaries

One process per configuration, two decode timing requests and one prefill timing request; two warmup streams are retained separately in TIMINGS.csv. Token-weighted decode uses post-first-chunk token counts and first-to-last token-bearing arrivals. Prefill is prompt tokens divided by client TTFT, not isolated server/kernel PP. Full precision, token geometry, cache counts, wall throughput and tails are in the CSV/JSON files. No cross-sampler ranking or new filter speed threshold is applied.

| Original configuration | Sampling | Ready s | Decode tok/s | Decode wall s (2 requests) | Decode worst chunk gap s | Prefill client tok/s | Prefill/decode tok/s | Prefill wall s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| control-bf16-m0 | comparison-T1.0-p0.95-k20 | 138.451 | 33.315 | 15.904 | 0.110591 | 971.774 | 33.117 | 6.932 |
| combined-bf16-m0 | comparison-T1.0-p0.95-k20 | 176.511 | 33.515 | 15.754 | 0.036474 | 960.585 | 33.302 | 6.946 |
| control-bf16-m3 | comparison-T1.0-p0.95-k20 | 146.485 | 39.363 | 13.664 | 0.079845 | 930.210 | 47.898 | 5.886 |
| combined-bf16-m3 | comparison-T1.0-p0.95-k20 | 196.498 | 39.638 | 13.514 | 0.072621 | 923.304 | 47.988 | 5.906 |
| octave_k3v4-m0 | model-T0.7-p0.8-k20 | 422.538 | 31.446 | 16.807 | 0.042820 | 747.498 | 31.008 | 8.123 |
| octave_k3v4-m3 | model-T0.7-p0.8-k20 | 456.545 | 35.295 | 15.167 | 0.073770 | 712.767 | 39.745 | 7.417 |
| octave_k3v3-m0 | model-T0.7-p0.8-k20 | 410.571 | 30.896 | 17.095 | 0.137270 | 757.751 | 30.878 | 8.084 |
| octave_k3v3-m3 | model-T0.7-p0.8-k20 | 462.569 | 36.132 | 14.860 | 0.072435 | 715.800 | 36.087 | 7.723 |
| octave_k3v3_compact-m0 | model-T0.7-p0.8-k20 | 428.633 | 31.419 | 16.772 | 0.039133 | 757.242 | 30.967 | 8.075 |
| octave_k3v3_compact-m3 | model-T0.7-p0.8-k20 | 166.504 | 38.353 | 13.970 | 0.078431 | 726.152 | 38.287 | 7.461 |
| int8_per_token_head-m0 | model-T0.7-p0.8-k20 | 436.551 | 32.003 | 16.503 | 0.042630 | 959.622 | 31.764 | 7.134 |
| int8_per_token_head-m3 | model-T0.7-p0.8-k20 | 472.537 | 40.008 | 13.452 | 0.073065 | 919.699 | 46.378 | 6.010 |
| int4_per_token_head-m0 | model-T0.7-p0.8-k20 | 416.540 | 32.255 | 16.429 | 0.057720 | 943.381 | 32.318 | 7.120 |
| int4_per_token_head-m3 | model-T0.7-p0.8-k20 | 458.563 | 38.051 | 14.132 | 0.069806 | 904.897 | 39.174 | 6.567 |
| ple16m-bf16-m0 | comparison-T1.0-p0.95-k20 | 134.644 | 32.847 | 16.105 | 0.047688 | 973.233 | 33.301 | 6.906 |
| ple16m-bf16-m3 | comparison-T1.0-p0.95-k20 | 144.522 | 39.597 | 13.558 | 0.072945 | 931.000 | 48.131 | 5.872 |
| native-default-m3 | model-T0.7-p0.8-k20 | 446.810 | 43.819 | 12.398 | 0.070055 | 937.502 | 45.150 | 6.024 |
| native-seeded-m3 | model-T0.7-p0.8-k20 | 143.003 | 43.706 | 12.412 | 0.072099 | 936.174 | 45.043 | 6.034 |

Native fresh/seeded readiness was 446.810/143.003 seconds in this single observed pair, with exact matched outputs and no seeded .best_config additions/removals/changes. This is scoped cache-reuse evidence, not a universal startup-time guarantee.

## Important limits

single user, context32768, 2GiB KV pool; ordinary+MTP3. No broad task accuracy, filled context or concurrency claim.

Timing streams use forced output caps, not a natural-prose throughput test. The separate model-default sampling probe allowed EOS but all 18 responses reached its 128-token cap (`finish_reason=length`); it establishes nonempty bounded generation, NOT natural completion. The tool-result replies did finish with `stop`. Historical shorthand calling that prose probe “normal-stop” must not be read as an observed natural-stop result.

No broad task-accuracy result, concurrency qualification, filled-window test or new long-context capacity claim is made. User-configurable options remain available; tested scope is not a restriction on the parser. Optional Octave KV is not required to match BF16 speed. No public push, release tag, accepted-s8 pointer change or production deployment is part of this packet.
