# HTR v0.2.3 release verification

The selected grouped-MoE, packed dense-WMMA and asynchronous-PLE stack is integrated by default, with the quantized-KV binding repair. Weights, checkpoint metadata and sampling defaults are unchanged.

Executable source: `ff9897df3d5b5405af6ee2353a10b0fcd8b92394`. Image: `ghcr.io/hawgauto/vllm-htr@sha256:417cd5d14ce241128379614de7dc128bbc4f236b8c0f3853d6d91c5c4d84b221`. Release documentation may be a documentation-only successor.

7 final-image configurations; 112 requests, including 61 reconstructed SSE streams. All seven configurations were exercised on the repaired final image; earlier-candidate receipts remain separate historical evidence. The separate selected-winner comparison adds two one-token requests and is not counted in these totals.

| Cell | MTP | Max sequences | Nonstream requests | SSE streams | Ready seconds |
|---|---:|---:|---:|---:|---:|
| native-bf16-m0 | 0 | 1 | 13 | 5 | 420.527 |
| int8_per_token_head-m0-s32 | 0 | 32 | 5 | 35 | 488.721 |
| native-default-m3 | 3 | 1 | 13 | 5 | 460.852 |
| int4_per_token_head-m3-s4 | 3 | 4 | 5 | 7 | 492.547 |
| octave_k3v4-m3-s1 | 3 | 1 | 5 | 3 | 462.549 |
| octave_k3v3-m3-s1 | 3 | 1 | 5 | 3 | 470.542 |
| octave_k3v3_compact-m3-s1 | 3 | 1 | 5 | 3 | 478.565 |

All final-image cells use fresh workspaces. Ready times are not a matched startup benchmark. Serial and concurrent streams are labeled separately in TIMINGS.csv. Client TTFT includes queueing and first-token work; decode_tps excludes the first returned token chunk. Do not relabel TTFT-derived rates as pure prefill speed.

## Numerical and delivery scope

In MTP0, the correct Lisbon continuation and returned logprobs match the already-selected winner exactly in both repetitions. MTP3 retains the predecessor's two-newline first token on this one-token probe, with changed returned logprobs. The earlier v0.2.2 first-token output was two newlines; a one-token output difference is not itself an answer-quality regression. The selected dense path is not claimed whole-model bitwise identical to every predecessor. Other retained ordinary greedy and teacher comparisons pass.

The literal public Docker command passed image-default environment, bridge/loopback mapping, readiness, five SSE streams and a structured tool-call/result round trip. INT8 MTP0/seq32 and INT4 MTP3/seq4 additionally exercised concurrent clients and grouped-MoE per-layer native exactness guards. Each optional KV format completed installed-image serving. This is not every mode/format cross-product or a filled32768-token test. Source and installed CPU checks remain separately labeled in QUALIFICATION.json.

## Evidence and limits

- [Machine-readable scope and source/runtime bindings](QUALIFICATION.json)
- [Public-command delivery](DELIVERY.json)
- [Raw per-stream timing rows](TIMINGS.csv), [per-cell summaries](CELLS.csv), [nonstream request timings](CALLS.json)
- [Current serving instructions](../../docs/optimized-serving.md)
- SHA256SUMS.txt verifies this evidence packet.

No new broad accuracy, maximum-context, multimodal or Bf16Core qualification is claimed. High-concurrency component selection does not establish universal or single-stream speedups; historical throughput targets remain unmet. No prebuilt compiler seed is shipped; new shapes may compile. Historical release evidence and tags are preserved.
