# vLLM-HTR v0.2.5 Release Notes (AMD `gfx1151` / Strix Halo)

`v0.2.5` delivers a full-stack clean-slate Wave32 HIP/HSACO + vLLM runtime optimization release for **Qwen3.8-Flash-Next-HTR** on AMD Strix Halo (`gfx1151`, 40 CU, 128 GiB UMA LPDDR5X-8000), shipping **18 new native `.so`/`.hsaco` binaries** and **38 runtime updates** enabled cleanly by default via `runtime-env.json`.

## Verified Served E2E A/B Results (Exclusive `gfx1151`, MTP3, `max-model-len=32768`)

### BF16 KV (`--max-num-seqs 4`, `--max-num-batched-tokens 3200`, `4 GiB` KV)
| Workload | `v0.2.4` | **`v0.2.5`** | **Gain** |
|---|---:|---:|---:|
| **Greedy `c=1` Decode** (`128 tok`) | `42.21 tok/s` (`55.7 ms/st`) | **`44.60 tok/s` (`51.8 ms/st`)** | **`+5.7%` tok/s / `+7.6%` step** |
| **`c=1` Decode Step Latency** (`256 tok`, 2 runs) | `58.0–58.4 ms/st` | **`54.6–55.0 ms/st`** | **`+6.2%–+6.3%`** |
| **`4K` Decode** (`5,812` ctx, `128 tok`) | `54.59 tok/s` | **`59.21 tok/s`** | **`+8.5%`** |
| **`4K` Prefill** (`5,812 tok`) | `977.7 tok/s` | **`983.9 tok/s`** | **`+0.6%`** |
| **`c=4` Concurrent (`t=0.7`, `4×256`) — Wall Agg** | `51.44 tok/s` (`19.91 s`) | **`69.99 tok/s` (`14.63 s`)** | **`+36.1%`** |
| **`c=4` Concurrent (`t=0.7`, `4×256`) — Sum Decode** | `71.33 tok/s` | **`91.61 tok/s`** | **`+28.4%`** |
| **`c=4` Greedy (`t=0.0`, `4×256`) — Sum Decode** | `80.37 tok/s` (`15.59 s`) | **`97.94 tok/s` (`12.73 s`)** *(up to `116.68 tok/s` / `10.97 s`)* | **`+21.9%` (`+22.4%` wall)** |
| **MTP3 Acceptance Rate** | `69.96%` (`3.10 tok/step`) | **`73.80%` (`3.21 tok/step`)** | **`+3.84 pp`** |

### INT8 KV Long-Context & High-Concurrency (`--kv-cache-dtype int8_per_token_head`, `--max-num-seqs 8`, `4 GiB` KV)
| Workload | `v0.2.4` KV8 | **`v0.2.5` KV8 (`+ V5-Q4`)** | **Gain** |
|---|---:|---:|---:|
| **Greedy `c=1` Decode** (`128 tok`) | `43.18 tok/s` | **`45.05 tok/s`** | **`+4.3%`** |
| **`13K` Long-Context Prefill** (`13,017 tok`) | `409.6 tok/s` (`31.78 s`) | **`551.9 tok/s` (`23.58 s`)** | **`+34.7%`** |
| **`13K` Long-Context Decode** (`13,017` ctx) | `43.39 tok/s` | **`63.92 tok/s`** | **`+47.3%`** |
| **`c=4 × 4K` (`16,702` ctx) — Wall Time / Agg** | `33.99 s` (`15.06 tok/s`) | **`23.82 s` (`21.49 tok/s`)** | **`+42.7%`** |
| **`c=8` High-Concurrency — Sum Decode** (`8×128`) | `103.96 tok/s` | **`132.18 tok/s`** | **`+27.1%`** |
| **`c=8` High-Concurrency — Wall Agg** (`8×128`) | `44.76 tok/s` (`22.88 s`) | **`53.80 tok/s` (`19.03 s`)** | **`+20.2%`** |

## Key Optimizations Enabled by Default
- **MoE & Dense (`V5-W1`, `V5-W2`, `V5-W3b`, `V5-F1`, `V5-F2`):** K6 routed-MoE decode `rows=1..16` with zero-realloc `arena.py` tail reservation for shared-expert folding (`T<=4`), HPF4 prefill (`M<=2048`), dense WMMA (`out_proj`, `qkvz`, `lm_head`, prefill), and nontemporal `global_load_b128` K4 `lm_head`.
- **Linear Attention / GDN (`V5-G1`, `V5-G4`, `V5-P8`):** Native Wave32 fused GDN MTP spec decode (`htr_gdn_spec_fused.so`), native `causal_conv1d_update`, and Mamba prefill checkpointing.
- **QSA Sparse Attention (`V5-Q2`, `V5-Q4`):** Fused `qsa_canon_expand_fused.hsaco` and native Wave32 `qsa_decode_union` across BF16, `int8_per_token_head`, and `int4_per_token_head` KV caches.
- **HyperConnection, PLE, MTP & Sampling (`V5-H1/H2`, `V5-P5/P6/P7`, `V5-M1/M2/M3/M5`, `V5-S1/S2/S3`):** `hiphc_v2.so` with fused combine, `ple_gather_v25.so` + native PLE short-conv & gate-norm, MTP DDN v2/WMMA + BF16 GEMV, and native Q38 filter + Gumbel + sparse rejection sampling.
