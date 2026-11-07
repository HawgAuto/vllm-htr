# v0.2.2 qualification and delivery

The combined native runtime completed its integration matrix and the exact public Docker-command delivery gate. Executable source: `9a4917f41101926c8cb23d630018b8e5e4aec44c`. Image: `ghcr.io/hawgauto/vllm-htr@sha256:034898ebd182a240fa403ad5ea24bea3f647d933571e43420a27daf9af079d1c`.

- [Qualification and all 18 timing summaries](QUALIFICATION.md)
- [Complete deterministic raw-data reduction](REPORT.json)
- [Machine-readable scope, source/tests and parity bindings](QUALIFICATION.json)
- [All 90 stream rows, including labeled warmups](TIMINGS.csv)
- [All 18 configuration summaries](CELLS.csv)
- [Exact image/source/native-library build binding](SOURCE-BUILD.json)
- [Public-command delivery](DELIVERY.json): real-model readiness, streaming and tool-call/result execution through the documented loopback-only Docker bridge mapping, with image-default environment and complete owned cleanup.
- [Checksums](SHA256SUMS.txt)
- [Current native serving instructions](../../docs/native-serving.md)

The integration matrix contains 16 new-image loads plus two explicitly inherited unchanged controls,324 requests and90 independently reconstructed SSE streams. Five matched pairs preserve streamed/legacy tokens,text and finish reasons and teacher-probe logprobs; legacy probes did not request logprobs. The separate public-command run adds13 nonstreaming requests and5 reconstructed streams; it is delivery evidence, not a new speed campaign.

Scope is Htr5Core,one sequence,configured32768 context,2GiB KV. No new broad task-accuracy,filled-window,concurrency or multimodal score is claimed. The EOS-enabled prose probes reached their128-token cap; tool-result turns stopped normally. Native settings remain user-configurable; qualification scope is not a parser restriction.

The six original matrix/build artifacts are byte-preserved from the completed local qualification, including original CSV line endings. Their local-status/publication wording describes when those immutable records were captured; publication does not retroactively rewrite evidence. The release asset contains that original offline-verifiable local packet, including the exact source archive and verifier. Release documentation is a documentation-only successor, not newly compiled runtime code.
