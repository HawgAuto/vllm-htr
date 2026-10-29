# Runtime credits and notices

This integration builds on vLLM; upstream Apache-2.0 source notices remain in place. HTR's format and quantization/kernel implementation are separately documented at [HawgAuto/HTR](https://github.com/HawgAuto/HTR). Its MIT notice is retained in `licenses/HTR-MIT.txt`.

HTR's methodological and engineering lineage includes exlr8 / kindlingai, ExLlamaV3 / Turboderp and contributors, QTIP, QuIP/QuIP#, and GPTQ/LDLQ. Read the [full source-pinned lineage disclosure](https://github.com/HawgAuto/HTR/blob/bd0b92c4f748f696c525932d2db0a75cd8e409f6/docs/LINEAGE.md) for the relationships and limitations. Retained ExLlamaV3 and exlr8-loader MIT notices acknowledge that lineage; the latter is not a claim of a repository-wide exlr8 license.

This runtime integrates HTR and GPTQ8 operators with vLLM's model loader, PLE I/O, graph execution and speculative decoding. It is not an upstream vLLM release. The quantizer remains a separate project. External dependencies retain their own licenses, and these code licenses do not relicense Qwen Community model weights.

The three retained license texts were retrieved unchanged from HTR revision `bd0b92c4f748f696c525932d2db0a75cd8e409f6`. The source repository's root license and original per-file notices remain authoritative for upstream vLLM code.
