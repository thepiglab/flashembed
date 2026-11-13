# Quantization (experimental): int8 / int4 / bf16 on the CPU path

Status: **study only, nothing in production.** The fp32 runner stays the default. Whether an int8 variant can replace
it depends on a retrieval check on PigMed's corpus that runs once the corpus fp32 vectors exist (see
[Retrieval check](#retrieval-check-prepared-not-run)).

## Question

GEMMs (Linear layers) are 40-60 % of a flashembed forward pass ([PROFILING.md](PROFILING.md#per-op-time-breakdown));
attention runs in the Rust FlashAttention op, which stays fp32. Can quantizing the GEMMs make embedding faster on
the cluster's CPUs (AMD EPYC Zen 2 / Zen 3 without VNNI, Zen 5 with AVX-512 VNNI and BF16) without moving the
vectors enough to change retrieval?

## What was built (all Rust)

| piece | what |
|---|---|
| `flashembed quantize` ([src/quant.rs](../src/quant.rs)) | rewrites an fp32 graph dir from `py/export_onnx.py` into a variant dir with the same file names (the runner takes it as `--onnx`). Reads/writes ONNX protobuf directly (`prost`, vendored `proto/onnx.proto`), external data included; about 8 s for all four graphs. Only MatMuls with a constant weight are touched; attention (custom ops), norms, embeddings (Gather), RoPE and pooling stay fp32. |
| schemes | `dyn8`: `com.microsoft::DynamicQuantizeMatMul`, int8 per-output-channel weights, activations quantized per tensor at run time (uint8). `dyn8rr`: same with 7-bit weights (reduce range: ORT's advice for AVX2 without VNNI, where `vpmaddubsw` saturates int16 pair sums). `qdq8`: static int8, `QuantizeLinear` + `MatMulIntegerToFloat`, activation ranges from `flashembed calib` (global min/max, or `--calib-avg` = mean of per-batch min/max). `nbB-aL`: weight-only `MatMulNBits`, B = 8 or 4 bits, block 32 along K, accuracy level L (1 = fp32 compute, 4 = int8 compute). |
| selective | `--keep <regex>` on the torch module path (from the exporter's name-scope metadata) and `--keep-edge K` (first/last K layers of every layer stack) keep chosen Linears fp32; `--graphs` limits which graphs are rewritten (e.g. `-novis`: EmbeddingGemma 2 vision tower fp32). |
| `flashembed calib` | runs `bench` inputs through a `--scheme calib` graph dir (the selected MatMuls' activation inputs exposed as graph outputs) with the Rust custom ops, so no Python ORT or attention-free twin graph is needed; writes `{graph: {tensor: range}}`. |
| `flashembed fidelity` ([src/fidelity.rs](../src/fidelity.rs)) | compares two `bench --save` dirs: per-item cosine (min / p1 / median / mean; NeoMME late vectors per row), top-10 neighbour overlap inside the set, and caption -> figure retrieval (R@1, R@10, MRR) with a paired bootstrap CI of the difference. |
| `qgemm-bench` ([src/bin/qgemm_bench.rs](../src/bin/qgemm_bench.rs)) | one-node graphs built with the same rewrite, timed against fp32 MLAS at the models' layer shapes; plus single-core instruction-throughput probes. |
| [slurm/qbench.sbatch](../slurm/qbench.sbatch) | per node: model-card Python runtime (fp32), fp32 flashembed and every variant on the same inputs and CPUs; rep 0 = full set + vectors, later reps re-time a subset in rotated order; then `fidelity` of every variant vs fp32 flashembed and vs Python. |

## Kernel level: ONNX Runtime's quantized GEMMs vs fp32 MLAS

`qgemm-bench 16` (and `1`), ONNX Runtime 1.31.1, non-spinning pool, the models' Linear shapes (M = tokens per forward).
Speed-up over fp32 MatMul; in brackets the relative error of the output (synthetic activations with a few 30x outlier
channels, like the models' hidden states).

KERNEL_TABLE

- **Zen 5 (AVX-512 VNNI)**: dynamic int8 is 1.5-3.1x fp32 at 16 threads and 3.1-3.5x on one core (VNNI's 4x peak
  over fp32 FMA, minus quantize/scale overhead). Static int8 (`qdq8`) is no faster than dynamic.
- **Zen 3 / Zen 2 (AVX2, no VNNI)**: plain `dyn8` **saturates**: MLAS's AVX2 u8 x s8 kernel sums pairs in int16
  (`vpmaddubsw`), and full-range weights overflow it (output error 0.16-0.44 at four shapes, vs 0.03-0.09 on Zen 5
  with the same weights). `dyn8rr` (7-bit weights) avoids it at the same speed: 1.3-2.0x on Zen 3, 1.0-1.6x on Zen 2
  (about 1.0x on one Zen 2 core: its AVX2 int8 path has no throughput edge over fp32 FMA).
- **Weight-only `MatMulNBits` (int8 / int4)** does not help here: these GEMMs have M = 500-2,500 rows and are
  compute-bound; MatMulNBits is built for memory-bound LLM decoding (M ~ 1). Often slower than fp32 (0.4-0.9x).
- **bf16** (Zen 5 only): `vdpbf16ps` peaks at 1.45x fp32 FMA on one core (331 vs 228 GOP/s), against about 4x for
  int8 VNNI (904 GOP/s). ONNX Runtime has no x86 bf16 GEMM, so a bf16 path would mean writing a GEMM; with at most
  1.45x on 40-60 % of the time (<= 1.2x end to end) it was not built.

CPU probes (one core, GOP/s): PROBE_TABLE

## End to end

END_TO_END

## Retrieval check (prepared, not run)

RETRIEVAL
