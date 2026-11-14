# Quantization (experimental): int8 / int4 / bf16 on the CPU path

Status: **study only; the fp32 runner stays the default.** A Rust int8 kernel (`q8blk`, below) makes both models
1.2-1.6x faster with vectors at cosine >= 0.99 to fp32 (>= 0.996 on Zen 5) and no measurable retrieval change on
a held-out PubMed Central set. Before it is used for real embeddings, a retrieval check on PigMed's corpus has to show
the same at scale ([Retrieval check](#retrieval-check-prepared-not-run)).

Measured 2026-11-13 on AMD EPYC 7702 (Zen 2), 75F3 (Zen 3) and 9655P (Zen 5): shared batch nodes, 16 CPUs per run,
batch 4, ONNX Runtime 1.31.1. Every number in a table comes from one job that ran all its runtimes back to back on the
same node and inputs. Full per-run tables: [QUANTIZATION-RESULTS.md](QUANTIZATION-RESULTS.md).

## TL;DR

| CPU | model | fp32 flashembed vs model-card Python | `q8blk-b32` vs fp32 flashembed: text / images | cosine to fp32, worst item: text / images | caption -> figure MRR change [95 % CI] |
|---|---|---|---|---|---|
| Zen 5 (VNNI) | NeoMME | 2.6x / 2.3x | **1.57x / 1.56x** | 0.9995 / 0.9966 | -0.000 [-0.008, +0.008] |
| Zen 5 (VNNI) | EmbeddingGemma 2 | 3.7x / 2.1x | **1.21x / 1.28x** | 0.9981 / 0.9977 | +0.001 [-0.009, +0.012] |
| Zen 3 | NeoMME | 1.6x / 1.3x | **1.37x / 1.45x** | 0.9982 / 0.9914 | -0.000 [-0.012, +0.011] |
| Zen 3 | EmbeddingGemma 2 | 2.4x / 1.4x | **1.23x / 1.21x** | 0.9965 / 0.9760 | +0.032 [+0.012, +0.054] |
| Zen 2 | NeoMME | 1.9x / 1.7x | pending | pending | pending |
| Zen 2 | EmbeddingGemma 2 | 2.7x / 1.8x | pending | pending | pending |

`q8blk-b32` is `flashembed quantize --scheme q8blk --block 32`, run by the Rust op `flashembed::QMatMul`. Speed is
items/s relative to fp32 flashembed on the same node (999 real text chunks from PigMed's 512-word chunking, 200 PMC
figures).
Cosine is per item, the minimum over the set (median >= 0.998 everywhere). MRR is caption -> figure retrieval over
the 200 held-out figures (199 queries; fp32 MRR 0.411 NeoMME, 0.835 EmbeddingGemma 2).

- **ONNX Runtime's own int8 is not usable here.** Per-tensor dynamic or static activation quantization (`dyn8`,
  `qdq8`) is 1.1-1.5x faster but moves vectors to cosine 0.79-0.93 at the worst item. Top-10 neighbours overlap only
  ~0.5-0.6, and caption -> figure MRR drops 0.06 (NeoMME). On Zen 2/3 (no VNNI) the plain variants also **saturate**:
  EmbeddingGemma 2 caption -> figure MRR falls from 0.835 to 0.117.
- **Weight-only `MatMulNBits`** (int8, per-block activations) keeps the vectors (cos >= 0.996) but is **0.5-0.7x**:
  its kernels target LLM decoding (M ~ 1), not these compute-bound GEMMs (M = 500-10,000 rows).
- **The fix is per-block activation scales with a fast large-M kernel.** ONNX Runtime has no such kernel, so it was
  written in Rust (`src/qgemm.rs`): int8 x int8 with AVX-512 VNNI on Zen 5 and an AVX2 path for Zen 2/3.
- **bf16** was not built: Zen 5's bf16 dot product peaks at 1.46x fp32 FMA (int8 VNNI: 4x), and ONNX Runtime has
  no x86 bf16 GEMM.
- **EmbeddingGemma 2 gains less** (1.2-1.35x) than NeoMME (1.4-1.6x). Its text layers are narrow (hidden 512), and
  more of its time is attention, norms and the per-layer embedding inputs, which stay fp32.
- **Zen 2/3 vs Zen 5 vectors differ** (7-bit vs 8-bit activations). Gemma images on Zen 3 drift most: worst cosine
  0.976, and held-out caption -> figure MRR moved +0.032 (outside the CI, though in the good direction). Mixed-node
  runs should either accept this or force one path (`FLASHEMBED_QGEMM_ISA=avx2` everywhere).

## What was built (all Rust)

| piece | what |
|---|---|
| `flashembed::QMatMul` ([src/qgemm.rs](../src/qgemm.rs)) | int8 GEMM as an ONNX Runtime custom op, on ORT's intra-op pool. **Weights**: int8, symmetric, one scale per (K-block, column), packed at quantize time in 16-column panels (16 cols x 4 k = 64 bytes per step, the layout `vpdpbusd` / `vpmaddubsw` read). **Activations**: quantized at run time per (row, K-block) with an AVX2 quantizer, stored as unsigned bytes with an offset (8 bits + 128 with VNNI; 7 bits + 64 on AVX2, so `vpmaddubsw`'s int16 pair sums cannot saturate). **Compute**: int32 sums per K-block, offset removed with precomputed column sums, scaled by s_x[row, blk] * s_w[blk, col] into fp32 accumulators. VNNI microkernel 12 x 16; AVX2 microkernel 4 x 16. `FLASHEMBED_QGEMM_ISA=avx2|scalar` forces a path. |
| `flashembed quantize` ([src/quant.rs](../src/quant.rs)) | rewrites an fp32 graph dir (from `py/export_onnx.py`) into a variant dir with the same file names, used as `--onnx`. Reads and writes ONNX protobuf directly (`prost`, vendored `proto/onnx.proto`, external data); ~8 s for all four graphs. Only MatMuls with constant weights change. Attention (custom ops), norms, embeddings (Gather), RoPE and pooling stay fp32. Schemes: `q8blk` (the Rust op above, `--block 32/64`), and ORT's: `dyn8` / `dyn8rr` (`DynamicQuantizeMatMul`, per-channel int8 weights, per-tensor activations; `rr` = 7-bit weights), `qdq8` (static: `QuantizeLinear` + `MatMulIntegerToFloat`, ranges from `flashembed calib`), `nbB-aL` (weight-only `MatMulNBits`, B = 8/4 bits, accuracy level L). Selection: `--keep <regex>` on the torch module path (from the exporter's name-scope metadata), `--keep-edge K` (first/last K layers fp32), `--graphs` (which graphs to rewrite), `--fuse` (q8blk: same-input MatMuls as one op + Split; slower here, off by default). |
| `flashembed calib` | runs `bench` inputs through a `--scheme calib` graph dir (MatMul activations exposed as graph outputs), with the Rust custom ops, so no Python or attention-free twin graph is needed. |
| `flashembed fidelity` ([src/fidelity.rs](../src/fidelity.rs)) | compares two `bench --save` dirs: per-item cosine (NeoMME late vectors per row), top-10 neighbour overlap inside the set, and caption -> figure retrieval (R@1, R@10, MRR) with a paired bootstrap CI. |
| `flashembed qreport` ([src/qreport.rs](../src/qreport.rs)) | the tables in [QUANTIZATION-RESULTS.md](QUANTIZATION-RESULTS.md) from `slurm/qbench.sbatch` outputs. |
| `qgemm-bench` ([src/bin/qgemm_bench.rs](../src/bin/qgemm_bench.rs)) | each scheme as a one-node graph, built with the same rewrite, timed against fp32 MLAS at the models' layer shapes; plus single-core instruction-throughput probes. |
| [slurm/qbench.sbatch](../slurm/qbench.sbatch) | per node: model-card Python runtime (fp32), fp32 flashembed and each variant on the same inputs. Rep 0 = full set + vectors; later reps re-time a subset in rotated order. Then `fidelity` of each variant vs fp32 flashembed and vs Python. |

Inputs: 999 text chunks (567 body, 296 figure captions, 86 supplementary, 50 table text) from 660 PMC open-access
articles in 8 PigMed ingest bundles, in each model card's document format (EmbeddingGemma 2:
`title: none | text: ...`), plus 200 of those articles' figures (199 with their caption in the text set). Static
calibration used 2 other bundles (160 chunks, 32 figures).

## Kernel level

`qgemm-bench`, the models' Linear shapes (M x K x N; M = tokens in a batch), speed over fp32 MatMul (ONNX Runtime
MLAS), and in brackets the output's relative error (synthetic activations with four 30x outlier channels, as in
these models' hidden states).

| CPU | threads | `q8blk` (Rust) | `dyn8` (ORT) | `dyn8rr` (ORT) | `nb8-a4` (ORT) |
|---|---|---|---|---|---|
| Zen 5 9655P | 16 | 1.75-2.83x (0.007-0.024) | 1.79-2.71x (0.025-0.090) | 1.80-2.69x (0.026-0.091) | 0.83-1.51x (0.006-0.016) |
| Zen 5 9655P | 1 | 2.75-3.05x | 2.93-3.58x | 2.90-3.57x | 1.10-1.27x |
| Zen 3 75F3 | 16 | 1.60-2.71x (0.010-0.037) | 1.36-2.22x (**0.03-0.44**) | 1.25-2.41x (0.026-0.091) | 0.69-0.97x |
| Zen 3 75F3 | 1 | 1.63-1.78x | 1.78-1.91x | 1.77-1.91x | 0.61-0.66x |
| Zen 2 7702 | 16 | 0.87-1.93x | 0.94-1.61x (**0.03-0.44**) | 0.86-1.59x | 0.56-1.08x |
| Zen 2 7702 | 1 | 0.88-1.00x | 0.98-1.05x | 0.98-1.05x | 0.41-0.44x |

- On Zen 2/3, ORT's AVX2 u8 x s8 kernel sums pairs in int16 (`vpmaddubsw`), and full-range int8 weights overflow it:
  output error up to 0.44 vs 0.025-0.09 with the same weights on Zen 5. `dyn8rr` (7-bit weights) and the 7-bit
  activations of `q8blk`'s AVX2 path avoid it.
- A single Zen 2 core gains nothing from int8 (its AVX2 integer multiply-add has no throughput edge over fp32 FMA).
  Its multi-thread gains probably come from the 4x smaller weights (less memory traffic).
- Single-core instruction peaks (GOP/s, one core): Zen 5 fp32 FMA 512-bit 233-239, bf16 `vdpbf16ps` 342-349,
  int8 `vpdpbusd` 946-954; Zen 3 fp32 FMA 256-bit 124-125; Zen 2 81-84.

## End to end

Speed relative to fp32 flashembed in the same job (mean of the per-rep ratios); cosine = worst item / median.
Caption -> figure: 199 captions query the 200 held-out figures.

### Why ONNX Runtime's per-tensor int8 loses accuracy: one layer type

Keeping one class of Linear layers fp32 under `dyn8` (Zen 5, VNNI, so no saturation involved):

| kept fp32 | NeoMME speed | NeoMME cos text / images (worst) | NeoMME MRR change | Gemma speed | Gemma cos text / images (worst) |
|---|---|---|---|---|---|
| nothing (`dyn8`) | 1.50x | 0.915 / 0.832 | -0.046 [-0.086, -0.006] | 1.19x / 1.30x | 0.930 / 0.811 |
| MLP `down_proj` | 1.31x / 1.25x | **0.994 / 0.968** | **+0.001 [-0.013, +0.017]** | 1.24x / 1.24x | **0.973 / 0.966** |
| MLP `gate_proj`, `up_proj` | 1.29x / 1.23x | 0.918 / 0.845 | -0.041 | | |
| attention `o_proj` | 1.36x / 1.34x | 0.919 / 0.840 | -0.043 | | |
| attention `q/k/v_proj` | 1.46x / 1.36x | 0.915 / 0.842 | -0.054 | | |
| embeddings, heads, first/last layer | 1.45x / 1.37x | 0.907 / 0.836 | -0.047 | 1.27x / 1.26x | 0.952 / 0.854 |

The input of `down_proj` (the gated MLP activation) has a few channels far larger than the rest. One scale per tensor
then leaves the other channels with 1-3 int8 levels. Per-block scales (`q8blk`, `nb8-a4`) confine an outlier to its
block of 32/64 inputs.

### All variants

Zen 5 (from three jobs on different nodes: Python / ORT variants; `q8blk`; `q8blk-b32`):

| model | modality | Python | fp32 | `dyn8` | `dyn8rr` | `qdq8` | `nb8-a4` | `nb4-a4` | `q8blk` | `q8blk-b32` |
|---|---|---|---|---|---|---|---|---|---|---|
| NeoMME | text: speed | 0.40x | 1 (9.0/s) | 1.51x | 1.49x | 1.53x | 0.71x | 0.70x | 1.60x | 1.57x |
| | text: worst cos | 1.0000000 | | 0.915 | 0.928 | 0.915 | 0.9989 | 0.899 | 0.9984 | 0.9995 |
| | images: speed | 0.43x | 1 (8.3/s) | 1.54x | 1.51x | 1.51x | 0.72x | 0.70x | 1.61x | 1.56x |
| | images: worst cos | 1.0000000 | | 0.832 | 0.856 | 0.837 | 0.9959 | 0.861 | 0.9968 | 0.9966 |
| | caption -> figure MRR change | | | -0.046 | -0.063 | -0.054 | +0.003 | -0.099 | -0.008 [-0.015, -0.001] | -0.000 [-0.008, +0.008] |
| EmbeddingGemma 2 | text: speed | 0.28x | 1 (9.3/s) | 1.19x | 1.16x | 1.19x | 0.65x | 0.64x | 1.24-1.35x | 1.21x |
| | text: worst cos | 1.0000000 | | 0.930 | 0.935 | 0.902 | 0.9984 | 0.935 | 0.9934 | 0.9981 |
| | images: speed | 0.47x | 1 (1.7/s) | 1.30x | 1.29x | 1.33x | 0.73x | 0.71x | 1.31-1.33x | 1.28x |
| | images: worst cos | 1.0000000 | | 0.811 | 0.786 | 0.836 | 0.9975 | 0.945 | 0.9953 | 0.9977 |
| | caption -> figure MRR change | | | +0.029 | +0.021 | +0.006 | +0.006 | +0.035 | -0.001 [-0.013, +0.011] | +0.001 [-0.009, +0.012] |

Zen 3:

| model | modality | Python | fp32 | `dyn8` | `dyn8rr` | `qdq8` | `nb8-a4` | `q8blk` | `q8blk-b32` |
|---|---|---|---|---|---|---|---|---|---|
| NeoMME | text: speed | 0.66x | 1 (4.3/s) | 1.34x | 1.35-1.40x | 1.42x | 0.58x | 1.44x | 1.37x |
| | text: worst cos | | | 0.903 | 0.927 | 0.905 | 0.9993 | 0.9959 | 0.9982 |
| | images: speed | 0.77x | 1 (3.7/s) | 1.38x | 1.33-1.45x | 1.41x | 0.62x | 1.57x | 1.45x |
| | images: worst cos | | | **0.239** | 0.862 | 0.758 | 0.9968 | 0.9881 | 0.9914 |
| | caption -> figure MRR change | | | **-0.212** | -0.064 | -0.080 | +0.002 | -0.011 [-0.022, -0.002] | -0.000 [-0.012, +0.011] |
| EmbeddingGemma 2 | text: speed | 0.45x | 1 (5.0/s) | 1.20x | 1.22-1.28x | | 0.61x | 1.27x | 1.23x |
| | text: worst cos | | | **0.439** | 0.935 | | 0.9960 | 0.9967 | 0.9965 |
| | images: speed | 0.73x | 1 (0.9/s) | 1.27x | 1.25-1.27x | | 0.66x | 1.25x | 1.21x |
| | images: worst cos | | | **0.573** | 0.781 | | 0.9977 | 0.9717 | 0.9760 |
| | caption -> figure MRR change | | | **-0.718** | +0.016 | -0.134 | -0.001 | +0.032 [+0.010, +0.055] | +0.032 [+0.012, +0.054] |

Zen 2 (nodes at load 85-160 of 128 cores, so speeds are noisier):

| model | modality | Python | fp32 | `dyn8` | `dyn8rr` | `nb8-a4` | `q8blk` | `q8blk-b32` |
|---|---|---|---|---|---|---|---|---|
| NeoMME | text: speed | 0.53x | 1 (3.0/s) | 1.12x | 1.13x | 0.50x | pending | pending |
| | images: speed | 0.60x | 1 (2.7/s) | 1.13x | 1.13x | 0.55x | pending | pending |
| | caption -> figure MRR change | | | -0.212 | -0.064 | +0.002 | pending | pending |
| EmbeddingGemma 2 | text: speed | 0.39x | 1 (3.1/s) | 1.07x | 1.08x | 0.56x | pending | pending |
| | images: speed | 0.57x | 1 (0.6/s) | 1.04x | 1.05x | 0.57x | pending | pending |
| | caption -> figure MRR change | | | -0.718 | +0.016 | -0.001 | pending | pending |

ORT's int8 vectors are bit-identical on Zen 2 and Zen 3 (same MLAS AVX2 kernel), so their fidelity is the same.

Notes:
- **Neighbour overlap** (each item's top 10 among the other 1,198 items, variant vs fp32): `q8blk-b32` 0.93-0.98 (EmbeddingGemma 2 images on Zen 3: 0.86),
  `nb8-a4` 0.955-0.975, per-tensor int8 0.47-0.88. Vs the model-card Python runtime, fp32 flashembed stays at
  cosine >= 0.9999999 (as in [PROFILING.md](PROFILING.md)); the int8 variants are as far from Python as from fp32.
- **Peak RSS** falls with int8 weights: EmbeddingGemma 2 images 2.6 -> 1.9 GB, NeoMME images 3.4 -> 3.0 GB; text
  runs barely change (5.6 / 9.9 GB at batch 4, dominated by long-input activations).
- **Graph size** (MB, fp32 -> `q8blk-b32`): gemma_lm 1,088 -> 712 (its 512 MB embedding table stays fp32),
  gemma_vision_dyn 674 -> 256, neomme_text 1,021 -> 505, neomme_image 1,054 -> 515. `nb4-a4` (int4) is smaller still
  (627 / 161 / 388 / 393) but loses accuracy (worst cos 0.86-0.95) and speed (0.7x).
- **Gemma audio/video** were not quantized: the audio tower's MatMuls take their weights through `Clip`, not as
  constants, and audio/video are a tiny share of the corpus (4 videos, no audio, among ~7,200 embedded items in a sampled
  bundle). Video frames go
  through the vision tower, so they get the image speed-up when `gemma_vision_dyn` is quantized.
- **Static int8** (`qdq8`) is no faster than dynamic. Calibrated ranges either clip outliers (batch-averaged ranges:
  NeoMME images on Zen 3, MRR 0.411 -> 0.051) or waste levels on them (global min/max).

## Retrieval check (prepared, not run)

The decision: does `q8blk-b32` change PigMed retrieval metrics beyond noise? The held-out set above is too small
(199 queries, 200 figures) to settle it. The check reuses PigMed's own `scripts/embed_bundles.py` and
`scripts/eval_retrieval.py` unchanged, on 38 ingest bundles (~7,600 articles, ~30,000 figures):

1. Re-embed the 38 bundles with the variant (both models; 6 x 16-CPU burst tasks per model).
2. Once the 100k run has the same bundles' fp32 vectors, build a comparison corpus with model directories `gemma`
   (fp32), `gemma_q8` (int8 corpus + int8 queries), `gemma_q8q` (fp32 corpus + int8 queries), `gemma_q8c` (int8
   corpus + fp32 queries), and the same for NeoMME. Make caption -> figure and title -> article queries
   (`eval_retrieval.py queries --n 2000`), embed them both ways, and score each pair with `eval_retrieval.py score`
   (paired MRR@10 difference with a 95 % bootstrap CI; figure -> caption from the corpus vectors).
3. Proposed rule: int8 is usable if, for every task and model, the MRR@10 CI includes 0 or stays within +-0.005, and
   R@10 moves < 0.5 points. The mixed cases decide whether int8 daily updates can be searched with the fp32 corpus.

The ingest bundles are deleted after upload, so their text items were copied (2026-11-13) for step 1. The scripts
live with the site files (not in this repository, since they hold cluster paths). Run step 1 on the same CPU family
as the planned production run: Zen 2/3 and Zen 5 produce slightly different int8 vectors.

## Using it

```bash
flashembed quantize --src onnx --out onnx-q8 --scheme q8blk --block 32   # once; ~8 s
flashembed embed --model neomme --modality image --onnx onnx-q8 ...        # same runner, same flags
```

The runner picks VNNI or AVX2 when the graph loads. Graphs with `QMatMul` need a flashembed build that includes it
(this branch). The ORT-scheme variants (`dyn8`, `nb8-a4`, ...) load in any build.
