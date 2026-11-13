# GPU runner (`flashembed-gpu`)

flashembed's GPU path is a native CUDA engine (`cuda/fegpu.cu`), driven from Rust (`src/gpu.rs`,
`src/bin/flashembed_gpu.rs`, cargo feature `cuda`). It reuses the CPU runner's tokenization and image preprocessing.
It produces the same vectors as the model-card runtimes on GPU (sentence-transformers for EmbeddingGemma 2,
transformers `NeoMMEForRetrieval` for NeoMME-260M-Retriever) and is faster and lighter than them.

RESULTS_PLACEHOLDER

## Build and run

```bash
# needs the CUDA toolkit (>= 12.8 for Blackwell; nvcc on PATH or CUDA_HOME) and a GPU driver for that CUDA
FLASHEMBED_CUDA_ARCH=100,120 cargo build --release --features cuda --bin flashembed-gpu
target/release/flashembed-gpu embed --model gemma --modality text --input docs.jsonl --out vectors \
  --snapshot <HF snapshot> --dtype bf16 --prefix "title: none | text: "
```

Options: `--dtype bf16|fp32`, `--max-tokens` (tokens per forward, default 32768; also sizes the activation arena),
`--max-patches` (image patches per forward), `--max-items`, `--max-len`, `--side document|query` (NeoMME), `--no-late`,
`--dim` (EmbeddingGemma 2 Matryoshka size), `--warmup`, `--prep-only` (time the CPU preprocessing alone).
Output files are the same as `flashembed embed`'s, plus `embeddings.json` with throughput and GPU memory.
Environment: `FLASHEMBED_ATTN=simt` forces the FP32-math attention kernel in bf16 mode; `FLASHEMBED_FP32_GEMM=emu`
runs FP32 GEMMs through cuBLAS's BF16x9 FP32 emulation (see below).

Scope: text and images for both models (EmbeddingGemma 2 at the default 280 soft tokens per image). Audio and
video stay on the CPU runner for now. Video would need the vision tower at 140 tokens per frame; audio would need
the conformer.

## Design

### What runs where

```
                 model-card runtime (GPU)                        flashembed-gpu
tokenize         HF tokenizers (Python, in the encode loop)      tokenizers crate, prefetch thread (overlaps the GPU)
image prep       HF processors on the CPU (torchvision)          Rust decode on the CPU; resize + patching on the GPU
                                                                 (EmbeddingGemma 2), uint8 patches (NeoMME)
batching         fixed item count, padded to the longest item    token budget, packed sequences (no padding at all)
forward          PyTorch eager, ~40-60 kernels per layer          cuBLAS GEMMs + ~6 fused kernels per layer
attention        SDPA (cuDNN flash / mem-efficient with masks)    fused flash attention with window tile skipping
pooling          ST Pooling(mean) + Normalize / NeoMME heads      fused on the GPU, only the vectors are copied back
memory           caching allocator, per-batch growth             weights + one arena sized at startup
```

### Options evaluated

| option | measurement | verdict |
|---|---|---|
| **ONNX Runtime CUDA EP** on the stock-attention graphs (the CPU runner's ONNX path, `py/ort_cuda_bench.py`) | the PyPI `onnxruntime-gpu` 1.31.1 wheel has no kernels for the B200's sm_100 (`cudaErrorNoKernelImageForDevice` in `Expand`/`Cast`); the custom ops of the CPU graphs are CPU-only anyway, and the stock graphs materialise S x S scores and dense masks | not usable on this hardware without building ORT from source; would still be bound by per-op launches (1,000-1,400 ops per forward) |
| **TensorRT EP** | not tried: needs plugins for the sliding-window/GQA attention and NeoMME's exclusive attention, and dynamic shapes per batch | not pursued |
| **candle (CUDA)** | not measured: `candle-flash-attn` stops at head dim 256 (EmbeddingGemma 2's global layers use 512), has no FP32 path, and candle's elementwise ops are unfused (one kernel each, the cost the reference already pays) | rejected on capability |
| **native CUDA engine** (this) | see results: 7-9x the reference on bf16 text, 2-4x on fp32, lower memory everywhere | **chosen** |

The reference profile (`py/gpu_ref_profile.py`, bf16, batch 64, B200) is what made the choice easy. GEMMs are only
7 % (EmbeddingGemma 2) and 9 % (NeoMME) of kernel time. Elementwise kernels, copies and reductions take 70-80 %, and
the GPU is busy only 55-63 % of the wall time. Both models are small (hidden size 512 and 1024), so a
framework-style forward is dominated by memory traffic and launch overhead, not by matrix math. A faster GEMM
backend cannot fix that. Fusing everything around the GEMMs can.

### Kernels (`cuda/fegpu.cu`)

Per layer, EmbeddingGemma 2 runs 5 GEMMs and 6 kernels, and NeoMME runs 4 GEMMs and 4 kernels. Per-layer weights
are concatenated at load time: QKV (+ NeoMME's output gate) and gate|up, so each input row is read once.

- **QK(V)-norm + RoPE, in place** on the fused QKV output (one warp per token x head): RMS-norm with or without weight,
  then 1-D RoPE (EmbeddingGemma 2), 2-axis interleaved partial M-RoPE (NeoMME) or axial 2-D RoPE (Gemma 4 vision),
  plus NeoMME's value embeddings. Angles are computed as `pos * inv_freq` in FP32 with `sincosf`, as HF does.
- **Residual + norm**: one kernel per residual junction. It applies the post-norm of the sublayer output, the residual
  add (or NeoMME's `h + m * x`), the layer scalar, NeoMME's `lambda0 * h + lambda1 * h0` mix, and the next pre-norm.
  It keeps an FP32 residual stream and writes the next GEMM's input in the compute dtype (warp per row, vectorised).
- **Gated activations**: GELU-tanh gate (Gemma), ReLU^2 (NeoMME), and EmbeddingGemma 2's per-layer-embedding gate
  (`gelu(g) * rmsnorm(scale * p)`), each one vectorised pass. The PLE projection is computed one layer slice at a time
  from the input embeddings instead of materialising all 24 x 512 values per token up front.
- **Attention**: flash attention over packed variable-length sequences (`cu_seqlens`), with GQA, a symmetric sliding
  window `|i - j| <= w` (key tiles outside a query tile's window are skipped), scale, and NeoMME's epilogue
  (exclusive self-attention against the token's own value vector, then the sigmoid output gate).
  - BF16: tensor cores (`mma.sync.m16n8k16`, FP32 accumulate), 64 query rows x 32-64 keys per tile, K/V tiles
    double-buffered with `cp.async` when shared memory allows. Head dim 512 (EmbeddingGemma 2's global layers) runs as
    two 256-wide output halves, which keeps the accumulator in registers.
  - FP32: SIMT kernel with the same tiling logic and FP32 math throughout.
- **Pooling**: masked mean over each packed sequence and L2 normalisation on the GPU. EmbeddingGemma 2's 512 -> 768
  projection is applied after pooling (it is linear, so this equals projecting every token first) in FP32.
  NeoMME's late-interaction head is one GEMM plus a row-normalisation kernel.
- **Gemma 4 vision tower**: patch embedding + 2-D position tables, 16 layers with the same fused kernels, then the 3x3
  pooler + `sqrt(768)` scale + embed_vision pre-norm as one kernel and the 768 -> 512 projection. The resulting soft
  tokens stay on the device and are scattered into the text backbone's input embeddings.

### Precision

- `--dtype fp32`: FP32 storage and math everywhere. GEMMs use cuBLAS's BF16x9 FP32 emulation on Blackwell
  (`CUBLAS_COMPUTE_32F_EMULATED_16BFX9`). It splits each FP32 operand into three BF16 parts and runs the nine partial
  products on tensor cores with FP32 accumulation. Against the FP32 reference its parity is indistinguishable from
  plain SGEMM (min cosine 0.99999999999 either way), and it is 1.3-1.8x faster end to end.
  `FLASHEMBED_FP32_GEMM=sgemm` selects plain SGEMM; on GPUs where cuBLAS rejects the emulation it falls back to SGEMM
  automatically. TF32 is never used.
- `--dtype bf16`: BF16 weights and GEMM inputs/outputs, FP32 accumulation, FP32 residual stream, FP32 norms,
  softmax, RoPE angles and pooling. This is at least as precise as the reference's own bf16 run, which keeps the
  residual stream and norm outputs in BF16. The checkpoints are BF16, so BF16 weights lose nothing. The FP32 path
  converts them exactly.

### Batching and the host side

- **Packing, not padding.** A batch is a flat token list plus `cu_seqlens`. Every GEMM and kernel runs on exactly the
  real tokens. Positions restart per sequence, and attention never crosses sequences. Items are length-sorted (longest
  first, as sentence-transformers does) and cut into batches by a token budget (`--max-tokens`, default 32,768),
  not an item count.
- **One arena.** Activation buffers are sized for the token budget at startup. Nothing is allocated while running,
  and peak device memory is weights + arena + cuBLAS workspace, known before the first batch.
- **Overlap.** A preprocessing thread (tokenization, image decoding, rayon-parallel within a batch) prepares batch n+1
  while the GPU runs batch n. Inputs go through one pinned staging buffer and a single async copy per batch.
- **No CUDA graphs.** Every batch has a different packed shape, and at a 32k-token budget a forward is milliseconds
  of GPU work against roughly 300 launches, so launch overhead is a small share.

### Image preprocessing on the GPU

- **EmbeddingGemma 2 (default).** The CPU only decodes (libjpeg-turbo / libwebp, as before). The decoded RGB8 image
  and the resize coefficients go to the GPU, which runs the same separable integer resize as `src/pil_resize.rs`
  (Pillow's algorithm with ATen's int16 weight quantisation, horizontal pass then vertical) and writes the 16 px
  patches directly in the compute dtype. The integer arithmetic is identical, so the vectors are bit-identical to
  CPU preprocessing (checked: `--img-mb 0` gives the same vectors). Decoded images larger than the per-batch budget
  (`--img-mb`, default 128) are preprocessed on the CPU.
- **NeoMME.** Patches are uploaded as uint8 (padding 0, which normalises to the processor's -1) and normalised inside
  the patch-stem LayerNorm kernel: a quarter of the host copies and PCIe traffic of float patches.

