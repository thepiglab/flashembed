# Design

## What runs where

```
           Python reference (model-card runtime)              flashembed
text       sentence-transformers / NeoMMEProcessor            tokenizers crate (same tokenizer.json)
image      HF processors (torchvision uint8 resize)           src/prep.rs + src/pil_resize.rs (bit-identical resize)
audio      torchcodec + Gemma4AudioFeatureExtractor           ffmpeg -> src/media.rs (log-mel port)
video      torchcodec + EmbeddingGemma2VideoProcessor         ffprobe/ffmpeg -> src/media.rs (same frame sampling)
decoders   torchvision (libjpeg-turbo), Pillow, FFmpeg 9      mozjpeg (libjpeg-turbo), image + libwebp crates, FFmpeg 9
model      PyTorch eager (SDPA attention)                     ONNX Runtime graphs + Rust custom ops (src/flash.rs)
pooling    ST Pooling(mean)+Normalize / NeoMME heads          src/main.rs
```

The model forward runs in ONNX Runtime (C++/MLAS), loaded through the `ort` crate in `load-dynamic` mode, i.e. the
very `libonnxruntime.so` that the Python `onnxruntime` wheel ships (`ORT_DYLIB_PATH`). Attention, Gemma 4's audio
attention and one normalisation are not ONNX Runtime's: they are custom operators in the `flashembed` ONNX domain,
implemented in Rust (`src/flash.rs`) and executed on ONNX Runtime's own intra-op thread pool.

## Backend choice

Three options were measured, not assumed (FP32, same nodes, interleaved runs):

| option | result | verdict |
|---|---|---|
| **candle** (pure Rust), NeoMME text encoder hand-ported (`src/bin/candle_neomme.rs`) | exact parity, but **0.4-0.5x torch** at 512 tokens: candle's CPU elementwise/broadcast ops are unfused and mostly single-threaded, so its fast GEMM does not help end to end; would also mean hand-porting three new architectures | rejected |
| **ONNX Runtime**, graphs from `torch.onnx` (dynamo), stock attention | exact parity; 1.0-2.1x torch at <= 512 tokens, but **0.6-0.8x at 1-2k tokens, 2.6x torch's RSS at 8k, out of memory at NeoMME 16k**: the exported attention materialises every S x S score matrix and per-layer dense masks | not sufficient alone |
| **ONNX Runtime + Rust custom attention ops** | exact parity; faster and lighter than torch across token ranges ([PROFILING.md](PROFILING.md)) | **chosen** |

Why ONNX Runtime for the rest of the graph: MLAS GEMMs run at about torch's MKL/oneDNN throughput on these CPUs
(`py/gemm_bench.py`, `src/bin/linear_bench.rs`), all graphs export with exact parity, and ORT's optimizer already fuses
RMSNorm (after rewriting HF's `x * pow(ms, -0.5)` as `x / sqrt(ms)` so the pattern matches) and transposed MatMuls.
What ORT lacks for these models is memory-efficient attention for bidirectional sliding-window GQA and for Gemma 4's
chunked audio attention.

## Custom operators (`src/flash.rs`)

**`flashembed::FlashAttention(Q[B,Hq,S,D], K[B,Hkv,S,D], V[B,Hkv,S,Dv], key_mask[B,S]; scale, window) -> O[B,S,Hq,Dv]`**
reproduces transformers' eager attention for EmbeddingGemma 2 (text and vision) and NeoMME: score scale, GQA (kv head =
h / (Hq/Hkv)), key padding, symmetric sliding window |i-j| <= w, fp32 softmax. It is computed FlashAttention-style:
64-query x 256-key tiles with an online softmax, so no S x S tensor or dense mask ever exists, and key tiles outside a
query tile's window are skipped (4 of 5 EmbeddingGemma 2 layers and 14 of 17 NeoMME layers are sliding-window, so long
inputs cost O(S x window) there instead of O(S^2)). Tile GEMMs use the `gemm` crate (runtime AVX2/AVX-512 dispatch);
the softmax uses AVX2/FMA or AVX-512 intrinsics (8/16-wide Cephes expf, rel. error ~1e-7). `flash-check` compares it
with an f64 reference (max |diff| <= 4e-7 at every layout). Against torch's fused SDPA at the same shapes (16 threads,
Zen 3): EmbeddingGemma 2 sliding 8.5 vs 32 ms, global (d=512) 18 vs 26 ms, NeoMME sliding 3.8 vs 29 ms, global 5.5 vs
18 ms, vision (2,520 patches) 26 vs 33 ms.

**`flashembed::AudioChunkAttention(Q, K, V[B,T,H,D], RelK[13,H,D], key_mask[B,T]; left, softcap, invalid)`** is Gemma 4's
audio conformer attention (12-frame chunks, 12-frame left context, relative-position logits, logit soft-cap 50) as one
op. Besides speed, it is what makes the audio tower exportable with a dynamic length: `torch.export` never finished
(> 2 h) on HF's `unfold`/relative-shift chunking under symbolic shapes; with the op the export takes 25 s.

**`flashembed::ChannelLayerNorm(X[N,C,H,W], W[C]; eps, relu)`** normalises over the channel axis of an NCHW tensor
(ReLU fused). Gemma 4's audio subsampling convs write this as permute -> LayerNorm -> permute, and ONNX Runtime ran the
48 MB channel-last transpose of a 30 s clip single-threaded (~190 ms, 12 % of the audio tower).

## Export (`py/export_onnx.py --attn flash`)

A `torch.library` op with a dense reference implementation is registered as a transformers attention backend
(`AttentionInterface`), so the same HF modules either run the reference (for the wrapper-vs-HF parity checks) or emit
the custom ONNX node (for export). Because that backend is not one of transformers' mask types, transformers also stops
building dense [B,1,S,S] masks. Graph rewrites applied at export, all output-identical and checked against HF:

- RMSNorm as `x / sqrt(mean(x^2) + eps) * w`, which ORT fuses into one kernel.
- Gemma 4 "clippable" linears clamp with buffer bounds, which export as 216 `Min`/`Max` nodes per audio forward
  (single-threaded in ORT, 23 % of the audio tower); the bounds are read once and exported as `Clip` with constants.
- NeoMME's exclusive self-attention without materialising `repeat_kv(V)` and `F.normalize`'s `expand_as` (the attention
  output viewed as [B,S,Hkv,G,D], broadcast against V): 34 fewer `Expand` nodes per forward.

## Runner policies (`src/main.rs`)

- **Thread-pool autotune.** A forward pass is 1,000-1,400 small ORT ops, each a thread-pool sync. At startup the
  runner times the largest batch (best of up to 3 runs) under each candidate pool and keeps the fastest: all allotted
  CPUs pinned; half of them, largest socket first; and on CPU sets spanning sockets, only the largest socket's CPUs.
  All candidates are non-spinning: spinning pools won short tuning windows on shared nodes and then ran 2-3x slower
  once the node's load shifted. On 2-socket 32-CPU sets, 16 pinned threads beat 32 for short inputs. The choice and
  timings are recorded in each result JSON; `FLASHEMBED_ORT_POLICY=<name prefix>` forces a pool (including spinning
  ones).
- **Tune on the largest item.** Tuning on a small first image picked too few threads for the rest of a mixed set.
- **No ORT memory arena** (`FLASHEMBED_ORT_ARENA=1` restores it): same speed, 25-45 % lower peak RSS on batched long
  inputs.
- **Length-sorted batching** (longest first, outputs restored to input order), as sentence-transformers does.
- **Vision patch trimming.** The HF processors pad every image to the full patch budget (2,520 patches for 280 soft
  tokens); padded patches are masked keys and pool to nothing, so the runner feeds the vision graph only up to the
  batch's largest real patch count. Same output; a square image is 9 % fewer patches, a 16:9 video frame 14 % fewer.
  `FLASHEMBED_VISION_FULL=1` pads like the processor.

## Preprocessing parity

Matching the Python runtime bit for bit took more than porting the math:

- **Resize**: the HF processors resize with torchvision's uint8 antialias kernel. `src/pil_resize.rs` is Pillow's
  algorithm with ATen's int16 weight quantisation and its precision search; bit-identical up- and downscaling.
- **Decoders**: the `image` crate's JPEG decoder differs from libjpeg-turbo by +-1-2 per pixel (NeoMME late cosine 0.91
  on photos), so JPEG goes through libjpeg-turbo (`mozjpeg` crate) and WebP through libwebp. Images the `image` crate
  cannot read (e.g. palette TIFFs) fall back to FFmpeg.
- **Audio/video**: decode with the same FFmpeg major version as PyAV/torchcodec (FFmpeg 9; FFmpeg 6.1 gave AAC/Vorbis
  cosines of 0.997-0.999). `slurm/ffmpeg.sbatch` builds one. Audio decoding stops at 30 s (the extractor's cap).
- **Video frame sampling** must use torchcodec's content-scanned metadata (frame count, `average_fps` = frames /
  (last pts + duration - first pts)), not header fields: MKV/WebM headers have no frame count, and a 30/1 header rate
  vs a scanned 29.999999999999996 shifts every sampled frame by one (cosines 0.945-0.998 before, 1.0000000 after).
