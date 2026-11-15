# GPU runner (`flashembed-gpu`)

flashembed's GPU path is a native CUDA engine (`cuda/fegpu.cu`), driven from Rust (`src/gpu.rs`,
`src/bin/flashembed_gpu.rs`, cargo feature `cuda`). It reuses the CPU runner's tokenization and image preprocessing.
It produces the same vectors as the model-card runtimes on GPU (sentence-transformers for EmbeddingGemma 2,
transformers `NeoMMEForRetrieval` for NeoMME-260M-Retriever) and is faster and lighter than them.

## Results

Same inputs on both sides: 4,000 real text units and 400 real figures from one PubMed Central bundle (median text
~250 tokens, 90th percentile ~800, truncated at 1,024 tokens like the reference runs). Each runtime runs once warm,
on one GPU with 3-4 CPUs, in 2026-11 Slurm jobs. The card runtimes are sentence-transformers 6.3.0 (EmbeddingGemma 2)
and transformers 5.22.2 `NeoMMEForRetrieval` (NeoMME) on torch 2.15 / CUDA 13.2 (`py/gpu_ref_bench.py`). Their best
batch size (64 or 256 texts, 16 or 64 images) is shown. "x" is flashembed-gpu / reference items per second.
Memory is what each runtime's allocator holds: torch's peak reserved memory, and flashembed-gpu's weights +
activation arena + cuBLAS workspace. The CUDA context comes on top for both.

#### NVIDIA B200 (183 GB, sm_100)

| case | reference items/s (memory) | flashembed-gpu items/s (memory) | x | low-memory mode items/s (memory) | x |
|---|---|---|---|---|---|
| EmbeddingGemma 2 text, bf16 | 391 (49.8 GB) | **3,156** (1.6 GB) | **8.1** | 2,293 (0.86 GB) | 5.9 |
| NeoMME text, bf16 | 436 (17.4 GB) | **3,961** (1.9 GB) | **9.1** | 2,887 (0.93 GB) | 6.6 |
| EmbeddingGemma 2 images, bf16 | 66 (8.6 GB) | **366** (3.6 GB) | **5.5** | 273 (1.6 GB) | 4.1 |
| NeoMME images, bf16 | 179 (10.0 GB) | **877** (2.2 GB) | **4.9** | 765 (1.0 GB) | 4.3 |
| EmbeddingGemma 2 text, fp32 | 163 (13.6 GB) | **511** (2.7 GB) | **3.1** | 452 (1.3 GB) | 2.8 |
| NeoMME text, fp32 | 150 (7.4 GB) | **734** (3.2 GB) | **4.9** | 705 (1.5 GB) | 4.7 |
| EmbeddingGemma 2 images, fp32 | 29 (14.2 GB) | **74** (6.1 GB) | **2.5** | 67 (2.6 GB) | 2.3 |
| NeoMME images, fp32 | 112 (4.3 GB) | **503** (3.7 GB) | **4.5** | 430 (1.6 GB) | 3.8 |

#### NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB, sm_120)

| case | reference items/s (memory) | flashembed-gpu items/s (memory) | x |
|---|---|---|---|
| EmbeddingGemma 2 text, bf16 | 226 (12.9 GB) | **1,508** (0.86 GB) | **6.7** |
| NeoMME text, bf16 | 296 (4.8 GB) | **1,736** (0.93 GB) | **5.9** |
| EmbeddingGemma 2 images, bf16 | 45 (8.6 GB) | **206** (3.6 GB) | **4.6** |
| NeoMME images, bf16 | 180 (2.7 GB) | **967** (2.2 GB) | **5.4** |
| EmbeddingGemma 2 text, fp32 | 128 (13.6 GB) | **319** (2.7 GB) | **2.5** |
| NeoMME text, fp32 | 130 (7.4 GB) | **351** (1.5 GB) | **2.7** |
| EmbeddingGemma 2 images, fp32 | 25 (4.6 GB) | **43** (6.1 GB) / 41 (2.6 GB) | **1.7** |
| NeoMME images, fp32 | 111 (4.3 GB) | **246** (1.6 GB) | **2.2** |

Notes:

- **Reference column.** It is the faster of the two batch sizes measured, at that batch's memory. At its smallest
  batch the reference needs less memory, but it is also slower: EmbeddingGemma 2 images in fp32 on the B200 run at
  27.8 items/s in 4.6 GB at batch 16. flashembed-gpu's low-memory mode needs less memory than even the reference's
  smallest batch in every case (for example 2.6 vs 4.6 GB there, 0.86 vs 12.9 GB for EmbeddingGemma 2 text bf16), and
  it is still 2.3-6.6x faster than the reference's best.
- **Budgets.** Default budgets are `--max-tokens` 32,768 on B200-class GPUs and 8,192 elsewhere. Images use
  40,320 patches (EmbeddingGemma 2) and 16,384 (NeoMME). Low-memory mode is `--max-tokens 8192`, plus
  `--max-patches 10080 --img-mb 32` (EmbeddingGemma 2 images) or `--max-patches 4096` (NeoMME images). On the
  RTX PRO 6000 the 8,192-token budget is as fast as or faster than larger ones for text, so it is the default
  there, and the table shows the faster setting per row.
- **CPUs.** The reference had 4 CPUs on the B200 and 3 on the RTX PRO 6000; flashembed-gpu had 3 on both. On the
  B200, the NeoMME image path is limited by JPEG decoding on those 3 CPUs; the GPU alone does about 2,700 images/s.
- **Full context.** At 8,192 / 16,384 tokens (the 250 longest texts), the reference needs 13.6 GB (EmbeddingGemma 2)
  and 56.7 GB (NeoMME) reserved at batch 8 in FP32. flashembed-gpu holds 2.7 / 3.2 GB at any length up to its token
  budget, because memory depends on the token budget, not on the longest item.

### Other GPU runtimes on the same graphs

ONNX Runtime 1.31.1 (`onnxruntime-gpu`, CUDA 13 build) on the stock-attention ONNX graphs (`py/export_onnx.py
--attn sdpa`, the CPU runner's export without the custom ops), batch 64, the same 4,000 texts
(`py/ort_cuda_bench.py`):

| GPU | runtime | EmbeddingGemma 2 items/s (min cos vs reference fp32, peak device memory) | NeoMME items/s (min cos, memory) |
|---|---|---|---|
| B200 | ORT CUDA EP, fp32 | fails: the wheel ships sm_75-sm_90a and sm_120a kernels but none for sm_100 and no PTX | fails |
| B200 | ORT TensorRT EP 10.16, fp32 (TF32 inside TensorRT) | 364 (0.999998, 10.9 GB) | engine build fails (no implementation for a masked-fill node) |
| B200 | ORT TensorRT EP 10.16, bf16 | 462 (**0.952**, 1,936 items < 0.999, 7.2 GB) | 470 (**0.951**, 1,678 items < 0.999, 10.0 GB) |
| B200 | **flashembed-gpu fp32 / bf16** | **511 / 3,156** (1.0 / 0.99990, 1.6-2.7 GB) | **734 / 3,961** (1.0 / 0.99993, 1.9-3.2 GB) |
| RTX PRO 6000 | ORT CUDA EP, fp32 | 148 (0.999999, 28.9 GB) | 125 (0.999999, 43.4 GB) |
| RTX PRO 6000 | ORT TensorRT EP 10.16, fp32 | 237 (0.999998, 10.5 GB) | engine build fails |
| RTX PRO 6000 | ORT TensorRT EP 10.16, bf16 | 351 (**0.952**, 6.7 GB) | 326 (**0.951**, 9.6 GB) |
| RTX PRO 6000 | **flashembed-gpu fp32 / bf16** | **319 / 1,508** (1.0 / 0.99992, 0.9-2.7 GB) | **351 / 1,736** (1.0 / 0.99993, 0.9-1.5 GB) |

Peak device memory for ONNX Runtime is NVML's device-wide use (CUDA context included) sampled during the run.
The TensorRT engines are built once per shape profile (batch 1-64, 2-1,024 tokens) and cached; build time (20-90 s)
is not counted. TensorRT's bf16 engines lose most of the parity. The likely cause is that the whole network,
softmax and RMSNorm reductions included, runs in bf16 (not verified layer by layer). flashembed-gpu keeps those
in FP32.



### Parity

Per-item cosine to the reference's **FP32** vectors on the same GPU, minimum over all items (`py/gpu_compare.py`):

| case | B200 fp32 | B200 bf16 | RTX PRO 6000 fp32 | RTX PRO 6000 bf16 | reference bf16 vs reference fp32 |
|---|---|---|---|---|---|
| EmbeddingGemma 2 text (4,000) | 0.99999999999 | 0.99990 | 0.99999999995 | 0.99992 | 0.99968 |
| NeoMME text (4,000) | 0.9999999999 | 0.99993 | 0.9999999999 | 0.99993 | 0.99941 |
| EmbeddingGemma 2 images (400) | 0.999999985 | 0.99856 (1 < 0.999) | 0.999999987 | 0.99858 (1 < 0.999) | 0.99893 (1 < 0.999) |
| NeoMME images (400) | 0.9999999998 | 0.99981 | 0.9999999998 | 0.99981 | 0.99955 |
| EmbeddingGemma 2 text, 250 longest at 8,192 tokens | 0.9999999998 | 0.99998 | | | |
| NeoMME text, 250 longest at 16,384 tokens | 0.999999995 | 0.99999 | | | |
| NeoMME late vectors, 434,475 token rows at 16,384 tokens | 0.9999997 | 0.972 (median 0.99996) | | | 0.813 (median 0.99978) |

- FP32 is FP32-exact in practice: 1 - cos <= 2e-8 everywhere. The FP32 GEMMs use cuBLAS's BF16x9 emulation (B200)
  or SGEMM (RTX PRO 6000); attention uses 3xTF32 tensor-core MMAs. Against the CPU runner's vectors (computed at the
  same 1,024-token truncation) the GPU FP32 vectors agree to cos >= 0.99999996.
- BF16 stays above 0.999 for every text item. One of the 400 figures falls to 0.9986, on both GPUs. The reference's
  own bf16 run is no better on these inputs: its min cosine to its FP32 run is 0.9989 for figures (one below 0.999)
  and 0.9994-0.9997 for text. In this mode flashembed-gpu is as close to FP32 as the reference's bf16 is.
- GPU image preprocessing gives the same pixels as CPU preprocessing: the vectors are bit-identical in both
  precisions (`--img-mb 0` vs default, same batches).


### GPU utilisation and concurrency

On a B200 with 3 CPUs, the inputs were repeated to 20,000 texts and 2,000 figures. NVML was sampled every 100 ms
(`nvidia-smi` "utilization.gpu" is the share of time any kernel runs). Several flashembed-gpu processes then shared
the one GPU:

| case | 1 process: items/s (GPU busy) | 2 processes: total items/s | 4 processes: total items/s | reference alone: items/s (GPU busy) |
|---|---|---|---|---|
| EmbeddingGemma 2 text, bf16 | 2,834 (72 %) | 2,733 (93 %) | 2,822 (95 %) | 329 (55 %) |
| NeoMME text, bf16 | 3,625 (78 %) | 3,471 (93 %) | 3,487 (95 %) | 344 (61 %) |
| EmbeddingGemma 2 images, bf16 | 329 (69 %) | 325 (89 %) | 336 (93 %) | 58 (41 %) |
| NeoMME images, bf16 | 510 (20 %) | 608 (25 %) | 629 (26 %) | 103 (15 %) |
| EmbeddingGemma 2 text, fp32 | 507 (98 %) | 441 (99 %) | 441 (99 %) | |

- **One process is enough.** A second or fourth process raises the busy figure but not the throughput, so a single
  flashembed-gpu process uses the GPU fully. The one exception is NeoMME figures, which are bound by JPEG decoding on
  3 CPUs: the GPU is 80 % idle, and more processes or more CPUs per GPU are the lever there.
- **The reference leaves the GPU 39-85 % idle** at its batch 64 / 16, waiting on Python-side tokenization,
  preprocessing and kernel launches.
- Memory grows linearly with processes (2.3-4.4 GB each), so packing several workers on one GPU buys nothing for text.

### Narrow-precision GEMMs (`--quant fp8|nvfp4`)

`--quant` runs every layer GEMM in a block-scaled narrow format on cuBLASLt. Embeddings, patch stems, output heads,
attention, norms and the residual stream stay as in bf16 mode. The formats are:

- MXFP8: E4M3 values with one UE8M0 scale per 32 values.
- NVFP4: E2M1 values with one UE4M3 scale per 16 values, plus a per-tensor pre-scale on the weights.

Weights are quantised once at load. Activations are quantised per block by a small kernel before each GEMM, with the
scale rounding of the cuBLAS docs ("1D Block Scaling", tiled scale-factor layout).

| GPU | case | bf16 items/s | MXFP8 items/s (median / min cos vs reference fp32) | NVFP4 items/s (median / min cos) |
|---|---|---|---|---|
| B200 | EmbeddingGemma 2 text | 3,162 | 3,042 (0.9983 / 0.987) | 2,364 (0.979 / 0.940) |
| B200 | NeoMME text | 3,961 | 4,149 (0.9977 / 0.985) | 3,472 (0.969 / 0.904) |
| B200 | EmbeddingGemma 2 images | 358 | 366 (0.9980 / 0.987) | 314 (0.980 / 0.944) |
| B200 | NeoMME images | 700 | 722 (0.9956 / 0.978) | 786 (0.938 / 0.832) |
| RTX PRO 6000 | EmbeddingGemma 2 text | 1,404 | 1,538 (0.9983 / 0.991) | 1,502 (0.979 / 0.936) |
| RTX PRO 6000 | NeoMME text | 1,676 | 2,076 (0.9977 / 0.985) | 2,189 (0.969 / 0.904) |
| RTX PRO 6000 | EmbeddingGemma 2 images | 203 | 229 (0.9981 / 0.988) | 247 (0.980 / 0.948) |
| RTX PRO 6000 | NeoMME images | 806 | 881 (0.9956 / 0.978) | 909 (0.938 / 0.832) |

For reference, bf16 is at median 0.99999, min 0.9986-0.9999.

- **Not worth it for these models.** GEMMs are only 25-30 % of the bf16 kernel time on the B200 and about half on
  the RTX PRO 6000. Halving or quartering them gives at most +5 % (B200) or +9-31 % (RTX PRO 6000) end to end, and on
  the B200 the activation-quantisation pass eats most of that.
- **Parity falls below the 0.999 bar.** MXFP8 lands at a median cosine of 0.996-0.998 to the FP32 vectors (all
  4,000 texts below 0.999). NVFP4 lands at 0.94-0.98. That is a 100-1000x larger error than bf16 mode, for embeddings
  that will be compared by cosine.
- **Kept as an option, not a default.** The remaining headroom is in fusing the activation quantisation into the
  producing kernels (one pass fewer). That cannot fix the parity.
- **INT8 was not built.** On these GPUs INT8 tensor-core throughput is the same as FP8, and cuBLASLt offers only
  tensor-wide scaling for INT8 (no block scaling), so it would be no faster than MXFP8 and less accurate.

### Video and audio (EmbeddingGemma 2)

`--modality video|audio` runs the rest of EmbeddingGemma 2 on the GPU, so a corpus run needs no CPU-only stage.

- **Video** follows `EmbeddingGemma2VideoProcessor`, as the CPU runner does:
  - ffmpeg decodes on the CPU at 1 fps, up to 32 frames (uniform overflow), 140 soft tokens per frame.
  - Each frame goes through the same GPU resize + vision tower + 3x3 pooler as an image. A clip whose decoded frames
    exceed the per-batch byte budget is resized on the CPU instead.
  - The text row is `<|image>` + `<|video|>` x n + `<image|>` per frame.
- **Audio** follows the Gemma 4 audio tower, a port of the CPU runner's ops:
  - ffmpeg decodes to 16 kHz mono (first 30 s), and log-mel is computed on the CPU (`src/media.rs`).
  - Two 3x3 stride-2 subsampling convs, each with a channel LayerNorm + ReLU, feed the input projection.
  - 12 conformer layers follow, each: FF (x0.5) / chunked attention / light conv / FF (x0.5) / norm. The attention
    uses 12-frame chunks with 12 frames of left context, relative-position logits and a soft cap of 50. The light conv
    is a GLU, a causal depthwise conv of width 5, an RMSNorm and SiLU. The clippable linears' bounds are applied.
  - `output_proj` and the `embed_audio` projection come last.
  - All of it runs as packed clips with per-stage row maps, one fused kernel per non-GEMM step. The subsampling convs
    always run in FP32.

Results on one B200 (job limited to 1 CPU core, so video is bound by ffmpeg decoding; the GPU time is shown
separately). Real PMC media: 120 videos (96 mp4, 18 avi, 4 mov, 2 mpg) and 46 audio recordings. The card runtime is
sentence-transformers on the same GPU, batch 4; CPU flashembed is the CPU runner (16 cores, EPYC 75F3).

| case | flashembed-gpu items/s (GPU s) | card runtime GPU items/s | CPU flashembed items/s | flashembed-gpu device memory | card runtime peak alloc / reserved |
|---|---|---|---|---|---|
| video bf16 | 4.49 (3.3 s for 120) | 1.15 | - | 3.5 GB | 5.2 / 9.3 GB |
| video fp32 | 3.39 (11.9 s) | 1.03 | 0.10 | 6.1 GB | 9.6 / 19.4 GB |
| audio bf16 | 77.4 | 9.07 | - | 2.5 GB | 1.5 / 1.8 GB |
| audio fp32 | 57.6 | 10.6 | 1.57 | 4.3 GB | 2.8 / 3.0 GB |

Audio's device memory is the arena for `--max-frames 12000` (about 10 clips of 30 s per forward); a smaller value
trades it for throughput.

Parity (cosine, per item):

| set | flashembed-gpu fp32 | flashembed-gpu bf16 |
|---|---|---|
| benchmark videos (34) vs CPU card runtime / CPU flashembed (fp32) | min 0.9999999995 | min 0.99988 |
| real videos (120) vs card runtime GPU fp32 | median 0.99999999998, 2 below 0.999 (min 0.9869), 3 below 0.9999 | same 2 below 0.999 |
| benchmark audio (40) vs CPU card runtime / CPU flashembed | min 0.99999999 | min 0.99989 |
| real audio (46) vs card runtime GPU fp32 | min 0.9999997 | min 0.99987 |

The three real videos below 0.9999 are container/decoder effects, not the model:
- two MPEG-PS `.mpg` files (0.987, 0.989): when torchcodec seeks in these open-GOP streams it drops frames, so the
  card runtime gets a later frame than the one it asked for (e.g. 0.91 s instead of 0.23 s). flashembed decodes
  sequentially and gets the requested frame;
- one DV `.avi` (0.9997): the same frames, colour conversion of 4:1:1 chroma differs by at most 14/255.

Video frame selection, shared with the CPU runner (`src/media.rs`), now reproduces torchcodec's exact mode:
- packet pts, or dts when the container has none (AVI);
- seconds as `ts * num / den`;
- frame i is the first decoded frame whose pts is at or after the i-th smallest packet pts. This uses ffmpeg with
  `-copyts`, a `select` filter and `showinfo`.

Before this fix, about 5-7 % of real videos (rounded sampling indices, AVI without pts, streams starting at pts 1000)
got shifted frames, at cos 0.98-0.998 vs the card runtime. That includes CPU-runner vectors built before the fix.

Adding video and audio does not change text or images: their vectors are bit-identical to the previous build, and
alternating runs on the same inputs give the same speed within noise (text 1,944 vs 1,951 items/s, images 246 vs
243 items/s, at 1 CPU core).

### Persistent mode (`serve`)

`flashembed-gpu serve` loads the model once and then takes one request per stdin line, for corpus runs that
embed many small bundles. It uses the same protocol as the CPU runner's `serve`.

```bash
flashembed-gpu serve --model gemma --snapshot <snapshot> --dtype bf16 --modalities text,image
# stdout: {"ready": true, "model": "gemma", "modalities": ["text", "image"], "load_s": ..., ...}
# stdin:  {"id": "b1/text", "modality": "text", "input": "/data/b1/text.jsonl", "out": "/out/b1/text"}
# stdout: {"id": "b1/text", "ok": true, "items": 5261, "failed": [], "seconds": 2.1, "items_per_s": ..., "out": ...}
```

- **Outputs:** each request writes the same files as `embed`.
- **Failures:**
  - Item-level failures (bad JSON row, undecodable file, too many tokens) are listed in `failed`. The other items are
    still embedded and written.
  - A request that can't be read, or embeds nothing, returns `ok: false` with an `error`.
  - A GPU error fails that batch's items.
  - The server keeps going after any failure and exits at end of input.
- **Pipelining:** a preprocessing thread works through the requests in order. It prepares the next request's batches
  while the GPU runs the current one.
- **Output parity:** vectors are bit-identical to `embed` on the same inputs.

On 20 real corpus bundles (124,250 text and figure units per model, B200, 8 CPU cores), compared with one `embed`
process per bundle and modality:

| model | per-bundle `embed` | `serve` | speed-up |
|---|---|---|---|
| EmbeddingGemma 2 | 598 items/s (208 s) | 1,376 items/s (90 s) | 2.3x |
| NeoMME | 657 items/s (189 s) | 1,356 items/s (92 s) | 2.1x |

Startup (load + warm-up) takes 2-3 s and happens once instead of once per invocation.

`FLASHEMBED_ATTN=cudnn` (experimental) in `serve` on the same 20 bundles: EmbeddingGemma 2 1,555-1,665 items/s vs
1,358-1,381 with the built-in attention (+14-20 %; 66 cuDNN graphs, 4-7 s of graph building per process), NeoMME no
gain (1,383 vs 1,414-1,427). Vectors vs the built-in bf16 kernels: min cos 0.9953 (EmbeddingGemma 2) / 0.9986
(NeoMME); vs the card runtime fp32 on the benchmark sets the cuDNN path is as close as the built-in one (min cos
0.9999 text, 0.9990 figures). It stays opt-in until the per-item outliers are understood.

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
Environment: `FLASHEMBED_ATTN=simt` forces the plain SIMT attention kernels; `FLASHEMBED_ATTN=cudnn` (experimental,
needs a build with `FLASHEMBED_CUDNN_INCLUDE` pointing at the cuDNN >= 9.10 and cudnn-frontend headers, and
`libcudnn.so.9` at run time) uses cuDNN's fused attention for bf16; `FLASHEMBED_FP32_GEMM=sgemm` runs FP32
GEMMs as plain SGEMM instead of cuBLAS's BF16x9 FP32 emulation (see Precision); `--quant fp8|nvfp4` is described
above.

Scope: text and images for both models (EmbeddingGemma 2 at the default 280 soft tokens per image), and video and
audio for EmbeddingGemma 2 (`--max-frames`, audio frames per forward, default 12000). Video and audio need `ffmpeg` /
`ffprobe` (or `FLASHEMBED_FFMPEG` / `FLASHEMBED_FFPROBE`), as on the CPU runner.

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
  - FP32: the same tiling on tensor cores with 3xTF32: each FP32 operand is split into a TF32 high part and a TF32
    remainder, and three `mma.sync.m16n8k8` products accumulate in FP32. The dropped low x low term is ~2^-22
    relative, so the parity is FP32-level (min cos >= 0.99999998 vs torch FP32). This is 1.3-1.8x faster end to end
    than the SIMT kernel it replaced, which remains as `FLASHEMBED_ATTN=simt`.
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
- **No CUDA graphs.** Every batch has a different packed shape, and at a 32k-token budget a forward is tens of
  milliseconds of GPU work against roughly 300 launches. The concurrency test below shows no idle GPU capacity for
  graphs to recover.

### Image preprocessing on the GPU

- **EmbeddingGemma 2 (default).** The CPU only decodes (libjpeg-turbo / libwebp, as before). The decoded RGB8 image
  and the resize coefficients go to the GPU, which runs the same separable integer resize as `src/pil_resize.rs`
  (Pillow's algorithm with ATen's int16 weight quantisation, horizontal pass then vertical) and writes the 16 px
  patches directly in the compute dtype. The integer arithmetic is identical, so the vectors are bit-identical to
  CPU preprocessing (checked: `--img-mb 0` gives the same vectors). Decoded images larger than the per-batch budget
  (`--img-mb`, default 128) are preprocessed on the CPU.
- **NeoMME.** Patches are uploaded as uint8 (padding 0, which normalises to the processor's -1) and normalised inside
  the patch-stem LayerNorm kernel: a quarter of the host copies and PCIe traffic of float patches.

