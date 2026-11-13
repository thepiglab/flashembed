# flashembed

Fast FP32 CPU inference for two multimodal embedding models, with the same vectors as their reference Python runtimes:

- **EmbeddingGemma 2** ([google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2)): text, images, audio and video, up to 8,192 tokens.
- **NeoMME-260M-Retriever** ([Hcompany/NeoMME-260M-Retriever](https://huggingface.co/Hcompany/NeoMME-260M-Retriever-ST-dense): text and images (documents and queries); 1024-d dense
  and 128-d late-interaction (multi-vector) embeddings, up to 16,384 tokens.

flashembed is a Rust runner around ONNX Runtime. The models are exported to ONNX once. Attention and two other hot
spots are custom operators written in Rust (a FlashAttention-style tiled kernel with sliding-window tile skipping and
AVX2/AVX-512 softmax). Tokenization, image/audio/video decoding and preprocessing, batching and pooling are Rust too.

**About the name.** "flash" refers to the attention kernel's algorithm, the tiling and online softmax introduced by
FlashAttention, reimplemented from scratch in Rust for CPUs. flashembed does not use, wrap or depend on the
FlashAttention-2/3 libraries (those are CUDA kernels for NVIDIA GPUs), runs on CPUs only, and is not affiliated with
the FlashAttention project. See [docs/DESIGN.md](docs/DESIGN.md#custom-operators-srcflashrs) for what the kernel does.

## Results at a glance

Compared with the runtime each model card documents (sentence-transformers / transformers on PyTorch, CPU, FP32) on
the same inputs, nodes and CPUs, over 353 cases measured on both runtimes, on three AMD EPYC generations:

- **Same vectors.** Cosine >= 0.9999999 to the reference in every case. Covers text from 128 tokens to each model's
  maximum, PNG/JPEG/GIF/WebP images and real published figures, WAV/MP3/FLAC/Ogg/Opus/M4A audio,
  MP4/MOV/AVI/MKV/WebM video, and NeoMME documents and queries (dense and late vectors).
  EmbeddingGemma 2's Matryoshka sizes (768/512/256/128, `--dim`) match sentence-transformers' `truncate_dim` at
  every size and modality (cosine >= 0.9999999, identical top-5 neighbours).
- **Faster:** in 351 of 353 cases; median 1.82x.
  - Short text: 1.1-1.7x.
  - Long text: 2.2-3.1x at EmbeddingGemma 2's 8,192 tokens and 4.4-6.1x at NeoMME's 16,384.
  - Images: about 1.4x in the format/size groups and 1.9x in the batch grid.
  - Video: 2.2x. Audio: 1.0-8x.
  - The two exceptions (0.97x and 0.99x) are mid-size images on heavily oversubscribed shared nodes; repeated runs
    put them above 1x.
- **Lighter:** lower peak memory in every case, median 0.59x (0.35x at 16k tokens).

| CPU | thread x batch grid | context length 128 ... max | modalities / file types |
|---|---|---|---|
| AMD EPYC 7702 (Zen 2, 2 sockets) | 1.67x | 2.04x | 1.88x |
| AMD EPYC 75F3 (Zen 3, 2 sockets) | 1.44x | 1.67x | 1.41x |
| AMD EPYC 9655P (Zen 5, 1 socket, AVX-512) | 2.18x | 2.39x | 2.02x |

(median speed over the cases in each group)

## What was profiled

All in [docs/PROFILING.md](docs/PROFILING.md), with the method and hardware:

- **Parity**: per-item cosine against the reference runtime, for every model, modality, file type, length and CPU,
  and for each EmbeddingGemma 2 Matryoshka size.
- **Throughput and memory**: items/s and peak RSS over a thread (4/8/16/32) x batch (1/2/4) grid for text and images.
- **Context length**: 128 to 16,384 tokens, mixed-length batches, over-maximum inputs (truncation), short queries.
- **Modalities and file types**: image formats and sizes, image token budgets, audio formats and lengths, video
  containers and lengths, real published media.
- **Per-op breakdown**: where the time goes (GEMMs, attention, norms, RoPE, other) in torch vs flashembed.
- **Close cases**: repeated runs of the slowest cases, plus a GEMM microbenchmark (torch/MKL, ONNX Runtime/MLAS,
  the `gemm` crate).

How it works, why ONNX Runtime plus Rust kernels (and not candle or stock ONNX Runtime), and what it took to match
the reference preprocessing bit for bit: [docs/DESIGN.md](docs/DESIGN.md).

### Hardware tested

Batch jobs on shared compute nodes, with 16 CPUs / 64 GB per job (plus 32 CPUs / 128 GB on the Zen 2 nodes):

- AMD EPYC 7702 (Zen 2): 2 sockets x 64 cores, AVX2 + FMA.
- AMD EPYC 75F3 (Zen 3): 2 sockets x 32 cores, AVX2 + FMA.
- AMD EPYC 9655P (Zen 5): 1 socket x 96 cores, AVX-512.

### Main findings

- **Attention is the lever.** Exported attention materialises S x S scores and dense masks. With it, ONNX Runtime
  alone lost to torch above ~1k tokens and ran out of memory at 16k. The fused kernel never builds them, and it skips
  key tiles outside the sliding windows (most layers of both models).
- **Thread pools on shared nodes.** A forward is 1,000-1,400 small ops. Spinning thread pools collapsed 2-3x when
  neighbours' load shifted. flashembed autotunes a non-spinning, pinned pool at startup (all CPUs, half of them, or
  one socket) on the largest batch.
- **Bit-exact preprocessing needs the same decoders.** That means libjpeg-turbo, libwebp, and the same FFmpeg major
  version as the Python side, plus Pillow/ATen-exact resize weights and torchcodec's content-scanned video frame
  timing. Each mismatch cost 0.003-0.09 cosine before it was fixed.
- **Some ONNX Runtime ops are single-threaded.** Rewriting them at export removed up to a quarter of the audio tower's
  time: clamps exported as constant `Clip`, a channel LayerNorm as a Rust op.

## Quick start

Requirements: Rust (stable, >= 1.92), Python 3.12 with `onnxruntime==1.31.*` (flashembed loads that wheel's
`libonnxruntime.so`), and, for exporting the graphs, PyTorch (CPU), `transformers==5.22.2`, `onnx`, `onnxscript`.
Audio/video also need `ffmpeg`/`ffprobe`, ideally FFmpeg 9 (`FLASHEMBED_FFMPEG` / `FLASHEMBED_FFPROBE`).

```bash
# 1. export the ONNX graphs once (needs the HF snapshots; writes <dir>/gemma_*.onnx, neomme_*.onnx)
python py/export_onnx.py --attn flash --fuse-norm --out onnx --inputs inputs \
  --which gemma_lm gemma_vision gemma_vision_dyn gemma_audio neomme_text neomme_image

# 2. build
cargo build --release

# 3. embed: one item per line, {"id": ..., "text": ...} or {"id": ..., "path": ...} (path relative to the file)
export ORT_DYLIB_PATH=$(python -c "import onnxruntime,glob,os;print(glob.glob(os.path.dirname(onnxruntime.__file__)+'/capi/libonnxruntime.so*')[0])")
target/release/flashembed embed --model gemma --modality text --input docs.jsonl --out vectors \
  --onnx onnx --snapshot ~/.cache/huggingface/hub/models--google--embeddinggemma-2/snapshots/<rev>
```

`embed` writes `embeddings.dense.npy` ([N, dim], L2-normalised), `embeddings.ids.json` and, for NeoMME,
`embeddings.late.npy` plus `embeddings.late_offsets.npy` (row ranges per item). Options: `--modality
text|image|audio|video`, `--side document|query` (NeoMME), `--threads`, `--batch`, `--max-len`, `--max-soft-tokens`,
`--dim 768|512|256|128` (EmbeddingGemma 2 Matryoshka size: keeps the leading dimensions and re-normalises, as
sentence-transformers' `truncate_dim` does; queries and documents must use the same size).
Text is embedded as given: add the model card's prompt prefix yourself (EmbeddingGemma 2:
`title: {title} | text: {content}` for documents, `task: search result | query: {q}` for queries).
`flashembed bench` runs the benchmark harness; `flash-check` tests the attention kernel against an f64 reference.

### Running the benchmarks on a Slurm cluster

`slurm/*.sbatch` hold no site specifics. Copy `site.env.example` to `local/site.env` (gitignored), set
`EMBED_ROOT` (and optionally account/QOS/partition), sync the repository to `$EMBED_ROOT/code`, and submit with
`--export=ALL,EMBED_ENV=<path to site.env>`:

| step | script | what |
|---|---|---|
| setup | `setup.sbatch` | rustup toolchain, model snapshots, text/image inputs |
| venv | `venv.sbatch` | pinned CPU PyTorch venv; CPU torchcodec + FFmpeg shared libraries |
| ffmpeg | `ffmpeg.sbatch` | FFmpeg 9 CLI matching PyAV/torchcodec's |
| media | `media.sbatch` | audio/video/format/size inputs, acceptance probe |
| export | `export.sbatch` | ONNX graphs + wrapper-vs-HF parity checks |
| build | `build.sbatch` (`TAG=...`) | release build on node-local disk, installs `$EMBED_ROOT/bin/flashembed-$TAG` |
| smoke | smoke | `smoke.sbatch` (`BIN=... ONNX=...`) | quick check: kernel test, reference vs flashembed parity + speed for every model x modality, `embed` |
| mrl | `mrl.sbatch` (`BIN=... ONNX=...`) | EmbeddingGemma 2 Matryoshka sizes vs sentence-transformers' `truncate_dim`, all modalities |
| bench / ctx / mm / ops | `bench.sbatch`, `ctx.sbatch`, `mm.sbatch`, `ops.sbatch` | the benchmark families in PROFILING.md |

`py/final_report.py` collects a results tree into the tables of PROFILING.md (`py/fill_doc.py`).

## Layout

| path | what |
|---|---|
| `src/main.rs` | CLI (`embed`, `bench`), ONNX Runtime sessions, thread-pool autotune, batching, pooling |
| `src/flash.rs` | custom ops: FlashAttention, AudioChunkAttention, ChannelLayerNorm |
| `src/prep.rs`, `src/pil_resize.rs` | tokenization, EmbeddingGemma 2 / NeoMME image preprocessing, Pillow/ATen-exact resize |
| `src/media.rs` | FFmpeg audio/video decoding, Gemma 4 log-mel features, video frame sampling |
| `src/bin/` | `flash-check` (kernel test), `linear-bench` (GEMM), `candle-neomme` / `gemm-bench` (backend evidence, feature `candle`) |
| `py/export_onnx.py` | ONNX export with the custom ops and graph rewrites, parity checks against HF |
| `py/py_bench.py` | reference runtime benchmark (the model-card Python code path) |
| `py/make_*.py`, `py/scan_media.py` | benchmark inputs from the PMC open-access subset (+ synthetic media) |
| `py/final_report.py`, `py/mrl_report.py`, `py/compare.py`, `py/profile_ops.py`, ... | reports, parity, per-op profiles, diagnostics |
| `slurm/` | job scripts (site-neutral) |

## License

MIT, see [LICENSE](LICENSE). The models are separate downloads under their own licences.
