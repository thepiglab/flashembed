//! flashembed-gpu: GPU embedding runner (native CUDA engine, `cuda/fegpu.cu`) with the CPU runner's preprocessing.
//!
//!   flashembed-gpu embed --model gemma|neomme --modality text|image --input items.jsonl --out DIR --snapshot SNAP
//!                        [--dtype bf16|fp32] [--max-tokens 32768] [--max-items 512] [--max-len N] [--prefix STR]
//!                        [--side document|query] [--no-late] [--dim 768|512|256|128]
//!
//! Writes the same files as `flashembed embed` (embeddings.dense.npy, .ids.json, NeoMME .late.npy/.late_offsets.npy)
//! plus embeddings.json (throughput, GPU memory, settings). Items are length-sorted and packed into batches by a
//! token budget (no padding); a preprocessing thread prepares the next batch while the GPU runs the current one.
//!
//!   flashembed-gpu serve --model gemma|neomme --snapshot SNAP [--modalities text,image,video] [embed options]
//!
//! Persistent mode: loads the model once, warms up, prints `{"ready": true, ...}`, then reads one JSON request per
//! stdin line, `{"id": ..., "modality": "text|image|video|audio", "input": "items.jsonl", "out": "DIR"}`, writes the
//! same files as `embed` and prints one line per request, `{"id": ..., "ok": true, "items": N, "seconds": s, ...}` or
//! `{"id": ..., "ok": false, "error": "..."}`. A failing request never stops the server; it exits at end of input.
//! The next request's preprocessing overlaps the current request's GPU work.

use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand, ValueEnum};
use flashembed::gpu::{self, Arch, Dtype, Engine};
use flashembed::{media, prep};
use ndarray::{Array1, Array2};
use rayon::prelude::*;
use serde_json::json;

#[derive(Parser)]
#[command(version, about = "flashembed GPU runner")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    Embed(Args),
    /// Persistent server: model loaded once, newline-delimited JSON requests on stdin (see the module docs).
    Serve(ServeArgs),
    /// Time the attention kernel alone on random inputs (one line of JSON per shape).
    AttnBench {
        /// shapes as D:Hq:Hkv:S:nseq:window (window -1 = full), comma-separated
        #[arg(long)]
        shapes: String,
        #[arg(long, default_value_t = 20)]
        iters: i32,
        #[arg(long)]
        fp32: bool,
    },
}

#[derive(Clone, Copy, ValueEnum, PartialEq, Eq, Debug)]
enum Model {
    Gemma,
    Neomme,
}
#[derive(Clone, Copy, ValueEnum, PartialEq, Eq, Debug)]
enum Modality {
    Text,
    /// EmbeddingGemma 2 late chunking: rows {"id", "text", "prefix_chars", "spans": [{"id", "start", "end"}]} (char
    /// offsets into text); one vector per span: mean of the document's final hidden states over the span's tokens
    /// (prompt prefix excluded), then the model's projection and L2 normalisation. Documents longer than --max-len go
    /// through overlapping windows (--late-overlap tokens); each span is pooled in the window where it is most central.
    #[value(name = "text_late", alias = "text-late")]
    TextLate,
    Image,
    /// EmbeddingGemma 2 only: ffmpeg decode at 1 fps, <= 32 frames, 140 soft tokens per frame
    Video,
    /// EmbeddingGemma 2 only: ffmpeg decode to 16 kHz mono (first 30 s), log-mel, Gemma 4 audio conformer
    Audio,
}
#[derive(Clone, Copy, ValueEnum, PartialEq, Eq, Debug)]
enum Dt {
    Bf16,
    Fp32,
}

#[derive(clap::Args)]
struct ServeArgs {
    /// modalities to serve (comma-separated); the engine is sized for all of them
    #[arg(long, default_value = "text")]
    modalities: String,
    #[command(flatten)]
    base: Args,
}

impl Args {
    fn input_file(&self) -> &Path {
        self.input.as_deref().unwrap_or(Path::new(""))
    }
    fn out_dir(&self) -> &Path {
        self.out.as_deref().unwrap_or(Path::new(""))
    }
}

#[derive(clap::Args, Clone)]
struct Args {
    #[arg(long, value_enum)]
    model: Model,
    #[arg(long, value_enum, default_value = "text")]
    modality: Modality,
    /// items.jsonl (embed; serve takes it per request)
    #[arg(long)]
    input: Option<PathBuf>,
    #[arg(long)]
    out: Option<PathBuf>,
    #[arg(long)]
    snapshot: PathBuf,
    #[arg(long, value_enum, default_value = "bf16")]
    dtype: Dt,
    /// token budget per forward (packed, no padding); also sizes the activation arena. Default: 32768 on GPUs with
    /// B200-class shared memory (measured fastest there), 8192 elsewhere (RTX PRO 6000: faster and 4x less memory)
    #[arg(long)]
    max_tokens: Option<usize>,
    /// image patch budget per forward (EmbeddingGemma 2 vision tower: default 40320 = 16 images at 280 soft tokens;
    /// NeoMME patch stem: default 16384, at least 4096 = one 2048 x 2048 page); unused for text
    #[arg(long)]
    max_patches: Option<usize>,
    /// EmbeddingGemma 2: MB of decoded images per batch for GPU-side resize + patching (0 = preprocess on the CPU)
    #[arg(long, default_value_t = 128)]
    img_mb: usize,
    #[arg(long, default_value_t = 1024)]
    max_items: usize,
    /// token limit per text (default: model maximum, 8192 / 16384)
    #[arg(long)]
    max_len: Option<usize>,
    /// prepended to every text (e.g. EmbeddingGemma 2's "title: none | text: ")
    #[arg(long, default_value = "")]
    prefix: String,
    #[arg(long, default_value = "document")]
    side: String,
    #[arg(long)]
    no_late: bool,
    #[arg(long)]
    dim: Option<usize>,
    /// only the first N items
    #[arg(long)]
    n: Option<usize>,
    /// run the first batch once before timing (CUDA/cuBLAS warm-up), like the reference benchmark
    #[arg(long)]
    warmup: bool,
    /// write embeddings.* (default) or <tag>.* file names
    #[arg(long, default_value = "embeddings")]
    tag: String,
    /// audio: log-mel frames per forward (100 per second; a 30 s clip is 3,000)
    #[arg(long, default_value_t = 12000)]
    max_frames: usize,
    /// narrow-precision layer GEMMs (with --dtype bf16): none, fp8 (MXFP8: E4M3 + UE8M0 scale per 32 values) or
    /// nvfp4 (E2M1 + UE4M3 scale per 16 values); attention, norms and the residual stream are unchanged
    #[arg(long, default_value = "none")]
    quant: String,
    /// preprocessing only (no GPU): time the producer thread alone
    #[arg(long)]
    prep_only: bool,
    /// text_late: tokens shared by consecutive windows of a document longer than --max-len
    #[arg(long, default_value_t = 1024)]
    late_overlap: usize,
}

enum GPrep {
    Host(prep::GemmaImage),
    Dev(gpu::ImagePlan, Vec<u8>),
}

static DECODE_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static PREP_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// decode + model preprocessing of one image, with per-stage CPU time accounting
fn timed_image<T>(path: &Path, f: impl Fn(&image::RgbImage) -> Result<T>) -> Result<T> {
    use std::sync::atomic::Ordering::Relaxed;
    let t = Instant::now();
    let im = prep::load_rgb(path)?;
    DECODE_NS.fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
    let t = Instant::now();
    let r = f(&im)?;
    PREP_NS.fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
    Ok(r)
}

enum Batch {
    /// packed token ids, cu_seqlens, item indices
    Text { ids: Vec<i32>, cu: Vec<i32>, items: Vec<usize> },
    /// EmbeddingGemma 2 images: either host-preprocessed float patches, or (dev) decoded RGB8 sources + resize
    /// descriptors for GPU-side preprocessing
    /// EmbeddingGemma 2 audio clips: log-mel features + per-stage row maps (fe_gemma_audio), then the text rows
    Audio {
        cu0: Vec<i32>, feats: Vec<f32>, m0: Vec<i32>, rows1: Vec<i32>, m1: Vec<i32>, rows2: Vec<i32>, m2: Vec<i32>, soft_rows: Vec<i32>,
        ids: Vec<i32>, cu: Vec<i32>, mm: Vec<i32>, items: Vec<usize>,
    },
    GemmaImage {
        vcu: Vec<i32>, pos: Vec<i32>, patches: Vec<f32>, slot_ptr: Vec<i32>, slot_patch: Vec<i32>, ids: Vec<i32>, cu: Vec<i32>,
        mm: Vec<i32>, items: Vec<usize>, dev: bool, src: Vec<u8>, desc: Vec<i32>, coef: Vec<i32>, tmp: usize, max_pix: usize,
    },
    NeoImage { ids: Vec<i32>, cu: Vec<i32>, pos: Vec<i32>, patches: Vec<u8>, rows: Vec<i32>, items: Vec<usize> },
    /// late chunking: packed document windows; span s pools tokens lo[s]..hi[s] and belongs to (item, span index)
    Late { ids: Vec<i32>, cu: Vec<i32>, lo: Vec<i32>, hi: Vec<i32>, refs: Vec<(usize, usize)>, meta: Vec<SpanMeta>, items: Vec<usize> },
}

/// Per output span: its id, the document window it was pooled in, its token count, and whether it had to be
/// clipped to that window (span longer than a window) or widened to one token (no token starts inside it).
#[derive(Clone)]
struct SpanMeta {
    id: String,
    window: usize,
    windows: usize,
    tokens: usize,
    clipped: bool,
    widened: bool,
    /// spans of the document (a document's output is complete when all `of` spans are back)
    of: usize,
}

/// One text_late document, tokenized and laid out: the windows to run and, per span, (window, lo, hi) token
/// positions inside that window.
struct LateDoc {
    windows: Vec<Vec<i32>>,
    spans: Vec<(usize, usize, usize, SpanMeta)>,
}

/// Tokenize a text_late row and place its spans. A span covers the document tokens (not special, not in the prompt
/// prefix) whose first character is inside [start, end); a span in which no token starts takes the token containing
/// `start`. Up to `wmax` tokens: one window, the whole tokenized document. Longer: windows of [bos + prefix tokens] +
/// a body slice + [eos], consecutive slices overlapping by `overlap` tokens; a span goes to the window that contains it
/// with the largest margin to the slice edges (else the one overlapping it most, clipped to it).
fn late_doc(tok: &tokenizers::Tokenizer, row: &str, wmax: usize, overlap: usize) -> Result<LateDoc> {
    let v: serde_json::Value = serde_json::from_str(row)?;
    let text = v["text"].as_str().context("no text")?;
    let prefix = v["prefix_chars"].as_u64().unwrap_or(0) as usize;
    let enc = tok.encode_char_offsets(text, true).map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
    let ids: Vec<i32> = enc.get_ids().iter().map(|&x| x as i32).collect();
    let offs = enc.get_offsets();
    let special = enc.get_special_tokens_mask();
    let nchar = text.chars().count();
    let lead: Vec<usize> = (0..ids.len()).take_while(|&k| special[k] == 1).collect();
    let mut trail: Vec<usize> = (0..ids.len()).rev().take_while(|&k| special[k] == 1 && k >= lead.len()).collect();
    trail.reverse();
    let content: Vec<usize> = (lead.len()..ids.len() - trail.len()).collect();
    let hdr: Vec<usize> = content.iter().copied().take_while(|&k| offs[k].0 < prefix).collect();
    let body: Vec<usize> = content[hdr.len()..].to_vec();
    if body.is_empty() {
        bail!("no document tokens after the {prefix}-character prefix");
    }
    let spans = v["spans"].as_array().context("no spans")?;
    let of = spans.len();
    // body index range per span
    let mut placed: Vec<(usize, usize, String, bool)> = Vec::with_capacity(of);
    for (j, sp) in spans.iter().enumerate() {
        let id = sp["id"].as_str().map(str::to_string).unwrap_or_else(|| sp["id"].to_string());
        let (c0, c1) = (sp["start"].as_u64().with_context(|| format!("span {j}: no start"))? as usize, sp["end"].as_u64().with_context(|| format!("span {j}: no end"))? as usize);
        if c0 >= c1 || c1 > nchar || c0 < prefix {
            bail!("span {id}: [{c0}, {c1}) is empty, past the text ({nchar} chars) or inside the prefix ({prefix})");
        }
        let b0 = body.partition_point(|&k| offs[k].0 < c0);
        let b1 = body.partition_point(|&k| offs[k].0 < c1);
        if b0 < b1 {
            placed.push((b0, b1, id, false));
        } else {
            // no token starts inside: the token containing c0 (the previous one), else the next one
            let b = if b0 > 0 && offs[body[b0 - 1]].1 > c0 { b0 - 1 } else { b0.min(body.len() - 1) };
            placed.push((b, b + 1, id, true));
        }
    }
    let mut out = LateDoc { windows: vec![], spans: vec![] };
    if ids.len() <= wmax {
        out.windows.push(ids.clone());
        for (b0, b1, id, widened) in placed {
            let (lo, hi) = (body[b0], body[b1 - 1] + 1);
            out.spans.push((0, lo, hi, SpanMeta { id, window: 0, windows: 1, tokens: hi - lo, clipped: false, widened, of }));
        }
        return Ok(out);
    }
    let head: Vec<i32> = lead.iter().chain(&hdr).map(|&k| ids[k]).collect();
    let tail: Vec<i32> = trail.iter().map(|&k| ids[k]).collect();
    if head.len() + tail.len() + 16 > wmax {
        bail!("prompt prefix ({} tokens) leaves no room in a {wmax}-token window", head.len());
    }
    let cap = wmax - head.len() - tail.len();
    let ov = overlap.min(cap / 2);
    let stride = cap - ov;
    let mut starts = vec![0usize];
    while starts.last().unwrap() + cap < body.len() {
        starts.push(starts.last().unwrap() + stride);
    }
    let nw = starts.len();
    for &s0 in &starts {
        let e0 = (s0 + cap).min(body.len());
        let mut w = head.clone();
        w.extend(body[s0..e0].iter().map(|&k| ids[k]));
        w.extend_from_slice(&tail);
        out.windows.push(w);
    }
    for (b0, b1, id, widened) in placed {
        // most central window that holds the whole span, else the one overlapping it most
        // ties go to the earliest window (max_by_key would take the last)
        let first_max = |score: &dyn Fn(usize) -> Option<usize>| -> Option<usize> {
            let mut best: Option<(usize, usize)> = None;
            for w in 0..nw {
                if let Some(v) = score(w) {
                    if best.is_none_or(|b| v > b.1) {
                        best = Some((w, v));
                    }
                }
            }
            best.map(|b| b.0)
        };
        let end = |s0: usize| (s0 + cap).min(body.len());
        let full = first_max(&|w| { let s0 = starts[w]; (s0 <= b0 && b1 <= end(s0)).then(|| (b0 - s0).min(end(s0) - b1)) });
        let (w, clipped) = match full {
            Some(w) => (w, false),
            None => (first_max(&|w| Some(b1.min(end(starts[w])).saturating_sub(b0.max(starts[w])))).unwrap(), true),
        };
        let s0 = starts[w];
        let e0 = (s0 + cap).min(body.len());
        let (x0, x1) = (b0.max(s0), b1.min(e0));
        let (lo, hi) = (head.len() + x0 - s0, head.len() + x1 - s0);
        out.spans.push((w, lo, hi, SpanMeta { id, window: w, windows: nw, tokens: hi - lo, clipped, widened, of }));
    }
    Ok(out)
}

struct Ctx {
    tok: tokenizers::Tokenizer,
    image_id: i64,
    video_id: i64,
    audio_id: i64,
    boa: i64,
    eoa: i64,
    boi: i64,
    eoi: i64,
    doc_id: i64,
    row_id: i64,
    specials: Vec<i64>, // Gemma: encode("") with specials (bos/eos around multimodal rows)
}

fn wrap(ctx: &Ctx, middle: &[i64]) -> Vec<i32> {
    let e = &ctx.specials;
    let mut row: Vec<i64> = e[..e.len().min(1)].to_vec();
    row.extend_from_slice(middle);
    row.extend_from_slice(&e[e.len().min(1)..]);
    row.into_iter().map(|x| x as i32).collect()
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Embed(a) => run(a),
        Cmd::Serve(a) => serve(a),
        Cmd::AttnBench { shapes, iters, fp32 } => {
            for sh in shapes.split(',') {
                let v: Vec<i64> = sh.split(':').map(|x| x.trim().parse()).collect::<Result<_, _>>()?;
                anyhow::ensure!(v.len() == 6, "shape {sh}: want D:Hq:Hkv:S:nseq:window");
                let (d, hq, hkv, s, n, w) = (v[0], v[1], v[2], v[3], v[4], v[5]);
                let ms = flashembed::gpu::attn_bench(fp32, d as i32, hq as i32, hkv as i32, s as i32, n as i32, w as i32, iters)?;
                // attended (query, key) pairs: full S^2, or the symmetric band |i-j| <= w
                let pairs: i64 = if w < 0 || w >= s { s * s } else { (0..s).map(|i| (i + w).min(s - 1) - (i - w).max(0) + 1).sum() };
                let tflops = 4.0 * (pairs * n * hq * d) as f64 / (ms as f64 * 1e-3) / 1e12;
                println!("{{\"shape\": \"{sh}\", \"fp32\": {fp32}, \"ms\": {ms:.4}, \"tflops\": {tflops:.1}, \"gpu\": \"{}\"}}", flashembed::gpu::device_name());
            }
            Ok(())
        }
    }
}

/// Tokenizer and special token ids for one model / modality (text truncates at max_len, media rows do not).
fn make_ctx(a: &Args) -> Result<Ctx> {
    let max_len = a.max_len.unwrap_or(match a.model { Model::Gemma => 8192, Model::Neomme => 16384 });
    let trunc = if a.modality == Modality::Text { Some(max_len) } else { None };
    let tok = prep::load_tokenizer(&a.snapshot.join("tokenizer.json"), trunc)?;
    let (image_id, doc_id, row_id, boi, eoi) = match a.model {
        Model::Gemma => (prep::token_id(&tok, "<|image|>")?, -1, -1, prep::token_id(&tok, "<|image>")?, prep::token_id(&tok, "<image|>")?),
        Model::Neomme => (prep::token_id(&tok, "<img>")?, prep::token_id(&tok, "<doc>")?, prep::token_id(&tok, "<row>")?, -1, -1),
    };
    let specials = prep::encode(&tok, &[String::new()], true)?.remove(0);
    let tid = |t: &str| if a.model == Model::Gemma { prep::token_id(&tok, t) } else { Ok(-1) };
    let (video_id, audio_id, boa, eoa) = (tid("<|video|>")?, tid("<|audio|>")?, tid("<|audio>")?, tid("<audio|>")?);
    Ok(Ctx { tok, image_id, video_id, audio_id, boa, eoa, boi, eoi, doc_id, row_id, specials })
}

/// Fills the per-modality defaults (token / patch budgets) and checks the options.
fn resolve(a: &mut Args) -> Result<()> {
    if let Some(d) = a.dim {
        if a.model != Model::Gemma || ![128, 256, 512, 768].contains(&d) {
            bail!("--dim is EmbeddingGemma 2's Matryoshka size: 768, 512, 256 or 128");
        }
    }
    if !matches!(a.side.as_str(), "document" | "query") {
        bail!("--side {}", a.side);
    }
    a.max_tokens = Some(a.max_tokens.unwrap_or(if gpu::smem_optin() >= 200 * 1024 { 32768 } else { 8192 }));
    a.max_patches = Some(match (a.modality, a.model) {
        (Modality::Text, _) => 0,
        (Modality::TextLate, Model::Gemma) => 0,
        (Modality::Image, Model::Gemma) => a.max_patches.unwrap_or(40320),
        (Modality::Image, Model::Neomme) => a.max_patches.unwrap_or(16384),
        // one full 32-frame video (32 x 1260 patches) per batch at least
        (Modality::Video, Model::Gemma) => a.max_patches.unwrap_or(40320),
        (Modality::Audio, Model::Gemma) => 0,
        (_, Model::Neomme) => bail!("NeoMME-260M-Retriever has no {:?} encoder (text and images only)", a.modality),
    });
    Ok(())
}

/// Items of one input file: (ids, text or path per item, directory paths are relative to).
fn read_items(a: &Args) -> Result<(Vec<String>, Vec<String>, PathBuf)> {
    let (ids, raw, dir, bad) = read_items_lenient(a)?;
    if let Some((i, e)) = bad.first() {
        bail!("input row {i}: {e}");
    }
    Ok((ids, raw, dir))
}

/// serve: like read_items, but a bad row is reported as ("line K", error) instead of failing the request.
fn read_items_lenient(a: &Args) -> Result<(Vec<String>, Vec<String>, PathBuf, Vec<(String, String)>)> {
    let dir = a.input_file().parent().map(Path::to_path_buf).unwrap_or_else(|| PathBuf::from("."));
    let key = if a.modality == Modality::Text { "text" } else { "path" };
    let (mut ids, mut raw, mut bad) = (vec![], vec![], vec![]);
    let text = std::fs::read_to_string(&a.input_file()).with_context(|| format!("reading {}", a.input_file().display()))?;
    for (k, line) in text.lines().enumerate().filter(|(_, l)| !l.trim().is_empty()).take(a.n.unwrap_or(usize::MAX)) {
        if a.modality == Modality::TextLate {
            // the whole row (text, prefix_chars, spans) goes to the producer, which validates it per document
            match serde_json::from_str::<serde_json::Value>(line) {
                Ok(v) if v["text"].is_string() && v["spans"].is_array() => {
                    ids.push(v["id"].as_str().map(str::to_string).unwrap_or_else(|| v["id"].to_string()));
                    raw.push(line.to_string());
                }
                Ok(v) => bad.push((v["id"].as_str().map(str::to_string).unwrap_or(format!("line {}", k + 1)), "text_late row needs \"text\" and \"spans\"".into())),
                Err(e) => bad.push((format!("line {}", k + 1), format!("bad JSON: {e}"))),
            }
            continue;
        }
        match serde_json::from_str::<serde_json::Value>(line) {
            Ok(v) => match v[key].as_str() {
                Some(x) => {
                    ids.push(v["id"].as_str().unwrap_or_default().to_string());
                    raw.push(x.to_string());
                }
                None => bad.push((v["id"].as_str().map(str::to_string).unwrap_or(format!("line {}", k + 1)), format!("no \"{key}\""))),
            },
            Err(e) => bad.push((format!("line {}", k + 1), format!("bad JSON: {e}"))),
        }
    }
    Ok((ids, raw, dir, bad))
}

fn peak_rss_mb() -> f64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| s.lines().find(|l| l.starts_with("VmHWM:")).and_then(|l| l.split_whitespace().nth(1)?.parse::<f64>().ok()))
        .map_or(0.0, |kb| kb / 1024.0)
}

/// Processing order: longest first (sentence-transformers sorts by text length; images by pixel count).
fn item_order(a: &Args, raw: &[String], dir: &Path) -> Vec<usize> {
    let mut order: Vec<usize> = (0..raw.len()).collect();
    if matches!(a.modality, Modality::Text | Modality::TextLate) {
        order.sort_by_key(|&i| std::cmp::Reverse(raw[i].chars().count()));
    } else {
        let px: Vec<usize> = raw
            .par_iter()
            .map(|p| match a.modality {
                Modality::Image => image::image_dimensions(dir.join(p)).map(|(w, h)| (w * h) as usize).unwrap_or(0),
                _ => std::fs::metadata(dir.join(p)).map(|m| m.len() as usize).unwrap_or(0),
            })
            .collect();
        order.sort_by_key(|&i| std::cmp::Reverse(px[i]));
    }
    order
}

fn load_engine(a: &Args, vision: bool, max_patches: usize, audio_frames: usize) -> Result<Engine> {
    let arch = match a.model { Model::Gemma => Arch::Gemma, Model::Neomme => Arch::Neomme };
    let dtype = match a.dtype { Dt::Bf16 => Dtype::Bf16, Dt::Fp32 => Dtype::F32 };
    let quant = match a.quant.as_str() { "none" => 0, "fp8" => 1, "nvfp4" => 2, q => bail!("--quant {q}: none|fp8|nvfp4") };
    Engine::load(&a.snapshot, arch, dtype, vision, max_tokens_of(a), max_patches, a.img_mb << 20, quant, audio_frames)
}

/// Results of one input file collected from the GPU loop.
struct Acc {
    dense: Vec<Vec<f32>>,
    late: Vec<Vec<f32>>,
    want_late: bool,
    ntok: usize,
    gpu_s: f64,
    nb: usize,
    /// serve: (item index, error) for items that were skipped
    failed: Vec<(usize, String)>,
    /// text_late: per document, per span: vector + metadata (filled as windows come back)
    spans: Vec<Vec<Option<(Vec<f32>, SpanMeta)>>>,
}

impl Acc {
    fn new(n: usize, want_late: bool) -> Acc {
        Acc { dense: vec![vec![]; n], late: vec![vec![]; if want_late { n } else { 0 }], want_late, ntok: 0, gpu_s: 0.0, nb: 0, failed: vec![], spans: vec![vec![]; n] }
    }
    fn items(b: &Batch) -> &[usize] {
        match b {
            Batch::Text { items, .. } | Batch::NeoImage { items, .. } | Batch::GemmaImage { items, .. } | Batch::Audio { items, .. } | Batch::Late { items, .. } => items,
        }
    }
    fn add(&mut self, b: &Batch, res: (Vec<f32>, Vec<f32>), dt: f64) {
        self.gpu_s += dt;
        self.nb += 1;
        if let Batch::Late { ids, refs, meta, .. } = b {
            self.ntok += ids.len();
            let od = res.0.len() / refs.len().max(1);
            for (k, (&(it, si), m)) in refs.iter().zip(meta).enumerate() {
                let sp = &mut self.spans[it];
                if sp.len() <= si {
                    sp.resize(si + 1, None);
                }
                sp[si] = Some((res.0[k * od..(k + 1) * od].to_vec(), m.clone()));
            }
            return;
        }
        let (items, ids, cu) = match b {
            Batch::Text { items, ids, cu } | Batch::NeoImage { items, ids, cu, .. } | Batch::GemmaImage { items, ids, cu, .. } | Batch::Audio { items, ids, cu, .. } => (items, ids, cu),
            Batch::Late { .. } => unreachable!(),
        };
        self.ntok += ids.len();
        let od = res.0.len() / items.len();
        for (j, &it) in items.iter().enumerate() {
            self.dense[it] = res.0[j * od..(j + 1) * od].to_vec();
        }
        if self.want_late {
            for (j, &it) in items.iter().enumerate() {
                self.late[it] = res.1[cu[j] as usize * 128..cu[j + 1] as usize * 128].to_vec();
            }
        }
    }
}

/// Writes <tag>.dense.npy / .ids.json (/ NeoMME late files) and <tag>.json; returns the stats row.
#[allow(clippy::too_many_arguments)]
fn write_outputs(a: &Args, eng: &Engine, mut acc: Acc, ids_all: &[String], wall: f64, load_s: f64, total_s: f64, mem_loaded: &gpu::MemInfo) -> Result<serde_json::Value> {
    // text_late: one row per span, documents in input order, spans in row order; a document is written when all of
    // its spans came back (failed documents are reported by the caller)
    let late_rows: Vec<String>;
    let mut span_info = vec![];
    let ids_all: &[String] = if a.modality == Modality::TextLate {
        let failed: std::collections::HashSet<usize> = acc.failed.iter().map(|f| f.0).collect();
        let (mut vecs, mut sids) = (vec![], vec![]);
        for (i, sp) in acc.spans.iter().enumerate() {
            if failed.contains(&i) || sp.is_empty() {
                continue;
            }
            let of = sp.iter().flatten().next().map_or(0, |x| x.1.of);
            if sp.len() != of || sp.iter().any(Option::is_none) {
                bail!("document {}: {} of {of} spans embedded", ids_all[i], sp.iter().flatten().count());
            }
            for (v, m) in sp.iter().flatten() {
                vecs.push(v.clone());
                sids.push(m.id.clone());
                span_info.push(json!({"id": m.id, "doc": ids_all[i], "window": m.window, "windows": m.windows,
                    "tokens": m.tokens, "clipped": m.clipped, "widened": m.widened}));
            }
        }
        acc.dense = vecs;
        acc.want_late = false;
        late_rows = sids;
        &late_rows
    } else {
        ids_all
    };
    // items without a vector (serve: failed items) are left out of every output file
    let keep: Vec<usize> = (0..ids_all.len()).filter(|&i| !acc.dense[i].is_empty()).collect();
    if keep.len() < ids_all.len() {
        acc.dense = keep.iter().map(|&i| std::mem::take(&mut acc.dense[i])).collect();
        if acc.want_late {
            acc.late = keep.iter().map(|&i| std::mem::take(&mut acc.late[i])).collect();
        }
    }
    let ids_out: Vec<String> = keep.iter().map(|&i| ids_all[i].clone()).collect();
    let ids_out = &ids_out;
    let n = ids_out.len();
    let mem = eng.mem()?;
    if let Some(d) = a.dim {
        for v in acc.dense.iter_mut() {
            v.truncate(d);
            let nn = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
            v.iter_mut().for_each(|x| *x /= nn);
        }
    }
    std::fs::create_dir_all(&a.out_dir()).with_context(|| format!("creating {}", a.out_dir().display()))?;
    let d = a.dim.unwrap_or(eng.out_dim);
    let flat: Vec<f32> = acc.dense.iter().flatten().copied().collect();
    if flat.len() != n * d {
        bail!("missing outputs ({} of {})", flat.len() / d.max(1), n);
    }
    ndarray_npy::write_npy(a.out_dir().join(format!("{}.dense.npy", a.tag)), &Array2::from_shape_vec((n, d), flat)?)?;
    if acc.want_late {
        let mut offs = vec![0i64];
        for v in &acc.late {
            offs.push(offs.last().unwrap() + (v.len() / 128) as i64);
        }
        let rows = *offs.last().unwrap() as usize;
        ndarray_npy::write_npy(a.out_dir().join(format!("{}.late.npy", a.tag)), &Array2::from_shape_vec((rows, 128), acc.late.concat())?)?;
        ndarray_npy::write_npy(a.out_dir().join(format!("{}.late_offsets.npy", a.tag)), &Array1::from_vec(offs))?;
    }
    std::fs::write(a.out_dir().join(format!("{}.ids.json", a.tag)), serde_json::to_string(ids_out)?)?;
    if a.modality == Modality::TextLate {
        std::fs::write(a.out_dir().join(format!("{}.spans.json", a.tag)), serde_json::to_string(&span_info)?)?;
    }
    let max_len = a.max_len.unwrap_or(match a.model { Model::Gemma => 8192, Model::Neomme => 16384 });
    let res = json!({
        "runtime": "flashembed-gpu", "model": format!("{:?}", a.model).to_lowercase(), "modality": format!("{:?}", a.modality).to_lowercase(),
        "dtype": format!("{:?}", a.dtype).to_lowercase(), "n": n, "batches": acc.nb, "max_tokens": max_tokens_of(a), "max_patches": a.max_patches.unwrap_or(0),
        "max_items": a.max_items, "max_len": max_len, "warmup": a.warmup, "wall_s": wall, "items_per_s": n as f64 / wall,
        "tokens": acc.ntok, "tokens_per_s": acc.ntok as f64 / wall, "gpu_call_s": acc.gpu_s, "load_s": load_s,
        "total_s": total_s,
        "gpu_weights_mb": mem.weights as f64 / 1048576.0, "gpu_arena_mb": mem.arena as f64 / 1048576.0,
        "gpu_device_used_mb": mem.device_used as f64 / 1048576.0, "gpu_device_used_after_load_mb": mem_loaded.device_used as f64 / 1048576.0,
        "device": gpu::device_name(), "host": std::fs::read_to_string("/proc/sys/kernel/hostname").unwrap_or_default().trim(),
        "job": std::env::var("SLURM_JOB_ID").ok(), "side": a.side, "no_late": a.no_late, "dim": a.dim, "quant": a.quant,
    });
    std::fs::write(a.out_dir().join(format!("{}.json", a.tag)), serde_json::to_string_pretty(&res)?)?;
    Ok(res)
}

fn run(a: Args) -> Result<()> {
    let t_start = Instant::now();
    if a.input.is_none() || a.out.is_none() {
        bail!("embed needs --input and --out");
    }
    let mut a = a;
    resolve(&mut a)?;
    let a = a;
    let query = a.side == "query";
    let (ids_out, raw, dir) = read_items(&a)?;
    let n = raw.len();
    let ctx = make_ctx(&a)?;
    if a.prep_only {
        return prep_only(&a, &ctx, &raw, &dir, query);
    }
    let t_load = Instant::now();
    let vision = a.model == Model::Gemma && matches!(a.modality, Modality::Image | Modality::Video);
    let audio_frames = if a.modality == Modality::Audio { a.max_frames } else { 0 };
    let mut eng = load_engine(&a, vision, a.max_patches.unwrap_or(0), audio_frames)?;
    let load_s = t_load.elapsed().as_secs_f64();
    let mem_loaded = eng.mem()?;
    let want_late = a.model == Model::Neomme && !a.no_late;
    let order = item_order(&a, &raw, &dir);

    let mut acc = Acc::new(n, want_late);
    let mut warm = a.warmup;
    let mut t0 = Instant::now();
    let (txb, rxb) = mpsc::sync_channel::<Result<Batch>>(2);
    std::thread::scope(|s| -> Result<()> {
        let (ctx, raw, order, dir, a) = (&ctx, &raw, &order, &dir, &a);
        s.spawn(move || {
            let send = |b: Batch| -> Result<()> { txb.send(Ok(b)).map_err(|_| anyhow::anyhow!("GPU loop stopped")) };
            if let Err(e) = produce(a, ctx, raw, order, dir, query, &send, &|i: usize, e: String| -> Result<()> { bail!("item {i}: {e}") }) {
                let _ = txb.send(Err(e));
            }
        });
        for b in rxb {
            let b = b?;
            loop {
                let tg = Instant::now();
                let res = execute(&mut eng, &b, want_late)?;
                let dt = tg.elapsed().as_secs_f64();
                if warm {
                    warm = false;
                    t0 = Instant::now();
                    continue; // same batch again, timed
                }
                acc.add(&b, res, dt);
                break;
            }
        }
        Ok(())
    })?;
    let wall = t0.elapsed().as_secs_f64();
    let res = write_outputs(&a, &eng, acc, &ids_out, wall, load_s, t_start.elapsed().as_secs_f64(), &mem_loaded)?;
    println!("{res}");
    Ok(())
}

/// Messages from the serve producer (requests in order) to the GPU loop.
enum Msg {
    /// a request whose items were read: per-request options, ids, start time
    Start { id: serde_json::Value, a: Box<Args>, ids: Vec<String>, t0: Instant, bad: Vec<(String, String)> },
    Batch(Batch),
    /// one item skipped by preprocessing
    ItemFail(usize, String),
    /// the request's producer failed (bad item, unreadable file, ...): its remaining batches are not coming
    Fail(String),
    End,
    /// the request could not be started at all (bad JSON, unknown modality, missing input)
    Reject { id: serde_json::Value, error: String, t0: Instant },
}

fn reply(v: serde_json::Value) {
    use std::io::Write;
    let mut o = std::io::stdout().lock();
    let _ = writeln!(o, "{v}");
    let _ = o.flush();
}

fn panic_msg(p: Box<dyn std::any::Any + Send>) -> String {
    p.downcast_ref::<String>().cloned().or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_else(|| "panic".into())
}

fn serve(sa: ServeArgs) -> Result<()> {
    let t_start = Instant::now();
    let base = sa.base;
    let mut mods = vec![];
    for m in sa.modalities.split(',').map(str::trim).filter(|m| !m.is_empty()) {
        mods.push(<Modality as ValueEnum>::from_str(m, true).map_err(|e| anyhow::anyhow!("--modalities {m}: {e}"))?);
    }
    if mods.is_empty() {
        bail!("--modalities: none given");
    }
    // per-modality options and tokenizers; one engine sized for all of them
    let mut per: Vec<(Modality, Args, Ctx)> = vec![];
    for &m in &mods {
        let mut a = base.clone();
        a.modality = m;
        a.warmup = false;
        resolve(&mut a)?;
        let ctx = make_ctx(&a)?;
        per.push((m, a, ctx));
    }
    let vision = base.model == Model::Gemma && mods.iter().any(|m| matches!(m, Modality::Image | Modality::Video));
    let max_patches = per.iter().map(|p| p.1.max_patches.unwrap_or(0)).max().unwrap_or(0);
    let audio_frames = if mods.contains(&Modality::Audio) { base.max_frames } else { 0 };
    let t_load = Instant::now();
    let mut eng = load_engine(&per[0].1, vision, max_patches, audio_frames)?;
    let load_s = t_load.elapsed().as_secs_f64();
    let mem_loaded = eng.mem()?;
    let want_late = base.model == Model::Neomme && !base.no_late;

    // warm-up: one tiny item per text / image modality through the full path (CUDA, cuBLAS, kernel loading)
    let t_warm = Instant::now();
    let wdir = std::env::temp_dir().join(format!("flashembed-gpu-warmup-{}", std::process::id()));
    std::fs::create_dir_all(&wdir)?;
    for (m, a, ctx) in &per {
        let row = match m {
            Modality::Text => json!({"id": "w", "text": "warm up"}),
            Modality::Image => {
                image::RgbImage::from_fn(96, 64, |x, y| image::Rgb([(x * 2) as u8, (y * 3) as u8, 128])).save(wdir.join("w.png"))?;
                json!({"id": "w", "path": "w.png"})
            }
            _ => continue,
        };
        let input = wdir.join("w.jsonl");
        std::fs::write(&input, format!("{row}\n"))?;
        let mut wa = a.clone();
        wa.input = Some(input);
        let (_, raw, dir) = read_items(&wa)?;
        let order = item_order(&wa, &raw, &dir);
        let bs = std::cell::RefCell::new(vec![]);
        produce(&wa, ctx, &raw, &order, &dir, wa.side == "query", &|b: Batch| {
            bs.borrow_mut().push(b);
            Ok(())
        }, &|i: usize, e: String| -> Result<()> { bail!("item {i}: {e}") })?;
        for b in bs.borrow().iter() {
            execute(&mut eng, b, want_late)?;
        }
    }
    let _ = std::fs::remove_dir_all(&wdir);
    reply(json!({"ready": true, "runtime": "flashembed-gpu", "model": format!("{:?}", base.model).to_lowercase(),
        "max_len": base.max_len.unwrap_or(match base.model { Model::Gemma => 8192, Model::Neomme => 16384 }), "side": base.side,
        "dim": base.dim, "no_late": base.no_late, "max_tokens": max_tokens_of(&per[0].1), "peak_rss_mb": peak_rss_mb(),
        "dtype": format!("{:?}", base.dtype).to_lowercase(), "modalities": mods.iter().map(|m| format!("{m:?}").to_lowercase()).collect::<Vec<_>>(),
        "load_s": load_s, "warmup_s": t_warm.elapsed().as_secs_f64(), "startup_s": t_start.elapsed().as_secs_f64(),
        "device": gpu::device_name(), "gpu_weights_mb": mem_loaded.weights as f64 / 1048576.0,
        "gpu_arena_mb": mem_loaded.arena as f64 / 1048576.0, "pid": std::process::id()}));

    let (tx, rx) = mpsc::sync_channel::<Msg>(2);
    std::thread::scope(|s| -> Result<()> {
        let per = &per;
        // producer: requests in order; the next request's batches are prepared while the GPU runs the current one
        s.spawn(move || {
            use std::io::BufRead;
            for line in std::io::stdin().lock().lines() {
                let Ok(line) = line else { break };
                if line.trim().is_empty() {
                    continue;
                }
                let req: serde_json::Value = match serde_json::from_str(&line) {
                    Ok(v) => v,
                    Err(e) => {
                        if tx.send(Msg::Reject { id: serde_json::Value::Null, error: format!("bad request JSON: {e}"), t0: Instant::now() }).is_err() {
                            return;
                        }
                        continue;
                    }
                };
                let id = req.get("id").cloned().unwrap_or(serde_json::Value::Null);
                let t0 = Instant::now();
                let start = (|| -> Result<(Args, &Ctx, Vec<String>, Vec<String>, PathBuf, Vec<(String, String)>)> {
                    let ms = req["modality"].as_str().context("request needs \"modality\"")?;
                    let m = <Modality as ValueEnum>::from_str(ms, true).map_err(|e| anyhow::anyhow!("modality {ms}: {e}"))?;
                    let (_, pa, ctx) = per.iter().find(|p| p.0 == m).with_context(|| format!("modality {ms} not served (--modalities)"))?;
                    let mut a = pa.clone();
                    a.input = Some(PathBuf::from(req["input"].as_str().context("request needs \"input\"")?));
                    a.out = Some(PathBuf::from(req["out"].as_str().context("request needs \"out\"")?));
                    let (ids, raw, dir, bad) = read_items_lenient(&a)?;
                    Ok((a, ctx, ids, raw, dir, bad))
                })();
                let (a, ctx, ids, raw, dir, bad) = match start {
                    Ok(x) => x,
                    Err(e) => {
                        if tx.send(Msg::Reject { id, error: format!("{e:#}"), t0 }).is_err() {
                            return;
                        }
                        continue;
                    }
                };
                if tx.send(Msg::Start { id, a: Box::new(a.clone()), ids, t0, bad }).is_err() {
                    return;
                }
                let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let order = item_order(&a, &raw, &dir);
                    let send = |b: Batch| -> Result<()> { tx.send(Msg::Batch(b)).map_err(|_| anyhow::anyhow!("GPU loop stopped")) };
                    let fail = |i: usize, e: String| -> Result<()> { tx.send(Msg::ItemFail(i, e)).map_err(|_| anyhow::anyhow!("GPU loop stopped")) };
                    produce(&a, ctx, &raw, &order, &dir, a.side == "query", &send, &fail)
                }));
                let msg = match r {
                    Ok(Ok(())) => Msg::End,
                    Ok(Err(e)) => Msg::Fail(format!("{e:#}")),
                    Err(p) => Msg::Fail(format!("preprocessing panicked: {}", panic_msg(p))),
                };
                if tx.send(msg).is_err() {
                    return;
                }
            }
        });
        // GPU loop
        type Cur = (serde_json::Value, Box<Args>, Vec<String>, Instant, Acc, Option<String>, Vec<(String, String)>);
        let mut cur: Option<Cur> = None;
        let fail_json = |ids: &[String], acc: &Acc, bad: &[(String, String)]| -> Vec<serde_json::Value> {
            bad.iter()
                .map(|(i, e)| json!({"id": i, "error": e}))
                .chain(acc.failed.iter().map(|(k, e)| json!({"id": ids[*k], "error": e})))
                .collect()
        };
        for msg in rx {
            match msg {
                Msg::Reject { id, error, t0 } => reply(json!({"id": id, "ok": false, "error": error, "seconds": t0.elapsed().as_secs_f64()})),
                Msg::Start { id, a, ids, t0, bad } => {
                    let n = ids.len();
                    cur = Some((id, a, ids, t0, Acc::new(n, want_late), None, bad));
                }
                Msg::ItemFail(k, e) => {
                    if let Some(c) = cur.as_mut() {
                        c.4.failed.push((k, e));
                    }
                }
                Msg::Batch(b) => {
                    let Some((_, _, _, _, acc, err, _)) = cur.as_mut() else { continue };
                    if err.is_some() {
                        continue;
                    }
                    let tg = Instant::now();
                    let r = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| execute(&mut eng, &b, want_late))) {
                        Ok(Ok(res)) => Ok(res),
                        Ok(Err(e)) => Err(format!("GPU: {e:#}")),
                        Err(p) => Err(format!("GPU loop panicked: {}", panic_msg(p))),
                    };
                    match r {
                        Ok(res) => acc.add(&b, res, tg.elapsed().as_secs_f64()),
                        // the batch's items fail (a CUDA error usually leaves the context unusable: later batches
                        // then fail the same way and are reported per item too)
                        Err(e) => {
                            for &k in Acc::items(&b) {
                                acc.failed.push((k, e.clone()));
                            }
                        }
                    }
                }
                Msg::Fail(_) | Msg::End if cur.is_none() => {}
                m @ (Msg::Fail(_) | Msg::End) => {
                    let (id, a, ids, t0, acc, err, bad) = cur.take().unwrap();
                    let err = match m {
                        Msg::Fail(e) => Some(err.unwrap_or(e)),
                        _ => err,
                    };
                    let failed = fail_json(&ids, &acc, &bad);
                    let nok = if a.modality == Modality::TextLate {
                        acc.spans.iter().flatten().flatten().count()
                    } else {
                        acc.dense.iter().filter(|v| !v.is_empty()).count()
                    };
                    let secs = t0.elapsed().as_secs_f64();
                    if let Some(e) = err {
                        reply(json!({"id": id, "ok": false, "error": e, "seconds": secs, "failed": failed}));
                        continue;
                    }
                    if nok == 0 {
                        reply(json!({"id": id, "ok": false, "error": "no items embedded", "seconds": secs, "failed": failed}));
                        continue;
                    }
                    match write_outputs(&a, &eng, acc, &ids, secs, load_s, secs, &mem_loaded) {
                        Ok(r) => reply(json!({"id": id, "ok": true, "items": nok, "failed": failed, "seconds": secs,
                            "items_per_s": nok as f64 / secs, "out": a.out_dir(), "peak_rss_mb": peak_rss_mb(),
                            "gpu_call_s": r["gpu_call_s"], "batches": r["batches"], "tokens": r["tokens"]})),
                        Err(e) => reply(json!({"id": id, "ok": false, "error": format!("writing outputs: {e:#}"), "seconds": secs, "failed": failed})),
                    }
                }
            }
        }
        Ok(())
    })
}

fn prep_only(a: &Args, ctx: &Ctx, raw: &[String], dir: &Path, query: bool) -> Result<()> {
    let n = raw.len();
    let order: Vec<usize> = (0..n).collect();
    let t0 = Instant::now();
    let (tx, rx) = mpsc::sync_channel::<Result<Batch>>(2);
    let (mut nb, mut ntok) = (0usize, 0usize);
    std::thread::scope(|s| -> Result<()> {
        s.spawn(move || {
            let send = |b: Batch| -> Result<()> { tx.send(Ok(b)).map_err(|_| anyhow::anyhow!("consumer stopped")) };
            if let Err(e) = produce(a, ctx, raw, &order, dir, query, &send, &|i: usize, e: String| -> Result<()> { bail!("item {i}: {e}") }) {
                let _ = tx.send(Err(e));
            }
        });
        for b in rx {
            match b? {
                Batch::Text { ids, .. } | Batch::NeoImage { ids, .. } | Batch::GemmaImage { ids, .. } | Batch::Audio { ids, .. } | Batch::Late { ids, .. } => ntok += ids.len(),
            }
            nb += 1;
        }
        Ok(())
    })?;
    let wall = t0.elapsed().as_secs_f64();
    use std::sync::atomic::Ordering::Relaxed;
    println!("{}", json!({"prep_only": true, "model": format!("{:?}", a.model), "modality": format!("{:?}", a.modality), "n": n,
        "batches": nb, "tokens": ntok, "wall_s": wall, "items_per_s": n as f64 / wall, "threads": rayon::current_num_threads(),
        "decode_cpu_s": DECODE_NS.load(Relaxed) as f64 * 1e-9, "preprocess_cpu_s": PREP_NS.load(Relaxed) as f64 * 1e-9}));
    Ok(())
}

fn execute(eng: &mut Engine, b: &Batch, late: bool) -> Result<(Vec<f32>, Vec<f32>)> {
    match b {
        Batch::Text { ids, cu, .. } => match eng.arch {
            Arch::Gemma => Ok((eng.gemma_text(ids, cu, None)?, vec![])),
            Arch::Neomme => eng.neomme(ids, cu, None, &[], &[], late),
        },
        Batch::GemmaImage { vcu, pos, patches, slot_ptr, slot_patch, ids, cu, mm, dev, src, desc, coef, max_pix, .. } => {
            if *dev {
                eng.gemma_vision_u8(vcu, pos, src, desc, coef, *max_pix, slot_ptr, slot_patch)?;
            } else {
                eng.gemma_vision(vcu, pos, patches, slot_ptr, slot_patch)?;
            }
            Ok((eng.gemma_text(ids, cu, Some(mm))?, vec![]))
        }
        Batch::NeoImage { ids, cu, pos, patches, rows, .. } => eng.neomme(ids, cu, Some(pos), patches, rows, late),
        Batch::Late { ids, cu, lo, hi, .. } => Ok((eng.gemma_text_spans(ids, cu, lo, hi)?, vec![])),
        Batch::Audio { cu0, feats, m0, rows1, m1, rows2, m2, soft_rows, ids, cu, mm, .. } => {
            eng.gemma_audio(cu0, feats, m0, rows1, m1, rows2, m2, soft_rows)?;
            Ok((eng.gemma_text(ids, cu, Some(mm))?, vec![]))
        }
    }
}

/// Preprocessing thread: builds token-/patch-budgeted batches in `order` and sends them to the GPU loop.
/// Item-level problems (undecodable file, too many tokens / patches) go to `fail(item, error)` and the item is skipped;
/// `embed` turns the first one into an error, `serve` reports them per request.
#[allow(clippy::too_many_arguments)]
fn produce(a: &Args, ctx: &Ctx, raw: &[String], order: &[usize], dir: &Path, query: bool, send: &dyn Fn(Batch) -> Result<()>, fail: &dyn Fn(usize, String) -> Result<()>) -> Result<()> {
    const CHUNK: usize = 256;
    match a.modality {
        Modality::Text => {
            let mut cur: (Vec<i32>, Vec<i32>, Vec<usize>) = (vec![], vec![0], vec![]);
            for chunk in order.chunks(CHUNK) {
                let rows: Vec<Result<Vec<i32>>> = chunk
                    .par_iter()
                    .map(|&i| {
                        let t = match a.model {
                            Model::Gemma => format!("{}{}", a.prefix, raw[i]),
                            Model::Neomme if query => format!("<query>{}{}{}", a.prefix, raw[i], "<mask>".repeat(10)),
                            Model::Neomme => format!("<doc>{}{}", a.prefix, raw[i]),
                        };
                        let e = ctx.tok.encode(t.as_str(), a.model == Model::Gemma).map_err(|e| anyhow::anyhow!("{e}"))?;
                        Ok(e.get_ids().iter().map(|&x| x as i32).collect())
                    })
                    .collect();
                for (&i, r) in chunk.iter().zip(rows) {
                    let r = match r {
                        Ok(r) => r,
                        Err(e) => {
                            fail(i, format!("tokenize: {e:#}"))?;
                            continue;
                        }
                    };
                    if r.len() > max_tokens_of(a) {
                        fail(i, format!("item of {} tokens exceeds --max-tokens {}", r.len(), max_tokens_of(a)))?;
                        continue;
                    }
                    if !cur.2.is_empty() && (cur.0.len() + r.len() > max_tokens_of(a) || cur.2.len() >= a.max_items) {
                        let b = std::mem::replace(&mut cur, (vec![], vec![0], vec![]));
                        send(Batch::Text { ids: b.0, cu: b.1, items: b.2 })?;
                    }
                    cur.0.extend_from_slice(&r);
                    cur.1.push(cur.0.len() as i32);
                    cur.2.push(i);
                }
            }
            if !cur.2.is_empty() {
                send(Batch::Text { ids: cur.0, cu: cur.1, items: cur.2 })?;
            }
        }
        Modality::TextLate => {
            if a.model != Model::Gemma {
                bail!("text_late is EmbeddingGemma 2 only");
            }
            let wmax = a.max_len.unwrap_or(8192).min(max_tokens_of(a));
            let empty = || Batch::Late { ids: vec![], cu: vec![0], lo: vec![], hi: vec![], refs: vec![], meta: vec![], items: vec![] };
            let mut cur = empty();
            for chunk in order.chunks(64) {
                let docs: Vec<Result<LateDoc>> = chunk.par_iter().map(|&i| late_doc(&ctx.tok, &raw[i], wmax, a.late_overlap)).collect();
                for (&i, d) in chunk.iter().zip(docs) {
                    let d = match d {
                        Ok(d) => d,
                        Err(e) => {
                            fail(i, format!("{e:#}"))?;
                            continue;
                        }
                    };
                    for (w, win) in d.windows.iter().enumerate() {
                        // windows without spans are not run
                        if !d.spans.iter().any(|sp| sp.0 == w) {
                            continue;
                        }
                        let Batch::Late { ids, cu, items, .. } = &cur else { unreachable!() };
                        if !items.is_empty() && (ids.len() + win.len() > max_tokens_of(a) || cu.len() > a.max_items) {
                            send(std::mem::replace(&mut cur, empty()))?;
                        }
                        let Batch::Late { ids, cu, lo, hi, refs, meta, items } = &mut cur else { unreachable!() };
                        let base = ids.len();
                        ids.extend_from_slice(win);
                        cu.push(ids.len() as i32);
                        if items.last() != Some(&i) {
                            items.push(i);
                        }
                        for (si, sp) in d.spans.iter().enumerate().filter(|(_, sp)| sp.0 == w) {
                            lo.push((base + sp.1) as i32);
                            hi.push((base + sp.2) as i32);
                            refs.push((i, si));
                            meta.push(sp.3.clone());
                        }
                    }
                }
            }
            if let Batch::Late { items, .. } = &cur {
                if !items.is_empty() {
                    send(cur)?;
                }
            }
        }
        Modality::Audio => {
            if a.model != Model::Gemma {
                bail!("audio is EmbeddingGemma 2 only");
            }
            let mel = media::MelExtractor::new();
            let new_audio = || Batch::Audio {
                cu0: vec![0], feats: vec![], m0: vec![], rows1: vec![], m1: vec![], rows2: vec![], m2: vec![], soft_rows: vec![],
                ids: vec![], cu: vec![0], mm: vec![], items: vec![],
            };
            let mut ab: Option<Batch> = None;
            for chunk in order.chunks(32) {
                // decode (first 30 s + 0.25 s, as the CPU runner) and log-mel per clip, in parallel
                let clips: Vec<Result<(Vec<f32>, Vec<bool>)>> = chunk
                    .par_iter()
                    .map(|&k| {
                        use std::sync::atomic::Ordering::Relaxed;
                        let t = Instant::now();
                        let wav = media::decode_audio(&dir.join(&raw[k]), Some(480_000))?;
                        DECODE_NS.fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
                        let t = Instant::now();
                        let (len, valid) = media::pad_batch(std::slice::from_ref(&wav));
                        let mut x = wav[..valid[0]].to_vec();
                        x.resize(len, 0.0);
                        let r = mel.features(&x, valid[0]);
                        PREP_NS.fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
                        Ok(r)
                    })
                    .collect();
                for (&item, clip) in chunk.iter().zip(clips) {
                    let (f, mk) = match clip {
                        Ok(x) => x,
                        Err(e) => {
                            fail(item, format!("{}: {e:#}", raw[item]))?;
                            continue;
                        }
                    };
                    let t0 = mk.len();
                    // the two stride-2 subsampling stages: t_out = (t_in + 2 - 3) / 2 + 1, mask[::2]
                    let sub = |m: &[bool]| -> Vec<bool> { m.iter().step_by(2).take((m.len() + 2 - 3) / 2 + 1).copied().collect() };
                    let mk1 = sub(&mk);
                    let mk2 = sub(&mk1);
                    let nsoft = mk2.iter().filter(|&&b| b).count();
                    let mut mid = vec![ctx.boa];
                    mid.extend(std::iter::repeat_n(ctx.audio_id, nsoft));
                    mid.push(ctx.eoa);
                    let row = wrap(ctx, &mid);
                    if t0 > a.max_frames || row.len() > max_tokens_of(a) {
                        fail(item, format!("{}: {t0} frames / {} tokens exceed --max-frames / --max-tokens", raw[item], row.len()))?;
                        continue;
                    }
                    let flush = matches!(&ab, Some(Batch::Audio { cu0, ids, items, .. })
                        if *cu0.last().unwrap() as usize + t0 > a.max_frames || ids.len() + row.len() > max_tokens_of(a) || items.len() >= a.max_items);
                    if flush {
                        send(ab.take().unwrap())?;
                    }
                    let Batch::Audio { cu0, feats, m0, rows1, m1, rows2, m2, soft_rows, ids, cu, mm, items } = ab.get_or_insert_with(new_audio) else {
                        unreachable!()
                    };
                    let (b0, b1, b2) = (*cu0.last().unwrap(), m1.len() as i32, m2.len() as i32);
                    feats.extend_from_slice(&f);
                    m0.extend(mk.iter().map(|&b| b as i32));
                    cu0.push(b0 + t0 as i32);
                    for (lt, &b) in mk1.iter().enumerate() {
                        rows1.extend([b0, t0 as i32, lt as i32]);
                        m1.push(b as i32);
                    }
                    for (lt, &b) in mk2.iter().enumerate() {
                        rows2.extend([b1, mk1.len() as i32, lt as i32, b2, mk2.len() as i32]);
                        m2.push(b as i32);
                        if b {
                            soft_rows.push(b2 + lt as i32);
                        }
                    }
                    let soft_base = (soft_rows.len() - nsoft) as i32;
                    let mut k = 0;
                    for &t in &row {
                        if t as i64 == ctx.audio_id {
                            mm.push(soft_base + k);
                            k += 1;
                        } else {
                            mm.push(-1);
                        }
                    }
                    ids.extend(row);
                    cu.push(ids.len() as i32);
                    items.push(item);
                }
            }
            if let Some(b) = ab {
                send(b)?;
            }
        }
        Modality::Image | Modality::Video => {
            // decode + preprocess chunks of images in parallel, then pack them into batches under the patch,
            // token and item budgets
            let mut gb: Option<Batch> = None;
            let new_gemma = |dev: bool| Batch::GemmaImage {
                vcu: vec![0], pos: vec![], patches: vec![], slot_ptr: vec![0], slot_patch: vec![], ids: vec![], cu: vec![0], mm: vec![],
                items: vec![], dev, src: vec![], desc: vec![], coef: vec![], tmp: 0, max_pix: 0,
            };
            let new_neo = || Batch::NeoImage { ids: vec![], cu: vec![0], pos: vec![], patches: vec![], rows: vec![], items: vec![] };
            let img_budget = a.img_mb << 20;
            let maxp = a.max_patches.unwrap_or(0);
            for chunk in order.chunks(64) {
                match a.model {
                    Model::Gemma => {
                        // GPU preprocessing: decode on the CPU, resize + patch on the GPU (items whose decoded frames
                        // exceed the byte budget fall back to CPU preprocessing). An image is one frame at 280 soft
                        // tokens; a video is <= 32 frames sampled at 1 fps at 140 soft tokens each
                        // (EmbeddingGemma2VideoProcessor defaults, as in the CPU runner).
                        let video = a.modality == Modality::Video;
                        let soft_budget = if video { 140 } else { prep::G_SOFT };
                        let prep_frames = |ims: Vec<image::RgbImage>| -> Result<Vec<GPrep>> {
                            let tmp_total: usize = ims
                                .iter()
                                .map(|im| gpu::ImagePlan::new(im.width() as usize, im.height() as usize, soft_budget * 9).map(|p| p.tmp_bytes).unwrap_or(usize::MAX))
                                .fold(0usize, |a, b| a.saturating_add(b));
                            // a whole item (all frames of a video) must fit one batch's GPU byte budget, else it is
                            // preprocessed on the CPU (e.g. 32 frames of 1080p video, ~200 MB decoded)
                            let host = img_budget == 0 || ims.iter().map(|im| im.as_raw().len()).sum::<usize>() > img_budget;
                            ims.into_iter()
                                .map(|im| {
                                    if !host && tmp_total <= img_budget {
                                        let plan = gpu::ImagePlan::new(im.width() as usize, im.height() as usize, soft_budget * 9)?;
                                        return Ok(GPrep::Dev(plan, im.into_raw()));
                                    }
                                    Ok(GPrep::Host(prep::gemma_image_budget(&im, soft_budget)?))
                                })
                                .collect()
                        };
                        let items_frames: Vec<Result<Vec<GPrep>>> = chunk
                            .par_iter()
                            .map(|&k| {
                                let path = dir.join(&raw[k]);
                                use std::sync::atomic::Ordering::Relaxed;
                                let t = Instant::now();
                                let ims = if video {
                                    let m = media::probe_video(&path)?;
                                    let idx = media::sample_indices(&m, 1.0, 32);
                                    media::decode_frames(&path, &m, &idx)?
                                } else {
                                    vec![prep::load_rgb(&path)?]
                                };
                                DECODE_NS.fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
                                let t = Instant::now();
                                let r = prep_frames(ims);
                                PREP_NS.fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
                                r
                            })
                            .collect();
                        let mm_tok = if video { ctx.video_id } else { ctx.image_id };
                        for (&item, frames) in chunk.iter().zip(items_frames) {
                            let frames = match frames {
                                Ok(f) if !f.is_empty() => f,
                                Ok(_) => {
                                    fail(item, format!("{}: no frames decoded", raw[item]))?;
                                    continue;
                                }
                                Err(e) => {
                                    fail(item, format!("{}: {e:#}", raw[item]))?;
                                    continue;
                                }
                            };
                            let dev = matches!(frames[0], GPrep::Dev(..));
                            let geo: Vec<(usize, usize)> = frames
                                .iter()
                                .map(|g| match g {
                                    GPrep::Host(h) => (h.n_soft * 9, 0),
                                    GPrep::Dev(p, _) => ((p.tw / 16) * (p.th / 16), p.tw / 16),
                                })
                                .collect();
                            let np_all: usize = geo.iter().map(|x| x.0).sum();
                            // EmbeddingGemma2Processor: boi + mm * n + eoi per frame, frames concatenated
                            let mut mid = vec![];
                            for &(np, _) in &geo {
                                mid.push(ctx.boi);
                                mid.extend(std::iter::repeat_n(mm_tok, np / 9));
                                mid.push(ctx.eoi);
                            }
                            let row = wrap(ctx, &mid);
                            if np_all > maxp || row.len() > max_tokens_of(a) {
                                fail(item, format!("{} needs {np_all} patches / {} tokens: raise --max-patches / --max-tokens", raw[item], row.len()))?;
                                continue;
                            }
                            // pooling slots must be in range before anything of this item is appended
                            let slots_ok = frames.iter().zip(&geo).all(|(g, &(np, pw_patch))| {
                                let xy: Vec<(usize, usize)> = match g {
                                    GPrep::Host(h) => (0..np).map(|p| (h.positions[2 * p] as usize, h.positions[2 * p + 1] as usize)).collect(),
                                    GPrep::Dev(..) => (0..np).map(|p| (p % pw_patch.max(1), p / pw_patch.max(1))).collect(),
                                };
                                let pw = xy.iter().map(|q| q.0).max().unwrap_or(0) + 1;
                                xy.iter().all(|&(x, y)| x / 3 + (pw / 3) * (y / 3) < np / 9)
                            });
                            if !slots_ok {
                                fail(item, format!("{}: pooling slot out of range", raw[item]))?;
                                continue;
                            }
                            let (sb, tb) = frames.iter().fold((0, 0), |acc, g| match g {
                                GPrep::Dev(p, src) => (acc.0 + src.len(), acc.1 + p.tmp_bytes),
                                _ => acc,
                            });
                            let flush = matches!(&gb, Some(Batch::GemmaImage { vcu, ids, items, dev: d, src, tmp, .. })
                                if *vcu.last().unwrap() as usize + np_all > maxp || ids.len() + row.len() > max_tokens_of(a)
                                    || items.len() >= a.max_items || *d != dev || src.len() + sb > img_budget || *tmp + tb > img_budget);
                            if flush {
                                send(gb.take().unwrap())?;
                            }
                            let Batch::GemmaImage { vcu, pos, patches, slot_ptr, slot_patch, ids, cu, mm, items, src, desc, coef, tmp, max_pix, .. } =
                                gb.get_or_insert_with(|| new_gemma(dev))
                            else {
                                unreachable!()
                            };
                            let soft_base = (slot_ptr.len() - 1) as i32;
                            for (g, &(np, pw_patch)) in frames.into_iter().zip(&geo) {
                                let n_soft = np / 9;
                                let p0 = *vcu.last().unwrap();
                                // patch p = r * pw + c sits at (x, y) = (c, r); pooling slot = (x / 3) + (pw / 3) * (y / 3)
                                // (HF _avg_pool_by_positions with k = 3)
                                let xy: Vec<(usize, usize)> = match &g {
                                    GPrep::Host(h) => (0..np).map(|p| (h.positions[2 * p] as usize, h.positions[2 * p + 1] as usize)).collect(),
                                    GPrep::Dev(..) => (0..np).map(|p| (p % pw_patch, p / pw_patch)).collect(),
                                };
                                let pw = xy.iter().map(|q| q.0).max().unwrap_or(0) + 1;
                                let mut slots: Vec<Vec<i32>> = vec![vec![]; n_soft];
                                for (p, &(x, y)) in xy.iter().enumerate() {
                                    let sl = x / 3 + (pw / 3) * (y / 3);
                                    if sl >= n_soft {
                                        bail!("pooling slot {sl} out of range {n_soft}");
                                    }
                                    slots[sl].push(p0 + p as i32);
                                    pos.push(x as i32);
                                    pos.push(y as i32);
                                }
                                for sl in slots {
                                    slot_patch.extend(sl);
                                    slot_ptr.push(slot_patch.len() as i32);
                                }
                                match g {
                                    GPrep::Host(h) => patches.extend_from_slice(&h.patches[..np * prep::G_PATCH_DIM]),
                                    GPrep::Dev(plan, bytes) => {
                                        plan.append(src.len(), *tmp, p0 as usize, desc, coef);
                                        src.extend_from_slice(&bytes);
                                        *tmp += plan.tmp_bytes;
                                        *max_pix = (*max_pix).max(plan.max_pix);
                                    }
                                }
                                vcu.push(p0 + np as i32);
                            }
                            let mut k = 0;
                            for &t in &row {
                                if t as i64 == mm_tok {
                                    mm.push(soft_base + k);
                                    k += 1;
                                } else {
                                    mm.push(-1);
                                }
                            }
                            ids.extend(row);
                            cu.push(ids.len() as i32);
                            items.push(item);
                        }
                    }
                    Model::Neomme => {
                        let ims: Vec<Result<(usize, usize, Vec<u8>)>> = chunk.par_iter().map(|&k| timed_image(&dir.join(&raw[k]), prep::neomme_image_u8)).collect();
                        for (&item, im) in chunk.iter().zip(ims) {
                            let (gh, gw, im_patches) = match im {
                                Ok(x) => x,
                                Err(e) => {
                                    fail(item, format!("{}: {e:#}", raw[item]))?;
                                    continue;
                                }
                            };
                            let (tids, tpos, flags) = prep::neomme_image_layout(gh, gw, ctx.doc_id, ctx.image_id, ctx.row_id);
                            let np = gh * gw;
                            if np > maxp || tids.len() > max_tokens_of(a) {
                                fail(item, format!("{}: image needs {np} patches / {} tokens: raise --max-patches / --max-tokens", raw[item], tids.len()))?;
                                continue;
                            }
                            let flush = matches!(&gb, Some(Batch::NeoImage { ids, rows, items, .. })
                                if rows.len() + np > maxp || ids.len() + tids.len() > max_tokens_of(a) || items.len() >= a.max_items);
                            if flush {
                                send(gb.take().unwrap())?;
                            }
                            let Batch::NeoImage { ids, cu, pos, patches, rows, items } = gb.get_or_insert_with(new_neo) else { unreachable!() };
                            let base = ids.len() as i32;
                            for (t, (&id, (p, &f))) in tids.iter().zip(tpos.iter().zip(&flags)).enumerate() {
                                ids.push(id as i32);
                                pos.push(p[0] as i32);
                                pos.push(p[1] as i32);
                                if f {
                                    rows.push(base + t as i32);
                                }
                            }
                            patches.extend_from_slice(&im_patches);
                            cu.push(ids.len() as i32);
                            items.push(item);
                        }
                    }
                }
            }
            if let Some(b) = gb {
                send(b)?;
            }
        }
    }
    Ok(())
}

fn max_tokens_of(a: &Args) -> usize {
    a.max_tokens.unwrap_or(32768)
}
