//! flashembed-gpu: GPU embedding runner (native CUDA engine, `cuda/fegpu.cu`) with the CPU runner's preprocessing.
//!
//!   flashembed-gpu embed --model gemma|neomme --modality text|image --input items.jsonl --out DIR --snapshot SNAP
//!                        [--dtype bf16|fp32] [--max-tokens 32768] [--max-items 512] [--max-len N] [--prefix STR]
//!                        [--side document|query] [--no-late] [--dim 768|512|256|128]
//!
//! Writes the same files as `flashembed embed` (embeddings.dense.npy, .ids.json, NeoMME .late.npy/.late_offsets.npy)
//! plus embeddings.json (throughput, GPU memory, settings). Items are length-sorted and packed into batches by a
//! token budget (no padding); a preprocessing thread prepares the next batch while the GPU runs the current one.

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
}

#[derive(Clone, Copy, ValueEnum, PartialEq, Eq, Debug)]
enum Model {
    Gemma,
    Neomme,
}
#[derive(Clone, Copy, ValueEnum, PartialEq, Eq, Debug)]
enum Modality {
    Text,
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
struct Args {
    #[arg(long, value_enum)]
    model: Model,
    #[arg(long, value_enum, default_value = "text")]
    modality: Modality,
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    out: PathBuf,
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
    let Cmd::Embed(a) = Cli::parse().cmd;
    run(a)
}

fn run(a: Args) -> Result<()> {
    let t_start = Instant::now();
    let arch = match a.model { Model::Gemma => Arch::Gemma, Model::Neomme => Arch::Neomme };
    let dtype = match a.dtype { Dt::Bf16 => Dtype::Bf16, Dt::Fp32 => Dtype::F32 };
    if let Some(d) = a.dim {
        if a.model != Model::Gemma || ![128, 256, 512, 768].contains(&d) {
            bail!("--dim is EmbeddingGemma 2's Matryoshka size: 768, 512, 256 or 128");
        }
    }
    let query = match a.side.as_str() { "document" => false, "query" => true, s => bail!("--side {s}") };
    // items
    let dir = a.input.parent().map(Path::to_path_buf).unwrap_or_else(|| PathBuf::from("."));
    let key = if a.modality == Modality::Text { "text" } else { "path" };
    let mut ids_out = vec![];
    let mut raw: Vec<String> = vec![];
    for line in std::fs::read_to_string(&a.input)?.lines().filter(|l| !l.trim().is_empty()).take(a.n.unwrap_or(usize::MAX)) {
        let v: serde_json::Value = serde_json::from_str(line)?;
        ids_out.push(v["id"].as_str().unwrap_or_default().to_string());
        raw.push(v[key].as_str().context("bad input row")?.to_string());
    }
    let n = raw.len();
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
    let ctx = Ctx { tok, image_id, video_id, audio_id, boa, eoa, boi, eoi, doc_id, row_id, specials };

    let mut a = a;
    let max_tokens = a.max_tokens.unwrap_or(if gpu::smem_optin() >= 200 * 1024 { 32768 } else { 8192 });
    a.max_tokens = Some(max_tokens);
    let max_patches = match (a.modality, a.model) {
        (Modality::Text, _) => 0,
        (Modality::Image, Model::Gemma) => a.max_patches.unwrap_or(40320),
        (Modality::Image, Model::Neomme) => a.max_patches.unwrap_or(16384),
        // one full 32-frame video (32 x 1260 patches) per batch at least
        (Modality::Video, Model::Gemma) => a.max_patches.unwrap_or(40320),
        (Modality::Audio, Model::Gemma) => 0,
        (_, Model::Neomme) => bail!("NeoMME-260M-Retriever has no {:?} encoder (text and images only)", a.modality),
    };
    a.max_patches = Some(max_patches);
    let a = a;
    if a.prep_only {
        return prep_only(&a, &ctx, &raw, &dir, query);
    }
    let t_load = Instant::now();
    let vision = a.model == Model::Gemma && matches!(a.modality, Modality::Image | Modality::Video);
    let quant = match a.quant.as_str() { "none" => 0, "fp8" => 1, "nvfp4" => 2, q => bail!("--quant {q}: none|fp8|nvfp4") };
    let audio_frames = if a.modality == Modality::Audio { a.max_frames } else { 0 };
    let mut eng = Engine::load(&a.snapshot, arch, dtype, vision, max_tokens, max_patches, a.img_mb << 20, quant, audio_frames)?;
    let load_s = t_load.elapsed().as_secs_f64();
    let mem_loaded = eng.mem()?;
    let out_dim = eng.out_dim;
    let want_late = a.model == Model::Neomme && !a.no_late;

    // order: longest first (sentence-transformers sorts by text length; images by pixel count)
    let mut order: Vec<usize> = (0..n).collect();
    if a.modality == Modality::Text {
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

    let mut dense = vec![vec![]; n];
    let mut late: Vec<Vec<f32>> = vec![vec![]; if want_late { n } else { 0 }];
    let mut ntok_total = 0usize;
    let mut gpu_s = 0.0;
    let mut nb = 0usize;
    let mut warm = a.warmup;
    let mut t0 = Instant::now();
    let (txb, rxb) = mpsc::sync_channel::<Result<Batch>>(2);
    std::thread::scope(|s| -> Result<()> {
        let (ctx, raw, order, dir, a) = (&ctx, &raw, &order, &dir, &a);
        s.spawn(move || {
            let r = produce(a, ctx, raw, order, dir, query, &txb);
            if let Err(e) = r {
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
                gpu_s += dt;
                nb += 1;
                let (items, ntok) = match &b {
                    Batch::Text { items, ids, .. } | Batch::NeoImage { items, ids, .. } | Batch::GemmaImage { items, ids, .. } | Batch::Audio { items, ids, .. } => (items, ids.len()),
                };
                ntok_total += ntok;
                let od = res.0.len() / items.len();
                for (j, &it) in items.iter().enumerate() {
                    dense[it] = res.0[j * od..(j + 1) * od].to_vec();
                }
                if want_late {
                    let cu = match &b { Batch::Text { cu, .. } | Batch::NeoImage { cu, .. } | Batch::GemmaImage { cu, .. } | Batch::Audio { cu, .. } => cu };
                    for (j, &it) in items.iter().enumerate() {
                        late[it] = res.1[cu[j] as usize * 128..cu[j + 1] as usize * 128].to_vec();
                    }
                }
                break;
            }
        }
        Ok(())
    })?;
    let wall = t0.elapsed().as_secs_f64();
    let mem = eng.mem()?;
    if let Some(d) = a.dim {
        for v in dense.iter_mut() {
            v.truncate(d);
            let nn = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
            v.iter_mut().for_each(|x| *x /= nn);
        }
    }
    std::fs::create_dir_all(&a.out)?;
    let d = a.dim.unwrap_or(out_dim);
    let flat: Vec<f32> = dense.iter().flatten().copied().collect();
    if flat.len() != n * d {
        bail!("missing outputs ({} of {})", flat.len() / d.max(1), n);
    }
    ndarray_npy::write_npy(a.out.join(format!("{}.dense.npy", a.tag)), &Array2::from_shape_vec((n, d), flat)?)?;
    if want_late {
        let mut offs = vec![0i64];
        for v in &late {
            offs.push(offs.last().unwrap() + (v.len() / 128) as i64);
        }
        let rows = *offs.last().unwrap() as usize;
        ndarray_npy::write_npy(a.out.join(format!("{}.late.npy", a.tag)), &Array2::from_shape_vec((rows, 128), late.concat())?)?;
        ndarray_npy::write_npy(a.out.join(format!("{}.late_offsets.npy", a.tag)), &Array1::from_vec(offs))?;
    }
    std::fs::write(a.out.join(format!("{}.ids.json", a.tag)), serde_json::to_string(&ids_out)?)?;
    let res = json!({
        "runtime": "flashembed-gpu", "model": format!("{:?}", a.model).to_lowercase(), "modality": format!("{:?}", a.modality).to_lowercase(),
        "dtype": format!("{:?}", a.dtype).to_lowercase(), "n": n, "batches": nb, "max_tokens": max_tokens, "max_patches": max_patches,
        "max_items": a.max_items, "max_len": max_len, "warmup": a.warmup, "wall_s": wall, "items_per_s": n as f64 / wall,
        "tokens": ntok_total, "tokens_per_s": ntok_total as f64 / wall, "gpu_call_s": gpu_s, "load_s": load_s,
        "total_s": t_start.elapsed().as_secs_f64(),
        "gpu_weights_mb": mem.weights as f64 / 1048576.0, "gpu_arena_mb": mem.arena as f64 / 1048576.0,
        "gpu_device_used_mb": mem.device_used as f64 / 1048576.0, "gpu_device_used_after_load_mb": mem_loaded.device_used as f64 / 1048576.0,
        "device": gpu::device_name(), "host": std::fs::read_to_string("/proc/sys/kernel/hostname").unwrap_or_default().trim(),
        "job": std::env::var("SLURM_JOB_ID").ok(), "side": a.side, "no_late": a.no_late, "dim": a.dim, "quant": a.quant,
    });
    std::fs::write(a.out.join(format!("{}.json", a.tag)), serde_json::to_string_pretty(&res)?)?;
    println!("{res}");
    Ok(())
}

fn prep_only(a: &Args, ctx: &Ctx, raw: &[String], dir: &Path, query: bool) -> Result<()> {
    let n = raw.len();
    let order: Vec<usize> = (0..n).collect();
    let t0 = Instant::now();
    let (tx, rx) = mpsc::sync_channel::<Result<Batch>>(2);
    let (mut nb, mut ntok) = (0usize, 0usize);
    std::thread::scope(|s| -> Result<()> {
        s.spawn(move || {
            if let Err(e) = produce(a, ctx, raw, &order, dir, query, &tx) {
                let _ = tx.send(Err(e));
            }
        });
        for b in rx {
            match b? {
                Batch::Text { ids, .. } | Batch::NeoImage { ids, .. } | Batch::GemmaImage { ids, .. } | Batch::Audio { ids, .. } => ntok += ids.len(),
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
        Batch::Audio { cu0, feats, m0, rows1, m1, rows2, m2, soft_rows, ids, cu, mm, .. } => {
            eng.gemma_audio(cu0, feats, m0, rows1, m1, rows2, m2, soft_rows)?;
            Ok((eng.gemma_text(ids, cu, Some(mm))?, vec![]))
        }
    }
}

/// Preprocessing thread: builds token-/patch-budgeted batches in `order` and sends them to the GPU loop.
fn produce(a: &Args, ctx: &Ctx, raw: &[String], order: &[usize], dir: &Path, query: bool, tx: &mpsc::SyncSender<Result<Batch>>) -> Result<()> {
    const CHUNK: usize = 256;
    match a.modality {
        Modality::Text => {
            let mut cur: (Vec<i32>, Vec<i32>, Vec<usize>) = (vec![], vec![0], vec![]);
            for chunk in order.chunks(CHUNK) {
                let rows: Vec<Vec<i32>> = chunk
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
                    .collect::<Result<_>>()?;
                for (&i, r) in chunk.iter().zip(rows) {
                    if !cur.2.is_empty() && (cur.0.len() + r.len() > max_tokens_of(a) || cur.2.len() >= a.max_items) {
                        let b = std::mem::replace(&mut cur, (vec![], vec![0], vec![]));
                        tx.send(Ok(Batch::Text { ids: b.0, cu: b.1, items: b.2 }))?;
                    }
                    if r.len() > max_tokens_of(a) {
                        bail!("item of {} tokens exceeds --max-tokens {}", r.len(), max_tokens_of(a));
                    }
                    cur.0.extend_from_slice(&r);
                    cur.1.push(cur.0.len() as i32);
                    cur.2.push(i);
                }
            }
            if !cur.2.is_empty() {
                tx.send(Ok(Batch::Text { ids: cur.0, cu: cur.1, items: cur.2 }))?;
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
                let clips: Vec<(Vec<f32>, Vec<bool>)> = chunk
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
                    .collect::<Result<_>>()?;
                for (&item, (f, mk)) in chunk.iter().zip(clips) {
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
                        bail!("{}: {t0} frames / {} tokens exceed --max-frames / --max-tokens", raw[item], row.len());
                    }
                    let flush = matches!(&ab, Some(Batch::Audio { cu0, ids, items, .. })
                        if *cu0.last().unwrap() as usize + t0 > a.max_frames || ids.len() + row.len() > max_tokens_of(a) || items.len() >= a.max_items);
                    if flush {
                        tx.send(Ok(ab.take().unwrap()))?;
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
                tx.send(Ok(b))?;
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
                        let items_frames: Vec<Vec<GPrep>> = chunk
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
                            .collect::<Result<_>>()?;
                        let mm_tok = if video { ctx.video_id } else { ctx.image_id };
                        for (&item, frames) in chunk.iter().zip(items_frames) {
                            if frames.is_empty() {
                                bail!("{}: no frames decoded", raw[item]);
                            }
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
                                bail!("{} needs {np_all} patches / {} tokens: raise --max-patches / --max-tokens", raw[item], row.len());
                            }
                            let (sb, tb) = frames.iter().fold((0, 0), |acc, g| match g {
                                GPrep::Dev(p, src) => (acc.0 + src.len(), acc.1 + p.tmp_bytes),
                                _ => acc,
                            });
                            let flush = matches!(&gb, Some(Batch::GemmaImage { vcu, ids, items, dev: d, src, tmp, .. })
                                if *vcu.last().unwrap() as usize + np_all > maxp || ids.len() + row.len() > max_tokens_of(a)
                                    || items.len() >= a.max_items || *d != dev || src.len() + sb > img_budget || *tmp + tb > img_budget);
                            if flush {
                                tx.send(Ok(gb.take().unwrap()))?;
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
                        let ims: Vec<(usize, usize, Vec<u8>)> = chunk.par_iter().map(|&k| timed_image(&dir.join(&raw[k]), prep::neomme_image_u8)).collect::<Result<_>>()?;
                        for (&item, (gh, gw, im_patches)) in chunk.iter().zip(ims) {
                            let (tids, tpos, flags) = prep::neomme_image_layout(gh, gw, ctx.doc_id, ctx.image_id, ctx.row_id);
                            let np = gh * gw;
                            if np > maxp || tids.len() > max_tokens_of(a) {
                                bail!("image needs {np} patches / {} tokens: raise --max-patches / --max-tokens", tids.len());
                            }
                            let flush = matches!(&gb, Some(Batch::NeoImage { ids, rows, items, .. })
                                if rows.len() + np > maxp || ids.len() + tids.len() > max_tokens_of(a) || items.len() >= a.max_items);
                            if flush {
                                tx.send(Ok(gb.take().unwrap()))?;
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
                tx.send(Ok(b))?;
            }
        }
    }
    Ok(())
}

fn max_tokens_of(a: &Args) -> usize {
    a.max_tokens.unwrap_or(32768)
}
