//! flashembed: CPU embedding runner for EmbeddingGemma 2 and NeoMME-260M-Retriever on ONNX Runtime.
//!
//! `flashembed bench --model gemma|neomme --modality text|image --threads T --batch B ...`
//! encodes the fixed benchmark inputs (see py/make_inputs.py) and writes, like py/py_bench.py:
//!   <out>/rs-<model>-<modality>-t<T>-b<B>.json        timings, peak RSS, CPU model
//!   ... .dense.npy / .late.npy / .late_offsets.npy   vectors (with --save)

use flashembed::{flash, media, prep};

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, anyhow, bail};
use clap::{Parser, Subcommand, ValueEnum};
use image::RgbImage;
use ndarray::{Array1, Array2};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;
use serde_json::json;

#[derive(Parser)]
#[command(version, about)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Embed every row of a jsonl file ({"id", "text"} or {"id", "path"}) and write the vectors as .npy.
    Embed(EmbedArgs),
    /// Encode the benchmark inputs and record throughput / peak RSS / vectors.
    Bench(BenchArgs),
    /// Persistent server: load once, then embed one jsonl file per stdin request line (JSON in, JSON out).
    Serve(ServeArgs),
}

#[derive(clap::Args)]
struct EmbedArgs {
    #[arg(long, value_enum)]
    model: ModelKind,
    #[arg(long, value_enum, default_value = "text")]
    modality: Modality,
    /// jsonl with one item per line: {"id", "text"} for text, {"id", "path"} for image/audio/video
    /// (paths relative to the jsonl's directory)
    #[arg(long)]
    input: PathBuf,
    /// output dir: embeddings.dense.npy [N, dim], embeddings.ids.json; NeoMME also embeddings.late.npy (all
    /// items' late vectors, concatenated) + embeddings.late_offsets.npy; embeddings.json = timing / settings
    #[arg(long)]
    out: PathBuf,
    /// directory with the exported ONNX graphs (py/export_onnx.py)
    #[arg(long)]
    onnx: PathBuf,
    /// local Hugging Face snapshot of the model (tokenizer and configs)
    #[arg(long)]
    snapshot: PathBuf,
    /// CPUs to use (default: all available to this process); the pool shape is autotuned at startup
    #[arg(long)]
    threads: Option<usize>,
    #[arg(long, default_value_t = 8)]
    batch: usize,
    /// token limit (default: the model's maximum, 8192 EmbeddingGemma 2 / 16384 NeoMME)
    #[arg(long)]
    max_len: Option<usize>,
    /// NeoMME: document | query
    #[arg(long, default_value = "document")]
    side: String,
    /// EmbeddingGemma 2 image budget (70, 140, 280, 560, 1120 soft tokens)
    #[arg(long, default_value_t = 280)]
    max_soft_tokens: usize,
    /// EmbeddingGemma 2 Matryoshka (MRL) output size: 768 (native), 512, 256 or 128; keeps the leading dims
    /// and re-normalises, like sentence-transformers' `truncate_dim` + `normalize_embeddings`
    #[arg(long)]
    dim: Option<usize>,
    /// NeoMME: dense vectors only (no embeddings.late*.npy; late vectors are not kept in memory either)
    #[arg(long)]
    no_late: bool,
}

impl EmbedArgs {
    fn into_bench(self) -> Result<BenchArgs> {
        let dir = self.input.parent().map(|p| p.to_path_buf()).unwrap_or_else(|| PathBuf::from("."));
        let file = self.input.file_name().context("--input must be a file")?.to_string_lossy().to_string();
        let threads = self.threads.unwrap_or_else(|| cpu_topology().len().max(1));
        let max_len = self.max_len.unwrap_or(match self.model { ModelKind::Gemma => 8192, ModelKind::Neomme => 16384 });
        std::fs::create_dir_all(&self.out)?;
        Ok(BenchArgs {
            model: self.model, modality: self.modality, threads, batch: self.batch, n: usize::MAX, warmup: 0, max_len,
            inputs: dir, onnx: self.onnx, snapshot: self.snapshot, out: self.out, save: true, texts_file: file.clone(),
            images_file: file, side: self.side, tag_suffix: String::new(), max_soft_tokens: self.max_soft_tokens,
            decode_in_loop: true, no_sort: false, plain_names: true, dim: self.dim, no_late: self.no_late,
        })
    }
}

#[derive(Clone, Copy, ValueEnum, PartialEq, Eq, Debug)]
enum ModelKind {
    Gemma,
    Neomme,
}

#[derive(Clone, Copy, ValueEnum, PartialEq, Eq, Debug)]
enum Modality {
    Text,
    Image,
    /// EmbeddingGemma 2 only (ffmpeg decode, 16 kHz mono, log-mel, conformer)
    Audio,
    /// EmbeddingGemma 2 only (ffmpeg decode, 1 fps, <= 32 frames, 140 soft tokens per frame)
    Video,
}

#[derive(clap::Args)]
struct BenchArgs {
    #[arg(long, value_enum)]
    model: ModelKind,
    #[arg(long, value_enum)]
    modality: Modality,
    #[arg(long)]
    threads: usize,
    #[arg(long)]
    batch: usize,
    #[arg(long, default_value_t = 32)]
    n: usize,
    #[arg(long, default_value_t = 1)]
    warmup: usize,
    #[arg(long, default_value_t = 512)]
    max_len: usize,
    /// Directory with texts.jsonl / images.jsonl.
    #[arg(long)]
    inputs: PathBuf,
    /// Directory with the exported *.onnx graphs.
    #[arg(long)]
    onnx: PathBuf,
    /// HF snapshot directory (tokenizer.json).
    #[arg(long)]
    snapshot: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[arg(long)]
    save: bool,
    /// jsonl under --inputs with {id, text}
    #[arg(long, default_value = "texts.jsonl")]
    texts_file: String,
    /// jsonl under --inputs with {id, path}
    #[arg(long, default_value = "images.jsonl")]
    images_file: String,
    /// NeoMME template side: document (`<doc>` + text) or query (`<query>` + text + 10 x `<mask>`)
    #[arg(long, default_value = "document")]
    side: String,
    /// appended to the output tag (no '-'), e.g. _L2048
    #[arg(long, default_value = "")]
    tag_suffix: String,
    /// Gemma image soft-token budget (70/140/280/560/1120); 280 = processor default
    #[arg(long, default_value_t = 280)]
    max_soft_tokens: usize,
    /// decode image files inside the timed loop (formats test); default pre-decodes like py_bench
    #[arg(long)]
    decode_in_loop: bool,
    /// keep input order inside batches (default: longest-first, like sentence-transformers)
    #[arg(long)]
    no_sort: bool,
    /// `embed`: write embeddings.* instead of benchmark-tagged file names
    #[arg(skip)]
    plain_names: bool,
    /// EmbeddingGemma 2 Matryoshka output size (768/512/256/128): truncate dense vectors, then re-normalise
    #[arg(long)]
    dim: Option<usize>,
    /// NeoMME: dense vectors only (skip collecting and writing the late-interaction vectors)
    #[arg(long)]
    no_late: bool,
}

fn oe<T, E: std::fmt::Display>(r: std::result::Result<T, E>) -> Result<T> {
    r.map_err(|e| anyhow!("{e}"))
}

/// CPUs this process may run on: elsewhere than Linux, all logical CPUs on one "socket".
#[cfg(not(target_os = "linux"))]
fn cpu_topology() -> Vec<(usize, String)> {
    let n = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    (0..n).map(|c| (c, "0".to_string())).collect()
}

/// CPUs this process may run on (cgroup/cpuset) with their socket ids.
#[cfg(target_os = "linux")]
fn cpu_topology() -> Vec<(usize, String)> {
    let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    let ok = unsafe { libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set) } == 0;
    if !ok {
        return vec![];
    }
    (0..libc::CPU_SETSIZE as usize)
        .filter(|&c| unsafe { libc::CPU_ISSET(c, &set) })
        .map(|c| {
            let s = std::fs::read_to_string(format!("/sys/devices/system/cpu/cpu{c}/topology/physical_package_id"))
                .map(|x| x.trim().to_string())
                .unwrap_or_default();
            (c, s)
        })
        .collect()
}

/// A thread-pool shape for ORT's intra-op pool.
#[derive(Clone, Debug)]
struct Policy {
    name: String,
    threads: usize,
    cfg: Vec<(String, String)>,
}

/// Candidate policies (all with dynamic block scheduling):
///   spin          T threads, spin-waiting (ORT default)
///   pinned        T threads pinned to the job's CPUs, no spinning
///   socket        threads pinned to the job's CPUs on its largest socket only (fewer threads, no cross-socket sync)
///   socket-nospin same, without spinning
/// FLASHEMBED_ORT_POLICY picks one; "auto" (default) = spin on 1 socket; on several sockets, the runner autotunes.
fn policies(threads: usize) -> Vec<Policy> {
    let cpus = cpu_topology();
    let mut by: std::collections::BTreeMap<String, Vec<usize>> = Default::default();
    for (c, s) in &cpus {
        by.entry(s.clone()).or_default().push(*c);
    }
    let base = || vec![("session.intra_op.dynamic_block_base".to_string(), "4".to_string())];
    let aff = |list: &[usize]| list[1..].iter().map(|c| (c + 1).to_string()).collect::<Vec<_>>().join(";");
    let mut out = vec![Policy { name: "spin".into(), threads, cfg: base() }];
    if cpus.len() >= threads && threads > 1 {
        let mut c = base();
        c.push(("session.intra_op.allow_spinning".into(), "0".into()));
        c.push(("session.intra_op_thread_affinities".into(), aff(&cpus.iter().map(|x| x.0).take(threads).collect::<Vec<_>>())));
        out.push(Policy { name: "pinned".into(), threads, cfg: c });
    }
    if threads >= 8 && cpus.len() >= threads {
        // half the threads, biggest socket's CPUs first: on busy shared nodes short inputs ran up to 1.5x faster on
        // 16 of 32 pinned CPUs than on all 32 (each of the ~1,000 ops per forward is a pool-wide sync)
        let mut socks: Vec<&Vec<usize>> = by.values().collect();
        socks.sort_by_key(|v| std::cmp::Reverse(v.len()));
        let order: Vec<usize> = socks.into_iter().flatten().copied().collect();
        let t = threads / 2;
        let mut c = base();
        c.push(("session.intra_op_thread_affinities".into(), aff(&order[..t])));
        out.push(Policy { name: format!("half{t}-spin"), threads: t, cfg: c.clone() });
        c.push(("session.intra_op.allow_spinning".into(), "0".into()));
        out.push(Policy { name: format!("half{t}"), threads: t, cfg: c });
    }
    if by.len() > 1 {
        let best = by.values().max_by_key(|v| v.len()).unwrap().clone();
        let t = best.len().min(threads);
        if t > 1 {
            let mut c = base();
            c.push(("session.intra_op_thread_affinities".into(), aff(&best[..t])));
            out.push(Policy { name: format!("socket{t}"), threads: t, cfg: c.clone() });
            c.push(("session.intra_op.allow_spinning".into(), "0".into()));
            out.push(Policy { name: format!("socket{t}-nospin"), threads: t, cfg: c });
        }
    }
    out
}

fn vision_full() -> bool {
    std::env::var("FLASHEMBED_VISION_FULL").is_ok_and(|v| v == "1")
}

/// sockets spanned by this process's CPU set
#[allow(dead_code)]
fn n_sockets() -> usize {
    cpu_topology().iter().map(|x| x.1.clone()).collect::<std::collections::BTreeSet<_>>().len().max(1)
}

/// The ONNX Runtime environment can be created only once per process (autotune builds several runners).
fn ort_env() -> Result<ort::environment::Environment> {
    static ENV: std::sync::OnceLock<ort::environment::Environment> = std::sync::OnceLock::new();
    if let Some(e) = ENV.get() {
        return Ok(e.clone());
    }
    let dylib = std::env::var("ORT_DYLIB_PATH")?;
    let e = oe(oe(ort::init_from(&dylib))?.build())?;
    Ok(ENV.get_or_init(|| e).clone())
}

fn session(env: &ort::environment::Environment, path: &Path, pol: &Policy) -> Result<Session> {
    let b = oe(Session::builder(env))?;
    let b = oe(b.with_optimization_level(GraphOptimizationLevel::Level3))?;
    let b = oe(b.with_intra_threads(pol.threads))?;
    let mut b = oe(b.with_inter_threads(1))?;
    // Rust fused attention (flashembed::FlashAttention) used by the --attn flash exports
    b = oe(b.with_operators(oe(flash::domain())?))?;
    // Defaults chosen from measurements (docs/RUST_EMBED.md): dynamic block scheduling in ORT's thread pool keeps
    // many-small-op graphs fast at high thread counts. FLASHEMBED_ORT_CONFIG="k=v,..." adds/overrides entries.
    let mut cfg = pol.cfg.clone();
    if let Ok(extra) = std::env::var("FLASHEMBED_ORT_CONFIG") {
        for kv in extra.split(',').filter(|x| !x.is_empty()) {
            let (k, v) = kv.split_once('=').ok_or_else(|| anyhow!("FLASHEMBED_ORT_CONFIG: bad entry {kv}"))?;
            cfg.retain(|(k0, _)| k0 != k);
            cfg.push((k.into(), v.into()));
        }
    }
    for (k, v) in &cfg {
        b = oe(b.with_config_entry(k, v))?;
    }
    if let Ok(prefix) = std::env::var("FLASHEMBED_ORT_PROFILE") {
        // per-op ORT profile (JSON) for the real Rust runs, custom attention op included
        b = oe(b.with_profiling(format!("{prefix}-{}", path.file_stem().unwrap_or_default().to_string_lossy())))?;
    }
    if !std::env::var("FLASHEMBED_ORT_ARENA").is_ok_and(|v| v == "1") {
        // default: no CPU arena / memory pattern -> intermediates are freed as soon as they die (same speed in our
        // measurements, 25-45% lower peak RSS on batched long inputs). FLASHEMBED_ORT_ARENA=1 restores ORT's default.
        b = oe(b.with_execution_providers([ort::ep::CPU::default().with_arena_allocator(false).build()]))?;
        b = oe(b.with_memory_pattern(false))?;
    }
    oe(b.commit_from_file(path)).with_context(|| format!("loading {}", path.display()))
}

fn peak_rss_mb() -> f64 {
    let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
    unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut ru) };
    ru.ru_maxrss as f64 / 1024.0 // KiB on Linux
}

fn cpu_model() -> String {
    std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|s| s.lines().find(|l| l.starts_with("model name")).map(|l| l.split(':').nth(1).unwrap_or("").trim().to_string()))
        .unwrap_or_default()
}

fn l2_normalize(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    v.iter_mut().for_each(|x| *x /= n);
}

/// Output of one batch: dense [B, D]; late = per-item valid rows (NeoMME only); token counts.
struct BatchOut {
    dense: Vec<Vec<f32>>,
    late: Vec<Vec<f32>>,
    late_rows: Vec<usize>,
    ntok: usize,
}

enum Item {
    Text(String),
    Image(RgbImage),
    ImagePath(PathBuf),
    Audio(PathBuf),
    Video(PathBuf),
}

struct Runner {
    _env: ort::environment::Environment,
    kind: ModelKind,
    modality: Modality,
    tok: tokenizers::Tokenizer,
    /// Gemma: gemma_lm; NeoMME: neomme_text (None when serving NeoMME images only)
    main: Option<Session>,
    /// NeoMME image graph
    nimg: Option<Session>,
    /// Gemma vision tower used for images (fixed or dynamic patch graph, see `new`)
    vision: Option<Session>,
    /// dynamic-patch vision graph for video frames, when `vision` is the fixed-shape one (else video uses `vision`)
    vision_vid: Option<Session>,
    audio: Option<Session>,
    mel: media::MelExtractor,
    soft_budget: usize,
    video_id: i64,
    audio_id: i64,
    // special ids
    pad_id: i64,
    image_id: i64,
    doc_id: i64,
    row_id: i64,
    query: bool,
    no_late: bool,
    /// seconds inside Session::run (model forward), accumulated; the rest of a batch is Rust prep/post
    infer_s: f64,
}

impl Runner {
    /// Load the graphs for `mods` (one runner can serve several modalities; set `modality` before `run`).
    fn new(a: &BenchArgs, mods: &[Modality], pol: &Policy) -> Result<Self> {
        let env = ort_env()?;
        let tok_path = a.snapshot.join("tokenizer.json");
        // truncation only affects text: multimodal rows are built from ids (wrap_specials encodes "")
        let trunc = if mods.contains(&Modality::Text) { Some(a.max_len) } else { None };
        let tok = prep::load_tokenizer(&tok_path, trunc)?;
        let has = |m: Modality| mods.contains(&m);
        let sess = |name: &str| session(&env, &a.onnx.join(name), pol);
        let (mut main, mut nimg, mut vision, mut vision_vid, mut audio) = (None, None, None, None, None);
        match a.model {
            ModelKind::Gemma => {
                main = Some(sess("gemma_lm.onnx")?);
                // fixed-shape graph for the 280-token default with FLASHEMBED_VISION_FULL=1; dynamic-patch graph for
                // other budgets, patch trimming (default) and video frames
                let img_dyn = a.max_soft_tokens != prep::G_SOFT || !vision_full();
                if has(Modality::Image) {
                    vision = Some(sess(if img_dyn { "gemma_vision_dyn.onnx" } else { "gemma_vision.onnx" })?);
                }
                if has(Modality::Video) {
                    if vision.is_none() {
                        vision = Some(sess("gemma_vision_dyn.onnx")?);
                    } else if !img_dyn {
                        vision_vid = Some(sess("gemma_vision_dyn.onnx")?);
                    }
                }
                if has(Modality::Audio) {
                    audio = Some(sess("gemma_audio.onnx")?);
                }
            }
            ModelKind::Neomme => {
                if let Some(m) = mods.iter().find(|m| !matches!(m, Modality::Text | Modality::Image)) {
                    bail!("NeoMME-260M-Retriever has no {m:?} encoder (text and images only)");
                }
                if has(Modality::Text) {
                    main = Some(sess("neomme_text.onnx")?);
                }
                if has(Modality::Image) {
                    nimg = Some(sess("neomme_image.onnx")?);
                }
            }
        }
        let (pad_id, image_id, doc_id, row_id) = match a.model {
            ModelKind::Gemma => (prep::token_id(&tok, "<pad>")?, prep::token_id(&tok, "<|image|>")?, -1, -1),
            ModelKind::Neomme => (
                prep::token_id(&tok, "<pad>")?,
                prep::token_id(&tok, "<img>")?,
                prep::token_id(&tok, "<doc>")?,
                prep::token_id(&tok, "<row>")?,
            ),
        };
        let query = match a.side.as_str() {
            "document" => false,
            "query" => true,
            s => bail!("--side {s}: expected document|query"),
        };
        let (video_id, audio_id) = match a.model {
            ModelKind::Gemma => (prep::token_id(&tok, "<|video|>")?, prep::token_id(&tok, "<|audio|>")?),
            ModelKind::Neomme => (-1, -1),
        };
        Ok(Self {
            _env: env, kind: a.model, modality: mods[0], tok, main, nimg, vision, vision_vid, audio,
            mel: media::MelExtractor::new(), soft_budget: a.max_soft_tokens, video_id, audio_id,
            pad_id, image_id, doc_id, row_id, query, no_late: a.no_late, infer_s: 0.0,
        })
    }

    fn run(&mut self, items: &[&Item]) -> Result<BatchOut> {
        match (self.kind, self.modality) {
            (ModelKind::Gemma, Modality::Text) => {
                let texts: Vec<String> = items.iter().map(|i| match i { Item::Text(t) => t.clone(), _ => unreachable!() }).collect();
                let rows = prep::encode(&self.tok, &texts, true)?;
                self.gemma_lm(&rows, None)
            }
            (ModelKind::Gemma, Modality::Image) => self.gemma_image(items),
            (ModelKind::Gemma, Modality::Video) => self.gemma_video(items),
            (ModelKind::Gemma, Modality::Audio) => self.gemma_audio(items),
            (ModelKind::Neomme, Modality::Text) => {
                // NeoMME chat_template.jinja: document -> "<doc>" + text; query -> "<query>" + text + "<mask>" * 10
                let q = self.query;
                let texts: Vec<String> = items
                    .iter()
                    .map(|i| match i {
                        Item::Text(t) if q => format!("<query>{t}{}", "<mask>".repeat(10)),
                        Item::Text(t) => format!("<doc>{t}"),
                        _ => unreachable!(),
                    })
                    .collect();
                let rows = prep::encode(&self.tok, &texts, false)?;
                let (ids, mask, s) = prep::pad(&rows, self.pad_id);
                let b = rows.len();
                let t = Instant::now();
                let outs = oe(self.main.as_mut().unwrap().run(ort::inputs![
                    "input_ids" => oe(Tensor::from_array(([b, s], ids)))?,
                    "attention_mask" => oe(Tensor::from_array(([b, s], mask.clone())))?,
                ]))?;
                self.infer_s += t.elapsed().as_secs_f64();
                neomme_outputs(&outs, &mask, b, s, self.no_late)
            }
            (ModelKind::Neomme, Modality::Image) => self.neomme_image(items),
            (ModelKind::Neomme, _) => unreachable!(),
        }
    }

    /// Run gemma_lm; `mm` = per-row soft tokens (n x 512) to place at <|image|> positions.
    fn gemma_lm(&mut self, rows: &[Vec<i64>], mm: Option<&[Vec<f32>]>) -> Result<BatchOut> {
        const H: usize = 512;
        let (ids, mask, s) = prep::pad(rows, self.pad_id);
        let b = rows.len();
        let mut mm_embeds = vec![0f32; b * s * H];
        let mut mm_mask = vec![false; b * s];
        if let Some(mm) = mm {
            for (bi, soft) in mm.iter().enumerate() {
                let mut k = 0;
                for t in 0..s {
                    let id = ids[bi * s + t];
                    if id == self.image_id || id == self.video_id || id == self.audio_id {
                        mm_mask[bi * s + t] = true;
                        mm_embeds[(bi * s + t) * H..(bi * s + t + 1) * H].copy_from_slice(&soft[k * H..(k + 1) * H]);
                        k += 1;
                    }
                }
                if k * H != soft.len() {
                    bail!("placeholder tokens {k} != soft tokens {}", soft.len() / H);
                }
            }
        }
        let t = Instant::now();
        let outs = oe(self.main.as_mut().unwrap().run(ort::inputs![
            "input_ids" => oe(Tensor::from_array(([b, s], ids)))?,
            "attention_mask" => oe(Tensor::from_array(([b, s], mask.clone())))?,
            "mm_embeds" => oe(Tensor::from_array(([b, s, H], mm_embeds)))?,
            "mm_mask" => oe(Tensor::from_array(([b, s], mm_mask)))?,
        ]))?;
        self.infer_s += t.elapsed().as_secs_f64();
        let (shape, hs) = oe(outs["last_hidden_state"].try_extract_tensor::<f32>())?;
        let d = shape[2] as usize;
        // sentence-transformers Pooling(mean, include_prompt) + Normalize
        let mut dense = Vec::with_capacity(b);
        for bi in 0..b {
            let mut acc = vec![0f32; d];
            let mut cnt = 0f32;
            for t in 0..s {
                if mask[bi * s + t] != 0 {
                    let row = &hs[(bi * s + t) * d..(bi * s + t + 1) * d];
                    acc.iter_mut().zip(row).for_each(|(a, x)| *a += x);
                    cnt += 1.0;
                }
            }
            acc.iter_mut().for_each(|a| *a /= cnt.max(1e-9));
            l2_normalize(&mut acc);
            dense.push(acc);
        }
        Ok(BatchOut { dense, late: vec![], late_rows: vec![], ntok: mask.iter().sum::<i64>() as usize })
    }

    /// Vision tower on a set of frames/images at `soft` tokens each; returns per-frame soft tokens (n x 512).
    fn vision_soft(&mut self, imgs: &[&RgbImage], soft: usize) -> Result<Vec<Vec<f32>>> {
        let b = imgs.len();
        let gs: Vec<prep::GemmaImage> = imgs.iter().map(|im| prep::gemma_image_budget(im, soft)).collect::<Result<_>>()?;
        // The processor pads every image to soft*9 patches; padded patches are masked keys and pool to nothing, so
        // the dynamic graph can stop at the batch's largest real patch count (valid patches come first). Same
        // output; 9-15% fewer patches for square / 16:9 inputs. FLASHEMBED_VISION_FULL=1: always pad (static graph).
        let p = if vision_full() { soft * prep::G_POOL * prep::G_POOL } else { gs.iter().map(|g| g.n_soft).max().unwrap_or(1).max(1) * prep::G_POOL * prep::G_POOL };
        let mut pv = Vec::with_capacity(b * p * prep::G_PATCH_DIM);
        let mut pp = Vec::with_capacity(b * p * 2);
        let mut nsoft = Vec::with_capacity(b);
        for g in &gs {
            pv.extend_from_slice(&g.patches[..p * prep::G_PATCH_DIM]);
            pp.extend_from_slice(&g.positions[..p * 2]);
            nsoft.push(g.n_soft);
        }
        let vision = match (&mut self.vision_vid, self.modality) {
            (Some(v), Modality::Video) => v,
            _ => self.vision.as_mut().unwrap(),
        };
        let t = Instant::now();
        let outs = oe(vision.run(ort::inputs![
            "pixel_values" => oe(Tensor::from_array(([b, p, prep::G_PATCH_DIM], pv)))?,
            "image_position_ids" => oe(Tensor::from_array(([b, p, 2], pp)))?,
        ]))?;
        self.infer_s += t.elapsed().as_secs_f64();
        let (shape, sv) = oe(outs["soft_tokens"].try_extract_tensor::<f32>())?;
        let (l, h) = (shape[1] as usize, shape[2] as usize);
        let (_, smask) = oe(outs["soft_mask"].try_extract_tensor::<bool>())?;
        let mut res = Vec::with_capacity(b);
        for bi in 0..b {
            let valid: Vec<usize> = (0..l).filter(|&j| smask[bi * l + j]).collect();
            if valid.len() != nsoft[bi] {
                bail!("vision returned {} valid soft tokens, processor expects {}", valid.len(), nsoft[bi]);
            }
            let mut v = Vec::with_capacity(valid.len() * h);
            for j in valid {
                v.extend_from_slice(&sv[(bi * l + j) * h..(bi * l + j + 1) * h]);
            }
            res.push(v);
        }
        Ok(res)
    }

    /// Token row = tokenizer specials around `middle` (Gemma: <bos> ... <eos>), built from ids so that
    /// multimodal placeholders never go through the text tokenizer.
    fn wrap_specials(&self, middle: &[i64]) -> Result<Vec<i64>> {
        let e = prep::encode(&self.tok, &[String::new()], true)?.remove(0);
        let mut row = e[..e.len().min(1)].to_vec();
        row.extend_from_slice(middle);
        row.extend_from_slice(&e[e.len().min(1)..]);
        Ok(row)
    }

    fn gemma_image(&mut self, items: &[&Item]) -> Result<BatchOut> {
        let loaded: Vec<RgbImage> = items
            .iter()
            .map(|it| match it {
                Item::ImagePath(p) => Ok(prep::load_rgb(p)?),
                Item::Image(im) => Ok(im.clone()),
                _ => unreachable!(),
            })
            .collect::<Result<_>>()?;
        let refs: Vec<&RgbImage> = loaded.iter().collect();
        let mm = self.vision_soft(&refs, self.soft_budget)?;
        let (boi, eoi) = (prep::token_id(&self.tok, "<|image>")?, prep::token_id(&self.tok, "<image|>")?);
        let rows = mm
            .iter()
            .map(|v| {
                // EmbeddingGemma2Processor.replace_image_token: boi + image * n + eoi
                let mut mid = vec![boi];
                mid.extend(std::iter::repeat_n(self.image_id, v.len() / 512));
                mid.push(eoi);
                self.wrap_specials(&mid)
            })
            .collect::<Result<Vec<_>>>()?;
        self.gemma_lm(&rows, Some(&mm))
    }

    fn gemma_video(&mut self, items: &[&Item]) -> Result<BatchOut> {
        // EmbeddingGemma2VideoProcessor defaults (processor_config.json): fps 1, max_frames 32, 140 tokens/frame
        let mut frames = Vec::new();
        let mut per_item = Vec::with_capacity(items.len());
        for it in items {
            let Item::Video(p) = it else { unreachable!() };
            let m = media::probe_video(p)?;
            let idx = media::sample_indices(&m, 1.0, 32);
            if std::env::var_os("FLASHEMBED_VIDEO_DEBUG").is_some() {
                eprintln!("video {} frames={} fps={:?} dur={:?} idx={:?}", p.display(), m.total_frames, m.fps, m.duration, idx);
            }
            let f = media::decode_frames(p, &m, &idx)?;
            per_item.push(f.len());
            frames.extend(f);
        }
        let refs: Vec<&RgbImage> = frames.iter().collect();
        let soft = self.vision_soft(&refs, 140)?;
        let (boi, eoi) = (prep::token_id(&self.tok, "<|image>")?, prep::token_id(&self.tok, "<image|>")?);
        let (mut rows, mut mm, mut k) = (Vec::new(), Vec::new(), 0usize);
        for n in per_item {
            // replace_video_token without timestamps: per frame boi + video * n + eoi, frames concatenated
            let (mut mid, mut v) = (Vec::new(), Vec::new());
            for f in &soft[k..k + n] {
                mid.push(boi);
                mid.extend(std::iter::repeat_n(self.video_id, f.len() / 512));
                mid.push(eoi);
                v.extend_from_slice(f);
            }
            k += n;
            rows.push(self.wrap_specials(&mid)?);
            mm.push(v);
        }
        self.gemma_lm(&rows, Some(&mm))
    }

    fn gemma_audio(&mut self, items: &[&Item]) -> Result<BatchOut> {
        let wavs: Vec<Vec<f32>> = items
            .iter()
            .map(|it| match it {
                Item::Audio(p) => media::decode_audio(p, Some(480_000)),
                _ => unreachable!(),
            })
            .collect::<Result<_>>()?;
        let (len, valid) = media::pad_batch(&wavs);
        let b = wavs.len();
        let mut feats = Vec::new();
        let mut masks = Vec::new();
        let mut nsoft = Vec::with_capacity(b);
        let mut t_frames = 0;
        for (w, &v) in wavs.iter().zip(&valid) {
            let mut x = w[..v].to_vec();
            x.resize(len, 0.0);
            let (f, m) = self.mel.features(&x, v);
            t_frames = m.len();
            nsoft.push(media::audio_soft_tokens(&m));
            feats.extend(f);
            masks.extend(m);
        }
        let (boa, eoa) = (prep::token_id(&self.tok, "<|audio>")?, prep::token_id(&self.tok, "<audio|>")?);
        let audio = self.audio.as_mut().unwrap();
        let t = Instant::now();
        let outs = oe(audio.run(ort::inputs![
            "input_features" => oe(Tensor::from_array(([b, t_frames, 128], feats)))?,
            "input_features_mask" => oe(Tensor::from_array(([b, t_frames], masks)))?,
        ]))?;
        self.infer_s += t.elapsed().as_secs_f64();
        let (shape, sv) = oe(outs["soft_tokens"].try_extract_tensor::<f32>())?;
        let (l, h) = (shape[1] as usize, shape[2] as usize);
        let sv = sv.to_vec();
        let (_, sm) = oe(outs["soft_mask"].try_extract_tensor::<bool>())?;
        let sm = sm.to_vec();
        drop(outs);
        let (mut rows, mut mm) = (Vec::new(), Vec::new());
        for bi in 0..b {
            let valid: Vec<usize> = (0..l).filter(|&j| sm[bi * l + j]).collect();
            if valid.len() != nsoft[bi] {
                bail!("audio tower returned {} valid tokens, processor expects {}", valid.len(), nsoft[bi]);
            }
            let mut v = Vec::with_capacity(valid.len() * h);
            for j in &valid {
                v.extend_from_slice(&sv[(bi * l + j) * h..(bi * l + j + 1) * h]);
            }
            let mut mid = vec![boa];
            mid.extend(std::iter::repeat_n(self.audio_id, valid.len()));
            mid.push(eoa);
            rows.push(self.wrap_specials(&mid)?);
            mm.push(v);
        }
        self.gemma_lm(&rows, Some(&mm))
    }

    fn neomme_image(&mut self, items: &[&Item]) -> Result<BatchOut> {
        let b = items.len();
        let mut rows = Vec::with_capacity(b);
        let mut poss = Vec::with_capacity(b);
        let mut flags = Vec::with_capacity(b);
        let mut pats = Vec::with_capacity(b);
        for it in items {
            let owned;
            let im = match it {
                Item::Image(im) => im,
                Item::ImagePath(p) => {
                    owned = prep::load_rgb(p)?;
                    &owned
                }
                _ => unreachable!(),
            };
            let p = prep::neomme_image(im)?;
            let (ids, pos, is_patch) = prep::neomme_image_layout(p.gh, p.gw, self.doc_id, self.image_id, self.row_id);
            rows.push(ids);
            poss.push(pos);
            flags.push(is_patch);
            pats.push(p.patches);
        }
        let (ids, mask, s) = prep::pad(&rows, self.pad_id);
        let mut position_ids = vec![0i64; 2 * b * s];
        let mut patch_mask = vec![false; b * s];
        let mut patches = vec![0f32; b * s * prep::N_PATCH_DIM];
        for bi in 0..b {
            let mut k = 0;
            for (t, (p, &f)) in poss[bi].iter().zip(&flags[bi]).enumerate() {
                position_ids[bi * s + t] = p[0];
                position_ids[b * s + bi * s + t] = p[1];
                if f {
                    patch_mask[bi * s + t] = true;
                    patches[(bi * s + t) * prep::N_PATCH_DIM..(bi * s + t + 1) * prep::N_PATCH_DIM]
                        .copy_from_slice(&pats[bi][k * prep::N_PATCH_DIM..(k + 1) * prep::N_PATCH_DIM]);
                    k += 1;
                }
            }
        }
        drop(pats);
        let t = Instant::now();
        let outs = oe(self.nimg.as_mut().unwrap().run(ort::inputs![
            "input_ids" => oe(Tensor::from_array(([b, s], ids)))?,
            "attention_mask" => oe(Tensor::from_array(([b, s], mask.clone())))?,
            "position_ids" => oe(Tensor::from_array(([2, b, s], position_ids)))?,
            "patches" => oe(Tensor::from_array(([b, s, prep::N_PATCH_DIM], patches)))?,
            "patch_mask" => oe(Tensor::from_array(([b, s], patch_mask)))?,
        ]))?;
        self.infer_s += t.elapsed().as_secs_f64();
        neomme_outputs(&outs, &mask, b, s, self.no_late)
    }
}

fn neomme_outputs(outs: &ort::session::SessionOutputs, mask: &[i64], b: usize, s: usize, no_late: bool) -> Result<BatchOut> {
    let (ds, dense) = oe(outs["dense"].try_extract_tensor::<f32>())?;
    let d = ds[1] as usize;
    let (ls, late) = oe(outs["late"].try_extract_tensor::<f32>())?;
    let e = ls[2] as usize;
    let mut out = BatchOut { dense: vec![], late: vec![], late_rows: vec![], ntok: 0 };
    for bi in 0..b {
        out.dense.push(dense[bi * d..(bi + 1) * d].to_vec());
        if no_late {
            out.ntok += mask[bi * s..(bi + 1) * s].iter().filter(|&&m| m != 0).count();
            continue;
        }
        let mut v = Vec::new();
        let mut n = 0;
        for t in 0..s {
            if mask[bi * s + t] != 0 {
                v.extend_from_slice(&late[(bi * s + t) * e..(bi * s + t + 1) * e]);
                n += 1;
            }
        }
        out.ntok += n;
        out.late.push(v);
        out.late_rows.push(n);
    }
    Ok(out)
}

fn load_items(dir: &Path, modality: Modality, n: usize, a: &BenchArgs) -> Result<(Vec<String>, Vec<Item>)> {
    let (file, key) = match modality {
        Modality::Text => (a.texts_file.as_str(), "text"),
        Modality::Image | Modality::Audio | Modality::Video => (a.images_file.as_str(), "path"),
    };
    let s = std::fs::read_to_string(dir.join(file))?;
    let mut ids = vec![];
    let mut items = vec![];
    for line in s.lines().filter(|l| !l.trim().is_empty()).take(n) {
        let v: serde_json::Value = serde_json::from_str(line)?;
        ids.push(v["id"].as_str().unwrap_or_default().to_string());
        let x = v[key].as_str().context("bad input row")?;
        items.push(match modality {
            Modality::Text => Item::Text(x.to_string()),
            Modality::Image if a.decode_in_loop => Item::ImagePath(dir.join(x)),
            Modality::Image => Item::Image(prep::load_rgb(&dir.join(x))?),
            Modality::Audio => Item::Audio(dir.join(x)),
            Modality::Video => Item::Video(dir.join(x)),
        });
    }
    Ok((ids, items))
}

fn check_dim(a: &BenchArgs) -> Result<()> {
    if let Some(d) = a.dim {
        if a.model != ModelKind::Gemma || ![128, 256, 512, 768].contains(&d) {
            bail!("--dim is EmbeddingGemma 2's Matryoshka size: 768, 512, 256 or 128");
        }
    }
    Ok(())
}

fn modality_name(m: Modality) -> &'static str {
    match m { Modality::Text => "text", Modality::Image => "image", Modality::Audio => "audio", Modality::Video => "video" }
}

/// Size key for length-sorted batching and for picking the tuning batch.
fn item_size(it: &Item, for_tuning: bool) -> usize {
    match it {
        Item::Text(t) => t.chars().count(),
        Item::Image(im) => (im.width() * im.height()) as usize,
        Item::ImagePath(p) if for_tuning => image::image_dimensions(p).map(|(w, h)| (w * h) as usize).unwrap_or(0),
        Item::Audio(p) | Item::Video(p) if for_tuning => std::fs::metadata(p).map(|m| m.len() as usize).unwrap_or(0),
        _ => 0,
    }
}

/// Thread-pool policy: FLASHEMBED_ORT_POLICY, or autotune over the non-spinning candidates by timing `tune` (one runner
/// alive at a time) and keeping the fastest. `tune` = None: no timing, take "pinned" (else the first candidate).
fn pick_policy(a: &BenchArgs, mods: &[Modality], tune: Option<(Modality, Vec<&Item>)>) -> Result<(Policy, Vec<serde_json::Value>)> {
    let forced = std::env::var("FLASHEMBED_ORT_POLICY").unwrap_or_else(|_| "auto".into());
    let all = policies(a.threads);
    let cands: Vec<Policy> = if forced != "auto" {
        all.iter().filter(|p| p.name == forced || p.name.starts_with(&forced)).cloned().collect()
    } else {
        // automatic choice never spins: spinning pools won short tuning windows on shared nodes and then ran 2-3x
        // slower in the timed loop once the node's load shifted (spinning remains available via FLASHEMBED_ORT_POLICY)
        let quiet: Vec<Policy> = all.iter().filter(|p| p.cfg.iter().any(|(k, v)| k == "session.intra_op.allow_spinning" && v == "0")).cloned().collect();
        if quiet.is_empty() { all.clone() } else { quiet } // 1 thread: nothing to pin
    };
    if cands.is_empty() {
        bail!("FLASHEMBED_ORT_POLICY={forced}: no such policy (have {:?})", all.iter().map(|p| &p.name).collect::<Vec<_>>());
    }
    let mut tuning = vec![];
    let mut best: Option<(f64, Policy)> = None;
    match tune {
        Some((m, batch)) if cands.len() > 1 && !batch.is_empty() => {
            for p in &cands {
                let mut r = Runner::new(a, mods, p)?;
                r.modality = m;
                r.run(&batch)?;
                // best of up to 3 timed runs within ~1 s (one run for long inputs): busy nodes are noisy
                let (mut dt, t_c) = (f64::MAX, Instant::now());
                for _ in 0..3 {
                    let t = Instant::now();
                    r.run(&batch)?;
                    dt = dt.min(t.elapsed().as_secs_f64());
                    if t_c.elapsed().as_secs_f64() > 1.0 {
                        break;
                    }
                }
                // Spinning pools are fast in a short tuning window but collapse when a shared node's load shifts
                // (seen: 0.15 s/item while tuning, 0.40 s/item in the timed loop); take one only if >15% faster.
                let spins = !p.cfg.iter().any(|(k, v)| k == "session.intra_op.allow_spinning" && v == "0");
                let score = if spins { dt * 1.15 } else { dt };
                tuning.push(json!({"policy": p.name, "threads": p.threads, "s": dt, "score": score}));
                if best.as_ref().is_none_or(|b| score < b.0) {
                    best = Some((score, p.clone()));
                }
            }
        }
        _ => {}
    }
    let pol = best.map(|b| b.1).unwrap_or_else(|| {
        cands.iter().find(|p| p.name.starts_with("pinned")).cloned().unwrap_or_else(|| cands[0].clone())
    });
    Ok((pol, tuning))
}

/// Embed `items` with `r` (modality already set) and write the outputs like `embed`/`bench`.
/// tolerant (serve): a failing batch is retried item by item; items that still fail are left out of the outputs and
/// returned as (id, error). Otherwise the first error is returned.
fn embed_run(r: &mut Runner, a: &BenchArgs, ids: &[String], items: &[Item], load_s: f64, pol: &Policy,
             tuning: &[serde_json::Value], tolerant: bool, extra: serde_json::Value)
             -> Result<(serde_json::Value, usize, Vec<(String, String)>)> {
    // Length-sorted batching (longest first), as sentence-transformers does: less padding, same per-item vectors.
    // Outputs are put back into input order before saving.
    let mut order: Vec<usize> = (0..items.len()).collect();
    if a.batch > 1 && !a.no_sort {
        order.sort_by_key(|&i| std::cmp::Reverse(item_size(&items[i], false)));
    }
    let chunks: Vec<&[usize]> = order.chunks(a.batch.max(1)).collect();
    for c in chunks.iter().take(a.warmup) {
        let b: Vec<&Item> = c.iter().map(|&i| &items[i]).collect();
        r.run(&b)?;
    }
    r.infer_s = 0.0;
    // per item: (input index, dense, late, late rows)
    let mut per: Vec<(usize, Vec<f32>, Vec<f32>, usize)> = Vec::with_capacity(items.len());
    let mut failed: Vec<(String, String)> = vec![];
    let mut lat = Vec::with_capacity(chunks.len());
    let mut ntok = 0usize;
    let mut push = |o: BatchOut, idx: &[usize], per: &mut Vec<(usize, Vec<f32>, Vec<f32>, usize)>| {
        ntok += o.ntok;
        for (j, dv) in o.dense.into_iter().enumerate() {
            let (lv, lr) = if o.late.is_empty() { (vec![], 0) } else { (o.late[j].clone(), o.late_rows[j]) };
            per.push((idx[j], dv, lv, lr));
        }
    };
    let t0 = Instant::now();
    for c in &chunks {
        let s = Instant::now();
        let b: Vec<&Item> = c.iter().map(|&i| &items[i]).collect();
        match r.run(&b) {
            Ok(o) => push(o, c, &mut per),
            Err(e) if !tolerant => return Err(e),
            Err(e) if c.len() == 1 => failed.push((ids[c[0]].clone(), format!("{e:#}"))),
            Err(_) => {
                for &i in c.iter() {
                    match r.run(&[&items[i]]) {
                        Ok(o) => push(o, &[i], &mut per),
                        Err(e) => failed.push((ids[i].clone(), format!("{e:#}"))),
                    }
                }
            }
        }
        lat.push(s.elapsed().as_secs_f64());
    }
    let wall = t0.elapsed().as_secs_f64();
    per.sort_by_key(|x| x.0);
    if let Some(d) = a.dim {
        // Matryoshka: keep the leading d dims of the (already normalised) vector, then re-normalise
        for x in per.iter_mut() {
            x.1.truncate(d);
            l2_normalize(&mut x.1);
        }
    }
    let model = match a.model { ModelKind::Gemma => "gemma", ModelKind::Neomme => "neomme" };
    let modality = modality_name(r.modality);
    let tag = if a.plain_names { "embeddings".to_string() } else { format!("rs-{model}-{modality}-t{}-b{}{}", a.threads, a.batch, a.tag_suffix) };
    let mut res = json!({
        "tag": tag, "runtime": "rust-ort", "model": model, "modality": modality, "threads": a.threads,
        "batch": a.batch, "dim": a.dim, "no_late": a.no_late, "n": per.len(), "failed": failed.len(), "warmup_batches": a.warmup, "wall_s": wall,
        "items_per_s": per.len() as f64 / wall, "batch_latency_s": lat, "load_s": load_s,
        "peak_rss_mb": peak_rss_mb(), "tokens_total": ntok, "dtype": "float32",
        "infer_s": r.infer_s, "prep_post_s": wall - r.infer_s, "side": a.side, "texts_file": a.texts_file,
        "ort": "load-dynamic (ORT_DYLIB_PATH)", "ort_arena": std::env::var("FLASHEMBED_ORT_ARENA").is_ok_and(|v| v == "1"), "ort_thread_policy": pol.name, "ort_intra_threads": pol.threads, "policy_tuning": tuning, "ort_config": std::env::var("FLASHEMBED_ORT_CONFIG").unwrap_or_default(), "ort_dylib": std::env::var("ORT_DYLIB_PATH").unwrap_or_default(),
        "cpu": cpu_model(), "host": std::fs::read_to_string("/proc/sys/kernel/hostname").unwrap_or_default().trim(),
        "job": std::env::var("SLURM_JOB_ID").ok(),
    });
    if let (Some(o), Some(x)) = (res.as_object_mut(), extra.as_object()) {
        o.extend(x.clone());
    }
    std::fs::create_dir_all(&a.out)?;
    if a.save && !per.is_empty() {
        let d = per[0].1.len();
        let flat: Vec<f32> = per.iter().flat_map(|x| x.1.iter().copied()).collect();
        ndarray_npy::write_npy(a.out.join(format!("{tag}.dense.npy")), &Array2::from_shape_vec((per.len(), d), flat)?)?;
        if per.iter().any(|x| x.3 > 0) {
            let late: Vec<f32> = per.iter().flat_map(|x| x.2.iter().copied()).collect();
            let mut offs = vec![0i64];
            for x in &per {
                offs.push(offs.last().unwrap() + x.3 as i64);
            }
            let rows = *offs.last().unwrap() as usize;
            ndarray_npy::write_npy(a.out.join(format!("{tag}.late.npy")), &Array2::from_shape_vec((rows, late.len() / rows), late)?)?;
            ndarray_npy::write_npy(a.out.join(format!("{tag}.late_offsets.npy")), &Array1::from_vec(offs))?;
        }
        let out_ids: Vec<&String> = per.iter().map(|x| &ids[x.0]).collect();
        std::fs::write(a.out.join(format!("{tag}.ids.json")), serde_json::to_string(&out_ids)?)?;
    }
    std::fs::write(a.out.join(format!("{tag}.json")), serde_json::to_string_pretty(&res)?)?;
    res.as_object_mut().unwrap().remove("batch_latency_s");
    Ok((res, per.len(), failed))
}

fn bench(a: BenchArgs) -> Result<()> {
    check_dim(&a)?;
    let t_load = Instant::now();
    let (ids, items) = load_items(&a.inputs, a.modality, a.n, &a)?;
    // tune on the largest batch (longest texts / largest images): it dominates the run time, and a small first
    // item picked too few threads for the rest (8 of 16 CPUs on a 9-item image set: 0.87x instead of 1.35x)
    let mut o: Vec<usize> = (0..items.len()).collect();
    o.sort_by_key(|&i| std::cmp::Reverse(item_size(&items[i], true)));
    let tune: Vec<&Item> = o.iter().take(a.batch).map(|&i| &items[i]).collect();
    let (pol, tuning) = pick_policy(&a, &[a.modality], Some((a.modality, tune)))?;
    let mut r = Runner::new(&a, &[a.modality], &pol)?;
    let load_s = t_load.elapsed().as_secs_f64();
    let (brief, _, _) = embed_run(&mut r, &a, &ids, &items, load_s, &pol, &tuning, false, json!({}))?;
    println!("{brief}");
    Ok(())
}

#[derive(clap::Args)]
struct ServeArgs {
    #[arg(long, value_enum)]
    model: ModelKind,
    /// directory with the exported ONNX graphs (py/export_onnx.py)
    #[arg(long)]
    onnx: PathBuf,
    /// local Hugging Face snapshot of the model (tokenizer and configs)
    #[arg(long)]
    snapshot: PathBuf,
    /// modalities to load, comma-separated (text,image,video,audio; NeoMME: text,image)
    #[arg(long, default_value = "text")]
    modalities: String,
    #[arg(long)]
    threads: Option<usize>,
    #[arg(long, default_value_t = 8)]
    batch: usize,
    /// token limit for text (default: the model's maximum)
    #[arg(long)]
    max_len: Option<usize>,
    #[arg(long, default_value = "document")]
    side: String,
    #[arg(long, default_value_t = 280)]
    max_soft_tokens: usize,
    #[arg(long)]
    dim: Option<usize>,
    #[arg(long)]
    no_late: bool,
}

/// Persistent server: load graphs and pick the thread pool once, then embed one request per stdin line.
///   request : {"id": <any>, "modality": "text|image|video|audio", "input": "/path/items.jsonl", "out": "/path/dir"}
///   response: {"id", "ok": true, "items", "failed": [{"id", "error"}], "seconds", ...} or {"id", "ok": false, "error"}
/// Writes the same files as `embed`. stdout carries only protocol lines; logs go to stderr. Exits at EOF.
fn serve(s: ServeArgs) -> Result<()> {
    use std::io::{BufRead, Write};
    let t_start = Instant::now();
    let mut mods: Vec<Modality> = vec![];
    for m in s.modalities.split(',').map(str::trim).filter(|m| !m.is_empty()) {
        let m = Modality::from_str(m, true).map_err(|e| anyhow!("--modalities {m}: {e}"))?;
        if !mods.contains(&m) {
            mods.push(m);
        }
    }
    if mods.is_empty() {
        bail!("--modalities: none given");
    }
    let threads = s.threads.unwrap_or_else(|| cpu_topology().len().max(1));
    let max_len = s.max_len.unwrap_or(match s.model { ModelKind::Gemma => 8192, ModelKind::Neomme => 16384 });
    let base = BenchArgs {
        model: s.model, modality: mods[0], threads, batch: s.batch.max(1), n: usize::MAX, warmup: 0, max_len,
        inputs: PathBuf::from("."), onnx: s.onnx, snapshot: s.snapshot, out: PathBuf::from("."), save: true,
        texts_file: String::new(), images_file: String::new(), side: s.side, tag_suffix: String::new(),
        max_soft_tokens: s.max_soft_tokens, decode_in_loop: true, no_sort: false, plain_names: true, dim: s.dim,
        no_late: s.no_late,
    };
    check_dim(&base)?;
    // synthetic tuning / warm-up batch: full-length texts if text is served, else large images
    let long_text = "The quick brown fox jumps over the lazy dog while the committee reviews the findings. "
        .repeat(max_len / 12 + 2);
    let synth_img = RgbImage::from_fn(1024, 1024, |x, y| image::Rgb([(x % 256) as u8, (y % 256) as u8, ((x + y) % 256) as u8]));
    let tune_items: Vec<Item> = if mods.contains(&Modality::Text) {
        (0..base.batch).map(|_| Item::Text(long_text.clone())).collect()
    } else if mods.contains(&Modality::Image) {
        (0..base.batch).map(|_| Item::Image(synth_img.clone())).collect()
    } else {
        vec![]
    };
    let tune_mod = if mods.contains(&Modality::Text) { Modality::Text } else { Modality::Image };
    let tune = if tune_items.is_empty() { None } else { Some((tune_mod, tune_items.iter().collect::<Vec<_>>())) };
    let (pol, tuning) = pick_policy(&base, &mods, tune)?;
    let mut r = Runner::new(&base, &mods, &pol)?;
    // warm up each text/image path once (first runs allocate and plan); outputs are discarded
    for &m in &mods {
        let w = match m {
            Modality::Text => Item::Text("warm up".into()),
            Modality::Image => Item::Image(RgbImage::from_pixel(256, 256, image::Rgb([128, 128, 128]))),
            _ => continue,
        };
        r.modality = m;
        r.run(&[&w])?;
    }
    let load_s = t_start.elapsed().as_secs_f64();
    let model = match base.model { ModelKind::Gemma => "gemma", ModelKind::Neomme => "neomme" };
    let mut out = std::io::stdout().lock();
    writeln!(out, "{}", json!({
        "ready": true, "model": model, "modalities": mods.iter().map(|&m| modality_name(m)).collect::<Vec<_>>(),
        "threads": threads, "batch": base.batch, "max_len": max_len, "side": base.side, "dim": base.dim,
        "no_late": base.no_late, "ort_thread_policy": pol.name, "ort_intra_threads": pol.threads,
        "policy_tuning": tuning, "load_s": load_s, "peak_rss_mb": peak_rss_mb(), "pid": std::process::id(),
    }))?;
    out.flush()?;
    let stdin = std::io::stdin();
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        if line.trim().is_empty() {
            continue;
        }
        let t = Instant::now();
        let mut rid = serde_json::Value::Null;
        let res: Result<serde_json::Value> = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<serde_json::Value> {
            let req: serde_json::Value = serde_json::from_str(&line).context("request is not JSON")?;
            rid = req.get("id").cloned().unwrap_or(serde_json::Value::Null);
            let ms = req.get("modality").and_then(|v| v.as_str()).context("request needs \"modality\"")?;
            let m = Modality::from_str(ms, true).map_err(|e| anyhow!("modality {ms}: {e}"))?;
            if !mods.contains(&m) {
                bail!("modality {ms} not loaded (server has {})", s.modalities);
            }
            let input = PathBuf::from(req.get("input").and_then(|v| v.as_str()).context("request needs \"input\"")?);
            let outdir = PathBuf::from(req.get("out").and_then(|v| v.as_str()).context("request needs \"out\"")?);
            let dir = input.parent().map(|p| p.to_path_buf()).unwrap_or_else(|| PathBuf::from("."));
            // rows: bad lines are reported per item, not fatal
            let text = std::fs::read_to_string(&input).with_context(|| format!("read {}", input.display()))?;
            let key = if m == Modality::Text { "text" } else { "path" };
            let (mut ids, mut items, mut bad) = (vec![], vec![], vec![]);
            for (ln, l) in text.lines().enumerate().filter(|(_, l)| !l.trim().is_empty()) {
                let v: serde_json::Value = match serde_json::from_str(l) {
                    Ok(v) => v,
                    Err(e) => { bad.push((format!("line {}", ln + 1), format!("bad JSON: {e}"))); continue; }
                };
                let id = match &v["id"] { serde_json::Value::String(x) => x.clone(), serde_json::Value::Null => String::new(), x => x.to_string() };
                let Some(x) = v[key].as_str() else { bad.push((id, format!("missing \"{key}\""))); continue; };
                ids.push(id);
                items.push(match m {
                    Modality::Text => Item::Text(x.to_string()),
                    Modality::Image => Item::ImagePath(dir.join(x)),
                    Modality::Audio => Item::Audio(dir.join(x)),
                    Modality::Video => Item::Video(dir.join(x)),
                });
            }
            let load_req = t.elapsed().as_secs_f64();
            r.modality = m;
            let mut a = base.clone_for(m, &dir, &input, &outdir);
            a.plain_names = true;
            let (_, n, mut failed) = if items.is_empty() {
                (json!({}), 0, vec![])
            } else {
                embed_run(&mut r, &a, &ids, &items, load_req, &pol, &tuning, true,
                          json!({"server_load_s": load_s, "request_id": rid.clone()}))?
            };
            failed.extend(bad);
            if n == 0 {
                bail!("no items embedded ({} failed{})", failed.len(),
                      failed.first().map(|f| format!(", first: {}: {}", f.0, f.1)).unwrap_or_default());
            }
            let secs = t.elapsed().as_secs_f64();
            Ok(json!({"id": rid.clone(), "ok": true, "items": n, "failed": failed.iter().map(|f| json!({"id": f.0, "error": f.1})).collect::<Vec<_>>(),
                      "seconds": secs, "items_per_s": n as f64 / secs, "out": outdir, "peak_rss_mb": peak_rss_mb()}))
        }))
        .unwrap_or_else(|p| {
            let msg = p.downcast_ref::<String>().cloned().or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default();
            Err(anyhow!("panic: {msg}"))
        });
        let resp = match res {
            Ok(v) => v,
            Err(e) => json!({"id": rid, "ok": false, "error": format!("{e:#}"), "seconds": t.elapsed().as_secs_f64()}),
        };
        writeln!(out, "{resp}")?;
        out.flush()?;
    }
    Ok(())
}

impl BenchArgs {
    /// Per-request copy of the server's settings.
    fn clone_for(&self, m: Modality, dir: &Path, input: &Path, out: &Path) -> BenchArgs {
        let file = input.file_name().map(|f| f.to_string_lossy().to_string()).unwrap_or_default();
        BenchArgs {
            model: self.model, modality: m, threads: self.threads, batch: self.batch, n: usize::MAX, warmup: 0,
            max_len: self.max_len, inputs: dir.to_path_buf(), onnx: self.onnx.clone(), snapshot: self.snapshot.clone(),
            out: out.to_path_buf(), save: true, texts_file: file.clone(), images_file: file, side: self.side.clone(),
            tag_suffix: String::new(), max_soft_tokens: self.max_soft_tokens, decode_in_loop: true, no_sort: false,
            plain_names: true, dim: self.dim, no_late: self.no_late,
        }
    }
}

fn main() -> Result<()> {
    if std::env::var("ORT_DYLIB_PATH").is_err() {
        bail!("set ORT_DYLIB_PATH to libonnxruntime.so (the Python onnxruntime wheel's copy)");
    }
    match Cli::parse().cmd {
        Cmd::Embed(a) => bench(a.into_bench()?),
        Cmd::Bench(a) => bench(a),
        Cmd::Serve(s) => serve(s),
    }
}
