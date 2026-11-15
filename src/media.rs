//! Audio and video decoding (ffmpeg / ffprobe subprocesses) and EmbeddingGemma 2 audio features.
//! Mirrors transformers 5.22.2: `Gemma4AudioFeatureExtractor` (log-mel), `EmbeddingGemma2Processor.replace_audio_token`,
//! `EmbeddingGemma2VideoProcessor.sample_frames` with the PyAV metadata HF reads (nb_frames, average_rate).

use std::io::Read;
use std::path::Path;
use std::process::{Command, Stdio};

use anyhow::{Context, Result, anyhow, bail};
use image::RgbImage;

pub const SR: usize = 16_000;

/// ffmpeg / ffprobe binaries: FLASHEMBED_FFMPEG / FLASHEMBED_FFPROBE, else from PATH. Use the FFmpeg release the Python
/// side decodes with (PyAV/torchcodec bundle) for bit-identical decoding (slurm/ffmpeg.sbatch).
fn ffmpeg_bin() -> String {
    std::env::var("FLASHEMBED_FFMPEG").unwrap_or_else(|_| "ffmpeg".into())
}
fn ffprobe_bin() -> String {
    std::env::var("FLASHEMBED_FFPROBE").unwrap_or_else(|_| "ffprobe".into())
}
const FRAME: usize = 320; // 20 ms
const HOP: usize = 160; // 10 ms
const NFFT: usize = 512;
const NMEL: usize = 128;
const MAX_SAMPLES: usize = 480_000; // feature extractor default max_length (30 s) -> longer audio is truncated
const PAD_MULT: usize = 128;

/// Decode any audio ffmpeg understands to mono f32 at 16 kHz (ffmpeg's default swresample).
/// Mono 16 kHz f32 samples. `max_samples`: stop decoding a little past that many output samples (the feature
/// extractor truncates at 480,000 = 30 s; decoding a 24-minute MP3 to the end only to drop it cost seconds per file).
/// The resampler is causal up to a few ms of look-ahead, so the kept samples are unchanged.
pub fn decode_audio(path: &Path, max_samples: Option<usize>) -> Result<Vec<f32>> {
    let mut cmd = Command::new(ffmpeg_bin());
    cmd.args(["-nostdin", "-hide_banner", "-loglevel", "error", "-i"]).arg(path).args(["-vn", "-ac", "1", "-ar", "16000"]);
    if let Some(m) = max_samples {
        cmd.args(["-t", &format!("{:.3}", m as f64 / 16000.0 + 0.25)]);
    }
    let out = cmd
        .args(["-f", "f32le", "-"])
        .output()
        .context("running ffmpeg (is it on PATH?)")?;
    if !out.status.success() {
        bail!("ffmpeg failed on {}: {}", path.display(), String::from_utf8_lossy(&out.stderr));
    }
    Ok(out.stdout.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect())
}

fn hz_to_mel(f: f64) -> f64 {
    2595.0 * (1.0 + f / 700.0).log10()
}
fn mel_to_hz(m: f64) -> f64 {
    700.0 * (10f64.powf(m / 2595.0) - 1.0)
}

pub struct MelExtractor {
    window: Vec<f32>,
    filters: Vec<f64>, // [257 * 128]
    fft: std::sync::Arc<dyn rustfft::Fft<f32>>,
}

impl MelExtractor {
    pub fn new() -> Self {
        // periodic Hann: np.hanning(321)[:-1]
        let window = (0..FRAME).map(|n| (0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / FRAME as f64).cos()) as f32).collect();
        let nbins = NFFT / 2 + 1;
        let (mmin, mmax) = (hz_to_mel(0.0), hz_to_mel(8000.0));
        let ff: Vec<f64> = (0..NMEL + 2).map(|i| mel_to_hz(mmin + (mmax - mmin) * i as f64 / (NMEL + 1) as f64)).collect();
        let fft_freqs: Vec<f64> = (0..nbins).map(|i| 8000.0 * i as f64 / (nbins - 1) as f64).collect();
        let mut filters = vec![0f64; nbins * NMEL];
        for (b, &f) in fft_freqs.iter().enumerate() {
            for m in 0..NMEL {
                let down = -(ff[m] - f) / (ff[m + 1] - ff[m]);
                let up = (ff[m + 2] - f) / (ff[m + 2] - ff[m + 1]);
                filters[b * NMEL + m] = down.min(up).max(0.0);
            }
        }
        let fft = rustfft::FftPlanner::<f32>::new().plan_fft_forward(NFFT);
        Self { window, filters, fft }
    }

    /// One item: returns (features [n_frames * 128] zeroed where masked, mask [n_frames]).
    /// `wav` is padded/truncated by the caller to the batch length; `valid` = number of real samples.
    pub fn features(&self, wav: &[f32], valid: usize) -> (Vec<f32>, Vec<bool>) {
        let pad_left = FRAME / 2;
        let mut x = vec![0f32; pad_left];
        x.extend_from_slice(wav);
        let n = if x.len() >= FRAME + 1 { (x.len() - (FRAME + 1)) / HOP + 1 } else { 0 };
        let nbins = NFFT / 2 + 1;
        let mut feats = vec![0f32; n * NMEL];
        let mut mask = vec![false; n];
        let mut buf = vec![rustfft::num_complex::Complex32::new(0.0, 0.0); NFFT];
        let mut mag = vec![0f32; nbins];
        for t in 0..n {
            let s = t * HOP;
            for (i, b) in buf.iter_mut().enumerate() {
                *b = if i < FRAME { rustfft::num_complex::Complex32::new(x[s + i] * self.window[i], 0.0) } else { Default::default() };
            }
            self.fft.process(&mut buf);
            for (k, m) in mag.iter_mut().enumerate() {
                *m = buf[k].norm();
            }
            // frame end index (in the left-padded signal) decides validity
            let end = t * HOP + FRAME;
            let valid_frame = end >= pad_left && end - pad_left < valid;
            mask[t] = valid_frame;
            if valid_frame {
                for m in 0..NMEL {
                    let mut acc = 0f64;
                    for k in 0..nbins {
                        acc += mag[k] as f64 * self.filters[k * NMEL + m];
                    }
                    feats[t * NMEL + m] = (acc + 1e-3).ln() as f32;
                }
            }
        }
        (feats, mask)
    }
}

/// Feature-extractor padding for a batch: truncate to 30 s, pad to the longest, rounded up to 128 samples.
pub fn pad_batch(wavs: &[Vec<f32>]) -> (usize, Vec<usize>) {
    let valid: Vec<usize> = wavs.iter().map(|w| w.len().min(MAX_SAMPLES)).collect();
    let longest = *valid.iter().max().unwrap_or(&0);
    (longest.div_ceil(PAD_MULT) * PAD_MULT, valid)
}

/// EmbeddingGemma2Processor.replace_audio_token: soft tokens after two stride-2 subsamplings of the mask.
pub fn audio_soft_tokens(mask: &[bool]) -> usize {
    let mut m: Vec<bool> = mask.to_vec();
    for _ in 0..2 {
        let t_out = (m.len() + 2 - 3) / 2 + 1;
        m = m.iter().step_by(2).take(t_out).copied().collect();
    }
    m.iter().filter(|&&b| b).count()
}

// ------------------------------------------------------------------------------------------------- video
pub struct VideoMeta {
    pub total_frames: usize,
    /// presentation timestamps (stream time base) of the frames found by the packet scan, sorted: frame index i of
    /// torchcodec's exact mode is the frame with the i-th smallest timestamp (empty: header fallback)
    pub frame_pts: Vec<i64>,
    pub fps: f64,
    /// Stream duration in seconds.
    pub duration: f64,
    pub width: u32,
    pub height: u32,
}

/// Decode a still image with FFmpeg (fallback for formats the image crate cannot read), first frame as RGB24.
pub fn decode_image_ffmpeg(path: &Path) -> Result<RgbImage> {
    let out = Command::new(ffprobe_bin())
        .args(["-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0"])
        .arg(path)
        .output()
        .context("running ffprobe")?;
    let txt = String::from_utf8_lossy(&out.stdout);
    let (w, h) = txt.trim().split_once(',').ok_or_else(|| anyhow!("ffprobe: no image stream in {}", path.display()))?;
    let (w, h): (u32, u32) = (w.trim().parse()?, h.trim().parse()?);
    let raw = Command::new(ffmpeg_bin())
        .args(["-nostdin", "-hide_banner", "-loglevel", "error", "-i"])
        .arg(path)
        .args(["-frames:v", "1", "-sws_flags", "bilinear", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
        .output()
        .context("running ffmpeg")?;
    if !raw.status.success() || raw.stdout.len() < (w * h * 3) as usize {
        bail!("ffmpeg could not decode {}: {}", path.display(), String::from_utf8_lossy(&raw.stderr).trim());
    }
    RgbImage::from_raw(w, h, raw.stdout[..(w * h * 3) as usize].to_vec()).ok_or_else(|| anyhow!("ffmpeg image buffer"))
}

/// Stream metadata the way torchcodec's `seek_mode="exact"` scan derives it (what the HF video
/// processor samples from): frames = packets with a pts (minus discarded ones),
/// duration = max(pts + dur) - min(pts) in seconds (pts * av_q2d(time_base)), fps = frames / duration.
/// Header fields are only a fallback: MKV/WebM carry no frame count, and header rates differ in the
/// last bits from the content rate (30/1 vs 29.999999999999996), which shifts sampled frame indices.
pub fn probe_video(path: &Path) -> Result<VideoMeta> {
    let out = Command::new(ffprobe_bin())
        .args(["-v", "error", "-select_streams", "v:0", "-show_entries"])
        .arg("stream=nb_frames,avg_frame_rate,width,height,time_base:packet=pts,dts,duration,flags")
        .args(["-of", "json"])
        .arg(path)
        .output()
        .context("running ffprobe")?;
    let v: serde_json::Value = serde_json::from_slice(&out.stdout)?;
    let s = &v["streams"][0];
    let width = s["width"].as_u64().ok_or_else(|| anyhow!("no video stream in {}", path.display()))? as u32;
    let height = s["height"].as_u64().unwrap_or(0) as u32;
    let ratio = |x: &str| -> f64 {
        let (a, b) = x.split_once('/').unwrap_or((x, "1"));
        a.parse::<f64>().unwrap_or(0.0) / b.parse::<f64>().unwrap_or(1.0)
    };
    // torchcodec's scan: per packet pts, or dts when the container gives no pts (AVI), discarded packets skipped;
    // seconds = ts * num / den (in that order: num/den first changes the last bits, and with them sampled indices)
    let (tn, td) = s["time_base"].as_str().and_then(|x| x.split_once('/')).map_or((0i64, 1i64), |(a, b)| {
        (a.parse().unwrap_or(0), b.parse().unwrap_or(1))
    });
    let secs = |x: i64| -> f64 { x as f64 * tn as f64 / td as f64 };
    let (mut lo, mut hi) = (i64::MAX, i64::MIN);
    let mut pts = Vec::new();
    for p in v["packets"].as_array().map(|a| a.as_slice()).unwrap_or(&[]) {
        let Some(ts) = p["pts"].as_i64().or_else(|| p["dts"].as_i64()) else { continue };
        if p["flags"].as_str().is_some_and(|f| f.as_bytes().get(1) == Some(&b'D')) {
            continue;
        }
        pts.push(ts);
        lo = lo.min(ts);
        hi = hi.max(ts + p["duration"].as_i64().unwrap_or(0));
    }
    let n = pts.len();
    if n > 0 && tn > 0 && hi > lo {
        pts.sort_unstable();
        let d = secs(hi) - secs(lo);
        return Ok(VideoMeta { total_frames: n, frame_pts: pts, fps: n as f64 / d, duration: d, width, height });
    }
    let total = s["nb_frames"].as_str().and_then(|x| x.parse().ok()).unwrap_or(0usize);
    let fps = ratio(s["avg_frame_rate"].as_str().unwrap_or("0/1"));
    let duration = if fps > 0.0 { total as f64 / fps } else { 0.0 };
    Ok(VideoMeta { total_frames: total, frame_pts: vec![], fps, duration, width, height })
}

/// EmbeddingGemma2VideoProcessor.sample_frames (fps=1, max_frames=32, overflow "uniform").
pub fn sample_indices(m: &VideoMeta, fps: f64, max_frames: usize) -> Vec<usize> {
    let step = m.fps / fps;
    let n = ((m.duration * fps) as usize).max(1);
    let mut idx: Vec<usize> = (0..n).map(|i| ((i as f64 * step) as usize).min(m.total_frames.saturating_sub(1))).collect();
    if idx.len() > max_frames {
        let l = idx.len();
        // np.linspace(0, l-1, max_frames, dtype=int): i*step, last element exactly l-1
        let st = (l - 1) as f64 / (max_frames - 1) as f64;
        idx = (0..max_frames)
            .map(|i| idx[if i + 1 == max_frames { l - 1 } else { (i as f64 * st) as usize }])
            .collect();
    }
    idx
}

/// Decode the selected frames as RGB24 (bilinear swscale). Frames are picked by presentation timestamp, as
/// torchcodec's exact seek mode does (frame i = i-th smallest pts of the scan), with an ffmpeg `select` filter; if
/// that yields a different number of frames (e.g. frames the decoder never outputs), fall back to taking the i-th
/// output frame in order.
pub fn decode_frames(path: &Path, m: &VideoMeta, indices: &[usize]) -> Result<Vec<RgbImage>> {
    if !m.frame_pts.is_empty() {
        // torchcodec exact mode: frame i is the first decoded frame with pts >= the i-th smallest packet pts. Keep
        // container timestamps (-copyts) and select, per wanted pts t, the first frame with prev_selected < t <= pts;
        // showinfo reports the pts of each selected frame so frames the decoder never emits map to the next one.
        let mut want: Vec<i64> = indices.iter().map(|&i| m.frame_pts[i.min(m.frame_pts.len() - 1)]).collect();
        want.sort_unstable();
        want.dedup();
        let prev = "if(isnan(prev_selected_pts),-1e18,prev_selected_pts)";
        let expr = want.iter().map(|t| format!("gte(pts,{t})*lt({prev},{t})")).collect::<Vec<_>>().join("+");
        let (got, pts) = run_ffmpeg_frames(path, m, Some(&format!("select='{expr}',showinfo")), want.len())?;
        if !got.is_empty() && pts.len() >= got.len() {
            let pick = |t: i64| pts[..got.len()].iter().position(|&p| p >= t);
            if want.iter().all(|&t| pick(t).is_some()) {
                return Ok(indices
                    .iter()
                    .map(|&i| got[pick(m.frame_pts[i.min(m.frame_pts.len() - 1)]).unwrap()].clone())
                    .collect());
            }
        }
    }
    let last = *indices.iter().max().unwrap_or(&0);
    let (all, _) = run_ffmpeg_frames(path, m, None, last + 1)?;
    let frames: Vec<RgbImage> = indices.iter().filter(|&&k| k < all.len()).map(|&k| all[k].clone()).collect();
    if frames.is_empty() {
        bail!("no frames decoded from {}", path.display());
    }
    Ok(frames)
}

/// Up to `max` RGB24 frames of the first video stream, optionally through a video filter (timestamps kept as in the
/// container); with a showinfo filter, also the pts of each output frame.
fn run_ffmpeg_frames(path: &Path, m: &VideoMeta, vf: Option<&str>, max: usize) -> Result<(Vec<RgbImage>, Vec<i64>)> {
    let mut cmd = Command::new(ffmpeg_bin());
    let info = vf.is_some_and(|f| f.contains("showinfo"));
    cmd.args(["-nostdin", "-hide_banner", "-loglevel", if info { "info" } else { "error" }, "-copyts", "-i"])
        .arg(path)
        .args(["-map", "0:v:0"]);
    if let Some(f) = vf {
        cmd.args(["-vf", f]);
    }
    let mut child = cmd
        .args(["-fps_mode", "passthrough", "-sws_flags", "bilinear", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
        .stdout(Stdio::piped())
        .stderr(if info { Stdio::piped() } else { Stdio::inherit() })
        .spawn()
        .context("running ffmpeg")?;
    let err = child.stderr.take().map(|e| {
        std::thread::spawn(move || {
            let mut s = String::new();
            let _ = std::io::BufReader::new(e).read_to_string(&mut s);
            s.lines()
                .filter(|l| l.contains("showinfo") && l.contains(" n:"))
                .filter_map(|l| l.split("pts:").nth(1)?.split_whitespace().next()?.parse::<i64>().ok())
                .collect::<Vec<i64>>()
        })
    });
    let mut so = child.stdout.take().unwrap();
    let fsz = (m.width * m.height * 3) as usize;
    let mut frames = Vec::new();
    let mut buf = vec![0u8; fsz];
    while frames.len() < max {
        if so.read_exact(&mut buf).is_err() {
            break;
        }
        frames.push(RgbImage::from_raw(m.width, m.height, buf.clone()).unwrap());
    }
    drop(so);
    if !info {
        let _ = child.kill();
    }
    let _ = child.wait();
    let pts = err.map(|h| h.join().unwrap_or_default()).unwrap_or_default();
    Ok((frames, pts))
}
