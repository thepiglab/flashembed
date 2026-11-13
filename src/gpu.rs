//! CUDA engine bindings (feature `cuda`): the native GPU forward passes in `cuda/fegpu.cu`, plus weight loading
//! from the Hugging Face safetensors snapshot and the model configs.

use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use serde_json::Value;

unsafe extern "C" {
    fn fe_error() -> *const c_char;
    fn fe_new(arch: c_int, dtype: c_int, device: c_int) -> *mut c_void;
    fn fe_cfg(h: *mut c_void, key: *const c_char, v: f64) -> c_int;
    fn fe_add(h: *mut c_void, name: *const c_char, src_dtype: c_int, shape: *const i64, ndim: c_int, data: *const c_void) -> c_int;
    fn fe_build(h: *mut c_void, max_tokens: c_int, max_patches: c_int) -> c_int;
    fn fe_gemma_text(h: *mut c_void, ntok: c_int, nseq: c_int, ids: *const i32, cu: *const i32, mm_idx: *const i32, out: *mut f32) -> c_int;
    fn fe_gemma_vision(
        h: *mut c_void, npatch: c_int, nimg: c_int, vcu: *const i32, pos: *const i32, patches: *const f32, nslot: c_int,
        slot_ptr: *const i32, slot_patch: *const i32,
    ) -> c_int;
    fn fe_gemma_vision_u8(
        h: *mut c_void, npatch: c_int, nimg: c_int, vcu: *const i32, pos: *const i32, src: *const u8, src_bytes: u64,
        desc: *const i32, coef: *const i32, ncoef: u64, max_pix: c_int, nslot: c_int, slot_ptr: *const i32,
        slot_patch: *const i32,
    ) -> c_int;
    fn fe_gemma_audio(
        h: *mut c_void, nclip: c_int, cu0: *const i32, feats: *const f32, m0: *const i32, n1: c_int, rows1: *const i32,
        m1: *const i32, n2: c_int, rows2: *const i32, m2: *const i32, nsoft: c_int, soft_rows: *const i32,
    ) -> c_int;
    fn fe_neomme(
        h: *mut c_void, ntok: c_int, nseq: c_int, ids: *const i32, cu: *const i32, pos2: *const i32, npatch: c_int,
        patches: *const u8, patch_rows: *const i32, dense: *mut f32, late: *mut f32,
    ) -> c_int;
    fn fe_mem(h: *mut c_void, out: *mut u64) -> c_int;
    fn fe_device_name(buf: *mut c_char, n: c_int) -> c_int;
    fn fe_smem_optin() -> c_int;
    fn fe_free(h: *mut c_void);
}

fn ck(r: c_int) -> Result<()> {
    if r == 0 {
        Ok(())
    } else {
        Err(anyhow!("cuda engine: {}", unsafe { CStr::from_ptr(fe_error()) }.to_string_lossy()))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arch {
    Gemma,
    Neomme,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dtype {
    F32,
    Bf16,
}

pub struct Engine {
    h: *mut c_void,
    pub arch: Arch,
    pub dtype: Dtype,
    pub max_tokens: usize,
    pub max_patches: usize,
    pub out_dim: usize,
}

unsafe impl Send for Engine {}

impl Drop for Engine {
    fn drop(&mut self) {
        unsafe { fe_free(self.h) }
    }
}

pub struct MemInfo {
    pub weights: u64,
    pub arena: u64,
    pub device_used: u64,
    pub device_total: u64,
}

/// Opt-in shared memory per block (bytes) of the current GPU.
pub fn smem_optin() -> usize {
    unsafe { fe_smem_optin() }.max(0) as usize
}

pub fn device_name() -> String {
    let mut b = vec![0 as c_char; 256];
    if unsafe { fe_device_name(b.as_mut_ptr(), 256) } != 0 {
        return String::new();
    }
    unsafe { CStr::from_ptr(b.as_ptr()) }.to_string_lossy().to_string()
}

/// Reads a safetensors file and calls `f(name, dtype, shape, bytes)` for every tensor whose name passes `keep`.
fn for_tensors(path: &Path, keep: &dyn Fn(&str) -> bool, f: &mut dyn FnMut(&str, &str, &[i64], &[u8]) -> Result<()>) -> Result<()> {
    let file = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let map = unsafe { memmap2::Mmap::map(&file)? };
    let n = u64::from_le_bytes(map[..8].try_into()?) as usize;
    let header: Value = serde_json::from_slice(&map[8..8 + n])?;
    let base = 8 + n;
    for (name, v) in header.as_object().context("safetensors header")? {
        if name == "__metadata__" || !keep(name) {
            continue;
        }
        let dtype = v["dtype"].as_str().context("dtype")?;
        let shape: Vec<i64> = v["shape"].as_array().context("shape")?.iter().map(|x| x.as_i64().unwrap_or(0)).collect();
        let off = v["data_offsets"].as_array().context("offsets")?;
        let (a, b) = (off[0].as_u64().unwrap_or(0) as usize, off[1].as_u64().unwrap_or(0) as usize);
        f(name, dtype, &shape, &map[base + a..base + b])?;
    }
    Ok(())
}

impl Engine {
    /// Load a model from its HF snapshot. `vision`: also build EmbeddingGemma 2's vision tower.
    /// `img_bytes` > 0: GPU image preprocessing (EmbeddingGemma 2) with this many bytes of decoded images per batch.
    /// `quant`: 0 = off, 1 = MXFP8, 2 = NVFP4 for the layer GEMMs (bf16 only). `audio_frames` > 0: also build
    /// EmbeddingGemma 2's audio tower, with that many log-mel frames per forward.
    #[allow(clippy::too_many_arguments)]
    pub fn load(snapshot: &Path, arch: Arch, dtype: Dtype, vision: bool, max_tokens: usize, max_patches: usize, img_bytes: usize, quant: u32, audio_frames: usize) -> Result<Self> {
        let cfg: Value = serde_json::from_str(&std::fs::read_to_string(snapshot.join("config.json"))?)?;
        let h = unsafe {
            fe_new(match arch { Arch::Gemma => 0, Arch::Neomme => 1 }, match dtype { Dtype::F32 => 0, Dtype::Bf16 => 1 }, 0)
        };
        if h.is_null() {
            bail!("cuda engine init: {}", unsafe { CStr::from_ptr(fe_error()) }.to_string_lossy());
        }
        let mut e = Engine { h, arch, dtype, max_tokens, max_patches, out_dim: 0 };
        let set = |k: &str, v: f64| -> Result<()> {
            let c = CString::new(k)?;
            ck(unsafe { fe_cfg(h, c.as_ptr(), v) })
        };
        set("quant", quant as f64)?;
        let num = |v: &Value, k: &str| -> Result<f64> { v[k].as_f64().with_context(|| format!("config: {k}")) };
        match arch {
            Arch::Gemma => {
                let t = &cfg["text_config"];
                set("hidden", num(t, "hidden_size")?)?;
                set("layers", num(t, "num_hidden_layers")?)?;
                set("heads", num(t, "num_attention_heads")?)?;
                set("inter", num(t, "intermediate_size")?)?;
                set("ple", num(t, "hidden_size_per_layer_input")?)?;
                set("eps", num(t, "rms_norm_eps")?)?;
                set("sliding_window", num(t, "sliding_window")?)?;
                set("rope_theta_full", num(&t["rope_parameters"]["full_attention"], "rope_theta")?)?;
                set("rope_theta_sliding", num(&t["rope_parameters"]["sliding_attention"], "rope_theta")?)?;
                for (i, lt) in t["layer_types"].as_array().context("layer_types")?.iter().enumerate() {
                    set(&format!("full.{i}"), if lt.as_str() == Some("full_attention") { 1.0 } else { 0.0 })?;
                }
                e.out_dim = num(t, "embedding_dim")? as usize;
                if audio_frames > 0 {
                    let a = &cfg["audio_config"];
                    set("audio", 1.0)?;
                    set("max_frames", audio_frames as f64)?;
                    for (k, j) in [("a_hidden", "hidden_size"), ("a_heads", "num_attention_heads"), ("a_layers", "num_hidden_layers"),
                        ("a_chunk", "attention_chunk_size"), ("a_left", "attention_context_left"), ("a_right", "attention_context_right"),
                        ("a_kernel", "conv_kernel_size"), ("a_softcap", "attention_logit_cap"), ("a_invalid", "attention_invalid_logits_value"),
                        ("a_resw", "residual_weight"), ("a_eps", "rms_norm_eps")]
                    {
                        set(k, num(a, j)?)?;
                    }
                    if a["subsampling_conv_channels"] != serde_json::json!([128, 32]) || a["hidden_act"].as_str() != Some("silu") {
                        bail!("unsupported Gemma 4 audio config");
                    }
                }
                if vision {
                    let v = &cfg["vision_config"];
                    set("vision", 1.0)?;
                    set("img_bytes", img_bytes as f64)?;
                    set("v_hidden", num(v, "hidden_size")?)?;
                    set("v_layers", num(v, "num_hidden_layers")?)?;
                    set("v_heads", num(v, "num_attention_heads")?)?;
                    set("v_inter", num(v, "intermediate_size")?)?;
                    set("v_eps", num(v, "rms_norm_eps")?)?;
                    set("v_rope_theta", num(&v["rope_parameters"], "rope_theta")?)?;
                    if v["pooling_kernel_size"].as_u64() != Some(3) || v["use_clipped_linears"].as_bool() == Some(true)
                        || v["standardize"].as_bool() == Some(true)
                    {
                        bail!("unsupported vision config (pooling kernel 3, no clipped linears, no standardize expected)");
                    }
                }
            }
            Arch::Neomme => {
                set("hidden", num(&cfg, "hidden_size")?)?;
                set("layers", num(&cfg, "num_hidden_layers")?)?;
                set("heads", num(&cfg, "num_attention_heads")?)?;
                set("inter", num(&cfg, "intermediate_size")?)?;
                set("eps", num(&cfg, "norm_eps")?)?;
                set("resid_mult", num(&cfg, "residual_multiplier")?)?;
                set("head_dim", num(&cfg, "head_dim")?)?;
                if cfg["hidden_act"].as_str() != Some("relu2") || cfg["mlp_bias"].as_bool() == Some(true)
                    || cfg["attention_bias"].as_bool() == Some(true)
                {
                    bail!("unsupported NeoMME config");
                }
                let rp = &cfg["rope_parameters"];
                set("rope_theta_full", num(&rp["full_attention"], "rope_theta")?)?;
                set("rope_theta_sliding", num(&rp["sliding_attention"], "rope_theta")?)?;
                set("rope_frac_full", rp["full_attention"]["partial_rotary_factor"].as_f64().unwrap_or(1.0))?;
                set("rope_frac_sliding", rp["sliding_attention"]["partial_rotary_factor"].as_f64().unwrap_or(1.0))?;
                let sw = num(&cfg, "sliding_window")?;
                let types = cfg["layer_types"].as_array().context("layer_types")?;
                let full: Vec<usize> = types.iter().enumerate().filter(|(_, t)| t.as_str() == Some("full_attention")).map(|(i, _)| i).collect();
                for (i, lt) in types.iter().enumerate() {
                    let w = if lt.as_str() == Some("full_attention") {
                        -1.0
                    } else {
                        cfg["per_layer_config"][format!("{i:02}")]["sliding_window"].as_f64().unwrap_or(sw)
                    };
                    set(&format!("window.{i}"), w)?;
                    // value embeddings feed the first and the last global layer (NeoMMEModel.value_embedding_layers)
                    set(&format!("vemb.{i}"), if Some(&i) == full.first() || Some(&i) == full.last() { 1.0 } else { 0.0 })?;
                }
                e.out_dim = num(&cfg, "hidden_size")? as usize;
            }
        }
        let keep = |n: &str| match arch {
            Arch::Gemma => {
                n.starts_with("language_model.")
                    || (vision && (n.starts_with("vision_tower.") || n.starts_with("embed_vision.")))
                    || (audio_frames > 0 && (n.starts_with("audio_tower.") || n.starts_with("embed_audio.")))
            }
            Arch::Neomme => true,
        };
        for_tensors(&snapshot.join("model.safetensors"), &keep, &mut |name, dt, shape, bytes| {
            if dt != "BF16" {
                bail!("{name}: dtype {dt} (bf16 checkpoints expected)");
            }
            let c = CString::new(name)?;
            ck(unsafe { fe_add(h, c.as_ptr(), 1, shape.as_ptr(), shape.len() as c_int, bytes.as_ptr() as *const c_void) })
        })?;
        ck(unsafe { fe_build(h, max_tokens as c_int, max_patches as c_int) })?;
        Ok(e)
    }

    /// EmbeddingGemma 2 backbone on packed sequences; `mm_idx[t]` >= 0 takes soft token mm_idx[t] from the last
    /// `gemma_vision` call. Returns [nseq * out_dim] L2-normalised mean-pooled vectors.
    pub fn gemma_text(&mut self, ids: &[i32], cu: &[i32], mm_idx: Option<&[i32]>) -> Result<Vec<f32>> {
        let nseq = cu.len() - 1;
        let mut out = vec![0f32; nseq * self.out_dim];
        ck(unsafe {
            fe_gemma_text(self.h, ids.len() as c_int, nseq as c_int, ids.as_ptr(), cu.as_ptr(), mm_idx.map_or(std::ptr::null(), |m| m.as_ptr()), out.as_mut_ptr())
        })?;
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn gemma_vision(&mut self, vcu: &[i32], pos: &[i32], patches: &[f32], slot_ptr: &[i32], slot_patch: &[i32]) -> Result<()> {
        let npatch = pos.len() / 2;
        ck(unsafe {
            fe_gemma_vision(
                self.h, npatch as c_int, (vcu.len() - 1) as c_int, vcu.as_ptr(), pos.as_ptr(), patches.as_ptr(),
                (slot_ptr.len() - 1) as c_int, slot_ptr.as_ptr(), slot_patch.as_ptr(),
            )
        })
    }

    /// Vision tower with GPU-side preprocessing (see `ImagePlan`).
    #[allow(clippy::too_many_arguments)]
    pub fn gemma_vision_u8(&mut self, vcu: &[i32], pos: &[i32], src: &[u8], desc: &[i32], coef: &[i32], max_pix: usize, slot_ptr: &[i32], slot_patch: &[i32]) -> Result<()> {
        let npatch = pos.len() / 2;
        ck(unsafe {
            fe_gemma_vision_u8(
                self.h, npatch as c_int, (vcu.len() - 1) as c_int, vcu.as_ptr(), pos.as_ptr(), src.as_ptr(), src.len() as u64,
                desc.as_ptr(), coef.as_ptr(), coef.len() as u64, max_pix as c_int, (slot_ptr.len() - 1) as c_int,
                slot_ptr.as_ptr(), slot_patch.as_ptr(),
            )
        })
    }

    /// Gemma 4 audio tower on packed clips (see cuda/fegpu.cu fe_gemma_audio); leaves soft tokens for gemma_text.
    #[allow(clippy::too_many_arguments)]
    pub fn gemma_audio(&mut self, cu0: &[i32], feats: &[f32], m0: &[i32], rows1: &[i32], m1: &[i32], rows2: &[i32], m2: &[i32], soft_rows: &[i32]) -> Result<()> {
        ck(unsafe {
            fe_gemma_audio(
                self.h, (cu0.len() - 1) as c_int, cu0.as_ptr(), feats.as_ptr(), m0.as_ptr(), m1.len() as c_int, rows1.as_ptr(),
                m1.as_ptr(), m2.len() as c_int, rows2.as_ptr(), m2.as_ptr(), soft_rows.len() as c_int, soft_rows.as_ptr(),
            )
        })
    }

    /// NeoMME: returns (dense [nseq * 1024], late [ntok * 128] when `late`); patches are uint8 (prep::neomme_image_u8)
    #[allow(clippy::too_many_arguments)]
    pub fn neomme(&mut self, ids: &[i32], cu: &[i32], pos2: Option<&[i32]>, patches: &[u8], patch_rows: &[i32], late: bool) -> Result<(Vec<f32>, Vec<f32>)> {
        let nseq = cu.len() - 1;
        let mut dense = vec![0f32; nseq * self.out_dim];
        let mut lv = if late { vec![0f32; ids.len() * 128] } else { vec![] };
        ck(unsafe {
            fe_neomme(
                self.h, ids.len() as c_int, nseq as c_int, ids.as_ptr(), cu.as_ptr(), pos2.map_or(std::ptr::null(), |p| p.as_ptr()),
                patch_rows.len() as c_int, if patches.is_empty() { std::ptr::null() } else { patches.as_ptr() }, patch_rows.as_ptr(),
                dense.as_mut_ptr(), if late { lv.as_mut_ptr() } else { std::ptr::null_mut() },
            )
        })?;
        Ok((dense, lv))
    }

    pub fn mem(&self) -> Result<MemInfo> {
        let mut o = [0u64; 4];
        ck(unsafe { fe_mem(self.h, o.as_mut_ptr()) })?;
        Ok(MemInfo { weights: o[0], arena: o[1], device_used: o[2], device_total: o[3] })
    }
}

/// GPU resize plan for one EmbeddingGemma 2 image: the HF Gemma 4 processor's target size and the Pillow/ATen
/// bicubic coefficients (`pil_resize::coeffs`), in the descriptor layout `cuda/fegpu.cu` expects
/// (D_SRC, D_W, D_H, D_TW, D_TH, D_TMP, D_ROWS, D_YB0, D_CH, D_KSH, D_PH, D_CV, D_KSV, D_PV, D_PATCH, D_NEEDH, D_NEEDV).
pub struct ImagePlan {
    pub tw: usize,
    pub th: usize,
    pub tmp_bytes: usize,
    pub max_pix: usize,
    desc: [i32; 20],
    coef: Vec<i32>,
}

pub const DESC_N: usize = 20;

impl ImagePlan {
    pub fn new(w: usize, h: usize, max_patches: usize) -> Result<Self> {
        use crate::pil_resize::{Filter, coeffs};
        let (th, tw) = crate::prep::gemma_target_size(h, w, max_patches)?;
        let (need_h, need_v) = (tw != w, th != h);
        let (ksv, bv, kv, pv) = coeffs(h, th, Filter::Bicubic);
        let (yb0, yb1) = (bv[0].0, bv[th - 1].0 + bv[th - 1].1);
        let rows = if need_h { yb1 - yb0 } else { h };
        let mut coef = vec![];
        let mut desc = [0i32; 20];
        if need_h {
            let (ksh, bh, kh, ph) = coeffs(w, tw, Filter::Bicubic);
            desc[8] = 0;
            desc[9] = ksh as i32;
            desc[10] = ph as i32;
            for b in &bh {
                coef.push(b.0 as i32);
                coef.push(b.1 as i32);
            }
            coef.extend_from_slice(&kh);
        }
        desc[11] = coef.len() as i32;
        desc[12] = ksv as i32;
        desc[13] = pv as i32;
        for b in &bv {
            coef.push(b.0 as i32);
            coef.push(b.1 as i32);
        }
        coef.extend_from_slice(&kv);
        desc[1] = w as i32;
        desc[2] = h as i32;
        desc[3] = tw as i32;
        desc[4] = th as i32;
        desc[6] = rows as i32;
        desc[7] = yb0 as i32;
        desc[15] = need_h as i32;
        desc[16] = need_v as i32;
        let tmp_bytes = if need_h { tw * rows * 3 } else { 0 };
        Ok(ImagePlan { tw, th, tmp_bytes, max_pix: (tw * th).max(tw * rows), desc, coef })
    }

    /// Append to batch buffers: sets the source / tmp / coefficient / first-patch offsets.
    pub fn append(&self, src_off: usize, tmp_off: usize, patch_off: usize, desc: &mut Vec<i32>, coef: &mut Vec<i32>) {
        let mut d = self.desc;
        let c0 = coef.len() as i32;
        d[0] = src_off as i32;
        d[5] = tmp_off as i32;
        d[8] += c0;
        d[11] += c0;
        d[14] = patch_off as i32;
        desc.extend_from_slice(&d);
        coef.extend_from_slice(&self.coef);
    }
}
