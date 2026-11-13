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
    fn fe_neomme(
        h: *mut c_void, ntok: c_int, nseq: c_int, ids: *const i32, cu: *const i32, pos2: *const i32, npatch: c_int,
        patches: *const f32, patch_rows: *const i32, dense: *mut f32, late: *mut f32,
    ) -> c_int;
    fn fe_mem(h: *mut c_void, out: *mut u64) -> c_int;
    fn fe_device_name(buf: *mut c_char, n: c_int) -> c_int;
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
    pub fn load(snapshot: &Path, arch: Arch, dtype: Dtype, vision: bool, max_tokens: usize, max_patches: usize) -> Result<Self> {
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
                if vision {
                    let v = &cfg["vision_config"];
                    set("vision", 1.0)?;
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
                n.starts_with("language_model.") || (vision && (n.starts_with("vision_tower.") || n.starts_with("embed_vision.")))
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

    /// NeoMME: returns (dense [nseq * 1024], late [ntok * 128] when `late`)
    #[allow(clippy::too_many_arguments)]
    pub fn neomme(&mut self, ids: &[i32], cu: &[i32], pos2: Option<&[i32]>, patches: &[f32], patch_rows: &[i32], late: bool) -> Result<(Vec<f32>, Vec<f32>)> {
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
