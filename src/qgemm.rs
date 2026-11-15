//! Block-quantized int8 GEMM for the Linear layers: `flashembed::QMatMul` (docs/QUANTIZATION.md).
//!
//! Y[M,N] = X[M,K] W[K,N] with
//! * W: int8, symmetric, one scale per (K-block, column), packed in 16-column panels: for each panel, K/4 groups of
//!   16 columns x 4 consecutive k (64 bytes), the layout `vpdpbusd` / `vpmaddubsw` consume directly;
//! * X: quantized at run time per (row, K-block) to unsigned bytes with a fixed offset: 8 bits (q + 128) on CPUs with
//!   AVX-512 VNNI, 7 bits (q + 64) on AVX2, so `vpmaddubsw`'s int16 pair sums (<= 2 * 127 * 127) cannot saturate;
//! * int32 sums per K-block, corrected for the offset with precomputed column sums, scaled by s_x[row, blk] *
//!   s_w[blk, col] and accumulated in fp32.
//!
//! Per-block activation scales keep channel outliers (large in a few hidden dimensions of these models) from setting
//! the scale of a whole row or tensor, which is what makes per-tensor dynamic int8 lose accuracy here.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub const NR: usize = 16;
/// rows per microtile (VNNI kernel; the AVX2 kernel uses 4, which divides it)
pub const MR: usize = 12;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Isa {
    Vnni,
    Avx2,
    Scalar,
}

pub fn isa() -> Isa {
    #[cfg(target_arch = "x86_64")]
    {
        if std::env::var("FLASHEMBED_QGEMM_ISA").is_ok_and(|v| v == "avx2") && is_x86_feature_detected!("avx2") {
            return Isa::Avx2;
        }
        if std::env::var("FLASHEMBED_QGEMM_ISA").is_ok_and(|v| v == "scalar") {
            return Isa::Scalar;
        }
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vnni") {
            return Isa::Vnni;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return Isa::Avx2;
        }
    }
    Isa::Scalar
}

/// (activation offset, its log2, max |q|) for an ISA
fn act_params(i: Isa) -> (i32, u32, f32) {
    match i {
        Isa::Avx2 => (64, 6, 63.0),
        _ => (128, 7, 127.0),
    }
}

/// Packed weights: (panels, int8 [npad/16][kpad/4][16][4]), scales [nb][npad], column sums [nb][npad].
pub struct Packed {
    pub k: usize,
    pub n: usize,
    pub block: usize,
    pub kpad: usize,
    pub npad: usize,
    pub b: Vec<i8>,
    pub sb: Vec<f32>,
    pub cs: Vec<i32>,
}

/// Borrowed view of packed weights (the op reads them straight from the graph's initializers).
#[derive(Clone, Copy)]
pub struct W<'a> {
    pub k: usize,
    pub n: usize,
    pub block: usize,
    pub kpad: usize,
    pub npad: usize,
    pub b: &'a [i8],
    pub sb: &'a [f32],
    pub cs: &'a [i32],
}

impl Packed {
    pub fn view(&self) -> W<'_> {
        W { k: self.k, n: self.n, block: self.block, kpad: self.kpad, npad: self.npad, b: &self.b, sb: &self.sb, cs: &self.cs }
    }
}

/// Quantize and pack W [K, N] (row-major, as in the ONNX MatMul initializer).
pub fn pack(w: &[f32], k: usize, n: usize, block: usize) -> Packed {
    assert!(block % 4 == 0);
    let kpad = k.next_multiple_of(block);
    let npad = n.next_multiple_of(NR);
    let nb = kpad / block;
    let mut sb = vec![0f32; nb * npad];
    let mut cs = vec![0i32; nb * npad];
    let mut q = vec![0i8; kpad * npad]; // [kpad][npad] first
    for blk in 0..nb {
        for j in 0..n {
            let (k0, k1) = (blk * block, ((blk + 1) * block).min(k));
            let amax = (k0..k1).map(|kk| w[kk * n + j].abs()).fold(0f32, f32::max);
            let s = if amax > 0.0 { amax / 127.0 } else { 1.0 };
            sb[blk * npad + j] = s;
            let mut sum = 0i32;
            for kk in k0..k1 {
                let v = (w[kk * n + j] / s).round().clamp(-127.0, 127.0) as i8;
                q[kk * npad + j] = v;
                sum += v as i32;
            }
            cs[blk * npad + j] = sum;
        }
    }
    let mut b = vec![0i8; kpad * npad];
    let mut o = 0;
    for p in 0..npad / NR {
        for g in 0..kpad / 4 {
            for c in 0..NR {
                for e in 0..4 {
                    b[o] = q[(g * 4 + e) * npad + p * NR + c];
                    o += 1;
                }
            }
        }
    }
    Packed { k, n, block, kpad, npad, b, sb, cs }
}

/// Quantize rows [r0, r1) of X [M, K] into aq [rows][kpad] (u8 with offset) and sa [rows][nb]; rows >= m are padding.
pub fn quantize_rows(x: &[f32], m: usize, k: usize, block: usize, kpad: usize, i: Isa, r0: usize, r1: usize, aq: &mut [u8], sa: &mut [f32]) {
    let (off, _, qmax) = act_params(i);
    let nb = kpad / block;
    for r in r0..r1 {
        let dst = &mut aq[(r - r0) * kpad..(r - r0 + 1) * kpad];
        let sd = &mut sa[(r - r0) * nb..(r - r0 + 1) * nb];
        if r >= m {
            dst.fill(off as u8);
            sd.fill(0.0);
            continue;
        }
        let row = &x[r * k..(r + 1) * k];
        for blk in 0..nb {
            let (k0, k1) = (blk * block, ((blk + 1) * block).min(k));
            #[cfg(target_arch = "x86_64")]
            let amax = if i != Isa::Scalar && k1 - k0 == block && block % 32 == 0 {
                unsafe { absmax_avx2(row.as_ptr().add(k0), block) }
            } else {
                row[k0.min(k)..k1.max(k0.min(k))].iter().fold(0f32, |a, v| a.max(v.abs()))
            };
            #[cfg(not(target_arch = "x86_64"))]
            let amax = row[k0.min(k)..k1.max(k0.min(k))].iter().fold(0f32, |a, v| a.max(v.abs()));
            let s = if amax > 0.0 { amax / qmax } else { 1.0 };
            sd[blk] = if amax > 0.0 { s } else { 0.0 };
            let inv = 1.0 / s;
            #[cfg(target_arch = "x86_64")]
            if i != Isa::Scalar && k1 - k0 == block && block % 32 == 0 {
                unsafe { quant_block_avx2(row.as_ptr().add(k0), block, inv, qmax, off, dst.as_mut_ptr().add(k0)) };
                continue;
            }
            for kk in blk * block..(blk + 1) * block {
                let q = if kk < k { (row[kk] * inv).round().clamp(-qmax, qmax) as i32 } else { 0 };
                dst[kk] = (q + off) as u8;
            }
        }
    }
}

/// q = clamp(round_even(x * inv), -qmax, qmax) + off as u8, 32 values per step (all three CPUs have AVX2).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn quant_block_avx2(x: *const f32, n: usize, inv: f32, qmax: f32, off: i32, dst: *mut u8) {
    unsafe {
        let (vi, lo, hi, vo) = (_mm256_set1_ps(inv), _mm256_set1_ps(-qmax), _mm256_set1_ps(qmax), _mm256_set1_epi32(off));
        let q = |p: *const f32| _mm256_add_epi32(_mm256_cvtps_epi32(_mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(_mm256_loadu_ps(p), vi), lo), hi)), vo);
        for j in (0..n).step_by(32) {
            let (a, b, c, d) = (q(x.add(j)), q(x.add(j + 8)), q(x.add(j + 16)), q(x.add(j + 24)));
            let ab = _mm256_packs_epi32(a, b);
            let cd = _mm256_packs_epi32(c, d);
            // packs/packus interleave 4-byte groups per 128-bit lane: a0 b0 c0 d0 a1 b1 c1 d1 -> a0 a1 b0 b1 ...
            let u = _mm256_permutevar8x32_epi32(_mm256_packus_epi16(ab, cd), _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
            _mm256_storeu_si256(dst.add(j) as *mut _, u);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn absmax_avx2(x: *const f32, n: usize) -> f32 {
    unsafe {
        let mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fff_ffff));
        let mut m = _mm256_setzero_ps();
        for j in (0..n).step_by(8) {
            m = _mm256_max_ps(m, _mm256_and_ps(_mm256_loadu_ps(x.add(j)), mask));
        }
        let mut t = [0f32; 8];
        _mm256_storeu_ps(t.as_mut_ptr(), m);
        t.iter().fold(0f32, |a, &b| a.max(b))
    }
}

/// One microtile: rows aq[0..rows) (stride kpad) x panel `p` -> out (stride ldo), columns [0, ncols) of the panel.
/// `rows` = MR for VNNI, a multiple of 4 for AVX2 (padding rows beyond `valid` are computed but not stored).
#[allow(clippy::too_many_arguments)]
unsafe fn tile(i: Isa, aq: *const u8, sa: *const f32, w: &W, p: usize, out: *mut f32, ldo: usize, valid: usize, ncols: usize) {
    unsafe {
        match i {
            #[cfg(target_arch = "x86_64")]
            Isa::Vnni => tile_vnni(aq, sa, w, p, out, ldo, valid, ncols),
            #[cfg(target_arch = "x86_64")]
            Isa::Avx2 => {
                for r0 in (0..MR).step_by(4) {
                    if r0 >= valid {
                        break;
                    }
                    let nb = w.kpad / w.block;
                    tile_avx2(aq.add(r0 * w.kpad), sa.add(r0 * nb), w, p, out.add(r0 * ldo), ldo, valid - r0, ncols);
                }
            }
            _ => tile_scalar(aq, sa, w, p, out, ldo, valid, ncols),
        }
    }
}

unsafe fn tile_scalar(aq: *const u8, sa: *const f32, w: &W, p: usize, out: *mut f32, ldo: usize, valid: usize, ncols: usize) {
    let (off, _, _) = act_params(Isa::Scalar);
    let nb = w.kpad / w.block;
    for r in 0..valid.min(MR) {
        for c in 0..ncols {
            let col = p * NR + c;
            let mut acc = 0f32;
            for blk in 0..nb {
                let mut s = 0i32;
                for kk in blk * w.block..(blk + 1) * w.block {
                    let a = unsafe { *aq.add(r * w.kpad + kk) } as i32;
                    let b = w.b[(p * (w.kpad / 4) + kk / 4) * NR * 4 + c * 4 + kk % 4] as i32;
                    s += a * b;
                }
                s -= off * w.cs[blk * w.npad + col];
                acc += s as f32 * w.sb[blk * w.npad + col] * unsafe { *sa.add(r * nb + blk) };
            }
            unsafe { *out.add(r * ldo + c) = acc };
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn tile_vnni(aq: *const u8, sa: *const f32, w: &W, p: usize, out: *mut f32, ldo: usize, valid: usize, ncols: usize) {
    unsafe {
        let nb = w.kpad / w.block;
        let groups = w.block / 4;
        let mut facc = [_mm512_setzero_ps(); MR];
        let panel = w.b.as_ptr().add(p * w.kpad * NR);
        for blk in 0..nb {
            let mut acc = [_mm512_setzero_si512(); MR];
            let mut bp = panel.add(blk * w.block * NR);
            let mut ap = aq.add(blk * w.block);
            for _ in 0..groups {
                let b = _mm512_loadu_si512(bp as *const _);
                for r in 0..MR {
                    let a = _mm512_set1_epi32((ap.add(r * w.kpad) as *const i32).read_unaligned());
                    acc[r] = _mm512_dpbusd_epi32(acc[r], a, b);
                }
                bp = bp.add(64);
                ap = ap.add(4);
            }
            let c = _mm512_slli_epi32::<7>(_mm512_loadu_si512(w.cs.as_ptr().add(blk * w.npad + p * NR) as *const _));
            let s = _mm512_loadu_ps(w.sb.as_ptr().add(blk * w.npad + p * NR));
            for r in 0..MR {
                let t = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(acc[r], c)), s);
                facc[r] = _mm512_fmadd_ps(t, _mm512_set1_ps(*sa.add(r * nb + blk)), facc[r]);
            }
        }
        let mask: __mmask16 = if ncols >= 16 { 0xffff } else { ((1u32 << ncols) - 1) as u16 };
        for r in 0..valid.min(MR) {
            _mm512_mask_storeu_ps(out.add(r * ldo), mask, facc[r]);
        }
    }
}

/// 4 rows x 16 columns; fp32 accumulators kept in a stack buffer (AVX2 has only 16 registers).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn tile_avx2(aq: *const u8, sa: *const f32, w: &W, p: usize, out: *mut f32, ldo: usize, valid: usize, ncols: usize) {
    unsafe {
        let nb = w.kpad / w.block;
        let groups = w.block / 4;
        let mut fbuf = [0f32; 4 * NR];
        let ones = _mm256_set1_epi16(1);
        let panel = w.b.as_ptr().add(p * w.kpad * NR);
        for blk in 0..nb {
            let mut acc = [_mm256_setzero_si256(); 8];
            let mut bp = panel.add(blk * w.block * NR);
            let mut ap = aq.add(blk * w.block);
            for _ in 0..groups {
                let b0 = _mm256_loadu_si256(bp as *const _);
                let b1 = _mm256_loadu_si256(bp.add(32) as *const _);
                for r in 0..4 {
                    let a = _mm256_set1_epi32((ap.add(r * w.kpad) as *const i32).read_unaligned());
                    acc[2 * r] = _mm256_add_epi32(acc[2 * r], _mm256_madd_epi16(_mm256_maddubs_epi16(a, b0), ones));
                    acc[2 * r + 1] = _mm256_add_epi32(acc[2 * r + 1], _mm256_madd_epi16(_mm256_maddubs_epi16(a, b1), ones));
                }
                bp = bp.add(64);
                ap = ap.add(4);
            }
            let cp = w.cs.as_ptr().add(blk * w.npad + p * NR);
            let spp = w.sb.as_ptr().add(blk * w.npad + p * NR);
            for h in 0..2 {
                let c = _mm256_slli_epi32::<6>(_mm256_loadu_si256(cp.add(8 * h) as *const _));
                let s = _mm256_loadu_ps(spp.add(8 * h));
                for r in 0..4 {
                    let t = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(acc[2 * r + h], c)), s);
                    let f = fbuf.as_mut_ptr().add(r * NR + 8 * h);
                    _mm256_storeu_ps(f, _mm256_fmadd_ps(t, _mm256_set1_ps(*sa.add(r * nb + blk)), _mm256_loadu_ps(f)));
                }
            }
        }
        for r in 0..valid.min(4) {
            std::ptr::copy_nonoverlapping(fbuf.as_ptr().add(r * NR), out.add(r * ldo), ncols.min(NR));
        }
    }
}

/// Rows per task and panels per task for the parallel loop.
pub const ROWS_PER_TASK: usize = 4 * MR;
pub const PANELS_PER_TASK: usize = 8;

#[derive(Clone, Copy)]
struct P<T>(*const T);
unsafe impl<T> Send for P<T> {}
unsafe impl<T> Sync for P<T> {}
#[derive(Clone, Copy)]
struct PM<T>(*mut T);
unsafe impl<T> Send for PM<T> {}
unsafe impl<T> Sync for PM<T> {}

/// Y[M, N] = X[M, K] W, parallelised by `par(tasks, f)` (ORT's pool inside the op; a plain loop elsewhere).
pub fn gemm(x: &[f32], m: usize, w: &W<'_>, y: &mut [f32], i: Isa, par: &dyn Fn(usize, &(dyn Fn(usize) + Sync + Send))) {
    let mp = m.next_multiple_of(MR);
    let nb = w.kpad / w.block;
    let mut aq: Vec<u8> = Vec::with_capacity(mp * w.kpad);
    let mut sa: Vec<f32> = Vec::with_capacity(mp * nb);
    // SAFETY: quantize_rows writes every element of both buffers (padding rows included) before the GEMM reads them
    unsafe {
        aq.set_len(mp * w.kpad);
        sa.set_len(mp * nb);
    }
    // 1. quantize activations (parallel over row chunks)
    {
        let chunks = mp.div_ceil(ROWS_PER_TASK);
        let (aqp, sap, xp) = (PM(aq.as_mut_ptr()), PM(sa.as_mut_ptr()), P(x.as_ptr()));
        let (k, kpad, block) = (w.k, w.kpad, w.block);
        par(chunks, &move |t| {
            let (aqp, sap, xp) = (aqp, sap, xp);
            let (r0, r1) = (t * ROWS_PER_TASK, ((t + 1) * ROWS_PER_TASK).min(mp));
            let a = unsafe { std::slice::from_raw_parts_mut(aqp.0.add(r0 * kpad), (r1 - r0) * kpad) };
            let s = unsafe { std::slice::from_raw_parts_mut(sap.0.add(r0 * nb), (r1 - r0) * nb) };
            let xs = unsafe { std::slice::from_raw_parts(xp.0, m * k) };
            quantize_rows(xs, m, k, block, kpad, i, r0, r1, a, s);
        });
    }
    // 2. GEMM over (row chunk, panel chunk) tasks
    let np = w.npad / NR;
    let (rc, pc) = (mp.div_ceil(ROWS_PER_TASK), np.div_ceil(PANELS_PER_TASK));
    let (aqp, sap, yp) = (P(aq.as_ptr()), P(sa.as_ptr()), PM(y.as_mut_ptr()));
    let n = w.n;
    par(rc * pc, &move |t| {
        let (aqp, sap, yp) = (aqp, sap, yp);
        let (ri, pi) = (t / pc, t % pc);
        for p in pi * PANELS_PER_TASK..((pi + 1) * PANELS_PER_TASK).min(np) {
            let ncols = (n - p * NR).min(NR);
            for r0 in (ri * ROWS_PER_TASK..((ri + 1) * ROWS_PER_TASK).min(mp)).step_by(MR) {
                if r0 >= m {
                    break;
                }
                unsafe {
                    tile(i, aqp.0.add(r0 * w.kpad), sap.0.add(r0 * nb), w, p, yp.0.add(r0 * n + p * NR), n, m - r0, ncols);
                }
            }
        }
    });
}

// ------------------------------------------------------------------------------------------------- ORT op
use ort::operator::{ComputeContext, Kernel, KernelContext, Operator, OperatorInput, OperatorOutput};
use ort::value::TensorElementType;

/// `flashembed::QMatMul(X[..., K] f32, B int8 [npad/16, kpad/4, 16, 4], Sb f32 [nb, npad], Cs int32 [nb, npad];
/// K, N, block) -> Y[..., N]` (written by `flashembed quantize --scheme q8blk`).
pub struct QMatMul;

pub struct QMatMulKernel {
    k: usize,
    n: usize,
    block: usize,
    isa: Isa,
}

impl Operator for QMatMul {
    type Kernel<'attr> = QMatMulKernel;
    fn name(&self) -> &str {
        "QMatMul"
    }
    fn inputs(&self) -> impl IntoIterator<Item = OperatorInput> {
        [
            OperatorInput::required(TensorElementType::Float32),
            OperatorInput::required(TensorElementType::Int8),
            OperatorInput::required(TensorElementType::Float32),
            OperatorInput::required(TensorElementType::Int32),
        ]
    }
    fn outputs(&self) -> impl IntoIterator<Item = OperatorOutput> {
        [OperatorOutput::required(TensorElementType::Float32)]
    }
    fn create_kernel<'attr>(&self, k: &KernelContext<'attr>) -> ort::Result<Self::Kernel<'attr>> {
        Ok(QMatMulKernel {
            k: k.get::<i64>("K").unwrap_or(0) as usize,
            n: k.get::<i64>("N").unwrap_or(0) as usize,
            block: k.get::<i64>("block").unwrap_or(64) as usize,
            isa: isa(),
        })
    }
}

impl Kernel for QMatMulKernel {
    fn compute(&mut self, ctx: &ComputeContext) -> ort::Result<()> {
        let err = |m: &str| ort::Error::new(m.to_string());
        let inp = |i: usize| ctx.input(i).and_then(|x| x.ok_or_else(|| err("QMatMul: missing input")));
        let (x, b, sb, cs) = (inp(0)?, inp(1)?, inp(2)?, inp(3)?);
        let (xs, xd) = x.try_extract_tensor::<f32>()?;
        let (_, bd) = b.try_extract_tensor::<i8>()?;
        let (_, sbd) = sb.try_extract_tensor::<f32>()?;
        let (_, csd) = cs.try_extract_tensor::<i32>()?;
        let (k, n, block) = (self.k, self.n, self.block);
        if xs.is_empty() || *xs.last().unwrap() as usize != k {
            return Err(err("QMatMul: last input dim != K"));
        }
        let kpad = k.next_multiple_of(block);
        let npad = n.next_multiple_of(NR);
        if bd.len() != kpad * npad || sbd.len() != (kpad / block) * npad || csd.len() != sbd.len() {
            return Err(err("QMatMul: packed weight sizes do not match K/N/block"));
        }
        let m: usize = xs[..xs.len() - 1].iter().map(|&d| d as usize).product();
        let mut shape = xs.to_vec();
        *shape.last_mut().unwrap() = n as i64;
        let mut out = ctx.output(0, shape)?.ok_or_else(|| err("QMatMul: out"))?;
        let (_, yd) = out.try_extract_tensor_mut::<f32>()?;
        let w = W { k, n, block, kpad, npad, b: bd, sb: sbd, cs: csd };
        let status = std::cell::RefCell::new(Ok(()));
        gemm(xd, m, &w, yd, self.isa, &|tasks, f| {
            if status.borrow().is_ok() {
                let r = ctx.par_for(tasks, tasks, |t| f(t));
                *status.borrow_mut() = r;
            }
        });
        status.into_inner()
    }
}
