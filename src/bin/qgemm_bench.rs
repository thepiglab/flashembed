//! Linear-layer kernel throughput of the quantized ONNX Runtime ops vs fp32 MLAS at the models' layer shapes
//! (docs/QUANTIZATION.md), plus single-core instruction-throughput probes (fp32 FMA, int8 VNNI, bf16 dot).
//!   qgemm-bench <threads> [MxKxN,...]      (needs ORT_DYLIB_PATH)
//! One-node graphs are built in memory with the same rewrites as `flashembed quantize` (src/quant.rs).
use std::time::Instant;

use flashembed::quant::{self, onnx};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;
use prost::Message;

fn vi(name: &str, dims: &[i64]) -> onnx::ValueInfoProto {
    use onnx::tensor_shape_proto::{Dimension, dimension::Value as D};
    onnx::ValueInfoProto {
        name: Some(name.into()),
        r#type: Some(onnx::TypeProto {
            value: Some(onnx::type_proto::Value::TensorType(onnx::type_proto::Tensor {
                elem_type: Some(1),
                shape: Some(onnx::TensorShapeProto { dim: dims.iter().map(|&d| Dimension { value: Some(D::DimValue(d)), ..Default::default() }).collect() }),
            })),
            ..Default::default()
        }),
        ..Default::default()
    }
}

/// fp32 MatMul graph X[m,k] @ W[k,n], rewritten by `scheme` (fp32 | nb8-a4 | ... | dyn8 | qdq8)
fn model(m: usize, k: usize, n: usize, w: &[f32], scheme: &str, xrange: (f32, f32)) -> anyhow::Result<Vec<u8>> {
    let wt = onnx::TensorProto { name: Some("W".into()), data_type: Some(1), dims: vec![k as i64, n as i64], ..Default::default() };
    let node = onnx::NodeProto { op_type: Some("MatMul".into()), name: Some("mm".into()), input: vec!["X".into(), "W".into()], output: vec!["Y".into()], ..Default::default() };
    let g = onnx::GraphProto {
        name: Some("g".into()), node: vec![node], initializer: vec![wt],
        input: vec![vi("X", &[m as i64, k as i64])], output: vec![vi("Y", &[m as i64, n as i64])], ..Default::default()
    };
    let mdl = onnx::ModelProto {
        ir_version: Some(10), graph: Some(g),
        opset_import: vec![onnx::OperatorSetIdProto { domain: Some(String::new()), version: Some(20) }], ..Default::default()
    };
    let mut l = quant::Loaded { model: mdl, data: [("W".to_string(), w.iter().flat_map(|x| x.to_le_bytes()).collect())].into() };
    if scheme != "fp32" {
        let calib = (scheme == "qdq8").then(|| [("X".to_string(), quant::Range { lo: xrange.0, hi: xrange.1, lo_avg: xrange.0, hi_avg: xrange.1, batches: 1 })].into());
        quant::rewrite(&mut l, &quant::Opts { scheme: scheme.into(), block: if scheme == "q8blk" { 64 } else { 32 }, keep: None, keep_edge: 0, calib, calib_avg: false })?;
    }
    // inline all initializers
    let g = l.model.graph.as_mut().unwrap();
    for t in g.initializer.iter_mut() {
        t.raw_data = Some(l.data.remove(t.name()).unwrap_or_default());
    }
    Ok(l.model.encode_to_vec())
}

#[cfg(target_arch = "x86_64")]
mod probe {
    use std::arch::x86_64::*;
    use std::hint::black_box;
    const IT: usize = 50_000_000;

    /// ops/s (multiply-adds x2) on one core with 8 independent accumulators
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn fma256() -> f64 {
        let (a, b) = (_mm256_set1_ps(1.0001), _mm256_set1_ps(0.9999));
        let mut c = [_mm256_setzero_ps(); 8];
        let t = std::time::Instant::now();
        for _ in 0..IT / 8 {
            for x in c.iter_mut() { *x = _mm256_fmadd_ps(a, b, *x); }
        }
        black_box(c);
        (IT * 8 * 2) as f64 / t.elapsed().as_secs_f64()
    }
    #[target_feature(enable = "avx512f")]
    pub unsafe fn fma512() -> f64 {
        let (a, b) = (_mm512_set1_ps(1.0001), _mm512_set1_ps(0.9999));
        let mut c = [_mm512_setzero_ps(); 8];
        let t = std::time::Instant::now();
        for _ in 0..IT / 8 {
            for x in c.iter_mut() { *x = _mm512_fmadd_ps(a, b, *x); }
        }
        black_box(c);
        (IT * 16 * 2) as f64 / t.elapsed().as_secs_f64()
    }
    #[target_feature(enable = "avx512f,avx512vnni")]
    pub unsafe fn vnni512() -> f64 {
        let (a, b) = (_mm512_set1_epi8(3), _mm512_set1_epi8(2));
        let mut c = [_mm512_setzero_si512(); 8];
        let t = std::time::Instant::now();
        for _ in 0..IT / 8 {
            for x in c.iter_mut() { *x = _mm512_dpbusd_epi32(*x, a, b); }
        }
        black_box(c);
        (IT * 64 * 2) as f64 / t.elapsed().as_secs_f64()
    }
    #[target_feature(enable = "avx512f,avx512bf16")]
    pub unsafe fn bf16_512() -> f64 {
        let a = _mm512_cvtne2ps_pbh(_mm512_set1_ps(1.0001), _mm512_set1_ps(0.9999));
        let mut c = [_mm512_setzero_ps(); 8];
        let t = std::time::Instant::now();
        for _ in 0..IT / 8 {
            for x in c.iter_mut() { *x = _mm512_dpbf16_ps(*x, a, a); }
        }
        black_box(c);
        (IT * 32 * 2) as f64 / t.elapsed().as_secs_f64()
    }
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let t: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(16);
    // EmbeddingGemma 2 text (hidden 512, MLP 2048?) / vision (768, 3072), NeoMME (1024, 3584); M = tokens per forward
    let shapes: Vec<(usize, usize, usize)> = args
        .get(2)
        .map(|s| s.split(',').map(|x| { let v: Vec<usize> = x.split('x').map(|d| d.parse().unwrap()).collect(); (v[0], v[1], v[2]) }).collect())
        .unwrap_or_else(|| vec![(2048, 512, 2048), (2048, 2048, 512), (2520, 768, 3072), (2520, 3072, 768), (2048, 1024, 3584), (2048, 3584, 1024), (2048, 1024, 1024), (64, 1024, 3584)]);
    #[cfg(target_arch = "x86_64")]
    {
        let mut p = serde_json::Map::new();
        unsafe {
            if is_x86_feature_detected!("fma") { p.insert("fp32_fma256_gops".into(), (probe::fma256() / 1e9).into()); }
            if is_x86_feature_detected!("avx512f") { p.insert("fp32_fma512_gops".into(), (probe::fma512() / 1e9).into()); }
            if is_x86_feature_detected!("avx512vnni") { p.insert("int8_vnni512_gops".into(), (probe::vnni512() / 1e9).into()); }
            if is_x86_feature_detected!("avx512bf16") { p.insert("bf16_dp512_gops".into(), (probe::bf16_512() / 1e9).into()); }
        }
        println!("{}", serde_json::json!({"probe": "single-core peak", "results": p}));
    }
    let env = ort::init_from(std::env::var("ORT_DYLIB_PATH")?).map_err(|e| anyhow::anyhow!("{e}"))?.build().map_err(|e| anyhow::anyhow!("{e}"))?;
    let schemes = ["fp32", "dyn8", "dyn8rr", "qdq8", "nb8-a1", "nb8-a4", "nb4-a1", "nb4-a4", "q8blk"];
    for (m, k, n) in shapes {
        // activations with a few large channels, like the models' hidden states
        let x: Vec<f32> = (0..m * k).map(|i| { let v = ((i * 7919) % 1000) as f32 * 2e-3 - 1.0; if i % k < 4 { v * 30.0 } else { v } }).collect();
        let w: Vec<f32> = (0..k * n).map(|i| ((i * 104729) % 1000) as f32 * 1e-4 - 0.05).collect();
        let lo = x.iter().copied().fold(f32::MAX, f32::min);
        let hi = x.iter().copied().fold(f32::MIN, f32::max);
        let mut yref: Vec<f32> = vec![];
        for sch in schemes {
            let bytes = model(m, k, n, &w, sch, (lo, hi))?;
            let mut b = Session::builder(&env).map_err(|e| anyhow::anyhow!("{e}"))?
                .with_optimization_level(GraphOptimizationLevel::Level3).map_err(|e| anyhow::anyhow!("{e}"))?
                .with_intra_threads(t).map_err(|e| anyhow::anyhow!("{e}"))?
                .with_inter_threads(1).map_err(|e| anyhow::anyhow!("{e}"))?
                .with_config_entry("session.intra_op.allow_spinning", "0").map_err(|e| anyhow::anyhow!("{e}"))?
                .with_operators(flashembed::flash::domain().map_err(|e| anyhow::anyhow!("{e}"))?).map_err(|e| anyhow::anyhow!("{e}"))?;
            let mut s = match b.commit_from_memory(&bytes) {
                Ok(s) => s,
                Err(e) => { println!("{}", serde_json::json!({"scheme": sch, "shape": format!("{m}x{k}x{n}"), "error": e.to_string()})); continue; }
            };
            let run = |s: &mut Session| -> anyhow::Result<Vec<f32>> {
                let o = s.run(ort::inputs!["X" => Tensor::from_array(([m, k], x.clone())).map_err(|e| anyhow::anyhow!("{e}"))?]).map_err(|e| anyhow::anyhow!("{e}"))?;
                Ok(o["Y"].try_extract_tensor::<f32>().map_err(|e| anyhow::anyhow!("{e}"))?.1.to_vec())
            };
            let y = run(&mut s)?;
            for _ in 0..2 { run(&mut s)?; }
            let reps = 10;
            let st = Instant::now();
            for _ in 0..reps { run(&mut s)?; }
            let dt = st.elapsed().as_secs_f64() / reps as f64;
            if sch == "fp32" { yref = y.clone(); }
            let num: f64 = y.iter().zip(&yref).map(|(a, b)| ((a - b) as f64).powi(2)).sum();
            let den: f64 = yref.iter().map(|b| (*b as f64).powi(2)).sum();
            println!("{}", serde_json::json!({"scheme": sch, "threads": t, "shape": format!("{m}x{k}x{n}"), "ms": dt * 1e3,
                "gflops": 2.0 * (m * k * n) as f64 / dt / 1e9, "rel_err": (num / den.max(1e-30)).sqrt()}));
        }
    }
    Ok(())
}
