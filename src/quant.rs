//! Quantized variants of the exported fp32 graphs (experimental; docs/QUANTIZATION.md).
//!
//! `flashembed quantize --src <fp32 dir> --out <dir> --scheme <s>` rewrites every MatMul whose B input is a constant
//! weight [K, N] (the models' Linear layers) and writes a graph directory with the same file names, so the runner
//! uses it unchanged (`--onnx <dir>`). Attention (flashembed custom ops), norms, embeddings (Gather), RoPE and pooling
//! stay fp32. Schemes:
//!
//! * `nb8-a4`, `nb8-a1`, `nb4-a4`, `nb4-a1`: weight-only `com.microsoft::MatMulNBits`, symmetric block-wise int8/int4
//!   weights (`--block`, default 32 along K). `-aL` = ORT accuracy level: 1 = fp32 compute on dequantized weights,
//!   4 = int8 compute (activations quantized per block of K at run time).
//! * `dyn8`, `dyn8rr`: `com.microsoft::DynamicQuantizeMatMul`: int8 per-output-channel weights, activations quantized
//!   per tensor (uint8, asymmetric) at run time. `rr` = reduce-range 7-bit weights (ORT's advice for AVX2 without VNNI).
//! * `qdq8`: static int8: activation ranges from `flashembed calib` (`--calib`), `QuantizeLinear` +
//!   `com.microsoft::MatMulIntegerToFloat`, int8 per-output-channel weights.
//! * `q8blk`: `flashembed::QMatMul` (src/qgemm.rs, Rust kernel): int8 weights with a scale per (K-block, column),
//!   activations quantized per (row, K-block) at run time (8 bits with VNNI, 7 bits on AVX2); `--block` 64 recommended.
//! * `calib`: the fp32 graph with every selected MatMul's activation input added as a graph output (for `calib`).
//!
//! Selection: a MatMul is left fp32 if its torch module path (from the exporter's `pkg.torch.onnx.name_scopes`
//! metadata) matches `--keep <regex>`, or if it is in the first/last `--keep-edge` layers of its layer stack.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use prost::Message;

pub mod onnx {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}
use onnx::{AttributeProto, GraphProto, ModelProto, NodeProto, OperatorSetIdProto, TensorProto, ValueInfoProto};

pub const GRAPHS: [&str; 6] = ["gemma_lm", "gemma_vision", "gemma_vision_dyn", "gemma_audio", "neomme_text", "neomme_image"];

const F32: i32 = 1;
const U8: i32 = 2;
const I8: i32 = 3;
const I32: i32 = 6;

/// A loaded graph plus the bytes of its initializers (resolved from raw_data, float_data or external data).
pub struct Loaded {
    pub model: ModelProto,
    pub data: HashMap<String, Vec<u8>>,
}

pub fn load(path: &Path) -> Result<Loaded> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let model = ModelProto::decode(bytes.as_slice())?;
    let dir = path.parent().unwrap_or(Path::new("."));
    let mut files: HashMap<String, Vec<u8>> = HashMap::new();
    let mut data = HashMap::new();
    let g = model.graph.as_ref().ok_or_else(|| anyhow!("no graph"))?;
    for t in &g.initializer {
        let name = t.name().to_string();
        let raw = if t.data_location == Some(1) {
            let kv: HashMap<&str, &str> = t.external_data.iter().map(|e| (e.key(), e.value())).collect();
            let loc = kv.get("location").ok_or_else(|| anyhow!("{name}: external without location"))?;
            if !files.contains_key(*loc) {
                files.insert(loc.to_string(), std::fs::read(dir.join(loc))?);
            }
            let f = &files[*loc];
            let off: usize = kv.get("offset").map(|s| s.parse()).transpose()?.unwrap_or(0);
            let len: usize = kv.get("length").map(|s| s.parse()).transpose()?.unwrap_or(f.len() - off);
            f[off..off + len].to_vec()
        } else if let Some(r) = &t.raw_data {
            r.clone()
        } else if !t.float_data.is_empty() {
            t.float_data.iter().flat_map(|x| x.to_le_bytes()).collect()
        } else if !t.int64_data.is_empty() {
            t.int64_data.iter().flat_map(|x| x.to_le_bytes()).collect()
        } else if !t.int32_data.is_empty() {
            // int32_data also carries bool / int8 / uint8 / int16 / f16 element types, one per entry
            let w = elem_size(t.data_type());
            t.int32_data.iter().flat_map(|x| x.to_le_bytes()[..w].to_vec()).collect()
        } else if !t.double_data.is_empty() {
            t.double_data.iter().flat_map(|x| x.to_le_bytes()).collect()
        } else {
            vec![]
        };
        data.insert(name, raw);
    }
    Ok(Loaded { model, data })
}

fn elem_size(dt: i32) -> usize {
    match dt {
        1 | 6 | 12 => 4,
        2 | 3 | 9 => 1,
        4 | 5 | 10 | 16 => 2,
        7 | 11 | 13 => 8,
        _ => 1,
    }
}

/// Write `model` to `out` with all initializers > 1 KiB in one external file `<out>.data` (64-byte aligned).
pub fn save(mut model: ModelProto, mut data: HashMap<String, Vec<u8>>, out: &Path) -> Result<u64> {
    let loc = format!("{}.data", out.file_name().unwrap().to_string_lossy());
    let tmp_data = out.with_file_name(format!(".{loc}.tmp"));
    let mut f = std::io::BufWriter::new(std::fs::File::create(&tmp_data)?);
    let mut off = 0u64;
    let g = model.graph.as_mut().unwrap();
    for t in g.initializer.iter_mut() {
        let raw = data.remove(t.name()).unwrap_or_default();
        t.float_data.clear();
        t.int32_data.clear();
        t.int64_data.clear();
        t.double_data.clear();
        t.external_data.clear();
        if raw.len() <= 1024 {
            t.raw_data = Some(raw);
            t.data_location = Some(0);
            continue;
        }
        let pad = (64 - off % 64) % 64;
        f.write_all(&vec![0u8; pad as usize])?;
        off += pad;
        f.write_all(&raw)?;
        t.raw_data = None;
        t.data_location = Some(1);
        t.external_data = vec![kv("location", &loc), kv("offset", &off.to_string()), kv("length", &raw.len().to_string())];
        off += raw.len() as u64;
    }
    f.flush()?;
    drop(f);
    std::fs::rename(&tmp_data, out.with_file_name(&loc))?;
    let bytes = model.encode_to_vec();
    std::fs::write(out, &bytes)?;
    Ok(off + bytes.len() as u64)
}

fn kv(k: &str, v: &str) -> onnx::StringStringEntryProto {
    onnx::StringStringEntryProto { key: Some(k.into()), value: Some(v.into()) }
}

/// Torch module path of an exported node ('lm.layers.3.mlp.down_proj'), from the dynamo exporter's metadata.
pub fn module_path(n: &NodeProto) -> String {
    for p in &n.metadata_props {
        if p.key() == "pkg.torch.onnx.name_scopes" {
            let v = p.value().trim().trim_start_matches('[').trim_end_matches(']');
            let parts: Vec<String> = v.split(", ").map(|s| s.trim_matches(|c| c == '\'' || c == '"').to_string()).collect();
            if parts.len() >= 2 {
                return parts[parts.len() - 2].clone();
            }
        }
    }
    String::new()
}

pub struct Selection {
    /// node name -> module path, for all constant-weight MatMuls
    pub paths: BTreeMap<String, String>,
    pub chosen: BTreeSet<String>,
    pub kept: Vec<String>,
}

pub fn select(g: &GraphProto, keep: Option<&regex::Regex>, keep_edge: usize) -> Selection {
    let inits: BTreeSet<&str> = g.initializer.iter().map(|t| t.name()).collect();
    let layer = regex::Regex::new(r"^(.*?layers)\.(\d+)\.").unwrap();
    let mut paths = BTreeMap::new();
    for n in &g.node {
        if n.op_type() == "MatMul" && n.input.len() == 2 && inits.contains(n.input[1].as_str()) {
            paths.insert(n.name().to_string(), module_path(n));
        }
    }
    let mut stacks: BTreeMap<String, BTreeSet<usize>> = BTreeMap::new();
    for p in paths.values() {
        if let Some(c) = layer.captures(p) {
            stacks.entry(c[1].to_string()).or_default().insert(c[2].parse().unwrap());
        }
    }
    let (mut chosen, mut kept) = (BTreeSet::new(), vec![]);
    for (name, p) in &paths {
        let mut skip = keep.is_some_and(|r| r.is_match(p));
        if keep_edge > 0 {
            if let Some(c) = layer.captures(p) {
                let idx: Vec<usize> = stacks[&c[1]].iter().copied().collect();
                let i: usize = c[2].parse().unwrap();
                let n = idx.len();
                if idx[..keep_edge.min(n)].contains(&i) || idx[n.saturating_sub(keep_edge)..].contains(&i) {
                    skip = true;
                }
            }
        }
        if skip { kept.push(p.clone()) } else { chosen.insert(name.clone()); }
    }
    Selection { paths, chosen, kept }
}

fn f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn tensor(name: &str, dt: i32, dims: &[i64]) -> TensorProto {
    TensorProto { name: Some(name.into()), data_type: Some(dt), dims: dims.to_vec(), ..Default::default() }
}

fn attr_i(name: &str, v: i64) -> AttributeProto {
    AttributeProto { name: Some(name.into()), i: Some(v), r#type: Some(2), ..Default::default() }
}

fn node(op: &str, domain: &str, name: &str, inputs: &[&str], outputs: &[&str], attrs: Vec<AttributeProto>) -> NodeProto {
    NodeProto {
        op_type: Some(op.into()),
        domain: if domain.is_empty() { None } else { Some(domain.into()) },
        name: Some(name.into()),
        input: inputs.iter().map(|s| s.to_string()).collect(),
        output: outputs.iter().map(|s| s.to_string()).collect(),
        attribute: attrs,
        ..Default::default()
    }
}

/// Weight-only block quantization for MatMulNBits: W [K, N] -> B uint8 [N, K/blk, blk*bits/8], scales [N * K/blk].
/// Symmetric: q = round(w / s) in [-(2^(b-1)-1), 2^(b-1)-1], stored + 2^(b-1) (the op's default zero point).
pub fn pack_nbits(w: &[f32], k: usize, n: usize, bits: usize, blk: usize) -> (Vec<u8>, Vec<f32>) {
    let nb = k.div_ceil(blk);
    let blob = blk * bits / 8;
    let qmax = ((1 << (bits - 1)) - 1) as f32;
    let zp = 1i32 << (bits - 1);
    let mut b = vec![0u8; n * nb * blob];
    let mut sc = vec![0f32; n * nb];
    for j in 0..n {
        for kb in 0..nb {
            let (k0, k1) = (kb * blk, ((kb + 1) * blk).min(k));
            let amax = (k0..k1).map(|kk| w[kk * n + j].abs()).fold(0f32, f32::max);
            let s = if amax > 0.0 { amax / qmax } else { 1.0 };
            sc[j * nb + kb] = s;
            let base = (j * nb + kb) * blob;
            for e in 0..blk {
                let kk = k0 + e;
                let q = if kk < k1 { ((w[kk * n + j] / s).round() as i32).clamp(-(qmax as i32), qmax as i32) + zp } else { zp };
                if bits == 8 {
                    b[base + e] = q as u8;
                } else {
                    let byte = &mut b[base + e / 2];
                    if e % 2 == 0 { *byte |= q as u8 & 0x0f } else { *byte |= (q as u8 & 0x0f) << 4 }
                }
            }
        }
    }
    (b, sc)
}

/// int8 per-output-channel symmetric weights: W [K, N] -> (q [K, N], scales [N]); qmax 127 (63 with reduce range).
pub fn quant_cols(w: &[f32], k: usize, n: usize, qmax: f32) -> (Vec<u8>, Vec<f32>) {
    let mut sc = vec![0f32; n];
    for j in 0..n {
        let amax = (0..k).map(|kk| w[kk * n + j].abs()).fold(0f32, f32::max);
        sc[j] = if amax > 0.0 { amax / qmax } else { 1.0 };
    }
    let q = (0..k * n).map(|i| ((w[i] / sc[i % n]).round().clamp(-qmax, qmax) as i8) as u8).collect();
    (q, sc)
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, Default)]
pub struct Range {
    pub lo: f32,
    pub hi: f32,
    /// mean over batches of each batch's min / max (ORT's moving-average-like estimate; less outlier-driven)
    pub lo_avg: f32,
    pub hi_avg: f32,
    pub batches: usize,
}

pub struct Opts {
    pub scheme: String,
    pub block: usize,
    pub keep: Option<regex::Regex>,
    pub keep_edge: usize,
    /// qdq8: tensor name -> activation range (from `flashembed calib`), and whether to use the batch-averaged range
    pub calib: Option<HashMap<String, Range>>,
    pub calib_avg: bool,
    /// q8blk: fuse MatMuls that read the same activation (q/k/v, gate/up) into one QMatMul + Split
    pub fuse: bool,
}

/// Rewrite one graph; returns (quantized MatMuls, module paths kept fp32).
pub fn rewrite(l: &mut Loaded, o: &Opts) -> Result<(usize, Vec<String>)> {
    let g = l.model.graph.as_mut().unwrap();
    let sel = select(g, o.keep.as_ref(), o.keep_edge);
    let dims: HashMap<String, Vec<i64>> = g.initializer.iter().map(|t| (t.name().to_string(), t.dims.clone())).collect();
    let nb_re = regex::Regex::new(r"^nb(\d)-a(\d)$").unwrap();
    let mut nodes = Vec::with_capacity(g.node.len());
    let mut inits = vec![];
    let mut extra_out = vec![];
    let mut done = 0;
    let old: Vec<NodeProto> = std::mem::take(&mut g.node);
    // q8blk --fuse: chosen MatMuls sharing an activation input -> one QMatMul over concatenated weights + Split
    let mut groups: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    if o.fuse && o.scheme == "q8blk" {
        for (i, n) in old.iter().enumerate() {
            if sel.chosen.contains(n.name()) {
                groups.entry(n.input[0].clone()).or_default().push(i);
            }
        }
        groups.retain(|_, v| v.len() > 1);
    }
    let first_of: HashMap<usize, String> = groups.iter().map(|(a, v)| (v[0], a.clone())).collect();
    let in_group: BTreeSet<usize> = groups.values().flat_map(|v| v[1..].iter().copied()).collect();
    for (idx, n) in old.iter().enumerate() {
        if in_group.contains(&idx) {
            continue; // emitted with its group's first member
        }
        if let Some(a) = first_of.get(&idx) {
            let mem: Vec<&NodeProto> = groups[a].iter().map(|&i| &old[i]).collect();
            let k = dims[&mem[0].input[1]][0] as usize;
            let ns: Vec<usize> = mem.iter().map(|m| dims[&m.input[1]][1] as usize).collect();
            let ntot: usize = ns.iter().sum();
            let mut w = vec![0f32; k * ntot];
            let mut c0 = 0;
            for (m, &nn) in mem.iter().zip(&ns) {
                let wm = f32s(&l.data[&m.input[1]]);
                for kk in 0..k {
                    w[kk * ntot + c0..kk * ntot + c0 + nn].copy_from_slice(&wm[kk * nn..(kk + 1) * nn]);
                }
                c0 += nn;
            }
            let nm = format!("{}_fused", mem[0].name());
            let pk = crate::qgemm::pack(&w, k, ntot, o.block);
            let nb = pk.kpad / pk.block;
            let (bn, sn, cn, spn, yt) = (format!("{nm}_qB"), format!("{nm}_qS"), format!("{nm}_qC"), format!("{nm}_split"), format!("{nm}_y"));
            inits.push((tensor(&bn, I8, &[(pk.npad / crate::qgemm::NR) as i64, (pk.kpad / 4) as i64, crate::qgemm::NR as i64, 4]),
                        pk.b.iter().map(|&x| x as u8).collect()));
            inits.push((tensor(&sn, F32, &[nb as i64, pk.npad as i64]), pk.sb.iter().flat_map(|x| x.to_le_bytes()).collect()));
            inits.push((tensor(&cn, I32, &[nb as i64, pk.npad as i64]), pk.cs.iter().flat_map(|x| x.to_le_bytes()).collect()));
            inits.push((tensor(&spn, 7, &[ns.len() as i64]), ns.iter().flat_map(|&x| (x as i64).to_le_bytes()).collect()));
            let attrs = vec![attr_i("K", k as i64), attr_i("N", ntot as i64), attr_i("block", o.block as i64)];
            nodes.push(node("QMatMul", "flashembed", &format!("{nm}_q8"), &[a, &bn, &sn, &cn], &[&yt], attrs));
            let outs: Vec<&str> = mem.iter().map(|m| m.output[0].as_str()).collect();
            nodes.push(node("Split", "", &format!("{nm}_split_node"), &[&yt, &spn], &outs, vec![attr_i("axis", -1)]));
            done += mem.len();
            continue;
        }
        let n = n.clone();
        if !sel.chosen.contains(n.name()) {
            nodes.push(n);
            continue;
        }
        let (a, wname, y, nm) = (n.input[0].clone(), n.input[1].clone(), n.output[0].clone(), n.name().to_string());
        let d = &dims[&wname];
        let (k, nn) = (d[0] as usize, d[1] as usize);
        if o.scheme == "calib" {
            extra_out.push(a.clone());
            nodes.push(n);
            done += 1;
            continue;
        }
        let w = f32s(&l.data[&wname]);
        if let Some(c) = nb_re.captures(&o.scheme) {
            let (bits, level): (usize, i64) = (c[1].parse()?, c[2].parse()?);
            let (b, sc) = pack_nbits(&w, k, nn, bits, o.block);
            let nbk = k.div_ceil(o.block);
            let (bn, sn) = (format!("{nm}_qB"), format!("{nm}_qS"));
            inits.push((tensor(&bn, U8, &[nn as i64, nbk as i64, (o.block * bits / 8) as i64]), b));
            inits.push((tensor(&sn, F32, &[(nn * nbk) as i64]), sc.iter().flat_map(|x| x.to_le_bytes()).collect()));
            let attrs = vec![attr_i("K", k as i64), attr_i("N", nn as i64), attr_i("bits", bits as i64),
                             attr_i("block_size", o.block as i64), attr_i("accuracy_level", level)];
            nodes.push(node("MatMulNBits", "com.microsoft", &format!("{nm}_nbits"), &[&a, &bn, &sn], &[&y], attrs));
        } else if o.scheme == "q8blk" {
            // flashembed::QMatMul (src/qgemm.rs): int8 weights per (K-block, column), activations per (row, K-block)
            let pk = crate::qgemm::pack(&w, k, nn, o.block);
            let nb = pk.kpad / pk.block;
            let (bn, sn, cn) = (format!("{nm}_qB"), format!("{nm}_qS"), format!("{nm}_qC"));
            inits.push((tensor(&bn, I8, &[(pk.npad / crate::qgemm::NR) as i64, (pk.kpad / 4) as i64, crate::qgemm::NR as i64, 4]),
                        pk.b.iter().map(|&x| x as u8).collect()));
            inits.push((tensor(&sn, F32, &[nb as i64, pk.npad as i64]), pk.sb.iter().flat_map(|x| x.to_le_bytes()).collect()));
            inits.push((tensor(&cn, I32, &[nb as i64, pk.npad as i64]), pk.cs.iter().flat_map(|x| x.to_le_bytes()).collect()));
            let attrs = vec![attr_i("K", k as i64), attr_i("N", nn as i64), attr_i("block", o.block as i64)];
            nodes.push(node("QMatMul", "flashembed", &format!("{nm}_q8"), &[&a, &bn, &sn, &cn], &[&y], attrs));
        } else if o.scheme == "dyn8" || o.scheme == "dyn8rr" {
            let (q, sc) = quant_cols(&w, k, nn, if o.scheme == "dyn8rr" { 63.0 } else { 127.0 });
            let (bn, sn, zn) = (format!("{nm}_qB"), format!("{nm}_qS"), format!("{nm}_qZ"));
            inits.push((tensor(&bn, I8, &[k as i64, nn as i64]), q));
            inits.push((tensor(&sn, F32, &[nn as i64]), sc.iter().flat_map(|x| x.to_le_bytes()).collect()));
            inits.push((tensor(&zn, I8, &[nn as i64]), vec![0u8; nn]));
            nodes.push(node("DynamicQuantizeMatMul", "com.microsoft", &format!("{nm}_dq"), &[&a, &bn, &sn, &zn], &[&y], vec![]));
        } else if o.scheme == "qdq8" {
            let Some(r) = o.calib.as_ref().and_then(|c| c.get(&a)) else {
                nodes.push(n); // no statistics for this input: stays fp32
                continue;
            };
            let (lo, hi) = if o.calib_avg { (r.lo_avg, r.hi_avg) } else { (r.lo, r.hi) };
            let (lo, hi) = (lo.min(0.0), hi.max(0.0));
            let sx = ((hi - lo) / 255.0).max(1e-12);
            let zx = ((-lo / sx).round() as i32).clamp(0, 255) as u8;
            let (q, sc) = quant_cols(&w, k, nn, 127.0);
            let p = |s: &str| format!("{nm}_{s}");
            inits.push((tensor(&p("sx"), F32, &[]), sx.to_le_bytes().to_vec()));
            inits.push((tensor(&p("zx"), U8, &[]), vec![zx]));
            inits.push((tensor(&p("qB"), I8, &[k as i64, nn as i64]), q));
            inits.push((tensor(&p("qS"), F32, &[nn as i64]), sc.iter().flat_map(|x| x.to_le_bytes()).collect()));
            inits.push((tensor(&p("qZ"), I8, &[nn as i64]), vec![0u8; nn]));
            nodes.push(node("QuantizeLinear", "", &p("q"), &[&a, &p("sx"), &p("zx")], &[&p("xq")], vec![]));
            nodes.push(node("MatMulIntegerToFloat", "com.microsoft", &p("mmi"),
                            &[&p("xq"), &p("qB"), &p("sx"), &p("qS"), &p("zx"), &p("qZ")], &[&y], vec![]));
        } else {
            bail!("unknown scheme {}", o.scheme);
        }
        done += 1;
    }
    g.node = nodes;
    // drop weights no node reads any more
    let used: BTreeSet<String> = g.node.iter().flat_map(|n| n.input.iter().cloned()).chain(g.output.iter().map(|v| v.name().to_string())).collect();
    g.initializer.retain(|t| used.contains(t.name()));
    l.data.retain(|k, _| used.contains(k));
    for (t, b) in inits {
        l.data.insert(t.name().to_string(), b);
        g.initializer.push(t);
    }
    let have: BTreeSet<String> = g.output.iter().map(|v| v.name().to_string()).collect();
    for name in extra_out.into_iter().collect::<BTreeSet<_>>() {
        if have.contains(&name) {
            continue;
        }
        g.output.push(ValueInfoProto {
            name: Some(name),
            r#type: Some(onnx::TypeProto {
                value: Some(onnx::type_proto::Value::TensorType(onnx::type_proto::Tensor { elem_type: Some(F32), shape: None })),
                ..Default::default()
            }),
            ..Default::default()
        });
    }
    if o.scheme == "q8blk" && !l.model.opset_import.iter().any(|o| o.domain() == "flashembed") {
        l.model.opset_import.push(OperatorSetIdProto { domain: Some("flashembed".into()), version: Some(1) });
    }
    if !l.model.opset_import.iter().any(|o| o.domain() == "com.microsoft") {
        l.model.opset_import.push(OperatorSetIdProto { domain: Some("com.microsoft".into()), version: Some(1) });
    }
    Ok((done, sel.kept))
}

/// Names of the activation tensors a calib graph exposes (graph outputs that are MatMul A inputs).
pub fn calib_outputs(path: &Path) -> Result<Vec<String>> {
    let m = ModelProto::decode(std::fs::read(path)?.as_slice())?;
    let g = m.graph.unwrap();
    let a_inputs: BTreeSet<String> = g.node.iter().filter(|n| n.op_type() == "MatMul").map(|n| n.input[0].clone()).collect();
    Ok(g.output.iter().map(|v| v.name().to_string()).filter(|n| a_inputs.contains(n)).collect())
}

#[derive(clap::Args)]
pub struct QuantizeArgs {
    /// fp32 graph directory (py/export_onnx.py)
    #[arg(long)]
    pub src: PathBuf,
    #[arg(long)]
    pub out: PathBuf,
    /// nb8-a4 | nb8-a1 | nb4-a4 | nb4-a1 | dyn8 | dyn8rr | qdq8 | q8blk | calib
    #[arg(long)]
    pub scheme: String,
    /// graphs to rewrite; the others are symlinked to the fp32 files
    #[arg(long, num_args = 1.., default_values_t = ["gemma_lm".to_string(), "gemma_vision_dyn".into(), "neomme_text".into(), "neomme_image".into()])]
    pub graphs: Vec<String>,
    /// MatMulNBits block size along K
    #[arg(long, default_value_t = 32)]
    pub block: usize,
    /// regex on torch module paths to keep fp32 (e.g. 'embed|projection|head')
    #[arg(long)]
    pub keep: Option<String>,
    /// keep the first/last K layers of every layer stack fp32
    #[arg(long, default_value_t = 0)]
    pub keep_edge: usize,
    /// qdq8: calibration json from `flashembed calib` ({graph: {tensor: range}})
    #[arg(long)]
    pub calib: Option<PathBuf>,
    /// qdq8: use the batch-averaged min/max instead of the global min/max
    #[arg(long)]
    pub calib_avg: bool,
    /// q8blk: one QMatMul (+ Split) for MatMuls that share an input (q/k/v, gate/up): one activation quantization
    #[arg(long)]
    pub fuse: bool,
}

pub fn quantize_cmd(a: QuantizeArgs) -> Result<()> {
    std::fs::create_dir_all(&a.out)?;
    let calib: Option<HashMap<String, HashMap<String, Range>>> =
        a.calib.as_ref().map(|p| -> Result<_> { Ok(serde_json::from_str(&std::fs::read_to_string(p)?)?) }).transpose()?;
    let mut summary = serde_json::json!({"scheme": a.scheme, "block": a.block, "keep": a.keep, "keep_edge": a.keep_edge,
                                         "src": a.src, "calib_avg": a.calib_avg, "fuse": a.fuse, "graphs": {}});
    for g in GRAPHS {
        let src = a.src.join(format!("{g}.onnx"));
        if !src.exists() {
            continue;
        }
        if !a.graphs.iter().any(|x| x == g) {
            for f in [format!("{g}.onnx"), format!("{g}.onnx.data")] {
                let dst = a.out.join(&f);
                let _ = std::fs::remove_file(&dst);
                std::os::unix::fs::symlink(std::fs::canonicalize(a.src.join(&f))?, &dst)?;
            }
            summary["graphs"][g] = serde_json::json!("fp32 (symlink)");
            continue;
        }
        let t = std::time::Instant::now();
        let mut l = load(&src)?;
        let o = Opts {
            scheme: a.scheme.clone(), block: a.block, keep: a.keep.as_deref().map(regex::Regex::new).transpose()?,
            keep_edge: a.keep_edge, calib: calib.as_ref().and_then(|c| c.get(g).cloned()), calib_avg: a.calib_avg, fuse: a.fuse,
        };
        if a.scheme == "qdq8" && o.calib.is_none() {
            bail!("qdq8: no calibration for {g} in --calib");
        }
        let (n, kept) = rewrite(&mut l, &o)?;
        let dst = a.out.join(format!("{g}.onnx"));
        let _ = std::fs::remove_file(&dst);
        let _ = std::fs::remove_file(a.out.join(format!("{g}.onnx.data")));
        let bytes = save(l.model, l.data, &dst)?;
        let src_bytes: u64 = [src.clone(), a.src.join(format!("{g}.onnx.data"))].iter().filter_map(|p| p.metadata().ok()).map(|m| m.len()).sum();
        summary["graphs"][g] = serde_json::json!({"quantized": n, "kept_fp32": kept, "bytes": bytes, "src_bytes": src_bytes,
                                                  "s": t.elapsed().as_secs_f64()});
        println!("{g}: {}", summary["graphs"][g]);
    }
    std::fs::write(a.out.join("quant-summary.json"), serde_json::to_string_pretty(&summary)?)?;
    Ok(())
}
