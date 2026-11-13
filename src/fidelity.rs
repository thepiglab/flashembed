//! `flashembed fidelity`: how far a variant's vectors are from fp32 on the same inputs (docs/QUANTIZATION.md).
//!
//! Reads two `flashembed bench --save` directories (`<tag>.dense.npy`, `<tag>.ids.json`, NeoMME `<tag>.late.npy`) and
//! reports, per tag present in both:
//! * `dense_cos` / `late_cos`: per-item (per-row for late vectors) cosine variant vs fp32: min, p1, median, mean;
//! * `knn`: every item queries all others in its set; mean overlap of the top-10 neighbour sets fp32 vs variant,
//!   share of items whose fp32 nearest neighbour is still in the variant's top 10, mean rank shift of the fp32 top 10;
//! * with `--pairs` ({caption id: figure id}) and text + image tags of one model: caption -> figure retrieval (R@1,
//!   R@10, MRR) for fp32 and the variant, the paired difference and its 95 % bootstrap interval over captions.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use anyhow::{Result, ensure};
use ndarray::{Array2, Axis, s};
use ndarray_npy::read_npy;
use serde_json::{Value, json};

#[derive(clap::Args)]
pub struct FidelityArgs {
    /// fp32 `bench --save` output dir
    #[arg(long = "ref")]
    pub reference: PathBuf,
    /// variant `bench --save` output dir (same inputs, same tags)
    #[arg(long)]
    pub var: PathBuf,
    /// json {caption unit id: figure unit id} for caption -> figure retrieval
    #[arg(long)]
    pub pairs: Option<PathBuf>,
    #[arg(long)]
    pub json: Option<PathBuf>,
    /// bootstrap resamples
    #[arg(long, default_value_t = 2000)]
    pub boot: usize,
}

struct Set {
    dense: Array2<f32>,
    ids: Vec<String>,
    late: Option<Array2<f32>>,
}

fn load(d: &Path, tag: &str) -> Result<Set> {
    let dense: Array2<f32> = read_npy(d.join(format!("{tag}.dense.npy")))?;
    let ids: Vec<String> = serde_json::from_str(&std::fs::read_to_string(d.join(format!("{tag}.ids.json")))?)?;
    let lp = d.join(format!("{tag}.late.npy"));
    let late = if lp.exists() { Some(read_npy(lp)?) } else { None };
    Ok(Set { dense, ids, late })
}

fn rowcos(a: &Array2<f32>, b: &Array2<f32>) -> Vec<f64> {
    a.outer_iter()
        .zip(b.outer_iter())
        .map(|(x, y)| {
            let (mut d, mut nx, mut ny) = (0f64, 0f64, 0f64);
            for (p, q) in x.iter().zip(y.iter()) {
                d += *p as f64 * *q as f64;
                nx += *p as f64 * *p as f64;
                ny += *q as f64 * *q as f64;
            }
            d / (nx * ny).sqrt().max(1e-24)
        })
        .collect()
}

fn pct(sorted: &[f64], p: f64) -> f64 {
    let i = (p / 100.0) * (sorted.len() - 1) as f64;
    let (lo, hi) = (i.floor() as usize, i.ceil() as usize);
    sorted[lo] + (sorted[hi] - sorted[lo]) * (i - lo as f64)
}

fn stats(mut c: Vec<f64>) -> Value {
    c.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = c.iter().sum::<f64>() / c.len() as f64;
    json!({"n": c.len(), "min": c[0], "p1": pct(&c, 1.0), "median": pct(&c, 50.0), "mean": mean})
}

fn normalize(x: &Array2<f32>) -> Array2<f32> {
    let mut y = x.clone();
    for mut r in y.outer_iter_mut() {
        let n = r.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
        r.mapv_inplace(|v| v / n);
    }
    y
}

/// Indices of the top-k most similar other rows, best first.
fn topk(x: &Array2<f32>, k: usize) -> Vec<Vec<usize>> {
    let n = x.nrows();
    let mut out = Vec::with_capacity(n);
    for s0 in (0..n).step_by(512) {
        let blk = x.slice(s![s0..(s0 + 512).min(n), ..]);
        let sim = blk.dot(&x.t());
        for (i, row) in sim.outer_iter().enumerate() {
            let mut idx: Vec<usize> = (0..n).filter(|&j| j != s0 + i).collect();
            let kk = k.min(idx.len());
            idx.select_nth_unstable_by(kk - 1, |&a, &b| row[b].partial_cmp(&row[a]).unwrap());
            idx.truncate(kk);
            idx.sort_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap());
            out.push(idx);
        }
    }
    out
}

fn knn(r: &Array2<f32>, v: &Array2<f32>) -> Value {
    let n = r.nrows();
    if n < 12 {
        return Value::Null;
    }
    let (r, v) = (normalize(r), normalize(v));
    let tr = topk(&r, 10);
    let tv = topk(&v, 100.min(n - 1));
    let (mut ov, mut top1, mut shift, mut cnt) = (0f64, 0f64, 0f64, 0f64);
    for i in 0..n {
        let v10: HashSet<usize> = tv[i].iter().take(10).copied().collect();
        ov += tr[i].iter().filter(|j| v10.contains(j)).count() as f64 / 10.0;
        top1 += v10.contains(&tr[i][0]) as u8 as f64;
        let pos: HashMap<usize, usize> = tv[i].iter().enumerate().map(|(p, &j)| (j, p)).collect();
        for (rk, j) in tr[i].iter().enumerate() {
            shift += (*pos.get(j).unwrap_or(&100) as f64 - rk as f64).abs();
            cnt += 1.0;
        }
    }
    json!({"items": n, "overlap@10": ov / n as f64, "fp32_top1_in_var_top10": top1 / n as f64, "mean_rank_shift_top10": shift / cnt})
}

/// rank (1-based) of the gold doc for each query
fn ranks(q: &Array2<f32>, docs: &Array2<f32>, gold: &[usize]) -> Vec<f64> {
    let sim = normalize(q).dot(&normalize(docs).t());
    sim.outer_iter().zip(gold).map(|(row, &g)| 1.0 + row.iter().filter(|&&x| x > row[g]).count() as f64).collect()
}

fn metrics(rk: &[f64]) -> Value {
    let n = rk.len() as f64;
    json!({"R@1": rk.iter().filter(|&&r| r <= 1.0).count() as f64 / n, "R@10": rk.iter().filter(|&&r| r <= 10.0).count() as f64 / n,
           "MRR": rk.iter().map(|r| 1.0 / r).sum::<f64>() / n})
}

/// paired bootstrap of variant - fp32 per metric
fn boot(a: &[f64], b: &[f64], reps: usize) -> Value {
    let mut st = 0x9e37_79b9_7f4a_7c15u64;
    let mut rnd = move || {
        st = st.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = st;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    };
    let fs: [(&str, fn(f64) -> f64); 3] = [("R@1", |r| (r <= 1.0) as u8 as f64), ("R@10", |r| (r <= 10.0) as u8 as f64), ("MRR", |r| 1.0 / r)];
    let mut out = serde_json::Map::new();
    for (name, f) in fs {
        let d: Vec<f64> = a.iter().zip(b).map(|(x, y)| f(*y) - f(*x)).collect();
        let n = d.len();
        let mut bs: Vec<f64> = (0..reps).map(|_| (0..n).map(|_| d[(rnd() % n as u64) as usize]).sum::<f64>() / n as f64).collect();
        bs.sort_by(|x, y| x.partial_cmp(y).unwrap());
        out.insert(name.into(), json!({"diff": d.iter().sum::<f64>() / n as f64, "ci95": [pct(&bs, 2.5), pct(&bs, 97.5)]}));
    }
    Value::Object(out)
}

pub fn run(a: FidelityArgs) -> Result<()> {
    let mut tags: Vec<String> = std::fs::read_dir(&a.var)?
        .filter_map(|e| e.ok()?.file_name().to_str()?.strip_suffix(".dense.npy").map(String::from))
        .filter(|t| a.reference.join(format!("{t}.dense.npy")).exists())
        .collect();
    tags.sort();
    let mut res = json!({"ref": a.reference, "var": a.var, "tags": {}});
    let mut sets: HashMap<String, (Set, Set)> = HashMap::new();
    for t in &tags {
        let (r, v) = (load(&a.reference, t)?, load(&a.var, t)?);
        ensure!(r.ids == v.ids, "{t}: item order differs between ref and var");
        let mut o = json!({"dense_cos": stats(rowcos(&r.dense, &v.dense)), "knn": knn(&r.dense, &v.dense)});
        if let (Some(lr), Some(lv)) = (&r.late, &v.late) {
            o["late_cos"] = stats(rowcos(lr, lv));
        }
        println!("{t} {o}");
        res["tags"][t] = o;
        sets.insert(t.clone(), (r, v));
    }
    if let Some(p) = &a.pairs {
        let pairs: HashMap<String, String> = serde_json::from_str(&std::fs::read_to_string(p)?)?;
        let re = regex::Regex::new(r"^rs-(\w+)-text-(.*)$")?;
        for t in &tags {
            let Some(c) = re.captures(t) else { continue };
            let ti = format!("rs-{}-image-{}", &c[1], &c[2]);
            let (Some((tr, tv)), Some((ir, iv))) = (sets.get(t), sets.get(&ti)) else { continue };
            let fpos: HashMap<&str, usize> = ir.ids.iter().enumerate().map(|(i, f)| (f.as_str(), i)).collect();
            let qi: Vec<usize> = (0..tr.ids.len()).filter(|&i| pairs.get(&tr.ids[i]).is_some_and(|f| fpos.contains_key(f.as_str()))).collect();
            if qi.len() < 20 {
                continue;
            }
            let gold: Vec<usize> = qi.iter().map(|&i| fpos[pairs[&tr.ids[i]].as_str()]).collect();
            let rr = ranks(&tr.dense.select(Axis(0), &qi), &ir.dense, &gold);
            let rv = ranks(&tv.dense.select(Axis(0), &qi), &iv.dense, &gold);
            let same = rr.iter().zip(&rv).filter(|(x, y)| x == y).count() as f64 / rr.len() as f64;
            let o = json!({"queries": qi.len(), "figures": ir.ids.len(), "fp32": metrics(&rr), "var": metrics(&rv),
                           "var_minus_fp32": boot(&rr, &rv, a.boot), "same_rank_share": same});
            println!("caption->figure {} {o}", &c[1]);
            res["caption_to_figure"][&c[1]] = o;
        }
    }
    if let Some(j) = &a.json {
        std::fs::write(j, serde_json::to_string_pretty(&res)?)?;
    }
    Ok(())
}
