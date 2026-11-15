//! `flashembed qreport <qbench out dirs...>`: markdown tables for docs/QUANTIZATION.md from slurm/qbench.sbatch
//! outputs (`rep<r>/<runtime>/<tag>.json`, `fid-<variant>-vs-fp32.json`, `fid-<variant>-vs-py.json`).
//!
//! Speed is reported as items/s over the full set (rep 0) and as the ratio to fp32 flashembed measured in the same
//! rep, averaged over reps (each rep runs every runtime back to back on the same node, so the ratio cancels most of
//! a shared node's load drift).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::Result;
use serde_json::Value;

#[derive(clap::Args)]
pub struct QreportArgs {
    /// qbench output dirs (one per CPU x model)
    pub dirs: Vec<PathBuf>,
    /// graph root with <variant>/quant-summary.json, for model sizes
    #[arg(long)]
    pub qroot: Option<PathBuf>,
}

fn read(p: &Path) -> Option<Value> {
    serde_json::from_str(&std::fs::read_to_string(p).ok()?).ok()
}

fn cpu_short(c: &str) -> String {
    for (k, v) in [("7702", "Zen 2 (EPYC 7702)"), ("75F3", "Zen 3 (EPYC 75F3)"), ("9655P", "Zen 5 (EPYC 9655P)")] {
        if c.contains(k) {
            return v.into();
        }
    }
    c.into()
}

fn f(v: &Value, k: &[&str]) -> Option<f64> {
    let mut x = v;
    for s in k {
        x = x.get(*s)?;
    }
    x.as_f64()
}

pub fn run(a: QreportArgs) -> Result<()> {
    let mut c2f = String::new();
    for d in &a.dirs {
        // runtime -> modality -> rep -> json
        let mut runs: BTreeMap<String, BTreeMap<String, BTreeMap<usize, Value>>> = BTreeMap::new();
        let mut cpu = String::new();
        let mut model = String::new();
        for rep in std::fs::read_dir(d)?.flatten() {
            let rn = rep.file_name().to_string_lossy().to_string();
            let Some(r) = rn.strip_prefix("rep").and_then(|x| x.parse::<usize>().ok()) else { continue };
            for rt in std::fs::read_dir(rep.path())?.flatten() {
                let rtn = rt.file_name().to_string_lossy().to_string();
                if rtn.ends_with("-as-rs") {
                    continue;
                }
                for j in std::fs::read_dir(rt.path())?.flatten() {
                    let p = j.path();
                    let name = p.file_name().unwrap().to_string_lossy().to_string();
                    if !name.ends_with(".json") || name.ends_with(".ids.json") {
                        continue;
                    }
                    let Some(v) = read(&p) else { continue };
                    let Some(m) = v.get("modality").and_then(|x| x.as_str()) else { continue };
                    if let Some(c) = v.get("cpu").and_then(|x| x.as_str()) { cpu = cpu_short(c); }
                    if let Some(c) = v.get("model").and_then(|x| x.as_str()) { model = c.into(); }
                    runs.entry(rtn.clone()).or_default().entry(m.into()).or_default().insert(r, v);
                }
            }
        }
        let fp = runs.get("fp32").cloned().unwrap_or_default();
        let mods: Vec<String> = fp.keys().cloned().collect();
        for m in &mods {
            println!("\n**{model} {m}, {cpu}** (16 threads, batch 4)\n");
            println!("| runtime | items/s | vs fp32 flashembed | vs Python | peak RSS MB | dense cos vs fp32: min / p1 / median | top-10 overlap | fp32 top-1 in top 10 | dense cos vs Python: min |");
            println!("|---|---|---|---|---|---|---|---|---|");
            let mut order: Vec<String> = vec!["py".into(), "fp32".into()];
            order.extend(runs.keys().filter(|k| *k != "py" && *k != "fp32").cloned());
            for rt in order {
                let Some(reps) = runs.get(&rt).and_then(|x| x.get(m)) else { continue };
                let ratio = |base: &str| -> Option<f64> {
                    let b = runs.get(base)?.get(m)?;
                    let rs: Vec<f64> = reps.iter().filter_map(|(r, v)| Some(f(v, &["items_per_s"])? / f(b.get(r)?, &["items_per_s"])?)).collect();
                    (!rs.is_empty()).then(|| rs.iter().sum::<f64>() / rs.len() as f64)
                };
                let r0 = reps.get(&0);
                let ips = r0.and_then(|v| f(v, &["items_per_s"])).map(|x| format!("{x:.2}")).unwrap_or("-".into());
                let rss = reps.values().filter_map(|v| f(v, &["peak_rss_mb"])).fold(0f64, f64::max);
                let fid = read(&d.join(format!("fid-{rt}-vs-fp32.json")));
                let fpy = read(&d.join(format!("fid-{rt}-vs-py.json")));
                let tag = |fd: &Option<Value>| -> Option<Value> {
                    let t = fd.as_ref()?.get("tags")?.as_object()?;
                    t.iter().find(|(k, _)| k.contains(&format!("-{m}-"))).map(|x| x.1.clone())
                };
                let (tf, tp) = (tag(&fid), tag(&fpy));
                let cos = tf.as_ref().map(|t| format!("{:.5} / {:.5} / {:.6}", f(t, &["dense_cos", "min"]).unwrap_or(f64::NAN),
                    f(t, &["dense_cos", "p1"]).unwrap_or(f64::NAN), f(t, &["dense_cos", "median"]).unwrap_or(f64::NAN))).unwrap_or("-".into());
                let ov = tf.as_ref().and_then(|t| f(t, &["knn", "overlap@10"])).map(|x| format!("{x:.3}")).unwrap_or("-".into());
                let t1 = tf.as_ref().and_then(|t| f(t, &["knn", "fp32_top1_in_var_top10"])).map(|x| format!("{x:.3}")).unwrap_or("-".into());
                let cpy = tp.as_ref().and_then(|t| f(t, &["dense_cos", "min"])).map(|x| format!("{x:.5}")).unwrap_or("-".into());
                let label = match rt.as_str() { "py" => "Python (model card, fp32)".to_string(), "fp32" => "flashembed fp32".to_string(), x => format!("flashembed {x}") };
                println!("| {label} | {ips} | {} | {} | {rss:.0} | {cos} | {ov} | {t1} | {cpy} |",
                    ratio("fp32").map(|x| format!("{x:.2}x")).unwrap_or("-".into()),
                    ratio("py").map(|x| format!("{x:.2}x")).unwrap_or("-".into()));
            }
        }
        // caption -> figure retrieval on the held-out set
        let mut rows = vec![];
        for rt in runs.keys().filter(|k| *k != "py" && *k != "fp32") {
            let Some(v) = read(&d.join(format!("fid-{rt}-vs-fp32.json"))) else { continue };
            let Some(c) = v.get("caption_to_figure").and_then(|x| x.get(&model)) else { continue };
            rows.push(format!("| {model} | {cpu} | {rt} | {} | {:.3} / {:.3} | {:.3} / {:.3} | {:+.3} [{:+.3}, {:+.3}] | {:.3} |",
                c["queries"], f(c, &["fp32", "R@1"]).unwrap_or(0.0), f(c, &["var", "R@1"]).unwrap_or(0.0),
                f(c, &["fp32", "MRR"]).unwrap_or(0.0), f(c, &["var", "MRR"]).unwrap_or(0.0),
                f(c, &["var_minus_fp32", "MRR", "diff"]).unwrap_or(0.0),
                c["var_minus_fp32"]["MRR"]["ci95"][0].as_f64().unwrap_or(0.0), c["var_minus_fp32"]["MRR"]["ci95"][1].as_f64().unwrap_or(0.0),
                f(c, &["same_rank_share"]).unwrap_or(0.0)));
        }
        c2f.push_str(&rows.join("\n"));
        if !rows.is_empty() { c2f.push('\n'); }
    }
    if !c2f.is_empty() {
        println!("\n**Caption -> figure retrieval on the held-out set** (each caption queries all figures of the set)\n");
        println!("| model | CPU | variant | queries | R@1 fp32 / variant | MRR fp32 / variant | MRR variant - fp32 [95 % CI] | same rank as fp32 |");
        println!("|---|---|---|---|---|---|---|---|");
        print!("{c2f}");
    }
    if let Some(q) = &a.qroot {
        println!("\n**Graph sizes** (MB on disk; fp32 = the exported graphs)\n");
        println!("| variant | gemma_lm | gemma_vision_dyn | neomme_text | neomme_image |");
        println!("|---|---|---|---|---|");
        let mut vs: Vec<_> = std::fs::read_dir(q)?.flatten().filter(|e| e.path().join("quant-summary.json").exists() && e.file_name() != "calib").collect();
        vs.sort_by_key(|e| e.file_name());
        let mut first = true;
        for e in vs {
            let s = read(&e.path().join("quant-summary.json")).unwrap_or_default();
            let cell = |g: &str, k: &str| s["graphs"][g][k].as_f64().map(|b| format!("{:.0}", b / 1e6)).unwrap_or("fp32".into());
            if first {
                println!("| fp32 | {} | {} | {} | {} |", cell("gemma_lm", "src_bytes"), cell("gemma_vision_dyn", "src_bytes"), cell("neomme_text", "src_bytes"), cell("neomme_image", "src_bytes"));
                first = false;
            }
            println!("| {} | {} | {} | {} | {} |", e.file_name().to_string_lossy(), cell("gemma_lm", "bytes"), cell("gemma_vision_dyn", "bytes"), cell("neomme_text", "bytes"), cell("neomme_image", "bytes"));
        }
    }
    Ok(())
}
