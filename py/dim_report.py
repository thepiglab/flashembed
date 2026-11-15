"""Matryoshka --dim parity table from a dimtest output dir (local/ job or any run with the same file names).

  python py/dim_report.py DIR [--dims 512,256,128] [--mods text,image,video]

Files (each X.dense.npy + X.ids.json):
  ref-<mod>-<d>                 sentence-transformers fp32, truncate_dim=d, normalize_embeddings=True (py/gpu_ref_dim.py)
  gpu-<mod>-<dt>-<d>            flashembed-gpu embed --dim d (d = 768: full size)
  srv-<mod>-<dt>-<d>/embeddings flashembed-gpu serve --dim d
  cpu-<mod>-<d>/embeddings      flashembed (CPU) embed --dim d, on a subset
Prints one JSON row per (runtime, dtype, modality, dim) and a markdown table; the slice check compares
gpu-...-768[:, :d] re-normalised with gpu-...-d.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def load(d: Path, stem: str):
    f = d / f"{stem}.dense.npy"
    if not f.exists():
        return None
    return dict(zip(json.load(open(d / f"{stem}.ids.json")), np.load(f).astype(np.float64)))


def cos(a, b):
    ids = [i for i in a if i in b]
    if not ids:
        return None
    x = np.stack([a[i] for i in ids])
    y = np.stack([b[i] for i in ids])
    c = (x * y).sum(1) / np.linalg.norm(x, axis=1) / np.linalg.norm(y, axis=1)
    return {"n": len(ids), "min": float(c.min()), "median": float(np.median(c)), "below_0.999": int((c < 0.999).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", type=Path)
    ap.add_argument("--dims", default="512,256,128")
    ap.add_argument("--mods", default="text,image,video")
    a = ap.parse_args()
    rows = []
    for mod in a.mods.split(","):
        for d in [int(x) for x in a.dims.split(",")]:
            ref = load(a.dir, f"ref-{mod}-{d}")
            if ref is None:
                continue
            cases = [("flashembed-gpu embed", dt, load(a.dir, f"gpu-{mod}-{dt}-{d}")) for dt in ("fp32", "bf16")]
            cases += [("flashembed-gpu serve", dt, load(a.dir / f"srv-{mod}-{dt}-{d}", "embeddings")) for dt in ("fp32", "bf16")]
            cases += [("flashembed CPU embed", "fp32", load(a.dir / f"cpu-{mod}-{d}", "embeddings"))]
            for rt, dt, v in cases:
                if v is None:
                    continue
                r = cos(v, ref)
                if r is None:
                    continue
                row = {"runtime": rt, "dtype": dt, "modality": mod, "dim": d, **r}
                if rt == "flashembed-gpu embed":
                    full = load(a.dir, f"gpu-{mod}-{dt}-768")
                    if full is not None:
                        diffs = []
                        for i, x in v.items():
                            s = full[i][:d] / np.linalg.norm(full[i][:d])
                            diffs.append(float(np.abs(s - x).max()))
                        row["slice_768_max_abs_diff"] = max(diffs)
                if rt == "flashembed-gpu serve":
                    e = load(a.dir, f"gpu-{mod}-{dt}-{d}")
                    row["serve_bit_identical_to_embed"] = e is not None and all(np.array_equal(e[i], x) for i, x in v.items())
                rows.append(row)
                print(json.dumps(row))
    print()
    print("| modality | dim | runtime | dtype | n | min cos | median cos | slice of 768 = --dim |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        extra = ""
        if "slice_768_max_abs_diff" in r:
            extra = f"max abs diff {r['slice_768_max_abs_diff']:.1e}"
        if "serve_bit_identical_to_embed" in r:
            extra = "serve = embed bit for bit" if r["serve_bit_identical_to_embed"] else "serve differs from embed"
        print(f"| {r['modality']} | {r['dim']} | {r['runtime']} | {r['dtype']} | {r['n']} | {r['min']:.7f} | "
              f"{r['median']:.7f} | {extra} |")


if __name__ == "__main__":
    main()
