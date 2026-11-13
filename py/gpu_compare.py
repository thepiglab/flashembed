"""Per-item cosine between two sets of dense vectors matched by id (flashembed-gpu vs the reference or the CPU runner).

  python py/gpu_compare.py A.dense.npy A.ids.json B.dense.npy B.ids.json [--label X] [--out results.jsonl]

ids files: a JSON list of ids, or {"unit_id": [...]} (PigMed vector bundles). Prints min / p01 / median cosine, the
number of items below 0.999 and the top-5 neighbour overlap within the matched set.
"""
import argparse
import json
import sys

import numpy as np


def ids(p):
    d = json.load(open(p))
    return d["unit_id"] if isinstance(d, dict) else d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("a_ids")
    ap.add_argument("b")
    ap.add_argument("b_ids")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default=None)
    x = ap.parse_args()
    A, B = np.load(x.a).astype(np.float64), np.load(x.b).astype(np.float64)
    ia, ib = ids(x.a_ids), ids(x.b_ids)
    pos = {u: i for i, u in enumerate(ib)}
    pairs = [(i, pos[u]) for i, u in enumerate(ia) if u in pos]
    if not pairs:
        print(json.dumps({"label": x.label, "error": "no common ids"}))
        return 1
    ga, gb = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    a, b = A[ga], B[gb]
    if a.shape[1] != b.shape[1]:
        d = min(a.shape[1], b.shape[1])
        a, b = a[:, :d], b[:, :d]
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    cos = (a * b).sum(1)
    k = min(5, len(a) - 1)
    top = 1.0
    if k > 0 and len(a) <= 20000:
        sa, sb = a @ a.T, b @ b.T
        np.fill_diagonal(sa, -2)
        np.fill_diagonal(sb, -2)
        ta, tb = np.argsort(-sa, 1)[:, :k], np.argsort(-sb, 1)[:, :k]
        top = float(np.mean([len(set(ta[i]) & set(tb[i])) / k for i in range(len(a))]))
    r = {"label": x.label, "n": len(pairs), "min_cos": float(cos.min()), "p01_cos": float(np.percentile(cos, 1)),
         "median_cos": float(np.median(cos)), "below_0.999": int((cos < 0.999).sum()),
         "below_0.9999": int((cos < 0.9999).sum()), "top5_overlap": top}
    print(json.dumps(r))
    if x.out:
        with open(x.out, "a") as f:
            f.write(json.dumps(r) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
