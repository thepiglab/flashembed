"""EmbeddingGemma 2 Matryoshka parity report for slurm/mrl.sbatch outputs:  python mrl_report.py <out dir> <threads>"""
import glob, json, sys
import numpy as np
out, t = sys.argv[1], sys.argv[2]
rows, ok = [], True
for mod in ("text", "image", "audio", "video"):
    full = {}
    for d in (768, 512, 256, 128):
        f = lambda rt: f"{out}/{rt}-gemma-{mod}-t{t}-b4_d{d}.dense.npy"
        try:
            p, r = np.load(f("py")), np.load(f("rs"))
        except FileNotFoundError:
            continue
        if d == 768:
            full = {"py": p, "rs": r}
        cos = (p * r).sum(1) / np.linalg.norm(p, axis=1) / np.linalg.norm(r, axis=1)
        # truncate-then-renormalise identity: the d-dim output equals the 768-d output's prefix, re-normalised
        pref = full["py"][:, :d] / np.linalg.norm(full["py"][:, :d], axis=1, keepdims=True)
        dev = np.abs(pref - p).max()
        # neighbour agreement: each item's top-5 most similar other items, Python vs Rust
        def top(x):
            k = min(5, len(x) - 1); s = x @ x.T; np.fill_diagonal(s, -9); return np.argsort(-s, 1)[:, :k]
        agree = np.mean([len(set(a) & set(b)) / len(a) for a, b in zip(top(p), top(r))]) if len(p) > 2 else float("nan")
        rows.append((mod, d, len(p), p.shape[1], cos.min(), np.abs(np.linalg.norm(r, axis=1) - 1).max(), dev, agree))
        ok &= p.shape[1] == d == r.shape[1] and cos.min() >= 0.9999
print("| modality | dim | items | out dim | min cosine py vs rs | max |norm-1| (rs) | ST d-dim vs renormalised 768-d prefix, max |diff| | top-5 neighbour agreement |")
print("|---|---|---|---|---|---|---|---|")
for m, d, n, od, c, nd, dev, ag in rows:
    print(f"| {m} | {d} | {n} | {od} | {c:.7f} | {nd:.1e} | {dev:.1e} | {ag:.3f} |")
print("status", "OK" if ok and rows else "FAIL")
sys.exit(0 if ok and rows else 1)
