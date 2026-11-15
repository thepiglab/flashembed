"""flashembed-gpu serve vs per-bundle embed: same inputs, total wall time and bit-identical outputs.

  python py/serve_test.py --bin BIN --snapshot SNAP --model gemma|neomme --bundles DIR [--extra "..."] --out OUT

DIR/<bundle>/{text.jsonl | text-neomme.jsonl, image.jsonl}. Per-bundle mode runs `BIN embed` once per bundle and
modality (as the corpus pipeline did); serve mode starts one `BIN serve` and sends every (bundle, modality) request.
"""
import argparse
import json
import shlex
import subprocess
import time
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--bundles", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--extra", default="")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--serve-only", action="store_true", help="skip the per-bundle embed pass (speed only)")
    ap.add_argument("--compare", type=Path, help="with --serve-only: directory of an earlier run's serve/ outputs")
    a = ap.parse_args()
    extra = shlex.split(a.extra)
    tname = "text-neomme.jsonl" if a.model == "neomme" else "text.jsonl"
    jobs = []
    for b in sorted(p for p in a.bundles.iterdir() if (p / "done").exists()):
        for mod, f in (("text", tname), ("image", "image.jsonl")):
            if (b / f).exists():
                n = sum(1 for _ in open(b / f))
                jobs.append((b.name, mod, b / f, n))
    items = sum(j[3] for j in jobs)
    base = ["--model", a.model, "--snapshot", a.snapshot, "--dtype", a.dtype] + extra
    # per-bundle embed
    t = time.perf_counter()
    for name, mod, inp, n in ([] if a.serve_only else jobs):
        od = a.out / "embed" / name / mod
        p = subprocess.run([a.bin, "embed", "--modality", mod, "--input", str(inp), "--out", str(od)] + base,
                           capture_output=True, text=True)
        if p.returncode:
            print(json.dumps({"embed_failed": f"{name}/{mod}", "stderr": p.stderr[-500:]}))
    t_embed = time.perf_counter() - t
    # serve
    t = time.perf_counter()
    srv = subprocess.Popen([a.bin, "serve", "--modalities", "text,image"] + base, stdin=subprocess.PIPE,
                           stdout=subprocess.PIPE, text=True, bufsize=1)
    ready = json.loads(srv.stdout.readline())
    t_ready = time.perf_counter() - t
    for k, (name, mod, inp, n) in enumerate(jobs):
        srv.stdin.write(json.dumps({"id": f"{name}/{mod}", "modality": mod, "input": str(inp),
                                    "out": str(a.out / "serve" / name / mod)}) + "\n")
    # one request with a bad row and a missing file (ok, 2 failed), one bad request and one missing input (ok:false)
    img = next((j for j in jobs if j[1] == "image"), None)
    if img:
        bad = a.out / "bad-items" / "image.jsonl"
        bad.parent.mkdir(parents=True, exist_ok=True)
        rows = [json.loads(l) for l in open(img[2])][:3]
        for r in rows:
            r["path"] = str((img[2].parent / r["path"]).resolve())
        with open(bad, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
            f.write("{broken row\n")
            f.write(json.dumps({"id": "nofile", "path": "/nonexistent.png"}) + "\n")
        srv.stdin.write(json.dumps({"id": "bad-items", "modality": "image", "input": str(bad), "out": str(a.out / "bad-out")}) + "\n")
    srv.stdin.write("{not json\n")
    srv.stdin.write(json.dumps({"id": "missing", "modality": "text", "input": "/nonexistent.jsonl", "out": str(a.out / "x")}) + "\n")
    srv.stdin.close()
    replies = [json.loads(l) for l in srv.stdout]
    srv.wait()
    t_serve = time.perf_counter() - t
    bi = next((r for r in replies if r.get("id") == "bad-items"), None)
    replies = [r for r in replies if r.get("id") != "bad-items"]
    ok = [r for r in replies if r.get("ok")]
    bad = [r for r in replies if not r.get("ok")]
    # bit-identical outputs
    same, diff, mincos = 0, [], 1.0
    for name, mod, inp, n in jobs:
        e, s = (a.compare if a.serve_only and a.compare else a.out / "embed") / name / mod, a.out / "serve" / name / mod
        try:
            eq = all(np.array_equal(np.load(e / f), np.load(s / f)) for f in ["embeddings.dense.npy"] +
                     [f for f in ("embeddings.late.npy", "embeddings.late_offsets.npy") if (e / f).exists()])
            eq = eq and json.load(open(e / "embeddings.ids.json")) == json.load(open(s / "embeddings.ids.json"))
            x, y = np.load(e / "embeddings.dense.npy").astype(np.float64), np.load(s / "embeddings.dense.npy").astype(np.float64)
            if x.shape == y.shape and len(x):
                c = (x * y).sum(1) / np.linalg.norm(x, axis=1) / np.linalg.norm(y, axis=1)
                mincos = min(mincos, float(c.min()))
        except Exception as ex:  # missing output
            eq = False
        same += eq
        if not eq:
            diff.append(f"{name}/{mod}")
    print(json.dumps({"model": a.model, "requests": len(jobs), "items": items, "embed_wall_s": round(t_embed, 1),
                      "embed_items_per_s": round(items / t_embed, 1), "serve_wall_s": round(t_serve, 1),
                      "serve_items_per_s": round(items / t_serve, 1), "serve_ready_s": round(t_ready, 1),
                      "speedup": round(t_embed / t_serve, 2), "ready": ready, "replies_ok": len(ok),
                      "replies_failed": len(bad), "failed_ids": [r.get("id") for r in bad][:5],
                      "failed_errors": [r.get("error", "")[:120] for r in bad][:5],
                      "bit_identical": same, "min_cos_vs_reference": mincos, "not_identical": diff[:10],
                      "bad_items_reply": {k: bi.get(k) for k in ("ok", "items", "failed")} if bi else None}))


if __name__ == "__main__":
    main()
