"""EmbeddingGemma 2 video / audio with the model-card runtime (sentence-transformers) on a GPU: throughput, peak
memory and vectors, for comparison with flashembed-gpu.

  python py/gpu_ref_media.py --models-json models.json --input media.jsonl --modality video|audio --out OUT \
      [--dtype float32|bfloat16] [--batch 4]

input rows: {"id", "path"} (path relative to the jsonl). Items are passed as {"video": path} / {"audio": path}, as the
model card does; max_seq_length is 8192 (sentence-transformers' default for this model is unbounded). Writes
OUT/gemma-<modality>-<dtype>.dense.npy + .ids.json and appends a result row to OUT/results.jsonl.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-json", type=Path, required=True)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--modality", choices=["video", "audio"], required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ids", default="", help="comma-separated ids to keep")
    a = ap.parse_args()
    import torch
    from sentence_transformers import SentenceTransformer

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows = [json.loads(l) for l in open(a.input)]
    rows = [r for r in rows if r.get("kind", a.modality) == a.modality]
    if a.ids:
        keep = set(a.ids.split(","))
        rows = [r for r in rows if r["id"] in keep]
    if a.n:
        rows = rows[: a.n]
    root = a.input.parent
    items = [{a.modality: str(root / r["path"])} for r in rows]
    snap = json.load(open(a.models_json))["google/embeddinggemma-2"]["path"]
    cfg = {"video": {"audio_config": None}, "audio": {"vision_config": None}}[a.modality]
    t = time.perf_counter()
    m = SentenceTransformer(snap, device=a.device, config_kwargs=cfg, model_kwargs={"torch_dtype": getattr(torch, a.dtype)})
    m.max_seq_length = 8192
    m.eval()
    load_s = time.perf_counter() - t
    m.encode(items[: a.batch], batch_size=a.batch, normalize_embeddings=True, show_progress_bar=False)  # warm-up
    cuda = a.device.startswith("cuda")
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    v = m.encode(items, batch_size=a.batch, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    if cuda:
        torch.cuda.synchronize()
    wall = time.perf_counter() - t
    a.out.mkdir(parents=True, exist_ok=True)
    tag = f"gemma-{a.modality}-{a.dtype}" + ("" if cuda else f"-{a.device}-b{a.batch}")
    np.save(a.out / f"{tag}.dense.npy", np.asarray(v, np.float32))
    json.dump([r["id"] for r in rows], open(a.out / f"{tag}.ids.json", "w"))
    res = {"runtime": "sentence-transformers", "modality": a.modality, "dtype": a.dtype, "batch": a.batch, "n": len(rows),
           "wall_s": wall, "items_per_s": len(rows) / wall, "load_s": load_s,
           "gpu_peak_mb": torch.cuda.max_memory_allocated() / 2**20 if cuda else None,
           "gpu_peak_reserved_mb": torch.cuda.max_memory_reserved() / 2**20 if cuda else None,
           "gpu": torch.cuda.get_device_name() if cuda else a.device, "job": os.environ.get("SLURM_JOB_ID")}
    print(json.dumps(res))
    with open(a.out / "results.jsonl", "a") as f:
        f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
