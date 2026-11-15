"""EmbeddingGemma 2 Matryoshka reference on a GPU: sentence-transformers encode(..., truncate_dim=d,
normalize_embeddings=True) in fp32, for each requested dim and modality (the model card's code path).

  python py/gpu_ref_dim.py --snapshot SNAP --input items.jsonl --modality text|image|video --dims 768,512,256,128 \
      --out DIR [--max-len 1024] [--batch 16]

input rows {"id", "text"} or {"id", "path"} (path relative to the jsonl). Writes DIR/ref-<modality>-<dim>.dense.npy
and .ids.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--modality", required=True)
    ap.add_argument("--dims", default="768,512,256,128")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    import torch
    from sentence_transformers import SentenceTransformer

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows = [json.loads(l) for l in open(a.input) if l.strip()]
    root = a.input.parent
    if a.modality == "text":
        items = [r["text"] for r in rows]
    else:
        items = [{a.modality: str(root / r["path"])} for r in rows]
    cfg = {"text": {"audio_config": None}, "image": {"audio_config": None}, "video": {"audio_config": None}}[a.modality]
    m = SentenceTransformer(a.snapshot, device="cuda", config_kwargs=cfg, model_kwargs={"torch_dtype": torch.float32})
    m.max_seq_length = a.max_len if a.modality == "text" else 8192
    m.eval()
    a.out.mkdir(parents=True, exist_ok=True)
    for d in (int(x) for x in a.dims.split(",")):
        v = m.encode(items, batch_size=a.batch, normalize_embeddings=True, truncate_dim=d, convert_to_numpy=True,
                     show_progress_bar=False)
        np.save(a.out / f"ref-{a.modality}-{d}.dense.npy", np.asarray(v, np.float32))
        json.dump([r["id"] for r in rows], open(a.out / f"ref-{a.modality}-{d}.ids.json", "w"))
        print(json.dumps({"modality": a.modality, "dim": d, "n": len(rows), "shape": list(np.asarray(v).shape)}), flush=True)


if __name__ == "__main__":
    main()
