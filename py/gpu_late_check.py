"""NeoMME late-interaction vectors: flashembed-gpu vs transformers NeoMMEForRetrieval (FP32, CUDA), row by row.

  python py/gpu_late_check.py --snapshot SNAP --input texts.jsonl --vectors DIR --tag TAG [--n 200] [--max-len 1024]

DIR/TAG.late.npy + .late_offsets.npy + .ids.json come from `flashembed-gpu embed --model neomme` (without --no-late).
Prints the min / median per-row cosine over all tokens of the first --n items.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--vectors", type=Path, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--max-len", type=int, default=1024)
    a = ap.parse_args()
    from transformers import NeoMMEForRetrieval, NeoMMEProcessor

    torch.backends.cuda.matmul.allow_tf32 = False
    proc = NeoMMEProcessor.from_pretrained(a.snapshot)
    m = NeoMMEForRetrieval.from_pretrained(a.snapshot, dtype=torch.float32).cuda().eval()
    mb = NeoMMEForRetrieval.from_pretrained(a.snapshot, dtype=torch.bfloat16).cuda().eval()  # context: torch's own bf16
    rows = [json.loads(l) for l in open(a.input)]
    ids = json.load(open(a.vectors / f"{a.tag}.ids.json"))
    late = np.load(a.vectors / f"{a.tag}.late.npy")
    offs = np.load(a.vectors / f"{a.tag}.late_offsets.npy")
    pos = {u: i for i, u in enumerate(ids)}
    cos_all, cos_tb = [], []
    for r in rows[: a.n]:
        i = pos[r["id"]]
        mine = late[offs[i]: offs[i + 1]]
        msgs = [[{"role": "user", "content": r["text"]}]]
        inp = proc.apply_chat_template(msgs, task="document", tokenize=True, return_dict=True, return_tensors="pt",
                                       processor_kwargs={"padding": "longest", "truncation": True,
                                                         "max_length": a.max_len})
        inp = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in inp.items()}
        with torch.inference_mode():
            ref = m(**inp).embeddings[0].float().cpu().numpy()
            rb = mb(**inp).embeddings[0].float().cpu().numpy()
        keep = inp["attention_mask"][0].bool().cpu().numpy()
        ref, rb = ref[keep], rb[keep]
        cos_tb.append((ref * rb).sum(1) / (np.linalg.norm(ref, axis=1) * np.linalg.norm(rb, axis=1) + 1e-12))
        if ref.shape != mine.shape:
            print(json.dumps({"id": r["id"], "shape_mismatch": [list(ref.shape), list(mine.shape)]}))
            continue
        c = (ref * mine).sum(1) / (np.linalg.norm(ref, axis=1) * np.linalg.norm(mine, axis=1) + 1e-12)
        cos_all.append(c)
    for label, cc in ((f"neomme late {a.tag} vs transformers fp32", cos_all),
                      ("neomme late transformers bf16 vs transformers fp32", cos_tb)):
        c = np.concatenate(cc)
        print(json.dumps({"label": label, "items": len(cc), "rows": int(len(c)), "min_cos": float(c.min()),
                          "median_cos": float(np.median(c)), "below_0.999": int((c < 0.999).sum())}))


if __name__ == "__main__":
    main()
