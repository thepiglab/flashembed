"""Where the model-card runtimes spend GPU time (torch.profiler), to decide what a GPU runner must optimise.

  python py/gpu_ref_profile.py --models-json models.json --inputs DIR --model gemma|neomme --modality text|image \
      --dtype bfloat16|float32 --batch 64 [--max-len 1024] [--n 256]

Runs the reference code path (sentence-transformers for EmbeddingGemma 2, transformers NeoMMEForRetrieval for NeoMME)
on the first --n items (longest first, like the encoders' own sorting), profiles one pass after a warm-up and prints
CUDA kernel time grouped into GEMM / attention / norm+elementwise / copy+other, plus the top kernels.
"""
from __future__ import annotations

import argparse
import collections
import json
import time
from pathlib import Path


def classify(name: str) -> str:
    n = name.lower()
    if any(k in n for k in ("fmha", "flash", "attention", "efficient_attention", "softmax")):
        return "attention"
    if any(k in n for k in ("gemm", "cutlass", "cublas", "sm90_xmma", "sm100", "nvjet", "ampere_", "matmul")):
        return "gemm"
    if any(k in n for k in ("memcpy", "memset", "copy", "cat", "index", "gather", "scatter")):
        return "copy/index"
    if any(k in n for k in ("reduce", "norm", "mean", "pow", "rsqrt")):
        return "reduction/norm"
    return "elementwise/other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-json", type=Path, required=True)
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--model", choices=["gemma", "neomme"], required=True)
    ap.add_argument("--modality", choices=["text", "image"], default="text")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    import torch
    from PIL import Image
    from torch.profiler import ProfilerActivity, profile

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dtype = getattr(torch, a.dtype)
    snaps = json.load(open(a.models_json))
    rows = [json.loads(l) for l in open(a.inputs / ("texts.jsonl" if a.modality == "text" else "images.jsonl"))]
    if a.modality == "text":
        rows.sort(key=lambda r: -len(r["text"]))
    rows = rows[: a.n]
    if a.model == "gemma":
        from sentence_transformers import SentenceTransformer

        cfg = {"text": {"vision_config": None, "audio_config": None}, "image": {"audio_config": None}}[a.modality]
        m = SentenceTransformer(snaps["google/embeddinggemma-2"]["path"], device="cuda", config_kwargs=cfg,
                                model_kwargs={"torch_dtype": dtype})
        m.max_seq_length = a.max_len
        items = ([f"title: none | text: {r['text']}" for r in rows] if a.modality == "text"
                 else [{"image": str(a.inputs / r["path"])} for r in rows])

        def run():
            m.encode(items, batch_size=a.batch, normalize_embeddings=True, show_progress_bar=False)
    else:
        from transformers import NeoMMEForRetrieval, NeoMMEProcessor

        snap = snaps["Hcompany/NeoMME-260M-Retriever"]["path"]
        proc = NeoMMEProcessor.from_pretrained(snap)
        m = NeoMMEForRetrieval.from_pretrained(snap, dtype=dtype).cuda().eval()
        items = ([r["text"] for r in rows] if a.modality == "text"
                 else [Image.open(a.inputs / r["path"]).convert("RGB") for r in rows])

        def run():
            for s in range(0, len(items), a.batch):
                chunk = items[s: s + a.batch]
                if a.modality == "text":
                    msgs = [[{"role": "user", "content": x}] for x in chunk]
                    kw = {"padding": "longest", "truncation": True, "max_length": a.max_len}
                else:
                    msgs = [[{"role": "user", "content": [{"type": "image", "image": x}]}] for x in chunk]
                    kw = {"padding": "longest"}
                inp = proc.apply_chat_template(msgs, task="document", tokenize=True, return_dict=True,
                                               return_tensors="pt", processor_kwargs=kw)
                inp = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in inp.items()}
                with torch.inference_mode():
                    torch.nn.functional.normalize(m(**inp).dense_embeddings.float(), dim=-1).cpu()

    run()
    torch.cuda.synchronize()
    t = time.perf_counter()
    run()
    torch.cuda.synchronize()
    wall = time.perf_counter() - t
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        run()
        torch.cuda.synchronize()
    groups, per, launches = collections.Counter(), collections.Counter(), 0
    for e in prof.key_averages():
        dt = getattr(e, "device_time_total", None) or getattr(e, "cuda_time_total", 0)
        if e.device_type.name == "CUDA" and dt:
            groups[classify(e.key)] += dt
            per[e.key] += dt
            launches += e.count
    tot = sum(groups.values())
    res = {"model": a.model, "modality": a.modality, "dtype": a.dtype, "batch": a.batch, "n": len(items),
           "wall_s_unprofiled": wall, "items_per_s": len(items) / wall, "gpu_kernel_ms": tot / 1e3,
           "kernel_launches": launches, "gpu_busy_frac": tot / 1e6 / wall,
           "groups_pct": {k: round(100 * v / tot, 1) for k, v in groups.most_common()},
           "top": [(k[:90], round(v / 1e3, 2)) for k, v in per.most_common(25)]}
    print(json.dumps(res, indent=1))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "a") as f:
            f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
