"""ONNX Runtime CUDA execution provider on the stock-attention graphs (py/export_onnx.py --attn sdpa): the "reuse the
CPU runner's ONNX path on the GPU" option, measured on the same inputs as the GPU runner.

  python py/ort_cuda_bench.py --onnx DIR --model gemma|neomme --snapshot SNAP --input texts.jsonl --out OUT \
      [--batch 64] [--max-len 1024] [--provider cuda|tensorrt] [--fp16]

Length-sorted padded batches (like sentence-transformers), mean pooling + L2 norm on the host, vectors saved as
<model>-text-ort-<provider>.dense.npy + .ids.json for py/gpu_compare.py. Peak device memory is sampled with NVML.
"""
import argparse
import json
import threading
import time
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--model", choices=["gemma", "neomme"], required=True)
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--provider", default="cuda")
    ap.add_argument("--trt-precision", default="fp32", choices=["fp32", "fp16", "bf16"],
                    help="TensorRT EP builder precision (fp32 = TF32 off where the EP allows it)")
    ap.add_argument("--trt-cache", type=Path, default=None)
    a = ap.parse_args()
    import onnxruntime as ort
    import pynvml
    from tokenizers import Tokenizer

    try:
        ort.preload_dlls()
    except Exception:
        pass
    pynvml.nvmlInit()
    dev = pynvml.nvmlDeviceGetHandleByIndex(0)
    used0 = pynvml.nvmlDeviceGetMemoryInfo(dev).used
    peak = [used0]
    stop = [False]

    def sampler():
        while not stop[0]:
            peak[0] = max(peak[0], pynvml.nvmlDeviceGetMemoryInfo(dev).used)
            time.sleep(0.02)

    threading.Thread(target=sampler, daemon=True).start()
    rows = [json.loads(l) for l in open(a.input)]
    if a.n:
        rows = rows[: a.n]
    tok = Tokenizer.from_file(str(a.snapshot / "tokenizer.json"))
    tok.no_padding()
    tok.enable_truncation(a.max_len)
    if a.model == "gemma":
        texts = [f"title: none | text: {r['text']}" for r in rows]
        special = True
        graph = a.onnx / "gemma_lm.onnx"
    else:
        texts = [f"<doc>{r['text']}" for r in rows]
        special = False
        graph = a.onnx / "neomme_text.onnx"
    B, L = a.batch, a.max_len
    if a.model == "gemma":
        names = {"input_ids": "", "attention_mask": "", "mm_embeds": "x512", "mm_mask": ""}
    else:
        names = {"input_ids": "", "attention_mask": ""}
    shp = lambda b, l: ",".join(f"{n}:{b}x{l}{e}" for n, e in names.items())
    trt = {"trt_fp16_enable": a.trt_precision == "fp16", "trt_bf16_enable": a.trt_precision == "bf16",
           "trt_engine_cache_enable": True, "trt_engine_cache_path": str(a.trt_cache or a.out / "trt-cache"),
           "trt_timing_cache_enable": True, "trt_builder_optimization_level": 3,
           "trt_profile_min_shapes": shp(1, 2), "trt_profile_opt_shapes": shp(B, 512), "trt_profile_max_shapes": shp(B, L),
           "trt_max_workspace_size": 8 << 30}
    prov = {"cuda": [("CUDAExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"})],
            "tensorrt": [("TensorrtExecutionProvider", trt), ("CUDAExecutionProvider", {})]}[a.provider]
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    t_load = time.perf_counter()
    sess = ort.InferenceSession(str(graph), so, providers=prov)
    load_s = time.perf_counter() - t_load
    order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))

    def run(idx_all):
        out = {}
        for s in range(0, len(idx_all), a.batch):
            idx = idx_all[s: s + a.batch]
            enc = [tok.encode(texts[i], add_special_tokens=special).ids for i in idx]
            S = max(len(e) for e in enc)
            ids = np.zeros((len(idx), S), np.int64)
            mask = np.zeros((len(idx), S), np.int64)
            for j, e in enumerate(enc):
                ids[j, : len(e)] = e
                mask[j, : len(e)] = 1
            if a.model == "gemma":
                feeds = {"input_ids": ids, "attention_mask": mask,
                         "mm_embeds": np.zeros((len(idx), S, 512), np.float32), "mm_mask": np.zeros((len(idx), S), bool)}
                hs = sess.run(["last_hidden_state"], feeds)[0]
                m = mask[..., None].astype(np.float32)
                d = (hs * m).sum(1) / m.sum(1)
            else:
                d = sess.run(["dense"], {"input_ids": ids, "attention_mask": mask})[0]
            d = d / np.linalg.norm(d, axis=1, keepdims=True)
            for j, i in enumerate(idx):
                out[i] = d[j]
        return out

    t_w = time.perf_counter()
    run(order[: a.batch])  # warm-up (TensorRT: engine build for the profile)
    warm_s = time.perf_counter() - t_w
    t0 = time.perf_counter()
    out = run(order)
    wall = time.perf_counter() - t0
    stop[0] = True
    a.out.mkdir(parents=True, exist_ok=True)
    tag = f"{a.model}-text-ort-{a.provider}" + (f"-{a.trt_precision}" if a.provider == "tensorrt" else "")
    np.save(a.out / f"{tag}.dense.npy", np.stack([out[i] for i in range(len(texts))]).astype(np.float32))
    json.dump([r["id"] for r in rows], open(a.out / f"{tag}.ids.json", "w"))
    res = {"runtime": f"onnxruntime-{a.provider}", "ort": ort.__version__, "model": a.model, "batch": a.batch,
           "n": len(texts), "wall_s": wall, "warmup_s": warm_s, "precision": a.trt_precision if a.provider == "tensorrt" else "fp32", "items_per_s": len(texts) / wall, "load_s": load_s,
           "peak_device_used_mb": peak[0] / 2**20, "device_used_before_mb": used0 / 2**20, "max_len": a.max_len}
    print(json.dumps(res))
    with open(a.out / "results.jsonl", "a") as f:
        f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
