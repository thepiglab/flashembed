"""Prototype: cuDNN fused SDPA (cudnn-frontend graph API) on flashembed-gpu's attention layout, before wiring it into
the engine. Packed variable-length sequences (ragged offsets), one qkv buffer [ntok, (Hq + 2 Hkv) * D] as the engine
keeps it, GQA, optional symmetric sliding window |i - j| <= w, bf16 in / out, FP32 softmax.

Per case: correctness vs a torch FP32 reference (per sequence, explicit mask) and mean ms / TFLOP/s per call.

  python py/cudnn_sdpa_proto.py
"""
import inspect
import json
import math

import torch

import cudnn


def ref_attn(qkv, cu, Hq, Hkv, D, w):
    ntok = qkv.shape[0]
    out = torch.empty(ntok, Hq * D, dtype=torch.float32, device=qkv.device)
    x = qkv.float()
    for b in range(len(cu) - 1):
        s0, s1 = cu[b], cu[b + 1]
        n = s1 - s0
        q = x[s0:s1, : Hq * D].view(n, Hq, D).transpose(0, 1)
        k = x[s0:s1, Hq * D:(Hq + Hkv) * D].view(n, Hkv, D).transpose(0, 1).repeat_interleave(Hq // Hkv, 0)
        v = x[s0:s1, (Hq + Hkv) * D:].view(n, Hkv, D).transpose(0, 1).repeat_interleave(Hq // Hkv, 0)
        s = q @ k.transpose(1, 2) / math.sqrt(D)
        if w >= 0:
            i = torch.arange(n, device=qkv.device)
            s = s.masked_fill((i[:, None] - i[None, :]).abs() > w, float("-inf"))
        out[s0:s1] = (s.softmax(-1) @ v).transpose(0, 1).reshape(n, Hq * D)
    return out


BANDS = [(0, 0), (1, 0), (0, 1), (1, 1)]


def build(B, smax, Hq, Hkv, D, w, ld, ld_out, variant):
    g = cudnn.pygraph(io_data_type=cudnn.data_type.BFLOAT16, intermediate_data_type=cudnn.data_type.FLOAT,
                      compute_data_type=cudnn.data_type.FLOAT)
    rag = g.tensor(dim=[B + 1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT32)
    rago = g.tensor(dim=[B + 1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT32)
    q = g.tensor(dim=[B, Hq, smax, D], stride=[smax * ld, D, ld, 1], data_type=cudnn.data_type.BFLOAT16)
    k = g.tensor(dim=[B, Hkv, smax, D], stride=[smax * ld, D, ld, 1], data_type=cudnn.data_type.BFLOAT16)
    v = g.tensor(dim=[B, Hkv, smax, D], stride=[smax * ld, D, ld, 1], data_type=cudnn.data_type.BFLOAT16)
    for t in (q, k, v):
        t.set_ragged_offset(rag)
    sq = g.tensor(dim=[B, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT32)
    skv = g.tensor(dim=[B, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT32)
    kw = dict(q=q, k=k, v=v, attn_scale=1.0 / math.sqrt(D), use_padding_mask=True, seq_len_q=sq, seq_len_kv=skv)
    params = inspect.signature(g.sdpa).parameters if hasattr(inspect, "signature") else {}
    if "generate_stats" in params:
        kw["generate_stats"] = False
    else:
        kw["is_inference"] = True
    if w >= 0:
        # variant 0: left/right band bounds (keys j with i - L < j < i + R allowed); variant 1: sliding_window_length
        L, R = BANDS[variant]
        kw.update(diagonal_band_left_bound=w + L, diagonal_band_right_bound=w + R,
                  diagonal_alignment=cudnn.diagonal_alignment.TOP_LEFT)
    o, _ = g.sdpa(name="sdpa", **kw)
    o.set_output(True).set_dim([B, Hq, smax, D]).set_stride([smax * ld_out, D, ld_out, 1])
    o.set_ragged_offset(rago)
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    g.check_support()
    g.build_plans()
    return g, dict(q=q, k=k, v=v, o=o, rag=rag, rago=rago, sq=sq, skv=skv)


def run_case(label, lens, Hq, Hkv, D, w, iters=20):
    dev = "cuda"
    B, smax = len(lens), max(lens)
    cu = [0]
    for n in lens:
        cu.append(cu[-1] + n)
    ld, ld_out = (Hq + 2 * Hkv) * D, Hq * D
    torch.manual_seed(0)
    qkv = (torch.rand(cu[-1], ld, device=dev) * 2 - 1).to(torch.bfloat16)
    out = torch.zeros(cu[-1], ld_out, device=dev, dtype=torch.bfloat16)
    rag = torch.tensor([c * ld for c in cu], device=dev, dtype=torch.int32).view(-1, 1, 1, 1)
    rago = torch.tensor([c * ld_out for c in cu], device=dev, dtype=torch.int32).view(-1, 1, 1, 1)
    sl = torch.tensor(lens, device=dev, dtype=torch.int32).view(-1, 1, 1, 1)
    for variant in (range(len(BANDS)) if w >= 0 else (0,)):
        try:
            g, t = build(B, smax, Hq, Hkv, D, w, ld, ld_out, variant)
        except Exception as ex:  # unsupported variant / shape
            print(json.dumps({"case": label, "band": BANDS[variant] if w >= 0 else None, "error": str(ex).splitlines()[0][:200]}))
            continue
        measure(label, variant, g, t, qkv, out, rag, rago, sl, cu, lens, Hq, Hkv, D, w, iters)


def measure(label, variant, g, t, qkv, out, rag, rago, sl, cu, lens, Hq, Hkv, D, w, iters):
    dev = "cuda"
    ws = torch.empty(max(1, g.get_workspace_size()), device=dev, dtype=torch.uint8)
    pack = {t["q"]: qkv, t["k"]: qkv[:, Hq * D:], t["v"]: qkv[:, (Hq + Hkv) * D:], t["o"]: out, t["rag"]: rag,
            t["rago"]: rago, t["sq"]: sl, t["skv"]: sl}
    g.execute(pack, ws)
    torch.cuda.synchronize()
    ref = ref_attn(qkv, cu, Hq, Hkv, D, w)
    o = out.float()
    cos = torch.nn.functional.cosine_similarity(o, ref, dim=1)
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters):
        g.execute(pack, ws)
    e1.record()
    torch.cuda.synchronize()
    ms = e0.elapsed_time(e1) / iters
    pairs = 0
    for n in lens:
        pairs += n * n if w < 0 or w >= n else sum(min(i + w, n - 1) - max(i - w, 0) + 1 for i in range(n))
    tf = 4 * pairs * Hq * D / (ms * 1e-3) / 1e12
    print(json.dumps({"case": label, "band": BANDS[variant] if w >= 0 else None, "ms": round(ms, 4), "tflops": round(tf, 1),
                      "min_row_cos": round(float(cos.min()), 6), "max_abs_err": round(float((o - ref).abs().max()), 5),
                      "workspace_mb": round(g.get_workspace_size() / 2**20, 2)}))


def main():
    doc = (cudnn.pygraph.sdpa.__doc__ or "").splitlines()
    print("\n".join(l for l in doc if "band" in l or "window" in l or "alignment" in l)[:2000])
    print("cudnn backend", cudnn.backend_version(), "frontend", cudnn.__version__, torch.cuda.get_device_name())
    # same shapes as `flashembed-gpu attn-bench` (equal lengths), then ragged mixes
    run_case("small w3 check", [40, 17], 4, 2, 64, 3)
    run_case("gemma text d256 4/2 1024x16 w512", [1024] * 16, 4, 2, 256, 512)
    run_case("neomme d64 16/4 1024x16 full", [1024] * 16, 16, 4, 64, -1)
    run_case("neomme d64 16/4 1024x16 w256", [1024] * 16, 16, 4, 64, 256)
    run_case("gemma text d512 4/1 1024x16 full", [1024] * 16, 4, 1, 512, -1)
    run_case("ragged vision d64 mix", [2520, 1260, 777, 2016, 640, 1500, 2520, 99], 12, 12, 64, -1)
    run_case("ragged text d256 w512 mix", [1024, 37, 600, 1024, 513, 9, 880, 1024, 300, 777], 4, 2, 256, 512)
    run_case("ragged neomme d64 w256 mix", [1024, 37, 600, 1024, 513, 9, 880, 1024, 300, 777], 16, 4, 64, 256)


if __name__ == "__main__":
    main()
