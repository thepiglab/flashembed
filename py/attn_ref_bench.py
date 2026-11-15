"""torch SDPA backends (cuDNN, FlashAttention, memory-efficient) on the GPU at flashembed-gpu's attention shapes,
full bidirectional attention only, bf16: mean ms per call and TFLOP/s, to compare with `flashembed-gpu attn-bench`.

  python py/attn_ref_bench.py D:Hq:Hkv:S:nseq [...]
"""
import json
import sys

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

F = torch.nn.functional


def main():
    for sh in sys.argv[1:]:
        d, hq, hkv, s, n = (int(x) for x in sh.split(":"))
        q = torch.randn(n, hq, s, d, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(n, hkv, s, d, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(n, hkv, s, d, device="cuda", dtype=torch.bfloat16)
        kk, vv = k.repeat_interleave(hq // hkv, 1), v.repeat_interleave(hq // hkv, 1)
        for name, be in (("cudnn", SDPBackend.CUDNN_ATTENTION), ("flash", SDPBackend.FLASH_ATTENTION),
                         ("efficient", SDPBackend.EFFICIENT_ATTENTION)):
            try:
                with sdpa_kernel(be):
                    for _ in range(3):
                        F.scaled_dot_product_attention(q, kk, vv)
                    torch.cuda.synchronize()
                    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    e0.record()
                    for _ in range(20):
                        F.scaled_dot_product_attention(q, kk, vv)
                    e1.record()
                    torch.cuda.synchronize()
                    ms = e0.elapsed_time(e1) / 20
                tf = 4 * s * s * n * hq * d / (ms * 1e-3) / 1e12
                print(json.dumps({"shape": sh + ":-1", "backend": name, "ms": round(ms, 4), "tflops": round(tf, 1)}))
            except Exception as ex:  # backend does not support this shape / GPU
                print(json.dumps({"shape": sh + ":-1", "backend": name, "error": str(ex).splitlines()[0][:160]}))


if __name__ == "__main__":
    main()
