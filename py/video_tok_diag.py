"""What sentence-transformers feeds EmbeddingGemma 2 for a video: processor outputs (shapes) and the token row
(counts of each special token, decoded text with runs collapsed). Compare with flashembed's row:
per frame <|image> + n_soft x <|video|> + <image|>.

  python py/video_tok_diag.py SNAPSHOT video1 [...]
"""
import itertools
import sys

import numpy as np


def main():
    snap, paths = sys.argv[1], sys.argv[2:]
    import torch
    from sentence_transformers import SentenceTransformer

    m = SentenceTransformer(snap, device="cpu", config_kwargs={"audio_config": None})
    tr = m[0]
    proc = tr.processor
    orig = proc.__call__
    seen = {}

    def wrap(*a, **k):
        out = orig(*a, **k)
        seen["out"] = out
        vids = k.get("videos") or []
        seen["vid"] = [(type(v).__name__, getattr(v, "shape", None) or (v if isinstance(v, str) else None)) for v in vids]
        seen["vk"] = k.get("videos_kwargs")
        seen["kw"] = {kk: (type(v).__name__ if not isinstance(v, (str, int, float)) else v) for kk, v in k.items()}
        return out

    proc.__call__ = wrap
    type(proc).__call__ = lambda self, *a, **k: wrap(*a, **k)
    tok = proc.tokenizer
    for p in paths:
        seen.clear()
        f = m.tokenize([{"video": p}]) if hasattr(m, "tokenize") else None
        out = seen.get("out", f)
        print(p.split("/")[-1], "call kwargs", seen.get("kw"))
        for k, v in (out or {}).items():
            print("  ", k, getattr(v, "shape", v) if hasattr(v, "shape") else (v if not isinstance(v, list) else len(v)))
        print("   videos arg:", seen.get("vid"), "videos_kwargs:", {kk: str(vv)[:80] for kk, vv in (seen.get("vk") or {}).items()})
        ref = proc.video_processor(videos=[p], return_tensors="np")["pixel_values_videos"]
        st = out["pixel_values_videos"].float().numpy()
        n = min(len(ref), len(st))
        dd = np.abs(ref[:n].astype(np.float64) - st[:n])
        print("   ST pixels vs video_processor(path):", st.shape, ref.shape, "max|d|*255 %.2f mean %.4f" % (dd.max() * 255, dd.mean() * 255),
              "per frame max", [round(float(x) * 255, 1) for x in dd.reshape(n, -1).max(1)][:12])
        ids = out["input_ids"][0].tolist()
        runs = [(t, len(list(g))) for t, g in itertools.groupby(ids)]
        txt = " ".join(f"{tok.convert_ids_to_tokens(t)}x{n}" if n > 1 else tok.convert_ids_to_tokens(t) for t, n in runs)
        print("   row:", len(ids), "tokens:", txt[:1500])


if __name__ == "__main__":
    main()
