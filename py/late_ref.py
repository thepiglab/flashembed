"""Late-chunking reference for EmbeddingGemma 2 (flashembed-gpu --modality text_late) on the model-card runtime:
sentence-transformers' transformer module gives the final hidden states of the whole tokenized document (or of each
window); the span's tokens are averaged and the ST modules after Pooling (Dense, Dense, Normalize) are applied, i.e.
the span mean takes the place of the Pooling(mean) output.

Same layout rules as flashembed-gpu (src/bin/flashembed_gpu.rs, late_doc): a span covers the document tokens (not
special, not in the prompt prefix) whose first character is inside [start, end); a span in which no token starts
takes the token containing start. Documents longer than --max-len are split into windows of
[bos + prefix tokens] + body slice + [eos], slices overlapping by --overlap tokens, and each span is pooled in the
window that holds it with the largest margin (else the one overlapping it most, clipped).

  python py/late_ref.py --snapshot SNAP --input docs.jsonl --out DIR [--dtype float32] [--max-len 8192] [--overlap 1024]

Writes DIR/ref-late-<dtype>.dense.npy, .ids.json (span ids, input order) and .spans.json (window per span).
"""
import argparse
import bisect
import json
from pathlib import Path

import numpy as np


def layout(tok, row, wmax, overlap):
    text, prefix = row["text"], int(row.get("prefix_chars", 0))
    enc = tok(text, add_special_tokens=True, return_offsets_mapping=True, return_special_tokens_mask=True)
    ids, offs, special = enc["input_ids"], enc["offset_mapping"], enc["special_tokens_mask"]
    n = len(ids)
    lead = 0
    while lead < n and special[lead]:
        lead += 1
    trail = 0
    while trail < n - lead and special[n - 1 - trail]:
        trail += 1
    content = list(range(lead, n - trail))
    nh = 0
    while nh < len(content) and offs[content[nh]][0] < prefix:
        nh += 1
    hdr, body = content[:nh], content[nh:]
    starts_c = [offs[k][0] for k in body]
    placed = []
    for sp in row["spans"]:
        c0, c1 = sp["start"], sp["end"]
        b0, b1 = bisect.bisect_left(starts_c, c0), bisect.bisect_left(starts_c, c1)
        sid = sp["id"] if isinstance(sp["id"], str) else json.dumps(sp["id"])
        if b0 < b1:
            placed.append((b0, b1, sid))
        else:
            b = b0 - 1 if b0 > 0 and offs[body[b0 - 1]][1] > c0 else min(b0, len(body) - 1)
            placed.append((b, b + 1, sid))
    if n <= wmax:
        return [ids], [(0, body[b0], body[b1 - 1] + 1, sid) for b0, b1, sid in placed]
    head = [ids[k] for k in range(lead)] + [ids[k] for k in hdr]
    tail = [ids[k] for k in range(n - trail, n)]
    cap = wmax - len(head) - len(tail)
    stride = cap - min(overlap, cap // 2)
    starts = [0]
    while starts[-1] + cap < len(body):
        starts.append(starts[-1] + stride)
    end = lambda s: min(s + cap, len(body))
    wins = [head + [ids[k] for k in body[s:end(s)]] + tail for s in starts]
    out = []
    for b0, b1, sid in placed:
        best = None
        for w, s in enumerate(starts):
            if s <= b0 and b1 <= end(s):
                m = min(b0 - s, end(s) - b1)
                if best is None or m > best[0]:
                    best = (m, w)
        if best is None:
            ovl = [max(0, min(b1, end(s)) - max(b0, s)) for s in starts]
            w = int(np.argmax(ovl))
        else:
            w = best[1]
        s = starts[w]
        x0, x1 = max(b0, s), min(b1, end(s))
        out.append((w, len(head) + x0 - s, len(head) + x1 - s, sid))
    return wins, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--overlap", type=int, default=1024)
    a = ap.parse_args()
    import torch
    from sentence_transformers import SentenceTransformer

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    m = SentenceTransformer(a.snapshot, device="cuda", config_kwargs={"audio_config": None},
                            model_kwargs={"torch_dtype": getattr(torch, a.dtype)})
    m.eval()
    tok = m.tokenizer
    tok = getattr(tok, "tokenizer", tok)  # a processor wraps the tokenizer
    mods = list(m)
    pool_i = next(i for i, x in enumerate(mods) if type(x).__name__ == "Pooling")
    head = mods[pool_i + 1:]
    dtype = next(m.parameters()).dtype

    def hidden_states(ids):
        x = torch.tensor([ids], device="cuda")
        mask = torch.ones_like(x)
        try:
            return mods[0]({"input_ids": x, "attention_mask": mask})["token_embeddings"][0].float()
        except Exception:  # module without a plain text-only forward: call the underlying model
            am = getattr(mods[0], "auto_model", None) or getattr(mods[0], "model")
            return am(input_ids=x, attention_mask=mask).last_hidden_state[0].float()

    vecs, ids, info = [], [], []
    with torch.inference_mode():
        for line in open(a.input):
            if not line.strip():
                continue
            row = json.loads(line)
            wins, spans = layout(tok, row, a.max_len, a.overlap)
            hid = {}
            for w, lo, hi, sid in spans:
                if w not in hid:
                    hid[w] = hidden_states(wins[w])
                f = {"sentence_embedding": hid[w][lo:hi].mean(0, keepdim=True).to(dtype)}
                for mod in head:
                    f = mod(f)
                vecs.append(f["sentence_embedding"][0].float().cpu().numpy())
                ids.append(sid)
                info.append({"id": sid, "doc": row["id"], "window": w, "windows": len(wins), "tokens": hi - lo})
    a.out.mkdir(parents=True, exist_ok=True)
    tag = f"ref-late-{a.dtype}"
    np.save(a.out / f"{tag}.dense.npy", np.stack(vecs).astype(np.float32))
    json.dump(ids, open(a.out / f"{tag}.ids.json", "w"))
    json.dump(info, open(a.out / f"{tag}.spans.json", "w"))
    print(json.dumps({"reference": tag, "spans": len(ids), "long_doc_spans": sum(1 for i in info if i["windows"] > 1)}))


if __name__ == "__main__":
    main()
