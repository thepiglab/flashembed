#!/usr/bin/env bash
# EmbeddingGemma 2 Matryoshka (MRL) parity: for every modality and each supported output size (768/512/256/128),
# sentence-transformers' encode(truncate_dim=d, normalize_embeddings=True) vs `flashembed bench --dim d`.
# Reports min cosine per (modality, dim), unit norms, and top-k neighbour agreement (each item vs all others).
#   EMBED_ENV=<site env> [BIN=$EMBED_ROOT/bin/flashembed] [ONNX=$EMBED_ROOT/onnx] sbatch --export=ALL slurm/mrl.sbatch
# REPORT_ONLY=<out dir> re-runs only the report.
# Needs the inputs from setup.sbatch (texts.jsonl, images.jsonl) and media.sbatch (media.jsonl).
#SBATCH --job-name=flashembed-mrl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
set -euo pipefail
source "${EMBED_ENV:?export EMBED_ENV=<site env>}"
T=${THREADS:-16}; ONNX=${ONNX:-$EMBED_ROOT/onnx}; BIN=${BIN:-$EMBED_ROOT/bin/flashembed}
NT=${NTEXT:-32}; NI=${NIMAGE:-16}; NM=${NMEDIA:-4}
IN=$EMBED_ROOT/inputs; OUT=${OUT:-$EMBED_ROOT/out/mrl-${SLURM_JOB_ID:-$$}}; mkdir -p "$OUT/lists"
export ORT_DYLIB_PATH=$(ls "$VENV"/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.so.* | head -1)
export OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T
[[ -n "${REPORT_ONLY:-}" ]] && { cd "$EMBED_CODE/py"; exec python mrl_report.py "$REPORT_ONLY" "$T"; }
echo "cpu=$(lscpu | sed -n 's/^Model name: *//p')"
cd "$EMBED_CODE/py"
G=$(python -c "import json;print(json.load(open('$EMBED_ROOT/models.json'))['google/embeddinggemma-2']['path'])")
# inputs: documents and queries (card prompts), images, audio and video across formats
python - "$IN" "$OUT/lists" "$NT" "$NI" "$NM" <<'PY'
import json, sys
from pathlib import Path
inp, out, nt, ni, nm = Path(sys.argv[1]), Path(sys.argv[2]), *map(int, sys.argv[3:])
rows = lambda f: [json.loads(l) for l in open(inp / f)] if (inp / f).exists() else []
docs = rows("texts.jsonl")[:nt]
qs = [{"id": f"q-{r['id']}", "text": "task: search result | query: " + r["text"].split("text: ", 1)[-1][:200]}
      for r in docs[: nt // 2]]
w = lambda name, rs: (out / name).write_text("".join(json.dumps(r) + "\n" for r in rs))
w("texts.jsonl", docs + qs)
w("images.jsonl", rows("images.jsonl")[:ni])
m = rows("media.jsonl")
for kind in ("audio", "video"):
    sel, seen = [], set()
    for r in m:  # one per format first, then fill
        if r["kind"] == kind and r["format"] not in seen and int(r.get("seconds", 0)) <= 30:
            sel.append(r); seen.add(r["format"])
    w(f"{kind}.jsonl", sel[: max(nm, len(seen))])
PY
REL=$(realpath --relative-to="$IN" "$OUT/lists")
for spec in "text texts" "image images" "audio audio" "video video"; do
  set -- $spec; mod=$1 f=$REL/$2.jsonl
  n=$(wc -l < "$IN/$f"); [[ $n -eq 0 ]] && { echo "$mod: no inputs, skipped"; continue; }
  C="--model gemma --modality $mod --threads $T --batch 4 --n $n --warmup 0 --inputs $IN --out $OUT --save --max-len 8192 --texts-file $f --images-file $f"
  for d in 768 512 256 128; do
    PYD=""; [[ $d != 768 ]] && PYD="--truncate-dim $d"
    python py_bench.py $C $PYD --tag-suffix "_d$d" > /dev/null 2>"$OUT/py-$mod-$d.err" || { echo "py $mod $d FAILED"; tail -5 "$OUT/py-$mod-$d.err"; }
    "$BIN" bench $C --dim $d --tag-suffix "_d$d" --onnx "$ONNX" --snapshot "$G" > /dev/null 2>"$OUT/rs-$mod-$d.err" || { echo "rs $mod $d FAILED"; tail -5 "$OUT/rs-$mod-$d.err"; }
  done
done
python mrl_report.py "$OUT" "$T"
