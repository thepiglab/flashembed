#!/usr/bin/env bash
# Quick end-to-end check after a build/export: attention kernel vs f64 reference, then the reference Python runtime
# vs flashembed on a few inputs per model x modality (parity = min cosine, speed = items/s), and `flashembed embed`.
#   EMBED_ENV=<site env> [BIN=$EMBED_ROOT/bin/flashembed] [ONNX=$EMBED_ROOT/onnx] sbatch --export=ALL slurm/smoke.sbatch
# Needs the inputs from setup.sbatch (texts.jsonl, images.jsonl) and media.sbatch (media.jsonl) for audio/video.
#SBATCH --job-name=flashembed-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:30:00
set -euo pipefail
source "${EMBED_ENV:?export EMBED_ENV=<site env>}"
T=${THREADS:-16}; ONNX=${ONNX:-$EMBED_ROOT/onnx}; BIN=${BIN:-$EMBED_ROOT/bin/flashembed}
CHECK=${CHECK:-$(dirname "$BIN")/$(basename "$BIN" | sed 's/^flashembed/flash-check/')}
IN=$EMBED_ROOT/inputs; OUT=${OUT:-$EMBED_ROOT/out/smoke-${SLURM_JOB_ID:-$$}}; mkdir -p "$OUT/lists"
export ORT_DYLIB_PATH=$(ls "$VENV"/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.so.* | head -1)
export OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T
echo "cpu=$(lscpu | sed -n 's/^Model name: *//p') flags=$(grep -o -w -E 'avx2|avx512f' /proc/cpuinfo | sort -u | tr '\n' ' ')"
cd "$EMBED_CODE/py"
snap() { python -c "import json;print(json.load(open('$EMBED_ROOT/models.json'))['$1']['path'])"; }
G=$(snap google/embeddinggemma-2); N=$(snap Hcompany/NeoMME-260M-Retriever)
[[ -x "$CHECK" ]] && { echo "== flash-check"; "$CHECK" "$T" | tail -6; }
# small input lists: 4 texts, 4 images, 3 x 10 s WAV, 3 x 10 s MP4
python - "$IN" "$OUT/lists" <<'PY'
import json, sys
from pathlib import Path
inp, out = Path(sys.argv[1]), Path(sys.argv[2])
rows = lambda f: [json.loads(l) for l in open(inp / f)] if (inp / f).exists() else []
(out / "texts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows("texts.jsonl")[:4]))
(out / "images.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows("images.jsonl")[:4]))
m = rows("media.jsonl")
for kind, fmt, name in (("audio", "wav-16k", "audio"), ("video", "mp4", "video")):
    sel = [r for r in m if r["kind"] == kind and r["format"] == fmt and int(r.get("seconds", 0)) == 10][:3]
    (out / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in sel))
PY
REL=$(realpath --relative-to="$IN" "$OUT/lists")
cos() { python -c "
import numpy as np,sys
a=np.load(sys.argv[1]); b=np.load(sys.argv[2]); c=(a*b).sum(1)/np.linalg.norm(a,axis=1)/np.linalg.norm(b,axis=1); print('%.7f'%c.min())" "$1" "$2"; }
fail=0
for spec in "gemma text texts" "gemma image images" "gemma audio audio" "gemma video video" "neomme text texts" "neomme image images"; do
  set -- $spec; m=$1 mod=$2 f=$REL/$3.jsonl
  n=$(wc -l < "$IN/$f"); [[ $n -eq 0 ]] && { echo "$m $mod: no inputs, skipped"; continue; }
  S=$G; [[ $m == neomme ]] && S=$N
  C="--model $m --modality $mod --threads $T --batch 1 --n $n --inputs $IN --out $OUT --save --max-len 8192 --texts-file $f --images-file $f"
  py=$(python py_bench.py $C 2>&1 | grep -o '"items_per_s": [0-9.]*' | grep -o '[0-9.]*$' || true)
  rs=$("$BIN" bench $C --onnx "$ONNX" --snapshot "$S" 2>&1 | grep -o '"items_per_s":[0-9.]*' | grep -o '[0-9.]*$' || true)
  c=$(cos "$OUT/py-$m-$mod-t$T-b1.dense.npy" "$OUT/rs-$m-$mod-t$T-b1.dense.npy" 2>/dev/null || echo "n/a")
  echo "$m $mod: n=$n python ${py:-FAIL} items/s, flashembed ${rs:-FAIL} items/s, min cos $c"
  [[ -z "$rs" || $c == n/a ]] && fail=1
  python -c "import sys; sys.exit(0 if '$c' != 'n/a' and float('$c') >= 0.9999 else 1)" || fail=1
done
echo "== embed"
"$BIN" embed --model gemma --input "$OUT/lists/texts.jsonl" --out "$OUT/embed" --onnx "$ONNX" --snapshot "$G" \
  --threads "$T" > /dev/null 2>&1 && python -c "
import numpy as np, json
d = np.load('$OUT/embed/embeddings.dense.npy'); ids = json.load(open('$OUT/embed/embeddings.ids.json'))
print('embed:', d.shape, 'ids', len(ids), 'norms', np.round(np.linalg.norm(d, axis=1), 4).tolist())" || { echo "embed FAILED"; fail=1; }
echo "done $(date -Is) status=$([[ $fail == 0 ]] && echo OK || echo FAIL)"
exit $fail
