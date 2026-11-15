#!/usr/bin/env bash
# EmbeddingGemma 2 Matryoshka (--dim) parity on a GPU: sentence-transformers fp32 encode(truncate_dim=d,
# normalize_embeddings=True) vs flashembed-gpu `embed --dim d` and `serve --dim d` (fp32 and bf16), and vs the CPU
# runner's `embed --dim d` on a subset, for text, images and video; plus "slice the 768-d output and re-normalise"
# vs `--dim d`. Report: py/dim_report.py.
#   EMBED_ENV=<site env> GPU_BIN=<flashembed-gpu> [CPU_BIN=<flashembed> ONNX=<graphs> ORT_DYLIB_PATH=...] \
#   REF_PY=<python with sentence-transformers + torch CUDA> PY=<python with numpy> SNAP=<embeddinggemma-2 snapshot> \
#   TEXTS=<texts.jsonl> IMAGES=<images.jsonl> [VIDEOS=<video jsonl>] [GPU_SETUP="module load cuda"] \
#   sbatch --gpus=1 --export=ALL slurm/gpu-mrl.sbatch   (from the repository root)
#SBATCH --job-name=flashembed-gpu-mrl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:30:00
set -uo pipefail
source "${EMBED_ENV:?export EMBED_ENV=<site env>}"
: "${GPU_BIN:?}" "${REF_PY:?}" "${PY:?}" "${SNAP:?}" "${TEXTS:?}" "${IMAGES:?}"
NT=${NTEXT:-1000}; NI=${NIMAGE:-200}; NV=${NVIDEO:-34}; CT=${CPU_NTEXT:-200}; CI=${CPU_NIMAGE:-50}; CV=${CPU_NVIDEO:-8}
DIMS=${DIMS:-"512 256 128"}; OUT=${OUT:-$EMBED_ROOT/out/gpu-mrl-${SLURM_JOB_ID:-$$}}; mkdir -p "$OUT/in"
CODE=${CODE:-${SLURM_SUBMIT_DIR:-$PWD}}  # repository checkout (submit from it)
eval "${GPU_SETUP:-}"  # e.g. "module load cuda/13.2.1"
nvidia-smi --query-gpu=name --format=csv,noheader
# inputs with absolute paths; texts in EmbeddingGemma 2's document format (both runtimes get the same strings)
$PY - "$TEXTS" "$IMAGES" "${VIDEOS:-}" "$OUT/in" "$NT" "$NI" "$NV" "$CT" "$CI" "$CV" <<'PY'
import json, sys, os
t, i, v, out, nt, ni, nv, ct, ci, cv = sys.argv[1:4] + [sys.argv[4]] + [int(x) for x in sys.argv[5:]]
def rows(f, n, key):
    d = os.path.dirname(os.path.abspath(f)); r = []
    for l in open(f):
        if not l.strip(): continue
        x = json.loads(l)
        if key == "text": r.append({"id": x["id"], "text": "title: none | text: " + x["text"]})
        else:
            if x.get("kind", "video") != "video": continue
            r.append({"id": x["id"], "path": os.path.join(d, x["path"])})
        if len(r) >= n: break
    return r
sets = {"text": rows(t, nt, "text"), "image": rows(i, ni, "path")}
if v: sets["video"] = rows(v, nv, "path")
caps = {"text": ct, "image": ci, "video": cv}
for m, r in sets.items():
    open(f"{out}/{m}.jsonl", "w").writelines(json.dumps(x) + "\n" for x in r)
    open(f"{out}/{m}-cpu.jsonl", "w").writelines(json.dumps(x) + "\n" for x in r[:caps[m]])
    print(m, len(r), "items,", min(len(r), caps[m]), "on the CPU runner")
PY
MODS="text image"; [[ -f $OUT/in/video.jsonl ]] && MODS="text image video"
csv() { echo "$@" | tr ' ' ','; }
echo "== card runtime (sentence-transformers fp32, truncate_dim)"
for m in $MODS; do
  $REF_PY -I $CODE/py/gpu_ref_dim.py --snapshot $SNAP --input $OUT/in/$m.jsonl --modality $m --dims $(csv 768 $DIMS) \
    --out $OUT --max-len 1024 2>&1 | grep '^{'
done
echo "== flashembed-gpu embed"
for m in $MODS; do for dt in fp32 bf16; do for d in 768 $DIMS; do
  da=(); [[ $d != 768 ]] && da=(--dim $d)
  $GPU_BIN embed --model gemma --modality $m --input $OUT/in/$m.jsonl --out $OUT --snapshot $SNAP --dtype $dt \
    --max-len 1024 --tag gpu-$m-$dt-$d "${da[@]}" > $OUT/gpu-$m-$dt-$d.log 2>&1 || { echo "FAILED gpu $m $dt $d"; tail -3 $OUT/gpu-$m-$dt-$d.log; }
done; done; done
echo "== flashembed-gpu serve (one server per dtype x dim, all modalities)"
for dt in fp32 bf16; do for d in $DIMS; do
  for m in $MODS; do echo "{\"id\": \"$m\", \"modality\": \"$m\", \"input\": \"$OUT/in/$m.jsonl\", \"out\": \"$OUT/srv-$m-$dt-$d\"}"; done \
    | $GPU_BIN serve --model gemma --snapshot $SNAP --dtype $dt --max-len 1024 --dim $d --modalities $(csv $MODS) 2>$OUT/srv-$dt-$d.err \
    | $PY -c "import json,sys;[print('serve $dt $d', {k: json.loads(l).get(k) for k in ('id','ok','items','error')}) for l in sys.stdin if not l.startswith('{\"ready')]"
done; done
if [[ -n "${CPU_BIN:-}" ]]; then
  echo "== flashembed CPU embed (subset)"
  for m in $MODS; do for d in $DIMS; do
    $CPU_BIN embed --model gemma --modality $m --input $OUT/in/$m-cpu.jsonl --out $OUT/cpu-$m-$d --onnx $ONNX --snapshot $SNAP \
      --max-len 1024 --threads ${CPU_THREADS:-8} --dim $d > $OUT/cpu-$m-$d.log 2>&1 || { echo "FAILED cpu $m $d"; tail -3 $OUT/cpu-$m-$d.log; }
  done; done
fi
echo "== report"; $PY -I $CODE/py/dim_report.py $OUT --dims $(csv $DIMS) --mods $(csv $MODS) | tee $OUT/report.txt
