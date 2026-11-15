#!/usr/bin/env bash
# Quantized variants vs fp32 flashembed vs the model-card Python runtime, same node / CPUs / inputs
# (docs/QUANTIZATION.md). Rep 0 runs every runtime over the full input set and saves vectors; reps 1.. re-time
# on the first N_TIME items in rotated order. Then `flashembed fidelity` compares each variant (and fp32) with
# fp32 flashembed and with the Python runtime.
#   MODEL=gemma|neomme VARIANTS="nb8-a4 dyn8 ..." QROOT=<dir with one graph dir per variant> FP32=<fp32 graph dir>
#   INPUTS=<dir> TEXTS=<jsonl under INPUTS> IMAGES=<jsonl under INPUTS> [PAIRS=<json>] [MODS="text image"]
#   [THREADS=16] [BATCH=4] [REPS=2] [N_TIME=256] [PY=1] [BIN=...] [SNAP=...] [OUT=...]
#SBATCH --job-name=flashembed-qbench
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
set -euo pipefail
source "${EMBED_ENV:?export EMBED_ENV=<site env>}"
: "${MODEL:?}" "${VARIANTS:?}" "${QROOT:?}" "${FP32:?}" "${INPUTS:?}" "${TEXTS:?}" "${IMAGES:?}" "${BIN:?}"
MODS=${MODS:-"text image"}; THREADS=${THREADS:-16}; BATCH=${BATCH:-4}; REPS=${REPS:-2}; N_TIME=${N_TIME:-256}; PY=${PY:-1}
OUT=${OUT:-$EMBED_ROOT/out/qbench-$SLURM_JOB_ID}
export ORT_DYLIB_PATH=${ORT_DYLIB_PATH:-$(ls "$VENV"/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.so.* | head -1)}
SNAP=${SNAP:-$(python -c "import json;d=json.load(open('$EMBED_ROOT/models.json'));print(d['google/embeddinggemma-2' if '$MODEL'=='gemma' else 'Hcompany/NeoMME-260M-Retriever']['path'])")}
MAXLEN=$([[ $MODEL == gemma ]] && echo 8192 || echo 16384)
mkdir -p "$OUT"
lscpu > "$OUT/lscpu.txt"
echo "host=$(hostname) cpu=$(lscpu | sed -n 's/^Model name: *//p') vnni=$(grep -c avx512_vnni /proc/cpuinfo | head -1) bf16=$(grep -c avx512_bf16 /proc/cpuinfo | head -1) start=$(date -Is)"
echo "model=$MODEL variants=[$VARIANTS] mods=[$MODS] t=$THREADS b=$BATCH reps=$REPS out=$OUT"
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
runtimes=(py fp32 $VARIANTS); [[ $PY == 1 ]] || runtimes=(fp32 $VARIANTS)
run() {  # $1 runtime $2 modality $3 rep $4 n
  local rt=$1 mod=$2 o=$OUT/rep$3/$1 n=$4 sf=""
  [[ $3 == 0 ]] && sf=--save
  local c="--model $MODEL --modality $mod --threads $THREADS --batch $BATCH --n $n --max-len $MAXLEN --inputs $INPUTS --texts-file $TEXTS --images-file $IMAGES --out $o $sf"
  case $rt in
    py)   (cd "$EMBED_CODE/py" && python py_bench.py $c) ;;
    fp32) FLASHEMBED_VARIANT=fp32 "$BIN" bench $c --onnx "$FP32" --snapshot "$SNAP" ;;
    *)    FLASHEMBED_VARIANT=$rt "$BIN" bench $c --onnx "$QROOT/$rt" --snapshot "$SNAP" ;;
  esac 2>&1 | grep -E '^\{|Error|error|Traceback|panicked' | sed 's/"batch_latency_s": \[[^]]*\], //;s/"policy_tuning":\[[^]]*\],//' | cut -c1-400 \
    || echo "FAILED rt=$rt mod=$mod rep=$3"
}
for rep in $(seq 0 $((REPS - 1))); do
  k=$((rep % ${#runtimes[@]})); order=("${runtimes[@]:$k}" "${runtimes[@]:0:$k}")
  n=100000; [[ $rep -gt 0 ]] && n=$N_TIME
  for mod in $MODS; do
    for rt in "${order[@]}"; do echo "== rep=$rep mod=$mod rt=$rt $(date +%T) load=$(cut -d' ' -f1 /proc/loadavg)"; run $rt $mod $rep $n; done
  done
done
# fidelity: every variant vs fp32 flashembed; fp32 and every variant vs the Python runtime (tags renamed py- -> rs-)
P=${PAIRS:+--pairs $PAIRS}
if [[ $PY == 1 && -d $OUT/rep0/py ]]; then
  mkdir -p "$OUT/rep0/py-as-rs"
  for f in "$OUT"/rep0/py/py-*; do ln -sf "$f" "$OUT/rep0/py-as-rs/rs-${f##*/py-}"; done
fi
for v in fp32 $VARIANTS; do
  [[ $v != fp32 ]] && "$BIN" fidelity --ref "$OUT/rep0/fp32" --var "$OUT/rep0/$v" $P --json "$OUT/fid-$v-vs-fp32.json" | cut -c1-600
  [[ -d $OUT/rep0/py-as-rs ]] && "$BIN" fidelity --ref "$OUT/rep0/py-as-rs" --var "$OUT/rep0/$v" $P --json "$OUT/fid-$v-vs-py.json" | cut -c1-600
done
echo "done $(date -Is)"
