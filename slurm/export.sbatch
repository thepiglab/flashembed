#!/usr/bin/env bash
# Export ONNX graphs (FP32) with wrapper-vs-HF and ONNX-vs-wrapper checks.
#   EMBED_ENV=<site env> [WHICH="gemma_lm ..." (default: all six graphs)] [ATTN=flash|eager|sdpa] [OUTDIR=$EMBED_ROOT/onnx] sbatch --export=ALL slurm/export.sbatch
#SBATCH --job-name=flashembed-export
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
set -euo pipefail
source "${EMBED_ENV:?export EMBED_ENV=<site env>}"
WHICH=${WHICH:-"gemma_lm gemma_vision gemma_vision_dyn gemma_audio neomme_text neomme_image"}
cd "$EMBED_CODE/py"
# one Python process per graph: export-time monkeypatches (norm rewrite, attention backend) must not leak from one
# graph's export into the next one's HF reference check (it did: the audio tower's check dropped to cos 0.46)
for w in $WHICH; do
  python export_onnx.py --out "${OUTDIR:-$EMBED_ROOT/onnx}" --inputs "$EMBED_ROOT/inputs" --threads 16 \
    --attn ${ATTN:-flash} --fuse-norm ${LEGACY:+--legacy-export} --which $w 2>&1 | grep -v -i "warn" | grep -v "Loading weights" | cut -c1-400
  mv -f "${OUTDIR:-$EMBED_ROOT/onnx}/export-summary-${SLURM_JOB_ID:-x}.json" "${OUTDIR:-$EMBED_ROOT/onnx}/export-summary-$w.json" 2>/dev/null || true
done
echo "done $(date -Is)"
