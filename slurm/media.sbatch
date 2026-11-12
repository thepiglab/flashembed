#!/usr/bin/env bash
# Media setup + acceptance probe for EmbeddingGemma 2 (needs ffmpeg/ffprobe on PATH via the site env):
# installs audio/video decoders into the venv, builds format/resolution/audio/video inputs, then records which
# inputs the card runtime accepts and what it builds from them.
#   EMBED_ENV=<site env> sbatch --export=ALL slurm/media.sbatch
#SBATCH --job-name=flashembed-media
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=03:00:00
set -euo pipefail
source "${EMBED_ENV:?export EMBED_ENV=<site env>}"
echo "host=$(hostname) ffmpeg=$(ffmpeg -version | head -1)"
cd "$EMBED_CODE/py"
# real media from PMC open-access metadata ranges (EMBED_PMC_RANGES); REBUILD=1 regenerates the inputs
if [[ "${REBUILD:-0}" == 1 || ! -f $EMBED_ROOT/inputs/media.jsonl ]]; then
  RANGES=${EMBED_PMC_RANGES:-"9000000:30000 7000000:30000 5000000:20000"}
  python scan_media.py --pmc-ranges $RANGES \
    --out "$EMBED_ROOT/inputs/real-media.jsonl"
  python make_media_inputs.py --out "$EMBED_ROOT/inputs" --real "$EMBED_ROOT/inputs/real-media.jsonl"
fi
python probe_media.py --inputs "$EMBED_ROOT/inputs" --out "$EMBED_ROOT/out/media-probe-$SLURM_JOB_ID" 2>&1 | grep -v -i warn | cut -c1-400
echo "done $(date -Is)"
