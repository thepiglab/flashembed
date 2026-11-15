#!/usr/bin/env bash
# Build the Rust runner and install it atomically as $EMBED_ROOT/bin/flashembed-$TAG (running jobs keep theirs).
#   EMBED_ENV=<site env> TAG=v2 [BIN_DIR=...] [FEATURES=candle] sbatch --export=ALL slurm/build.sbatch
#SBATCH --job-name=flashembed-build
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
set -euo pipefail
source "${EMBED_ENV:?export EMBED_ENV=<site env>}"
: "${TAG:?}"
cd "$EMBED_CODE"
# link on node-local disk: lld mmaps its output, which hit SIGBUS on the shared filesystem once
export CARGO_TARGET_DIR="${TMPDIR:-/tmp}/flashembed-target-$SLURM_JOB_ID"
cargo build --release -j 8 ${FEATURES:+--features $FEATURES}
BIN_DIR="${BIN_DIR:-$EMBED_ROOT/bin}"   # install dir (BIN_DIR=... keeps experimental builds apart)
mkdir -p "$BIN_DIR"
for b in flashembed flash-check linear-bench qgemm-bench ${FEATURES:+candle-neomme gemm-bench}; do
  cp "$CARGO_TARGET_DIR/release/$b" "$BIN_DIR/.$b-$TAG.tmp" && mv -f "$BIN_DIR/.$b-$TAG.tmp" "$BIN_DIR/$b-$TAG"
done
ls -la "$BIN_DIR"
rm -rf "${TMPDIR:-/tmp}/flashembed-target-$SLURM_JOB_ID"
