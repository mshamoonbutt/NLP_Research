#!/usr/bin/env bash
# Exp 2 full Phase-1 sweep: 3 SLMs x 4 conditions x all finalized families
# (748 families -> 8,976 primary responses for the current snapshot), then
# aggregation. One Python process; each model is loaded once for all
# conditions. Resumable: re-run the same command after an interruption.
#
# Prereqs: scripts/exp0_finalize_data.py (finalized dataset + split) and a
# PASS judge validation manifest from scripts/calibrate_judge.py.
#
# Override via env: OUTDIR=outputs/exp2/main MODELS="qwen25 phi3 llama32"
set -euo pipefail
cd "$(dirname "$0")/.."

OUTDIR="${OUTDIR:-outputs/exp2/main}"
MODELS="${MODELS:-qwen25 phi3 llama32}"
CONDITIONS="${CONDITIONS:-CS EN RU UR}"

echo "[sweep] outdir: $OUTDIR  models: $MODELS  conditions: $CONDITIONS"
START=$(date +%s)
# shellcheck disable=SC2086
python -m csjail.run_eval --out-dir "$OUTDIR" --models $MODELS --conditions $CONDITIONS
python -m csjail.aggregate "$OUTDIR"
python scripts/exp3_isolation.py --results "$OUTDIR"
END=$(date +%s)
echo "[sweep] DONE in $(( (END - START) / 60 )) min -> $OUTDIR"
