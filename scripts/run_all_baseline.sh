#!/usr/bin/env bash
# Full baseline sweep: 3 SLMs x 4 conditions = 12 runs.
#
# Models loop is the OUTER loop so we only pay each weights-load cost once
# (~30s for Qwen2.5-1.5B, ~90s for Phi-3-mini, ~75s for Llama-3.2-3B).
# Conditions loop inside.
#
# Override defaults via env: DATASET=path/to/file.jsonl OUTDIR=results/run42
set -euo pipefail

cd "$(dirname "$0")/.."

DATASET="${DATASET:-data/csjail_v1.jsonl}"
OUTDIR="${OUTDIR:-results/baseline_$(date -u +%Y%m%dT%H%M%SZ)}"
SEED="${SEED:-0}"
TEMP="${TEMP:-0.0}"
MAX_TOK="${MAX_TOK:-512}"
MODELS="${MODELS_OVERRIDE:-qwen25 phi3 llama32}"
CONDITIONS="${CONDITIONS_OVERRIDE:-CS EN RU UR}"

if [ ! -f "$DATASET" ]; then
  echo "[sweep] FAIL: dataset not found at $DATASET" >&2
  echo "[sweep] hint: DATASET=/path/to/csjail_v0.jsonl bash scripts/run_all_baseline.sh" >&2
  exit 1
fi

mkdir -p "$OUTDIR"
echo "[sweep] dataset:    $DATASET"
echo "[sweep] outdir:     $OUTDIR"
echo "[sweep] models:     $MODELS"
echo "[sweep] conditions: $CONDITIONS"
echo "[sweep] sampling:   temp=$TEMP max_tokens=$MAX_TOK seed=$SEED"
START=$(date +%s)

for model in $MODELS; do
  for cond in $CONDITIONS; do
    OUT="${OUTDIR}/${model}_${cond}.jsonl"
    echo
    echo "[sweep] ===== ${model} x ${cond} ====="
    python -m csjail.run_eval \
      --model "$model" \
      --condition "$cond" \
      --dataset "$DATASET" \
      --out "$OUT" \
      --temperature "$TEMP" \
      --max-tokens "$MAX_TOK" \
      --seed "$SEED"
  done
done

echo
echo "[sweep] aggregating ..."
python -m csjail.aggregate "$OUTDIR"/*.jsonl \
  --out "${OUTDIR}/headline_table.csv" \
  --with-mcnemar

END=$(date +%s)
echo
echo "[sweep] DONE in $(( (END - START) / 60 )) min"
echo "[sweep] headline:  ${OUTDIR}/headline_table.csv"
echo "[sweep] mcnemar:   ${OUTDIR}/headline_table_mcnemar.csv"
echo "[sweep] raw runs:  ${OUTDIR}/*.jsonl"
