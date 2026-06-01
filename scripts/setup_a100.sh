#!/usr/bin/env bash
# One-shot host setup for Lambda Labs / Runpod A100 (Ubuntu 22.04, CUDA 12.x).
# Idempotent — safe to re-run. Fails fast on missing API keys.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "[setup] checking required env vars ..."
: "${OPENAI_API_KEY:?OPENAI_API_KEY must be set (export OPENAI_API_KEY=sk-...)}"
: "${HF_TOKEN:?HF_TOKEN must be set (export HF_TOKEN=hf_...)}"

echo "[setup] checking CUDA ..."
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[setup] FAIL: nvidia-smi not found — are you on a GPU host?" >&2
  exit 1
fi
{ nvidia-smi || true; } | head -n 5 || true

PY="${PY:-python3}"
echo "[setup] using $($PY --version)"

if [ ! -d ".venv" ]; then
  echo "[setup] creating .venv"
  $PY -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo "[setup] upgrading pip"
pip install --upgrade pip wheel setuptools >/dev/null

echo "[setup] installing project (this pulls vllm + torch — takes ~5 min)"
pip install -e ".[dev]"

echo "[setup] huggingface-cli login"
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential || true

echo "[setup] smoke-importing vllm + transformers"
python -c "import vllm, transformers, torch; \
  print('vllm', vllm.__version__); \
  print('transformers', transformers.__version__); \
  print('torch', torch.__version__, 'cuda?', torch.cuda.is_available())"

echo "[setup] running data + asr + judge-parser unit tests"
pytest tests/ -q

echo
echo "[setup] OK. Next steps:"
echo "  1. python scripts/smoke_test.py             # <90s sanity check"
echo "  2. python scripts/calibrate_judge.py        # gate: precision >= 0.90"
echo "  3. cp /path/to/csjail_v0.jsonl data/        # drop the real dataset"
echo "  4. bash scripts/run_all_baseline.sh         # full sweep (~30-50 min)"
echo "  5. python -m csjail.aggregate results/*.jsonl --out results/headline_table.csv --with-mcnemar"
