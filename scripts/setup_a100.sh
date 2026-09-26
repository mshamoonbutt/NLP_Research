#!/usr/bin/env bash
# One-shot GPU host setup (Linux + CUDA 12.x; name is legacy -- also used on
# the RTX 4080 / WSL2 box). Idempotent. Fails fast on missing API keys.
set -euo pipefail
cd "$(dirname "$0")/.."

: "${OPENAI_API_KEY:?OPENAI_API_KEY must be set (judge)}"
: "${HF_TOKEN:?HF_TOKEN must be set (gated Llama-3.2)}"
command -v nvidia-smi >/dev/null 2>&1 || { echo "[setup] FAIL: no nvidia-smi" >&2; exit 1; }
{ nvidia-smi || true; } | head -n 5 || true

PY="${PY:-python3}"
[ -d .venv ] || $PY -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip wheel setuptools >/dev/null
pip install -e ".[train,judge,dev]"
huggingface-cli login --token "$HF_TOKEN" || true
python -c "import vllm, transformers, torch, trl, peft; print('vllm', vllm.__version__, \
'transformers', transformers.__version__, 'torch', torch.__version__, 'trl', trl.__version__, \
'peft', peft.__version__, 'cuda', torch.cuda.is_available())"
pytest tests -q

echo "[setup] OK. Next (see RUNBOOK.md):"
echo "  python scripts/exp0_finalize_data.py --source-csv data/CS-Jail-UR_final_692.csv"
echo "  python scripts/smoke_test.py --model qwen25"
echo "  python scripts/exp1_sample_for_annotation.py --role development ..."
