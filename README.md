# CS-Jail-UR — Code-Switched Urdu-English Jailbreak Eval for SLMs

Track 2, slice 1: inference harness + ASR evaluation pipeline.

## Quickstart (Linux A100 host)

```bash
# 1. Clone / sync
git clone <repo> csjail && cd csjail

# 2. One-shot host setup (installs vllm, checks API keys)
bash scripts/setup_a100.sh

# 3. Smoke test (<90s, runs Qwen2.5-1.5B on 10 fixture prompts)
python scripts/smoke_test.py

# 4. Judge calibration gate — MUST pass before any paper run
python scripts/calibrate_judge.py
#   exits 0 iff precision >= 0.90 on the 30+30 held-out set

# 5. Drop the real dataset (JSONL) at data/csjail_v0.jsonl, then:
python -m csjail.run_eval --model qwen25 --condition EN \
    --dataset data/csjail_v0.jsonl \
    --out results/qwen25_EN.jsonl

# 6. Full baseline sweep (3 SLMs × 4 conditions, ~30-50 min)
bash scripts/run_all_baseline.sh

# 7. Aggregate to publication-ready CSV
python -m csjail.aggregate results/*.jsonl --out results/headline_table.csv
```

## Required environment variables

| Var | Where | Why |
|---|---|---|
| `OPENAI_API_KEY` | A100 host | LLM-as-Judge (gpt-4o-mini) |
| `HF_TOKEN` | A100 host | Download Llama-3.2 / Phi-3 (gated) |

## Layout

```
csjail/         # library code (data, models, judge, asr, run_eval)
configs/        # YAML configs for models, judge, eval
data/           # dataset + judge calibration set
scripts/        # host setup + one-shot runners
tests/          # pytest unit tests
results/        # per-run JSONL outputs (gitignored)
```

## Local Windows development

The full pipeline requires CUDA + vLLM and only runs on the Linux A100 host.
For local editing, install the slim deps:

```powershell
pip install -e ".[local]"
pytest tests/       # data + asr tests run anywhere; model/judge tests skip without env
```

## Reproducibility

Every results file embeds a provenance header (model rev, vllm version, judge
version, dataset SHA-256, CLI args, git commit). Bootstrap CIs and McNemar's
p-values are seeded; rerunning with `--seed 0` reproduces exactly.

## Status

- Track 2 slice 1 (this slice): inference + ASR — **in build**
- Track 2 slice 2: DPO-LoRA mitigation — **deferred**
- Track 3: RQ5 tokenizer fragmentation, MT-Bench, Urdu-QA — **deferred**
