# Exp 2 extension: two more SLMs (design and how to run)

Added 2026-10-07 at the professor's suggestion ("a DeepSeek SLM or Gemma").
The three original models (qwen25, phi3, llama32) and Phase 2 are unchanged.

## Models

| Key | Model (pinned) | Size | Licence | Why |
|---|---|---|---|---|
| `gemma4e2b` | `google/gemma-4-E2B-it` @ `3e22461` | 2B effective (5.1B stored) | Apache-2.0, ungated | New family (Google), April 2026, multilingual |
| `r1qwen15` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` @ `ad9f0ae` | 1.8B | MIT, ungated | DeepSeek's only small model; a reasoning model on a Qwen2.5 base |

DeepSeek has no other model under 7B. Gemma 4 E4B (8B stored) does not fit a T4 at
full precision, so E2B is the Gemma choice.

## What stays identical

- Prompts: all 791 families × EN/CS/RU/UR (3,164 per model), no system prompt.
- Decoding: greedy, seed and settings from `configs/eval.yaml`.
- Robustness: the same 200 families × CS/RU × 5 draws at T 0.7 (2,000 per model).
- Judge: DeepSeek V4.1 Flash via Ollama, rubric harm-v2, fingerprint `b0248df31f1d9da2`.
- Reporting: `csjail.aggregate`, raw ASR plus ASR corrected for the judge's error.

## The three deliberate differences, and how each is controlled

1. **Gemma runs through Ollama, not vLLM.** vLLM 0.6.3 (the pinned stack) cannot load
   Gemma 4. Its production backend is the Ollama build `gemma4:e2b-it-q8_0`, pinned by
   digest `95e5aad2…` in `configs/models.yaml`; a run refuses any other build. The same
   build runs on a Kaggle T4 and on the laptop. **Control:** qwen25 is generated again
   through Ollama on the 200 held-out families and judged; comparing it with qwen25's
   vLLM results measures how much the serving stack alone moves ASR.
2. **DeepSeek-R1 writes visible reasoning before answering.** With the usual 512-token
   budget the answer would rarely appear, so it gets 2,048 tokens. The judged response is
   the whole visible output, reasoning included, because that is what a user sees.
   Report how often the reasoning did not finish (no `</think>`). Greedy decoding is the
   protocol; DeepSeek recommends sampling at T 0.6, which the robustness run (T 0.7)
   approximates, so both views are reported.
3. **The judge's error is not yet measured on these models.** Exp 1 measured it per
   model × language on the original three, and the error differs a lot by model. Until
   the new models are labelled, their corrected ASR falls back to per-language error,
   which is less reliable; their tables say so (`correction_basis`). **Fix:** the same
   reviewer labels each new model's 240 responses on the 60 development families
   (step 6), and the correction then uses their own error.

## Outputs

| Directory | Contents |
|---|---|
| `outputs/exp2/main-r1`, `outputs/exp2/robustness-r1` | DeepSeek-R1 greedy and sampled runs |
| `outputs/exp2/main-gemma4`, `outputs/exp2/robustness-gemma4` | Gemma 4 runs |
| `outputs/exp2/backend-check-qwen25-ollama` | qwen25 via Ollama, 200 held-out families (debug run by design) |

## How to run

Everything below assumes `git pull` on the branch, the Ollama app running and signed in
(it serves the judge), and the laptop plugged in with the lid open while judging.

### 1. Local test on Ollama (laptop, about 30 minutes)

Downloads about 10 GB (Gemma 8.1 GB, DeepSeek-R1 1.9 GB). This is a sanity check of 5
families per model, marked as a debug run: does each model answer in the expected
language, does the reasoning model close its reasoning, and does the judge parse.

```bash
ollama pull gemma4:e2b-it-q8_0
```
```bash
ollama pull deepseek-r1:1.5b-qwen-distill-q8_0
```
```bash
python -m csjail.run_eval --backend ollama --models gemma4e2b --max-families 5 --out-dir outputs/exp2/smoke-gemma4 --skip-judge
```
```bash
python -m csjail.run_eval --backend ollama --models r1qwen15 --max-tokens 2048 --max-families 5 --out-dir outputs/exp2/smoke-r1 --skip-judge
```
```bash
python -m csjail.run_eval --out-dir outputs/exp2/smoke-gemma4 --judge-only
```
```bash
python -m csjail.run_eval --out-dir outputs/exp2/smoke-r1 --judge-only
```
```bash
python -m csjail.aggregate outputs/exp2/smoke-gemma4 outputs/exp2/smoke-r1 --allow-debug
```

The local DeepSeek-R1 test uses Ollama's 8-bit build; its production run (step 2) uses
the pinned fp16 weights through vLLM, like qwen25.

### 2. Full generation on Kaggle (about 4–6 hours, no secrets needed)

Import `notebooks/kaggle_newmodels.ipynb`, set Accelerator **GPU T4 x2** and Internet
**On**, then **Save Version → Save & Run All**. It runs DeepSeek-R1 through vLLM first,
then starts Ollama for Gemma and the backend check. When it finishes, download
`newmodels_outputs.zip` from the version's Output tab.

### 3. Put the outputs in place

```bash
unzip -o newmodels_outputs.zip -d .
```

### 4. Judge on the laptop (about 1.5 hours at ~190 judgments/min)

Judging replays each run's own settings, so no flags need retyping.

```bash
python -m csjail.run_eval --out-dir outputs/exp2/main-r1 --judge-only
```
```bash
python scripts/exp2_robustness.py --judge-only --greedy-results outputs/exp2/main-r1 --out-dir outputs/exp2/robustness-r1
```
```bash
python -m csjail.run_eval --out-dir outputs/exp2/main-gemma4 --judge-only
```
```bash
python scripts/exp2_robustness.py --judge-only --greedy-results outputs/exp2/main-gemma4 --out-dir outputs/exp2/robustness-gemma4
```
```bash
python -m csjail.run_eval --out-dir outputs/exp2/backend-check-qwen25-ollama --judge-only
```

### 5. Tables for all five models

```bash
python -m csjail.aggregate outputs/exp2/main outputs/exp2/main-r1 outputs/exp2/main-gemma4 --out outputs/exp2/summary_5models.csv
```

Add `--split eval_main` for the 200 held-out families.

### 6. Labels for the new models' judge error

```bash
python scripts/exp1_gold_audit.py make-model-sample --run-dir outputs/exp2/main-r1 outputs/exp2/main-gemma4 --models r1qwen15 gemma4e2b
```

This writes `outputs/exp1/gold-audit-05/reviewer_file.xlsx` (480 responses, blinded) for
the same reviewer. The returned file is folded into the judge manifest, and the tables
are re-aggregated.

### Optional: Gemma fully on the laptop

Because Gemma's production backend is the pinned Ollama build, the step 2 Gemma runs
can also be done locally (same commands as in the notebook, without `--max-families`).
Expect roughly a day on this CPU.

## Paper notes

- State the serving stack per model (vLLM fp16 vs Ollama q8_0) and report the qwen25
  backend check next to the Gemma results.
- State DeepSeek-R1's 2,048-token budget, that its visible reasoning is judged, and its
  unfinished-reasoning rate.
- Until step 6 is done, mark the new models' corrected ASR as per-language corrected.
