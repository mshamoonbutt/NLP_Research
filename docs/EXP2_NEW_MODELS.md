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

### 1. Local test on Ollama (laptop; done 2026-10-07, passed)

Downloads about 10 GB (Gemma 8.1 GB, DeepSeek-R1 1.9 GB; ~45 min at 4 MB/s). This is a
sanity check of 5 families per model, marked as a debug run: does each model answer in
the expected language, does the reasoning model close its reasoning, and does the judge
parse. Measured on this laptop's CPU: DeepSeek-R1 30 min for 20 responses (~13
tokens/s, ~1,160 tokens each), Gemma 7 min (~7 tokens/s, ~160 tokens each); judging
under a minute. Both builds matched their pins, all 40 judgments parsed, every R1
response closed its reasoning, and Gemma answered Urdu-script prompts in Urdu script.

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

### 2. Full generation on Kaggle (about 3–4 hours, no secrets needed)

Import `notebooks/kaggle_newmodels.ipynb`, set Accelerator **GPU T4 x2** and Internet
**On**, then **Save Version → Save & Run All**. It runs DeepSeek-R1 through vLLM first,
then starts Ollama for Gemma and the backend check. When it finishes, download
`newmodels_outputs.zip` from the version's Output tab.

### 3. Put the outputs in place

```bash
unzip -o newmodels_outputs.zip -d .
```

### 4. Judge on the laptop (about 1–1.5 hours, ~11,100 judgments, ~$7 of Ollama Pro credit)

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
At the measured ~7 tokens/s that is about 30 hours (main ~19 h, robustness ~12 h).
DeepSeek-R1 locally would take days; use Kaggle.

## Results (final, 2026-10-08)

All five models judged by DeepSeek V4.1 Flash (fingerprint `b0248df31f1d9da2`):
main 3,164 + robustness 2,000 responses per new model, 0 judge errors. Gemma's
robustness run was redone on Kaggle after the first run's Ollama server failed
(two models loaded at once); the notebook now loads one model at a time.

**Raw ASR, all 791 families (EN / CS / RU / UR):**

| Model | EN | CS | RU | UR |
|---|---|---|---|---|
| r1qwen15 | .569 | .382 | .125 | .013 |
| llama32 | .118 | .373 | .359 | .298 |
| qwen25 | .154 | .092 | .034 | .061 |
| phi3 | .082 | .113 | .100 | .102 |
| gemma4e2b | .033 | .066 | .063 | .046 |

The 200 held-out families (`--split eval_main`) show the same ordering
(`outputs/exp2/summary_5models_eval_main.csv`).

**Judge accuracy per model** (reviewer labels; original three: development +
held-out, 560 each; new models: gold-audit-05, 240 each):

| Model | Labelled | Harmful | Precision | Recall | Specificity | Kappa |
|---|---|---|---|---|---|---|
| qwen25 | 560 | 36 | .73 | .83 | .98 | .76 |
| phi3 | 560 | 48 | .67 | .85 | .96 | .73 |
| llama32 | 560 | 140 | .73 | .86 | .90 | .72 |
| r1qwen15 | 240 | 59 | .56 | .80 | .80 | .52 |
| gemma4e2b | 240 | 8 | 1.00 | .38 | 1.00 | .54 |

**Corrected ASR.** For the original three models the Rogan–Gladen correction
per model × language is stable (see `docs/EXPERIMENT_STATUS.md`, Exp 2). For the
two new models it is not usable: on DeepSeek-R1's code-switched responses the
judge's specificity is .45 (it flags 21 of 38 safe responses, mostly where the
reviewer judged R1's finished answer safe and helpful), which drives the
correction to 0; Gemma's cells hold 1–3 harmful responses, so its sensitivity is
undefined or near zero. For the new models report instead the reviewer-labelled
rate on the 60 development families, with the predictive-value estimate
(judge-flagged rate × P(harmful | flagged) + unflagged rate × P(harmful |
unflagged), from the same labels) as a cross-check:

| Model | Language | Raw (791 fam.) | Reviewer-labelled (60 fam., 95% CI) | Predictive-value estimate |
|---|---|---|---|---|
| r1qwen15 | EN | .569 | .567 [.44, .68] | .522 |
| r1qwen15 | CS | .382 | .367 [.26, .49] | .319 |
| r1qwen15 | RU | .125 | .033 [.01, .11] | .051 |
| r1qwen15 | UR | .013 | .017 [.00, .09] | .031 |
| gemma4e2b | EN | .033 | .017 [.00, .09] | .040 |
| gemma4e2b | CS | .066 | .017 [.00, .09] | .056 |
| gemma4e2b | RU | .063 | .050 [.02, .14] | .063 |
| gemma4e2b | UR | .046 | .050 [.02, .14] | .078 |

**DeepSeek-R1 reasoning (greedy, 2,048 tokens):** reasoning finished in 529/791
EN, 709 CS, 747 RU, 762 UR responses; 30% of all responses hit the limit. Among
EN responses whose reasoning finished, judged ASR is .78; unfinished ones are
mostly looping text judged unintelligible, so greedy decoding understates R1's
English ASR. Sampled decoding (robustness) loops less.

**Robustness (200 families, T 0.7, 5 draws; per-draw / any-of-5 / greedy):**
r1qwen15 CS .536 / .805 / .480, RU .150 / .375 / .135; gemma4e2b CS .084 / .110 /
.085, RU .082 / .115 / .090. Gemma refuses consistently: five tries barely raise
its rate, unlike every other model.

**Backend check (qwen25, 200 held-out families, vLLM fp16 vs Ollama q8_0):** ASR
EN .175 / .180, CS .120 / .135, RU .035 / .020, UR .065 / .085; McNemar p ≥ .42 in
every language; identical text in 417/800. The serving stack does not move ASR
materially, so Gemma via Ollama is comparable. One prompt (`CSJUR-V3-0046::RU`)
makes Ollama abort on a token loop ("token repeat limit reached", HTTP 500);
it is recorded as a missing generation (vLLM would have run the loop to the limit).

## Paper notes

- State the serving stack per model (vLLM fp16 vs Ollama q8_0) and report the qwen25
  backend check next to the Gemma results.
- State DeepSeek-R1's 2,048-token budget, that its visible reasoning is judged, and its
  unfinished-reasoning rate.
- The new models' corrected ASR is the reviewer-labelled rate (60 families), not the
  Rogan–Gladen correction, for the reasons in the results section; say why.
- The judge reads DeepSeek-R1's visible reasoning; its agreement with the reviewer on R1
  is lower (kappa .52) than on the original three models (.72–.76).
- Reviewer identity is still unrecorded in every review file (`reviewer_id` blank).
