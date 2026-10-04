# Running Exp 1 on Kaggle — step-by-step

Written 2026-10-04. Goal of this session: generate the judge-validation
samples (harm + benign) on the production backend, then finish Exp 1 on the
CPU box: AI-prefill → human verification → adjudication → judge calibration
→ frozen PASS manifests. Exp 0 is already done (`final-791-ddc14ecbc568`).

Production decisions in force (configs committed):
- **Backend:** vLLM, pinned revisions, **float16** (T4 has no bf16; vLLM
  cannot run on the P100 — always pick **GPU T4 ×2**, never P100).
- **Judge candidate:** `gpt-4o` (snapshot pinned before validation, below).
  Chosen for accuracy against human gold — better Urdu/Roman-Urdu reading and
  partial-compliance recall than mini — never for favorable ASR numbers.
- **Prefill model:** `gpt-4o-mini` (must differ from the judge; same-provider
  correlation is recorded, human verification is the control).

## 0. One-time prerequisites

1. Kaggle account, phone-verified (required for Internet + GPU).
2. Hugging Face: accept the license on `meta-llama/Llama-3.2-3B-Instruct`
   with your HF account, create a **read** token.
3. In the Kaggle notebook editor: **Settings → Accelerator = GPU T4 x2**,
   **Internet = On**. Add-ons → Secrets: add `HF_TOKEN`.
4. OpenAI key stays on the CPU box (no judging happens on Kaggle).

## 1. Kaggle notebook cells (~60–90 min total)

Cell 1 — clone the branch:
```python
!git clone --branch revision/final-692-dataset https://github.com/mshamoonbutt/NLP_Research.git
%cd NLP_Research
```

Cell 2 — pinned stack (~5–10 min; torch 2.4.0 downgrade is expected; run
this before anything imports torch):
```python
!pip install -q -e ".[gpu,dev]"
```

Cell 3 — secrets:
```python
from kaggle_secrets import UserSecretsClient
import os
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
```

Cell 4 — restore the frozen dataset and prove the environment (~5 min):
```python
!python scripts/exp0_finalize_data.py --restore
!python -m pytest tests -q
```
`--restore` must report a byte-identical `dataset_final.jsonl`; the tests
must all pass. If either fails, stop and report — do not generate.

Cell 5 — GPU generation smoke + Urdu sanity on the production precision:
```python
!python scripts/smoke_test.py --model qwen25
!python scripts/urdu_sanity_check.py --backend vllm --out outputs/checks/urdu_sanity_vllm.json
```
The Urdu check answers whether the CPU-batch degeneration was a quantization
artifact; its JSON goes home with the outputs either way.

Cell 6 — **Exp 1 harm validation sample** (60 untouched train families × 4
conditions × 3 models = 720 items; fixed out-dir so a rerun resumes instead
of resampling):
```python
!python scripts/exp1_sample_for_annotation.py --role validation \
  --models qwen25 phi3 llama32 \
  --exclude-sample-dirs outputs/exp1/development-cpu-20260926 outputs/exp6/feasibility-cpu-20260927 \
  --out-dir outputs/exp1/validation-kaggle-01
```

Cell 7 — **benign validation sample** (150 probes × phi3+llama32 = 300 items):
```python
!python scripts/benign_sample_for_annotation.py --role validation \
  --models phi3 llama32 --out-dir outputs/exp1/benign-validation-kaggle-01
```

Cell 8 — package for download:
```python
!zip -qr exp1_kaggle_outputs.zip outputs/exp1/validation-kaggle-01 \
  outputs/exp1/benign-validation-kaggle-01 outputs/checks
```
Download `exp1_kaggle_outputs.zip` from the notebook's Output panel and
unzip it into the local repo root (it lands under `outputs/`).

Cell 9 (optional, same warm session) — start the Exp 2 generation-only
sweep; metrics stay NA until the judge passes, and the run resumes across
sessions with the identical command:
```python
!python -m csjail.run_eval --out-dir outputs/exp2/main --skip-judge
!zip -qr exp2_main_partial.zip outputs/exp2/main
```

**Do not commit** the downloaded sample/response files to the public repo:
items.csv and the rater files contain harmful prompts and completions. Share
them with the raters privately; the manifests record their hashes.

## 2. Back on the CPU box — finish Exp 1

Step 1 — pin the judge snapshot (one line; then commit `configs/judge.yaml`):
```bash
python -c "from openai import OpenAI; print(OpenAI().chat.completions.create(model='gpt-4o', messages=[{'role':'user','content':'hi'}], max_tokens=1).model)"
```
Put the printed dated id into `judge.model_snapshot`. The snapshot is part
of the frozen fingerprint — set it BEFORE any calibration you intend to keep.

Step 2 — development feedback (needs the 15 dev adjudications done once):
```bash
python scripts/exp1_dev_gold.py --sample-dir outputs/exp1/development-cpu-20260926
python scripts/calibrate_judge.py --sample-dir outputs/exp1/development-cpu-20260926 \
    --gold-csv outputs/exp1/development-cpu-20260926/development_gold.csv
```
This is rubric feedback only (never a validation manifest). If the rubric
needs edits, edit now — any later edit re-opens validation. Then freeze
`configs/judge.yaml`.

Step 3 — AI-prefill the rater files (assisted verification):
```bash
python scripts/exp1_prefill_rater_files.py --sample-dir outputs/exp1/validation-kaggle-01 --model gpt-4o-mini
python scripts/exp1_prefill_rater_files.py --kind benign --sample-dir outputs/exp1/benign-validation-kaggle-01 --model gpt-4o-mini
```

Step 4 — human verification (the step that cannot be compressed):
- Two verifiers work **independently** — no discussion, no shared screen.
- Each reviews **every row** of their own `rater*_prefilled.csv`, corrects
  the labels (guide: `docs/exp1_rater_guide.md`; benign: label `refused` +
  `response_kind`), puts their real `rater_id` on every row, and saves the
  file back as `rater1.csv` / `rater2.csv` in the sample dir.
- This is **assisted verification, not blind annotation** — the paper must
  say so, and `prefill_metadata.json` records the prefill model.
- Workload: ~3–4 h (720 harm) + ~1–1.5 h (300 benign) per verifier.

Step 5 — adjudication: for every disagreement between the two saved files,
add a row to `adjudication.csv` (gold_* fields + `resolution` +
`adjudicator`). Pre-adjudication agreement/κ is computed automatically.

Step 6 — final calibration (the gate):
```bash
python scripts/calibrate_judge.py --sample-dir outputs/exp1/validation-kaggle-01 \
    --development-sample-dir outputs/exp1/development-cpu-20260926
python scripts/calibrate_judge.py --kind benign --sample-dir outputs/exp1/benign-validation-kaggle-01
```
Exit 0 = PASS (per-condition precision AND recall ≥ 0.90, support minimums,
no unresolved items) and writes `outputs/exp1/judge_validation_manifest.json`
(+ `_benign`). Commit the **manifests only**. Exit 1/2 = FAIL/INSUFFICIENT:
read the per-condition table it prints, fix the rubric or add support, and
re-run — never lower the gate after seeing results.

Cost for all of step 2–6: a few USD (720 gpt-4o + 300 benign judgments +
1,020 mini prefills).

## 3. What happens after Exp 1 passes

Exp 2 judging is now one command against the cached generations from Cell 9
(`python -m csjail.run_eval --out-dir outputs/exp2/main`, no `--skip-judge`),
then `aggregate`, `exp3_isolation.py`, and onward per `docs/TIMELINE.md`
Days 3–6. If a condition cannot pass validation, production judging stays
blocked for it — that is a finding to report, not a gate to bypass.
