# CS-Jail-UR — runner guide

These are exact commands from a fresh checkout. Every command is labelled:

- **[tested]**: executed on the CPU box (Windows 11, Python 3.12) for the current release.
- **[CPU/untested]**: expected to work there, not yet executed.
- **[GPU]** / **[API]**: needs the GPU host or API keys; **not** executed yet.

The protocol and its rationale are in `docs/PROTOCOL.md`; current statuses
are in `docs/EXPERIMENT_STATUS.md`.

## 0. Environment

| Where | What runs | Needs |
|---|---|---|
| CPU box | Exp 0, statistics, aggregation, tests, CPU smoke (Ollama) | Python 3.10–3.12; `pip install -e ".[dev]"`; optional Ollama for smoke |
| API | judge (Exp 1 scoring, Exp 2+), chosen generator (Exp 6) | `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` |
| GPU host | Exp 1 validation sample, Exp 2, Exp 4b, Exp 6–9 | Linux + CUDA 12.x GPU (16 GB is enough for these ≤3.8B models; QLoRA 4-bit for training); `pip install -e ".[train,judge,dev]"`; `HF_TOKEN` with access to meta-llama/Llama-3.2-3B-Instruct |

```bash
git clone https://github.com/mshamoonbutt/NLP_Research.git && cd NLP_Research
git checkout revision/final-692-dataset
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"                                 # CPU  [CPU/untested as a single command]
pip install -e ".[train,judge,dev]"                     # GPU host  [GPU]
cp .env.example .env    # fill in keys; then: set -a; source .env; set +a
```

- The CPU dependency set was tested as individual installs: numpy 2.5, scipy 1.18,
  **pandas 2.2.3**, statsmodels 0.15, pydantic 2.13, PyYAML 6, pytest 9.
- On Windows with Smart App Control, pandas 3.x's compiled module was blocked;
  pin `pandas<3` there.
- Very long install paths can break statsmodels' wheel (MAX_PATH).
- The GPU pins in `pyproject.toml` (vLLM 0.6.3.post1, torch 2.4.0,
  transformers 4.46.3, trl 0.12.2, peft 0.13.2) are compatibility **candidates**
  and have not been executed.

Place the release CSV locally (never commit it):
`data/CS-Jail-UR_final_approved_791.csv` (SHA-256 `4a34e81a…1ecf`).

## 1. Checks (CPU, no keys)
```bash
pytest tests -q                              # [tested] all pass
python scripts/smoke_pipeline_cpu.py         # [tested] stubbed end-to-end chain
python scripts/smoke_exp1_cpu.py             # [tested] validation-gate logic
```

## 2. Exp 0 — finalize the active release (CPU)
```bash
python scripts/exp0_finalize_data.py         # [tested] reads configs/dataset.yaml (active: final-791)
```

This does the following:
- Verifies the CSV checksum.
- Converts all ten columns (the three metadata columns are preserved).
- Runs the structure and QA checks.
- Extends the frozen split append-only, so the 200 eval families never move.
- Screens new families against eval families.
- Checks exposure of the development and pilot samples.
- Writes `outputs/exp0/final-791-ddc14ecbc568/`: `FINDINGS.md`,
  `split_manifest.json`, `exposure_report.json`, `dataset_manifest.json`,
  `dataset_stats.json`, `qa_review_flags.json`, and `dataset_final.jsonl`
  (gitignored).
- Updates `LATEST.json`.
- Re-running refuses to overwrite the frozen output.

On a fresh clone (for example the GPU host), the committed output folder lacks
the gitignored dataset file. Recreate it and verify it against the committed
manifest:
```bash
python scripts/exp0_finalize_data.py --restore   # [tested: byte-identical dataset_final.jsonl on final-791]
```

A later **training-only extension** leaves the core release, eval membership,
wording and `LATEST` untouched:
```bash
python scripts/exp0_finalize_data.py --training-extension-csv data/<ext>.csv --extension-name ext1   # [CPU/untested on real data; covered by tests]
```

## 3. Exp 1 — judge validation (status: development only; see EXPERIMENT_STATUS)

Development (rubric iteration) [API]:
```bash
# adjudicate the 15 flagged items in outputs/exp1/development-cpu-20260926/annotations/adjudication_questions.csv, then
python scripts/exp1_dev_gold.py --sample-dir outputs/exp1/development-cpu-20260926      # [tested]
python scripts/calibrate_judge.py --sample-dir outputs/exp1/development-cpu-20260926 \
    --gold-csv outputs/exp1/development-cpu-20260926/development_gold.csv              # [API] report only
```

Freeze `configs/judge.yaml`: any later edit changes the fingerprint and
invalidates validation.

Final validation, on families never used for development or the pilot, from
the production backend [GPU]:
```bash
python scripts/exp1_sample_for_annotation.py --role validation --models qwen25 phi3 llama32 \
    --exclude-sample-dirs outputs/exp1/development-cpu-20260926 outputs/exp6/feasibility-cpu-20260927
```

- That is 60 families × 4 conditions × 3 models = 720 blank items. Two models
  (480 items) is the minimum.
- Two independent humans fill `rater1.csv` and `rater2.csv` (blank label
  fields; guide: `docs/exp1_rater_guide.md`). Optionally AI-prefill them first
  with `scripts/exp1_prefill_rater_files.py` (model distinct from the judge;
  humans verify every row) — this is assisted verification, not blind
  annotation, and is documented as such. Full walkthrough:
  `docs/KAGGLE_EXP1_GUIDE.md`.
- Adjudicate disagreements in `adjudication.csv`, then run:
```bash
python scripts/calibrate_judge.py --sample-dir outputs/exp1/validation-<ts> \
    --development-sample-dir outputs/exp1/development-cpu-20260926             # [API] exit 0 = PASS
```

It writes `outputs/exp1/judge_validation_manifest.json`. PASS requires every
condition to reach precision and recall ≥ 0.90 with the declared support.

The benign (over-refusal) rubric needs its own validation before Exp 8
(`--kind benign`). Sample benign-probe responses with:
```bash
python scripts/benign_sample_for_annotation.py --role validation --models phi3 llama32   # [GPU]
# two raters label refused/response_kind, adjudicate, then:
python scripts/calibrate_judge.py --kind benign --sample-dir outputs/exp1/benign-validation-<ts>   # [API]
```

## 4. Exp 2 — evaluation

Smoke tests, all marked debug and written under `outputs/smoke/` (never paper results):
```bash
# live generation, CPU/quantized Ollama (ollama pull the tags in configs/models.yaml first)
python -m csjail.run_eval --backend ollama --skip-judge --families train_pool --max-families 2 \
    --models qwen25 phi3 llama32 --out-dir outputs/smoke/exp2-ollama-791          # [tested]
# live judge on the HARMLESS fixture only (unvalidated judge => debug)
python scripts/exp0_finalize_data.py --source-csv tests/fixtures/final_fixture.csv --eval-size 4 \
    --out-root outputs/smoke/exp0-fixture --no-latest --exposure-samples            # [tested]
python -m csjail.run_eval --exp0-dir outputs/smoke/exp0-fixture/<final-12-...> --backend ollama \
    --models qwen25 --max-families 2 --allow-unvalidated-judge --out-dir outputs/smoke/exp2-fixture-judge   # [ran: API reached, all calls 429 no credits]
python -m csjail.aggregate outputs/smoke/exp2-ollama-791 --allow-debug              # [tested]
```

Full runs [GPU], not executed:
```bash
# generation-only while Exp 1 is pending: responses are produced now, every metric stays NA
python -m csjail.run_eval --out-dir outputs/exp2/main --skip-judge
# after a PASS judge manifest exists: the same command without --skip-judge judges the cached generations
python -m csjail.run_eval --out-dir outputs/exp2/main
bash scripts/run_all_baseline.sh                              # = run_eval + aggregate + Exp 3
python -m csjail.aggregate outputs/exp2/main                  # full-core table (791 families)
python -m csjail.aggregate outputs/exp2/main --split eval_main   # the frozen 200-family held-out baseline
python scripts/exp2_robustness.py --greedy-results outputs/exp2/main   # 200 fams x CS/RU x 5 draws x 3 models
```

**Scope and expected counts.**

| Run | Per model | For 3 models |
|---|---:|---:|
| Full core: 791 families × 4 conditions | 3,164 | 9,492 |
| Held-out subset: 200 families × 4 conditions | 800 | 2,400 |
| Robustness: 200 × 2 × 5 | 2,000 | 6,000 |

- Never compare the full-core baseline against held-out-only post-training
  results; Exp 8 regenerates arm A on exactly the 200 held-out families.
- Training-exposed families are never held-out evidence.

**Resume and outputs.**
- Re-run the identical command after an interruption. Cached generations and
  judgments are reused, with no duplicate rows.
- A run folder made with a different dataset, split, sampling, backend or
  judge is refused.
- Results go to `outputs/exp2/<run>/results.<model>.jsonl`, plus
  `run_manifest.json`, `generations.jsonl` and `judgments.jsonl`. These hold
  prompt/response text and are **gitignored**: store raw outputs in the
  team's private storage and commit only `run_manifest.json` and
  `summary*.csv/json`.

**Rough estimates, unverified.** Exp 2 generation on one 16 GB GPU takes about
1–2 h. Judging 9,492 responses with gpt-4o-mini costs a few USD and is
rate-limit bound.

## 5. Later stages (commands unchanged; see PROTOCOL)
```bash
python scripts/exp3_isolation.py --results outputs/exp2/main                     # CPU
python scripts/prepare_capability_sets.py --urdummlu-dataset <hf-id-or-use---urdummlu-file>  # CPU+network, before Exp 8
python scripts/exp4b_comprehension.py --baseline-results outputs/exp2/main       # [GPU][API]
python scripts/prepare_external_english_pairs.py --n 1000                        # [untested] B_ext source
python scripts/exp6_build_prefdata.py --model phi3 --results outputs/exp2/main   # [API]
python scripts/exp7_train_arms.py --model phi3 --arm C --budget 100 --naturalness-csv <rated>   # [GPU]
python scripts/exp7_train_arms.py --model phi3 --arm B_ext --budget 100                          # [GPU]
python scripts/exp8_posteval.py --arms A B_ext C E                               # [GPU][API]
python scripts/exp9_ablations.py ncurve --model phi3                             # CPU
```

- The primary Phase 2 comparison is C vs B_ext at equal accepted-pair budgets.
- C_matched and B_matched are optional.
- Exp 8 needs a PASS judge manifest for the harm rubric **and** one for the
  benign rubric.
