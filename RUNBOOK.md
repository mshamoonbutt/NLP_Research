# CS-Jail-UR — Runbook

These are the commands, in order. The rationale for each step is in
`docs/PROTOCOL.md`.

There are two environments:

- **CPU box:** Exp 0, statistics, pair assembly, tests.
- **GPU host (RTX 4080 / WSL2):** generation, judging, training.

Every stage consumes the finalized Exp 0 artifact (`outputs/exp0/LATEST.json`)
and verifies its hashes.

## 0. Environment
```bash
pip install -e ".[dev]"                          # CPU
pip install -e ".[train,judge,dev]"              # GPU host (pins are candidates; see §9 smoke)
export OPENAI_API_KEY=... ANTHROPIC_API_KEY=... HF_TOKEN=...
pytest tests -q
python scripts/smoke_pipeline_cpu.py && python scripts/smoke_exp1_cpu.py
```

## Exp 0: finalize (CPU)
```bash
python scripts/exp0_finalize_data.py --source-csv data/CS-Jail-UR_final_692.csv \
    [--qa-ledger data/qa/qa_ledger.csv] [--independent-annotations data/qa/annotations.csv]
#   -> outputs/exp0/final-692-b11d22b34008/  (FINALIZED; split 9451ca61d16575fb: 200 eval / 492 train)
#   Frozen: re-running refuses to overwrite; add data later with
#   --extend-split outputs/exp0/<version>/split_manifest.json (append-only)
# On another machine (GPU host), the committed dir lacks the gitignored dataset
# file; recreate it from the same CSV and verify it against the manifest:
python scripts/exp0_finalize_data.py --source-csv data/CS-Jail-UR_final_692.csv --restore
python scripts/dry_run_report.py
```
Review `qa_review_flags.json`. It lists 60 UR items with Latin letters and
2 possible English clauses in RU, and inventories technical loans in RU. Also
review `duplicate_candidates.csv` (0 here). Record decisions in `data/qa/`.
Do not edit rows silently.

## GPU smoke (required before production)
```bash
python scripts/smoke_test.py --model qwen25     # pinned load, template probe, finish reasons
```
Then run a tiny end-to-end pass (`--max-families 4 --allow-unvalidated-judge`,
marked debug) through Exp 8 on a few pairs. See PROTOCOL §9.

## Exp 1: judge validation
```bash
# rubric development (train_pool families)
python scripts/exp1_sample_for_annotation.py --role development --models qwen25 phi3 \
    --n-per-model-condition 20
# ... iterate on configs/judge.yaml using ONLY the development sample ...
# final validation on DISJOINT families (60 per model per condition -> 480 responses)
python scripts/exp1_sample_for_annotation.py --role validation --models qwen25 phi3 \
    --exclude-sample-dirs outputs/exp1/development-<ts>
```
Two raters then fill `rater1.csv` / `rater2.csv` independently. Each row needs
`rater_id`, `unsafe`, `response_kind` and `compliance_degree`. Adjudicate the
disagreements in `adjudication.csv`, then score:
```bash
python scripts/calibrate_judge.py --sample-dir outputs/exp1/validation-<ts> \
    --development-sample-dir outputs/exp1/development-<ts>
# exit 0 PASS / 1 FAIL / 2 INSUFFICIENT_EVIDENCE -> outputs/exp1/judge_validation_manifest.json
# benign rubric: same flow on benign probe responses with `refused` labels, --kind benign
```

## Exp 2: sweep + robustness
```bash
bash scripts/run_all_baseline.sh                  # 3 x 4 x 692 = 8,304 responses; resumable
python -m csjail.aggregate outputs/exp2/main      # denominators, micro/macro ASR, behaviour rates
python scripts/exp2_robustness.py --greedy-results outputs/exp2/main   # 6,000 responses (CS/RU)
```

## Exp 3 / 4b
```bash
python scripts/exp3_isolation.py --results outputs/exp2/main
python scripts/exp4b_comprehension.py --baseline-results outputs/exp2/main
python scripts/exp4b_comprehension.py --baseline-results outputs/exp2/main \
    --score-review outputs/exp4b/review_sample.csv      # after human review
```

## Exp 6–8 (Phase 2; phi3 + llama32)
```bash
python scripts/dry_run_report.py --results outputs/exp2/main   # actual pair budgets
python scripts/exp6_build_prefdata.py --model phi3 --results outputs/exp2/main
#   rate outputs/exp6/phi3/naturalness_sample.csv (mean >= 4, clean refusals) before C/D
python scripts/exp7_train_arms.py --model phi3 --arm C --naturalness-csv <rated csv>
python scripts/exp7_train_arms.py --model phi3 --arm B          # matched English control
python scripts/exp7_train_arms.py --model phi3 --arm B_ext      # optional practical baseline
python scripts/exp8_posteval.py --arms A B C E                  # add B_ext / D if trained
```

## Exp 9: ablations
```bash
python scripts/exp9_ablations.py ncurve --model phi3
python scripts/exp9_ablations.py domain --domain <Dk> --attest-chosen-before-outcomes
#   then exp6 / exp7 / exp8 with --split-manifest outputs/exp9/ablation_<Dk>/split_manifest.json
#   --tag ablation_<Dk>   (fresh adapters from the base model)
```

## Gates vs. flags
- **Hard:**
  - Exp 0 structural gate.
  - Exp 1 per-condition precision and recall ≥ 0.90, with declared support
    (PASS only).
  - Production refuses any unvalidated judge.
  - Missing adapters or inputs stop Exp 8.
- **Not gates:**
  - Dataset κ: descriptive only.
  - Robustness ranking agreement: reported.
  - Exp 8 improvement thresholds: reporting flags. "NA" means undefined.

## Rough budget (RTX 4080)
- Exp 2: 8,304 generations in about 1 hour.
- Robustness: 6,000 generations.
- Judging is the bottleneck, so run it with concurrency.
- Each DPO arm: tens of minutes.
