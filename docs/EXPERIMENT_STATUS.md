# Experiment status (as of 2026-09-27)

This page records what actually ran, with artifacts and hashes. An experiment
is **not** marked complete just because its inputs exist. Commands are in
`RUNBOOK.md` and the design in `docs/PROTOCOL.md`.

## Exp 0 — dataset finalization: **DONE** (release `final-791`)

| Item | Value |
|---|---|
| Source | `data/CS-Jail-UR_final_approved_791.csv` (local), SHA-256 `4a34e81a38932fef769e563bbfe735606eadb4c473d24898ae249bd8480a1ecf` |
| Output | `outputs/exp0/final-791-ddc14ecbc568/` (`FINDINGS.md` has the full findings) |
| Normalized content | `ddc14ecbc568aec9efd41f042e5b26078698d1c2fdca4793852e9fa6a090b3d4` |
| `dataset_final.jsonl` (gitignored) | `070944241e91c7148aff77f6a052193de150f1858c287fe15755ed0e01adc53b`; `--restore` reproduces it byte-for-byte |
| Split | `51c2dd1bb7d72166`, an append-only extension of `9451ca61d16575fb` (692) → `7df6f9b8276fa856` (748) |
| Gates | ingest, structure, split and exposure: **PASS** |

| Check | Expected | Observed |
|---|---:|---:|
| Families / unique IDs | 791 / 791 | 791 / 791 |
| Prompt rows (4 per family) | 3,164 | 3,164 |
| `evaluation_stratum=harmful` / `approval_status=approved` | 791 / 791 | 791 / 791 |
| Domains D1–D6 | 122/123/137/128/137/144 | same |
| Existing families changed vs 748 | 0 | 0 (43 added: 3 X1, 40 X2) |
| Evaluation families (frozen since 692) | 200 | 200: identical set, D1–D6 35/31/33/34/32/35 |
| Training pool | 591 | 591 (87/92/104/94/105/109) |
| Exact duplicates / equal variants | — | 0 / 0 |
| Near-duplicate candidates (token-Jaccard ≥ 0.92) | — | 0 |
| New families resembling an eval family (≥ 0.5) | — | 0 quarantined |
| Exposure conflicts | 0 | 0: development 20 and pilot 100 families (11 shared) are all `train_pool` |
| Eligible for preference training | — | 591 (no reserved exclusions) |

**Open for human review** (flags only; nothing was changed):
- 69 UR items contain Latin letters.
- 2 RU items may contain English clauses.
- 347 RU items carry technical loanwords (informational).
- Semantic equivalence and harmfulness are not certified by Exp 0.
- Per-batch authorship and model assistance are unrecorded.

## Exp 1 — judge validation: **ANNOTATION RETURNED (2026-10-05); ADJUDICATION + TOP-UP PENDING**

Stage A (Kaggle T4, vLLM 0.6.3.post1, float16, pinned revisions; see
`docs/KAGGLE_EXP1_GUIDE.md`):
- **Judge frozen for validation:** gpt-4o, snapshot `gpt-4o-2024-08-06`, rubric
  harm-v2, harm fingerprint `b0d6676cf5d89d08` (benign `f9aec59096cb12cf`).
- **Live judge check (development sample, feedback only):** 160/160 parsed,
  0 failures. The first attempt lost 57/160 to `429` tokens-per-minute limits
  (organization limit 30,000 TPM for gpt-4o, ~22 judgments/min); fixed by
  hint-aware retries and concurrency 2. Development support is below the gate
  minimums, so its metrics are not evidence.
- **Harm validation sample** `outputs/exp1/validation-kaggle-01`: 720 items =
  60 train-pool families (D1–D6 9/10/10/9/11/11; 0 overlap with the
  development and pilot samples; 0 eval families) × 4 conditions × 3 models.
  Prompts byte-identical to the frozen release; items hash matches the
  manifest; 0 empty responses; finish `length` 276/720 (512-token cap).
- **Benign validation sample** `outputs/exp1/benign-validation-kaggle-01`:
  150 probes × phi3/llama32 = 300 items; probe-file hash matches.
- Rater files are blank, independently shuffled, with no judge output in the
  folders. Raw text is gitignored; only the manifests are committed.

Annotation (returned 2026-10-05, imported with `scripts/exp1_import_rater_xlsx.py`;
workbook hashes in each folder's `annotation_import.json`):
- **Harm, 720 items:** pre-adjudication agreement on `unsafe` 718/720
  (κ 0.978), on `response_kind` 682/720 (κ 0.925). 38 items need adjudication
  (2 binary, 36 kind-only). 0 internally inconsistent rows.
- **Benign, 300 items:** `refused` 300/300 (κ 1.000), `response_kind`
  298/300 (κ 0.964). 2 kind-only items need adjudication.
- **Support shortfall:** gold unsafe per condition (both raters) EN 6,
  CS 20, RU 14, UR 8. The declared minimum is 10, so EN and UR cannot pass on
  this sample. Low rates: qwen25/phi3 Urdu-script responses are almost all
  unintelligible (52/60, 55/60), and English compliance is rare.
- **Top-up (declared rule: new train-pool families, next seed):**
  `outputs/exp1/validation-kaggle-01-topup1`, 80 families (seed 43) × EN/UR ×
  3 models = 480 items, responses taken verbatim from the Exp 2 production
  run (`--from-results`; same release, split, sampling, revisions). Sized for
  P(EN gold unsafe ≥ 10) ≈ 0.96 at the observed rate. Raters fill blank
  files; then `exp1_merge_samples.py` combines both for one calibration.
- The annotation process (who, when, any tools or assistance) is still to be
  recorded in `annotation_import.json` from real records.
- **Adjudication returned (2026-10-05):** 38/38 harm and 2/2 benign resolved,
  all consistent, every row with a reason (harm gold sides with rater 1 on 15,
  rater 2 on 23; both binary cases resolved harmful). The `adjudicator` field
  reads `review`, not a person: record who adjudicated.
- **Top-up annotated:** 480/480 items; agreement unsafe 477/480 (κ 0.964),
  kind κ 0.956; 13 disagreements (3 binary) await adjudication.
- **Projected harm support** (base + top-up, before the 13): gold unsafe EN 30,
  CS 20, RU 16, UR 29; safe ≥ 160 in every condition. Pooled agreement over
  1,200 items: unsafe κ 0.972, kind κ 0.937.
- **Rate shift to check:** gold-unsafe rate in EN is 3.3% (6/180) in the base
  sample vs 10.1% (24/237) in the top-up (Fisher p = 0.007; all three models
  higher; 19/24 in D2/D4). UR 4.4% vs 9.1% (p = 0.081). Either family
  sampling or a labelling-standard shift between batches; after calibration,
  compare judge–gold agreement per batch.
- **Harm sample complete (2026-10-05):** top-up 13/13 adjudicated (`resolution`
  names the adopted rater; `adjudicator` = `adjudication_pass`). Merged folder
  `outputs/exp1/validation-kaggle-01-merged`: 1,200 items, 51 disagreements,
  0 unresolved; pooled agreement unsafe κ 0.972, kind κ 0.937; gold unsafe/safe
  EN 31/389, CS 20/160, RU 16/164, UR 29/391. Ready for the harm gate.
- **Gate runs on Kaggle** (paid key in Secrets): `notebooks/kaggle_stageC_judge.ipynb`
  with the private upload `outputs/stageC_upload.zip` (gitignored).
- **Benign support shortfall:** gold `refused` = 8 of 300 (< 10 minimum). Top-up:
  the same 150 probes answered by qwen25 (arm A). Its rater workbooks are back
  (24/150 refused in both), but the generated folder
  `benign-validation-kaggle-02` (items, manifest) has not been received yet.

Earlier development evidence (kept for provenance):

Evidence inspected:
- `outputs/exp1/judge_validation_manifest*.json`: **absent**. No validated
  judge exists and there is nothing compatible to reuse.
- **Development sample:** `outputs/exp1/development-cpu-20260926/`.
  - 160 items: 20 training families × 4 conditions × {qwen25, phi3}.
  - Generated with the Ollama Q8_0 CPU backend.
  - Its prompts match the 791 source exactly.
- **Returned labels:** two files, `rater_id` `AI_PRELABEL` and
  `AI_PRELABEL_R2`, preserved byte-for-byte.
  - They are **AI-prefilled** and **identical on all 160 items**, so they are
    not independent annotations.
  - Human verifier IDs, date and scope are **not recorded**
    (`annotations/verification_metadata.json` has them as `null`).
- **Development gold:** `development_gold.csv` is provisional.
  - 145 items are `ai_prefill` and 15 are `pending_adjudication`.
  - Unsafe/safe counts by condition: CS 7/33, EN 5/35, RU 3/37, UR 5/35.
  - With so few unsafe items, per-condition recall from this batch is fragile.
- **No judge predictions** exist on any sample. An attempted live call failed
  on billing (see Exp 2).
- **The 400-row AI-labelled feasibility file** (team-reported) is **not in
  this repo**. It is exploratory evidence only and must never be used as
  human gold, a second rater, or validation evidence.

Remaining steps:
1. A human adjudicates the 15 flagged development items, then runs
   `scripts/exp1_dev_gold.py`. Fill in the verification metadata from real
   records.
2. **Restore OpenAI credits.** Then run
   `calibrate_judge.py --sample-dir <dev> --gold-csv <dev>/development_gold.csv`
   for rubric feedback.
3. Freeze `configs/judge.yaml`. The current harm fingerprint is
   `fd5a40e535fba75a`; any edit changes it.
4. **On the GPU host,** draw the validation sample from untouched training
   families, excluding the development and pilot samples.
5. Two independent humans label blank files, then adjudicate. Run
   `calibrate_judge.py --sample-dir <val> --development-sample-dir <dev>`.
   PASS needs per-condition precision and recall ≥ 0.90 with the declared
   support.
6. Validate the benign (over-refusal) rubric separately before Exp 8:
   `scripts/benign_sample_for_annotation.py` (added 2026-10-04) generates the
   probe responses and blank rater files; gate with
   `calibrate_judge.py --kind benign`.

## Exp 2 — main evaluation: **GENERATION DONE (2026-10-05); JUDGING PENDING THE EXP 1 PASS**

- **Main sweep** `outputs/exp2/main` (raw text gitignored; `run_manifest.json`
  committed): 9,492/9,492 generations ok = 791 families × 4 conditions × 3
  models; vLLM fp16 on a Kaggle T4, pinned revisions, greedy, non-debug, git
  `729db4b`. The 200 eval families are 2,400 rows.
- **Share hitting the 512-token cap**, by condition (EN/CS/RU/UR): qwen25
  .09/.09/.10/.58; phi3 .06/.28/.55/.66; llama32 .07/.56/.76/.63. Mostly long
  or looping non-English output; report truncation with the results.
- **Reproducibility:** regenerating the 720 Exp 1 items in this run gave
  byte-identical text for 78% (qwen25 93%, phi3 76%, llama32 66%). Greedy fp16
  output depends on batch composition; the paper already states greedy is not
  bitwise reproducible.
- **Robustness** `outputs/exp2/robustness`: 6,000/6,000 ok = 200 families
  (44 eval / 156 train) × CS/RU × 5 draws × 3 models, T 0.7, top-p 0.9; the
  five draws use seeds 1234–1238, one request each (`SLMRunner.generate`;
  a single n=5 request stalled vLLM 0.6.3 on Phi-3).
- **Judging needs no GPU:** `--judge-only` (run_eval, exp2_robustness) replays the
  recorded provenance; offline key check found 9,492/9,492 and 6,000/6,000 cached.

Earlier smoke checks:

Summary: `outputs/checks/exp2_smoke_summary.json` (raw smoke outputs are
gitignored under `outputs/smoke/`).

| Check | Result |
|---|---|
| Offline CPU checks: 128 tests, `smoke_pipeline_cpu`, `smoke_exp1_cpu` | **PASSED** (tests include mixed string IDs, extra metadata columns, group leakage, incompatible resume, missing judgments ≠ safe, interrupted runs) |
| Live **generation** smoke: Ollama Q8_0, 3 models × 4 conditions × 2 training families, generation-only | **PASSED**: 24/24 records, 24 unique keys, 0 failures, finish 15 stop / 9 length; the resumed re-run produced **0** new generations; metrics correctly NA (0/2 scored, bounds 0–1) |
| Live **judge** smoke: gpt-4o-mini on the **harmless fixture** only | **BLOCKED**: API reached and authenticated, but all 8 calls returned `429 insufficient_quota` (no credits). They were handled as missing (never safe) and will be retried. Judge parsing on live output is **not verified** |
| Official chat templates at pinned revisions | Qwen2.5: default system "You are Qwen…" (same as the Ollama build); Phi-3: no default system. **Llama-3.2: not fetched (gated; needs `HF_TOKEN`)** (`outputs/checks/official_chat_templates.json`) |
| Benign Urdu-script controls (Ollama) | qwen25/phi3: wrong or degenerate Urdu-script answers, fluent English. llama32: mostly coherent Urdu script with some script contamination |
| Benign Urdu-script controls (vLLM fp16, production) | **Same pattern as Q8** (`outputs/checks/urdu_sanity_vllm.json`): repeat-word share and script share nearly identical to Ollama Q8. qwen25/phi3 loop or produce meaningless Urdu script and mistranslate a one-line sentence; llama32 answers correctly (e.g. Islamabad) with some Devanagari contamination. **Degeneration is a model property, not a quantization artifact.** Implication: low UR ASR for qwen25/phi3 must be read with the response-type and comprehension analyses |
| Production backend (vLLM, fp16, Kaggle T4) | **Working**: 3-model sequential loads without OOM; Exp 1 samples generated |
| Full Exp 2 (9,492 greedy + 6,000 robustness responses) | **NOT RUN** (by design in this task) |

## Exp 3–4b: not run (need Exp 2 results)
## Exp 5: cut

## Phase 2 (Exp 6–9): **NOT RUN**; design recorded
- **Primary (RQ4):** C (our CS pairs) vs B_ext (external English pairs) at
  **equal accepted-pair budgets** and identical optimisation.
  - This compares **training recipes**, not language alone.
  - The planning target is ~100 accepted pairs per model, with nested budgets
    of 25, 50 and 100.
- **Secondary (optional):** C_matched vs B_matched, only where same-family
  English and CS failures exist. It is never a gate.
- **Feasibility pilot:** `outputs/exp6/feasibility-cpu-20260927/`.
  - 400 responses: 100 training families × CS/EN × {phi3, llama32}.
  - Generated with Ollama Q8_0 on the CPU; the repo's review sheets are blank.
- **Team-reported AI-reviewed pilot counts** (exploratory, not gold):
  - CS failures: 18 (phi3) / 17 (llama32); strict 13 / 10.
  - Same-family CS-and-EN failures: 9 / 1; strict 6 / 1.
  - Yield must be re-validated in the production configuration.
- **Not yet run:**
  - `scripts/prepare_external_english_pairs.py` (the B_ext source)
  - any DPO training
  - Exp 8/9

## Next commands for the resource-owning collaborator (GPU + API)
```bash
git checkout revision/final-692-dataset && pip install -e ".[train,judge,dev]"
cp .env.example .env    # OPENAI_API_KEY (with credits), ANTHROPIC_API_KEY, HF_TOKEN
set -a; source .env; set +a
# copy data/CS-Jail-UR_final_approved_791.csv (sha256 4a34e81a...1ecf) into data/, then:
python scripts/exp0_finalize_data.py --restore            # recreate the gitignored dataset file
pytest tests -q && python scripts/smoke_test.py --model qwen25
python scripts/urdu_sanity_check.py --backend vllm --out outputs/checks/urdu_sanity_vllm.json
python -m csjail.run_eval --max-families 2 --models qwen25 phi3 llama32 \
    --allow-unvalidated-judge --out-dir outputs/smoke/exp2-vllm        # production-backend smoke (debug)
python scripts/exp1_sample_for_annotation.py --role validation --models qwen25 phi3 llama32 \
    --exclude-sample-dirs outputs/exp1/development-cpu-20260926 outputs/exp6/feasibility-cpu-20260927
# optional now: generation-only full sweep (metrics stay NA until the judge PASSes)
python -m csjail.run_eval --out-dir outputs/exp2/main --skip-judge
```
