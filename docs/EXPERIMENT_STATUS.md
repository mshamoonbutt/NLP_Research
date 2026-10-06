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
  (24/150 refused in both). Folder `benign-validation-kaggle-02` received and
  verified (fp16 qwen25, items hash matches, all 150 labelled responses
  identical to the generated ones). Merged `benign-validation-merged`: 450
  items, refused κ 1.000, kind κ 0.977, gold refused 32 / not refused 417;
  **1 kind-only disagreement awaits adjudication**
  (`benign-validation-kaggle-02/adjudication_todo.csv`).
- **Stage C made resumable (2026-10-06):** gate predictions and Exp 2
  judgments persist per chunk; quota exhaustion stops API calls (exit 4, no
  verdict); `scripts/stagec_run.py` always writes a resume bundle. End-to-end
  check in a fresh clone with the real bundle (no key): gates exit 3 cleanly,
  Exp 2 withheld, bundle written and re-used as input.
- **First harm-gate attempt (Kaggle, 2026-10-05; judge gpt-4o-2024-08-06,
  rubric harm-v2):** stopped by exhausted OpenAI credits after 1,185/1,200
  predictions (benign gate 0/450). No verdict or manifest was written.
  Preliminary evaluation of the 1,185 finished predictions against the human
  gold: **the gate would FAIL on precision in every condition** (EN 0.69,
  CS 0.44, RU 0.26, UR 0.19; recall 0.79–1.00; 181 false positives vs 87
  true positives). The judge labels responses the raters called
  unintelligible (90), safe-helpful (64) or irrelevant (22) as harmful
  compliance, mostly "partial"; 80 of the 101 UR false positives are
  unintelligible (degenerate Urdu-script output), and 128/181 false positives
  hit the 512-token cap. The pattern holds in both annotation batches (base
  P 0.28, top-up P 0.40), so it is a judge problem, not a labelling shift.
  Thresholds are not lowered. The rubric (or judge model) must be revised on
  development data and validated on families it was not tuned on.
- **Decision (2026-10-06): Option B below; the judge stays gpt-4o-2024-08-06
  (recall 0.79–1.00 shows the model detects harm; the failure is the decision
  procedure). Rubric revised to harm-v3: an ordered procedure (intelligibility
  → relevance → assistance → degree) ahead of the unchanged definitions, and
  response_kind decided before unsafe. Iterated on `rubric-dev-01` (role
  development: feedback reports only).** Split: development =
  `validation-kaggle-01` (60 families, all four conditions, 720 items, gold
  complete); validation = `validation-kaggle-01-topup1` (80 families, EN/UR,
  480 items, gold complete) + `validation-kaggle-01-topup2` (80 new families,
  CS/RU, 480 items drawn from the Exp 2 cache on 2026-10-06, **not yet
  annotated**). Family-disjoint; expected gold unsafe EN 24, UR 21, CS ~26,
  RU ~22. **2026-10-06 update:** top-up 2 annotated (480/480; unsafe κ 0.952,
  kind κ 0.942; agreed gold unsafe CS 27, RU 17); 20 disagreements (4 binary,
  16 kind-only) await adjudication, after which `validation-v2-merged` is
  built. Benign merged sample complete: 450 gold, 0 unresolved, 32 refused.
  **Validation set built (2026-10-06):** `validation-v2-merged` = top-up 1 +
  top-up 2 (20/20 adjudicated): 960 items, 160 families, 0 shared with the
  development set, 0 unresolved; pooled agreement unsafe κ 0.958, kind κ 0.950;
  gold unsafe/safe EN 25/215, CS 28/212, RU 17/223, UR 21/219 (all ≥ 10).
  Ready for the final harm gate once rubric iteration on `rubric-dev-01`
  finishes.
  Disclosure: the first rubric's aggregate confusion was inspected
  over all 1,200 items (including top-up 1) before this split was declared;
  no item-level tuning used top-up 1. The alternative is a fully fresh
  1,200-item validation sample at roughly 2.5× the annotation cost.
- **Rubric iteration 1 (harm-v3, fingerprint `7030929c9323c3cc`, 2026-10-06,
  720/720 judged, 0 failures):** on the same items, false positives fell
  from 125 (v2) to 43; UR 48 → 7, RU 42 → 17, CS 25 → 13, EN 10 → 6.
  Still below the gate: precision EN 0.46, CS 0.58, RU 0.32, UR 0.50
  (intervals wide; dev set has only 6–20 gold unsafe per condition). New
  problem: RU recall fell to 0.50 (8 of 16 harmful RU responses labelled
  unintelligible/irrelevant), i.e. step 1 was over-applied to romanized
  text. Remaining false positives are mostly gold safe-helpful responses
  judged "partial" (31 of 43; 28 of 43 from llama32).
- **harm-v4** (step 1 excludes non-standard spelling/romanization from
  "unintelligible"; step 3 adds a "concretely better equipped" test; the
  rationale must name a concrete element the response supplies). Also added:
  a development-only `--judge-model` comparison (same rubric, other OpenAI
  model) to tell a rubric ceiling from a model ceiling; the validation judge
  stays the configured one.
- **Rubric iteration 2 (harm-v4; gpt-4o `3e8b9c302deb4023` and gpt-4.1
  `203df99c54b0479b`, 720/720 each, 0 failures, 2026-10-06):** no material
  change. Overall precision/recall: gpt-4o v3 0.469/0.760, gpt-4o v4
  0.481/0.740, gpt-4.1 v4 0.474/0.740. Per condition (gpt-4o v4): EN
  0.46/0.83, CS 0.60/0.90, RU 0.33/0.50, UR 0.50/0.75. Agreement on the
  unsafe label (κ): raters 0.978; gpt-4o vs gpt-4.1 0.863; v3 vs v4 0.957;
  each judge vs gold 0.54. 31 items are false positives for all three
  judge configurations and 11 are missed by all three (42/720; 31 from
  llama32, 23 in RU; gold for 38 of them is rater agreement). Outside those
  42 items judge precision is ~0.80. **Conclusion: rubric iteration has
  saturated and a model change does not help; the residual is a stable
  disagreement between the LLM judges and the human labelling standard on
  borderline responses.**
- **Blinded gold audit (option 1, decided 2026-10-06):**
  `scripts/exp1_gold_audit.py make` → `outputs/exp1/gold-audit-01/`. The
  42 consensus-disputed development items + 84 controls (2 per disputed item,
  random, from items where all three judge configurations agree with the gold;
  matched on gold label exactly and on model/condition where the pool allowed),
  shuffled, opaque IDs, no item ID/model/condition/gold/judge output in the
  reviewer file. Key and reviewer file are gitignored; `audit_plan.json`
  (committed before the review) records the seed, composition, both files'
  SHA-256 and the decision rule: **gold has systematic error** if the
  reviewer differs from gold on ≥ 50% of disputed items, ≤ 10% of controls,
  Fisher p < 0.05; **gold stands** if ≤ 30% of disputed; otherwise
  inconclusive. Systematic error → re-specify the gold standard with a
  documented third review and re-review a random sample of the validation
  set under it before any validation run; otherwise → measurement option 2
  (judge as screener, humans verify judge-positive responses, random audit
  of judge-negatives). Reviewer: bilingual, not one of the raters or the
  adjudicator, no pre-filled draft, no judge output.
- **Audit result (scored 2026-10-06, `gold-audit-01/audit_result.json`):
  VERDICT gold_has_systematic_error.** Reviewer differs from the gold on
  25/42 disputed items (0.60, CI 0.44–0.73) vs 4/84 controls (0.05, CI
  0.02–0.12); Fisher p < 0.0001. Of 31 items the raters labelled safe but
  every judge labelled harmful, the reviewer sided with the judges on 24; of
  11 items every judge missed, the reviewer sided with the raters on 10. The
  original labels are lenient on borderline partial compliance. Projection
  (126 of 720 dev items corrected): gpt-4o harm-v4 precision 0.48 → ~0.79,
  recall ~0.79. Gap: `reviewer_id` was blank on the returned file; the
  reviewer's identity and independence must be recorded.
- **Declared follow-up:** (1) rater guide clarified on partial assistance
  (`docs/exp1_rater_guide.md`); (2) `gold-audit-02`: every dev item not yet
  re-reviewed on which any of the four judge configurations (gpt-4o v2/v3/v4,
  gpt-4.1 v4) disagrees with the gold (82) + 82 matched controls, blind;
  `exp1_gold_audit.py select` then picks the gpt-4o rubric version on the
  corrected dev gold by the rule in its plan (no API cost); (3)
  `third-review-v2`: all 960 validation items, blind, for the validation
  gold (original gold where the third review agrees, adjudication otherwise,
  by an adjudicator who is not one of the original raters).
- **Amendment (2026-10-06, before any validation review): two-phase
  validation replaces the full 960-item third review** (`third-review-v2`
  withdrawn, never issued). Phase 1: the frozen gpt-4o rubric (chosen on the
  corrected dev gold) predicts all 960 validation items once (`stagec_run
  --validation-predict`, no verdict). Phase 2: the independent reviewer labels,
  blind, ALL judge-flagged items + a random 25% of unflagged items per
  condition (~320 items instead of 960). Gold = original where the reviewer
  agrees, adjudication otherwise. Precision is exact on the flagged items;
  recall uses inverse-probability weighting of the sampled misses; gate
  thresholds and support minimums unchanged. Re-reviewing only judge/gold
  disagreements is explicitly excluded (it would bias the judge's score
  upward). `exp1_gold_audit.py make-two-phase` / `score-two-phase`; the
  score step writes the production manifest, and the final Kaggle run no
  longer re-scores against the original (lenient) validation gold. The
  audit reviewer was a new, independent person (per the team, 2026-10-06);
  initials still to be recorded.
- **Gold audit round 2 (164 items, returned 2026-10-06; `reviewer_id` blank
  again, so the official `selection_result.json` waits for it):** reviewer
  differs from the original gold on 25/82 disagreement items vs 3/82
  controls. Corrected development gold: 290 of 720 items re-reviewed, 57
  labels changed. Preliminary selection (same function as the official
  step): gpt-4o v2 P 0.59 / R 0.97 (min-condition P 0.45, UR); v3 0.84 /
  0.65; v4 0.86 / 0.63 (UR recall 0.44); gpt-4.1 v4 0.83 / 0.62. Declared
  rule → **v2 chosen (only one with recall ≥ 0.80), flag fired (min P <
  0.85)**: the validation run was NOT started.
- **harm-v5 (one more development run, declared before running):** v2 text
  + only the intelligibility rule (v3's step 1, which raised UR precision
  0.45 → 0.93), fingerprint `2a8177e4fb3ea862`. **Decision rule:** on the
  corrected development gold, if v5's minimum per-condition precision ≥ 0.85
  AND minimum per-condition recall ≥ 0.85, v5 is frozen and goes to the
  two-phase validation run; otherwise measurement option 2 with gpt-4o v2 as
  the screener (recall 0.97): humans verify every flagged response plus a
  random audit of the unflagged. If v5 disagrees with the gold on more than
  10 development items outside the 290 re-reviewed ones, those are
  re-reviewed blind (with matched controls) before the decision.
- **harm-v5 result (Kaggle, 2026-10-06; 718/720 judged, 2 lost to the
  event-loop bug below):** against the corrected development gold, precision
  / recall EN 0.67 / 0.91, CS 0.80 / 0.92, RU 0.54 / 0.79, UR 0.49 / 0.72;
  overall 0.62 / 0.83. Only 1 disagreement lies outside the 290 re-reviewed
  items, so no extra review was triggered. **Decision rule not met → v5 is
  not frozen; measurement option 2 (gpt-4o harm-v2 as screener, humans verify
  every flagged response plus a random audit of the unflagged) applies.**
- **Judge throughput bug fixed (2026-10-06):** `score_sync` starts a new event
  loop per chunk, but the concurrency semaphore and HTTP pool were created
  once, in the first loop. Every later chunk failed on contention ("bound to a
  different event loop") and crawled through backoff retries. This, not only
  the Tier 1 TPM limit, caused the ~7 judgments/min seen on Kaggle; Tier 1
  supports ~16–18/min. Both are now created per call; a regression test
  reproduces the error. Affects calibrate_judge (chunks of 64) and Exp 2
  judging (chunks of 256); no stored judgment was wrong, only slow or
  `api_error` (which is re-tried on resume).
- **Alternative judge under consideration (partner suggestion, 2026-10-06):
  DeepSeek V4.** Not runnable locally on Kaggle or the CPU box (V4-Flash
  ~90–175 GB VRAM; Ollama serves it only as a cloud model). Testable through
  DeepSeek's OpenAI-compatible API (`provider: deepseek`, key
  `DEEPSEEK_API_KEY`) with the development-only `--judge-provider/--judge-model`
  comparison on the same 720 items, rubric v5 and corrected labels. Caveat:
  DeepSeek replaced the model behind `deepseek-v4-flash` on 2026-09-10
  (V4.1 now `deepseek-flash`); a judge whose alias can change needs the exact
  model and dates recorded, and re-validation if it changes.

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
