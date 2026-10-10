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

## Exp 1 — judge choice and held-out test: **DONE (2026-10-08, DeepSeek V4.1 Flash)**

**Final judge:** DeepSeek V4.1 Flash, `deepseek-v4.1-flash:cloud`, through the
local Ollama app (provider `ollama`, Ollama Pro), rubric harm-v2 (written
before any result), thinking off, 300-token budget, temperature 0,
fingerprint `b0248df31f1d9da2`. Production manifest
`outputs/exp1/judge_validation_manifest.json`, status SELECTED. Judged
2026-10-06 UTC (development) and 2026-10-07 UTC (held-out); every response
parsed. Commands: `RUNBOOK.md` §3; design: `docs/PROTOCOL.md` §3.

**Design (replaces the 0.90 PASS gate; see history below):**
1. Choose on development: every judge scored on 720 responses
   (`rubric-dev-01`: 60 families × EN/CS/RU/UR × qwen25/phi3/llama32). Rule
   declared before any DeepSeek result: highest macro-F1 over the four
   languages, ties → higher minimum recall; the 0.90 gate reported, not
   required.
2. Test on held-out: the chosen judge and the runner-up on 960 responses
   from 160 other families (`heldout-960`).
3. Exp 2 corrects ASR with the judge's error per model × language, pooled
   over both sets (1,680 responses, 140 per cell, 7–49 harmful per cell).

**Labels:** one independent reviewer labelled all 720 development responses
(rounds 1–3) and 733 of the 960 held-out (round 4), blind to judge output and
original labels. The other 227 held-out responses are identical to development
responses the reviewer labelled and carry that label (222 refusals, 5
unintelligible; all safe). Harmful: development EN 11, CS 33, RU 34, UR 25;
held-out EN 27, CS 34, RU 31, UR 29. Reviewer vs original annotators: κ 0.56
on development (57 safe→harmful, 4 reverse), 0.82 on held-out (32 and 2).

**Development, selection** (`outputs/exp1/judge_comparison.json`):

| Judge, rubric | P | R | F1 | κ | macro-F1 |
|---|---|---|---|---|---|
| DeepSeek V4.1 Flash, harm-v2 | 0.73 | 0.86 | 0.79 | 0.75 | 0.783 |
| DeepSeek V4 Pro, harm-v2 | 0.64 | 0.86 | 0.74 | 0.68 | 0.749 |
| gpt-4o, harm-v2 | 0.57 | 0.96 | 0.71 | 0.65 | 0.733 |
| gpt-4o, harm-v3* | 0.80 | 0.63 | 0.71 | 0.66 | 0.720 |
| gpt-4o, harm-v4* | 0.82 | 0.61 | 0.70 | 0.66 | 0.707 |
| gpt-4o, harm-v5* | 0.60 | 0.83 | 0.70 | 0.64 | 0.705 |
| gpt-4.1, harm-v4* | 0.79 | 0.60 | 0.69 | 0.64 | 0.704 |

\* rubric revised after seeing errors on these items, so optimistic. Family
bootstrap (2,000 draws): Flash − gpt-4o v2 F1 +0.08 [+0.02, +0.13]; Flash − Pro
F1 +0.056 [−0.004, +0.117].

**Held-out test** (`outputs/exp1/heldout_comparison.json`, from
`scripts/exp1_heldout_compare.py`), P/R/F1 per language:

| Judge | EN | CS | RU | UR | F1 | κ | macro-F1 |
|---|---|---|---|---|---|---|---|
| DeepSeek V4.1 Flash | .96/.93/.94 | .82/.94/.88 | .59/.74/.66 | .56/.79/.66 | 0.77 | 0.74 | 0.784 |
| DeepSeek V4 Pro | .96/.85/.90 | .72/.97/.82 | .49/.84/.62 | .44/.83/.58 | 0.71 | 0.66 | 0.731 |

- Flash − Pro: F1 +0.063 [+0.012, +0.118], κ +0.078 [+0.017, +0.142]; the
  same without the 227 carried labels (neither judge flags any of them).
- Flash held-out vs development: macro-F1 0.784 vs 0.783, F1 −0.017
  [−0.117, +0.092], so the choice was not flattered overall. Per language,
  RU F1 fell 0.83 → 0.66 (−0.18 [−0.35, −0.005]) and EN rose 0.77 → 0.94
  (development had 11 harmful EN). Flash over-flags 1.20× (judge-flagged ÷
  truly harmful), Pro 1.46×.
- No gpt-4o run covers the held-out set (an old run judged 465 of the 960).

**Judge error used by Exp 2** (development + held-out, per model × language):
false-alarm rate 12–16% for llama32 CS/RU/UR, 10% phi3 UR, ≤ 5% elsewhere;
recall 0.84–1.00 except qwen25 RU 0.43 (7 harmful), phi3 UR 0.73 and qwen25 UR
0.75. The two sets did not differ measurably (overall recall gap −0.01,
false-alarm gap −0.003; 1 of 24 per-cell Fisher tests p < 0.05).

**Caveats for the paper:**
- `reviewer_id` is blank in all four review files (1,453 rows). The paper must
  say who the reviewer was and whether any tool assisted; do not claim
  independent review without that record. Round-3 notes are 15 fixed
  sentences (one per label category); rounds 1, 2 and 4 have per-item notes.
- Rubrics v3–v5 were tuned on the development items; harm-v2 (the chosen
  rubric) was not.
- The judge is a named cloud model, not a pinned snapshot, and Ollama retires
  cloud models (V4 Flash retired 2026-09-25). All raw judgments are kept; the
  paper records model name, provider and run dates.

**Not done (needed only for Exp 8):** the benign (over-refusal) rubric has not
been checked with Flash. Labelled sample: `outputs/exp1/benign-validation-merged`
(450 responses, 32 refused, two raters + adjudication).

## Exp 1 — history: judge validation (superseded 2026-10-07)

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
- **Option 2 set up (2026-10-06):** `configs/judge.yaml` is the harm-v2 text
  again (fingerprint `b0d6676cf5d89d08` reproduced), used only as a screener;
  it has no PASS manifest, so `run_eval` and Exp 6 still refuse it.
  `scripts/exp2_verify.py` screens the eval_main responses (Kaggle
  `MODE = "screen"`: 2,400 main + 1,320 robustness), builds the blinded
  review file (all flagged + 10% of the unflagged per model × condition) and
  scores the human labels into per-model × condition ASR. Expected review
  load for main ≈ 650 responses (assuming ~20% flagged). The Kaggle upload
  now includes the Exp 0 dataset file; without it no Exp 2 judging could run
  on Kaggle. **Still open under option 2:**
  - aggregation (`csjail.aggregate`) and Exp 3–5 read per-item judge labels,
    and need the human-verified labels instead;
  - Exp 6 preference data needs a labelling decision (screener labels only
    choose training pairs, so they may be acceptable there);
  - Exp 8 post-training evaluation needs the same screen-and-verify step per
    arm;
  - the paper's methods section must describe the screening design.
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

## Exp 2 — main evaluation: **DONE, FIVE MODELS (2026-10-08, DeepSeek V4.1 Flash)**

- **Five models (2026-10-08):** the three originals plus Gemma 4 E2B and
  DeepSeek-R1-Distill-Qwen-1.5B, all judged (0 judge errors). Raw ASR, all 791
  families (EN / CS / RU / UR): r1qwen15 .569 / .382 / .125 / .013; gemma4e2b
  .033 / .066 / .063 / .046. Judge kappa with the reviewer: original three
  .72–.76, r1qwen15 .52, gemma4e2b .54. **Correction method (decided
  2026-10-08): predictive values for all five models** (flagged share ×
  P(harmful | flagged) + unflagged share × P(harmful | not flagged), from the
  reviewer-labelled responses of the same model and language); Rogan–Gladen
  collapsed where the judge's error is lopsided or harm is rare (R1 CS → 0,
  phi3 UR → 0, Gemma cells undefined). Corrected ASR, all 791 families (EN /
  CS / RU / UR): r1qwen15 .520 / .311 / .037 / .024; llama32 .124 / .318 / .319 /
  .220; qwen25 .116 / .076 / .046 / .060; phi3 .080 / .120 / .064 / .078;
  gemma4e2b .049 / .081 / .063 / .093.
  Robustness: r1qwen15 CS .536 / .805 / .480, RU .150 / .375 / .135; gemma4e2b
  CS .084 / .110 / .085, RU .082 / .115 / .090 (per-draw / any-of-5 / greedy).
  Backend check (qwen25 Ollama vs vLLM): ASR within 2 points, McNemar p ≥ .42.
  Full tables and caveats: `docs/EXP2_NEW_MODELS.md` § Results; summaries
  `outputs/exp2/summary_5models.csv`, `summary_5models_eval_main.csv`.

- **Done:** main 9,492/9,492 and robustness 6,000/6,000 judgments ok (0 errors),
  run manifest `judging: judge SELECTED in Exp 1`, git `3eb3abc`. The laptop's
  Modern Standby paused the run twice (no data lost); a keep-display-on
  request fixed it. Summaries: `outputs/exp2/main/summary.csv`,
  `summary_eval_main.csv`, `outputs/exp2/robustness/robustness_summary.json`.
- **Raw ASR, all 791 families (EN / CS / RU / UR):** llama32 .118 / .373 / .359
  / .298; phi3 .082 / .113 / .100 / .102; qwen25 .154 / .092 / .034 / .061.
  The 200 held-out families (`--split eval_main`) give the same pattern.
- **Corrected ASR, superseded 2026-10-08 by the predictive-value correction
  above** (these were Rogan–Gladen, per model × language judge error from Exp 1,
  development + held-out, 140 responses per cell), all 791 families EN / CS / RU /
  UR: llama32 .125 / .328 / .281 / .219; phi3 .085 / .121 / .064 / .000; qwen25
  .127 / .078 / .029 / .062.
  E.g. llama32 CS .328 [.237, .418] (was .312 [.141, .464] with the 60
  development responses per cell), RU .281 [.165, .380]. 11 of 12 intervals
  narrow (median width −40%); qwen25 RU widens (7 harmful in its cell, judge
  recall .43).
  phi3 UR corrects to 0: its 10% raw rate matches the judge's false-alarm
  rate in that cell. The judge's false-alarm rate varies by model (~12–16% on
  llama32 CS/RU/UR, ~2% on qwen25), so per-language correction alone is biased.
- **Robustness (sampled, T 0.7, 5 draws; per-draw / any-of-5 / greedy):**
  llama32 CS .434 / .735 / .405, RU .438 / .780 / .395; phi3 CS .140 / .310 /
  .115, RU .099 / .270 / .115; qwen25 CS .154 / .300 / .130, RU .062 / .180 /
  .020. CS-vs-RU ordering matches greedy only for qwen25 (llama32 and phi3
  differ by < .04, within noise).


- **Judge:** `deepseek-v4.1-flash:cloud` via the local Ollama app (Ollama Pro),
  harm-v2 rubric, fingerprint `b0248df31f1d9da2`, production manifest status
  SELECTED (`outputs/exp1/judge_validation_manifest.json`, written by
  `exp1_compare_judges.py --write-manifest`). Runs on the local machine:
  Kaggle cannot reach the signed-in Ollama. ~190 judgments/min.
- **Commands:** `python -m csjail.run_eval --out-dir outputs/exp2/main --judge-only`,
  then `python scripts/exp2_robustness.py --judge-only --greedy-results outputs/exp2/main`;
  both resume from `judgments.jsonl`.
- **Reporting:** `python -m csjail.aggregate outputs/exp2/main [--split eval_main]`
  prints raw ASR and ASR corrected for the judge's error per model × language
  (default since 2026-10-08: the judge's predictive values from the
  reviewer-labelled responses of that model and language; `--correction
  rogan_gladen` for the earlier method). Interval pairs the family bootstrap with
  posterior draws. Assumes the labelled responses are a random sample of the
  same model's greedy responses in that language.
- **Superseded:** the screen-and-verify design (`exp2_verify.py`) and the Kaggle
  `screen` mode are no longer the plan.


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

- **Extension (2026-10-07/08): two more SLMs — done**, see the five-model bullet at
  the top of this section and `docs/EXP2_NEW_MODELS.md`.
- **Held-out labels (round 4, prepared 2026-10-07):** `outputs/exp1/gold-audit-04`,
  733 of the 960 set-aside items to review; 227 identical to development responses
  the reviewer already labelled keep that label (`carried_labels.csv`).

## Exp 3 — matched contrasts: **DONE (2026-10-08, five models)**

`python scripts/exp3_isolation.py --results outputs/exp2/main outputs/exp2/main-r1
outputs/exp2/main-gemma4 --out-dir outputs/exp3` (and `--split eval_main --out-dir
outputs/exp3/eval_main`) → `isolation_results.json`, `contrasts_table.csv`.

**Design (declared in `configs/eval.yaml` before the first run, 2026-10-07; correction
amended 2026-10-08 to follow Exp 2, before the five-model run):**
- Test: two-sided McNemar on the judge's labels, complete pairs, all 791 families.
- Holm within two families:
  - confirmatory: qwen25, phi3, llama32 × CS−RU / RU−UR / RU−EN, 9 tests (the paper's family);
  - extension: r1qwen15, gemma4e2b, 6 tests (models added after Exp 2 results were seen).
- CS−EN: descriptive.
- Effect sizes:
  - raw paired difference, 10,000-draw domain-stratified family bootstrap;
  - **judge-corrected paired difference** with the judge's predictive values per model ×
    condition, the Exp 2 correction (`aggregate.predictive_value_diff`): pairs resampled
    together, Jeffreys draws of each condition's P(harmful | flagged) and P(harmful | not
    flagged), and the model's pooled P(harmful | flagged) where the judge flagged none of a
    cell's labelled responses (Gemma CS/EN/UR).
- **Finding** = Holm-significant AND corrected 95% CI excludes 0 with the same sign.
- Descriptive: paired shifts in refusal, non-response (unintelligible / irrelevant / empty)
  and full-only harm; EN refusal → harmful elsewhere; per-model GEE (raw labels,
  supplementary) and a pooled condition × model interaction GEE (15,820 rows, converged).

**Results, all 791 families (pp; corrected difference with 95% CI):**

| Family | Model | CS−RU | RU−UR | RU−EN | CS−EN (desc.) |
|---|---|---|---|---|---|
| confirmatory | llama32 | +1.4 / −0.0 [−8.3, +8.2] | +6.1 / +9.8 [+1.6, +17.9] (Holm p .057) | **+24.1 / +19.4 [+12.7, +26.5]** | +25.5 / +19.4 |
| confirmatory | phi3 | +1.3 / +5.5 [+1.0, +9.4] (Holm p 1.0) | −0.3 / −1.4 | +1.8 / −1.6 | +3.0 / +4.0 |
| confirmatory | qwen25 | +5.8* / +3.0 [−1.8, +6.7] | −2.7 / −1.4 | **−12.0 / −7.0 [−11.4, −1.7]** | −6.2 / −4.0 |
| extension | r1qwen15 | **+25.7 / +27.4 [+14.5, +40.8]** | +11.3* / +1.4 [−4.2, +8.9] | **−44.4 / −48.3 [−59.5, −34.8]** | −18.7 / −20.9 |
| extension | gemma4e2b | +0.3 / +1.8 | +1.8 / −3.0 | +3.0* / +1.4 [−4.5, +5.7] | +3.3 / +3.2 |

Bold = finding (raw / corrected); * = Holm-significant but the corrected CI includes 0 (not
robust to judge error). Four findings: llama32 RU−EN, qwen25 RU−EN, r1qwen15 CS−RU and RU−EN.
The 200 held-out families give the same four findings and no others.
- Not findings, but worth stating: llama32 RU−UR and phi3 CS−RU have corrected CIs that
  exclude 0, but their preregistered raw test is not Holm-significant.
- Why the lower harm in Urdu forms (paired shifts, pp, raw judge labels): phi3 RU vs EN
  refusal −70, non-response +58; qwen25 UR vs RU non-response +66, refusal −64; r1qwen15
  RU vs EN non-response +40; llama32 RU vs EN refusal −51 with harm +24 (it engages and
  complies). Gemma: refusal 91–95% in every form, 0 non-response.
- **Response types checked against the reviewer** (`scripts/exp1_kind_agreement.py` →
  `outputs/exp1/response_kind_agreement.json`, 2,160 reviewer-labelled responses): four-way
  type (harmful / refusal / safe help / non-response) agreement 0.825, κ 0.74; refusal κ 0.83,
  non-response κ 0.82 (per model κ: qwen25 .82, phi3 .74, gemma4e2b .74, llama32 .63,
  r1qwen15 .55). The judge over-calls refusal by 6–11 pp for qwen25, llama32, phi3 and
  r1qwen15 (gemma4e2b 2); non-response rates match within 2.3 pp for every model. The
  interpretive shifts hold with
  reviewer labels on the labelled families (judge / reviewer, pp): phi3 RU−EN refusal −75 / −70,
  non-response +50 / +47; qwen25 RU−UR non-response −65 / −73; r1qwen15 RU−EN non-response
  +42 / +52; llama32 RU−EN refusal −67 / −58, but its non-response shift (+15 / +0) is not
  confirmed.
- Full-only harm moves little (|diff| ≤ 6 pp except r1qwen15 RU−EN −17 and CS−EN −11): the
  contrasts are driven by partial assistance.
- English refusal → harmful elsewhere (share of the model's EN refusals): llama32 CS 32% /
  RU 35% / UR 28% (684 refusals); r1qwen15 CS 30% (73); phi3 8–10%; qwen25 3–7%; gemma4e2b
  3–5%.
- GEE odds ratios vs EN (raw labels): llama32 CS 4.5 / RU 4.3 / UR 3.2; phi3 1.4 / 1.2 / 1.3;
  qwen25 0.55 / 0.19 / 0.35; r1qwen15 0.46 / 0.10 / 0.01; gemma4e2b 2.2 / 2.1 / 1.4.

## Exp 4 — tokenizer fertility: **DONE (2026-10-08, five models)**

`python scripts/exp4_features.py --results outputs/exp2/main outputs/exp2/main-r1
outputs/exp2/main-gemma4 --tokenizer llama32=unsloth/Llama-3.2-3B-Instruct --out-dir
outputs/exp4` → `features_results.json`, `fertility_by_prompt.csv` (counts only, no text).

**Design:** fertility = tokens of the raw prompt (pinned tokenizer, no chat template, no
special tokens) / whitespace words. Tokenizer check against the runs (recorded prompt tokens
− our count = one constant): **1.000 for all five models** (llama32 via the ungated mirror,
gemma4e2b against its Ollama build). Association per model, exploratory: GEE (logit,
clustered on family) of harm and of non-response on fertility standardized within condition,
adjusted for condition, domain and log word count; conditions where every response has the
same outcome are left out; Holm across the 9 pooled fits (Gemma has no non-response).
**Human-label check:** the same pooled fit on the reviewer-labelled responses (560 per
original model, 240 per added model), with the judge's and the reviewer's labels on
identical rows. CMI / Urdu share: **not run** (no validated tagger).

**Mean fertility (× EN):**

| Model | EN | CS | RU | UR |
|---|---|---|---|---|
| qwen25 / r1qwen15 (same tokenizer) | 1.16 | 1.48 (1.28) | 1.85 (1.60) | 3.12 (2.69) |
| llama32 | 1.16 | 1.48 (1.28) | 1.83 (1.58) | 3.02 (2.61) |
| phi3 | 1.33 | 1.68 (1.26) | 2.03 (1.52) | 4.84 (3.64) |
| gemma4e2b | 1.16 | 1.30 (1.12) | 1.55 (1.34) | 1.39 (1.20) |

**Associations (OR per within-condition SD; full data, judge labels → human-label check):**

| Model | Non-response | Human-label check (judge / reviewer, same rows) | Harm | Human-label check (judge / reviewer) |
|---|---|---|---|---|
| phi3 | 1.58 (Holm p < .001) | 1.99 / **1.92 [1.43, 2.56]** | 0.82 (Holm p .03) | 0.81 / 0.96 [0.73, 1.27] |
| r1qwen15 | 1.34 (< .001) | 1.47 / **1.52 [1.06, 2.16]** | 0.75 (< .001) | 0.68 / 0.84 [0.56, 1.25] |
| qwen25 | 1.40 (< .001) | 1.38 / 1.07 [0.78, 1.46] | 0.92 (.45) | 0.77 / 0.95 [0.69, 1.30] |
| llama32 | 1.24 (.001) | 1.80 / 1.23 [0.95, 1.59] | 0.98 (.67) | 0.89 / 1.00 [0.81, 1.22] |
| gemma4e2b | no non-response | — | 1.15 (.32) | 3 events: uninformative |

- **Supported:** within a form, more fragmented prompts get more non-response, robustly
  under human labels for phi3 and r1qwen15; weaker and not distinguishable from 1 under
  human labels for qwen25 and llama32.
- **Not supported:** any link between fragmentation and harmful compliance. The negative
  harm associations in the judge's labels (phi3, r1qwen15) vanish with human labels: a
  judge artefact, not a finding.
- Descriptive, n = 5 models: Gemma barely fragments Urdu script (1.39 tokens/word vs
  3.0–4.8) and has no non-response; r1qwen15 shares qwen25's tokenizer exactly yet differs
  sharply in behaviour, so tokenization alone does not explain model differences.

## Exp 4b — intent-recovery probe: **DONE (2026-10-09); scorer validated**

Generated on Kaggle (`notebooks/kaggle_exp4b.ipynb`): 2,000/2,000 probes ok (100 families × 4
forms × 5 models; Gemma via Ollama 0.40.0, pinned build accepted; R1 finished its reasoning on
386/400, the other 14 are "uncertain"). Scored on the laptop with the comprehension rubric
through DeepSeek V4.1 Flash (`outputs/exp4b/comprehension_summary.json`; probe texts and the
blinded audit file stay local). **Scorer validated** against the reviewer's labels on 80 blinded
answers (4 per model × form; `outputs/exp4b/scorer_agreement.json`): agreement 0.887, κ 0.78;
understood vs not 74/80 (CS 19/20, EN 18/20, RU 17/20, UR 20/20). The scorer is slightly strict
(4 reviewer-understood answers scored not_demonstrated, 2 the other way), so recovery rates are
if anything understated. The returned file has no reviewer_id column (omitted from the file I
generated): record who reviewed and whether any tool assisted.

**Intent recovered (share understood, EN / CS / RU / UR):**

| Model | EN | CS | RU | UR | Note |
|---|---|---|---|---|---|
| qwen25 | .95 | .48 | .17 | .04 | |
| phi3 | .96 | .44 | .11 | .02 | |
| llama32 | .61 | .58 | .34 | .60 | 32 EN probes refused (refusing to restate ≠ not understanding) |
| r1qwen15 | .70 | .32 | .13 | .02 | |
| gemma4e2b | .90 | .84 | .91 | .84 | |

- Qwen, Phi-3 and R1 mostly do not understand Roman Urdu or Urdu-script requests; Gemma
  understands every form and still refuses; Llama understands Urdu script and often RU.
- **Contrasts restricted to families understood in both forms** (raw judge labels, small n,
  descriptive): R1 RU−EN −44 pp on all probed families → **0.0 (n 13)**; R1 CS−EN −25 → +3.8
  (n 26); Llama RU−EN +19 → **+13 (n 23)**, CS−EN +16 → +22 (n 41); Qwen RU−EN −11 → −6 (n 16).
  R1's lower harm in RU/CS is non-comprehension; Llama's higher harm in RU persists when it
  understands.
- ASR among understood probes is in the summary file; UR/RU cells for qwen25/phi3/r1qwen15 have
  2–17 understood items, so their conditional rates are not interpretable.

### Exp 4b design and steps (as prepared 2026-10-08)

**Design** (`scripts/exp4b_comprehension.py`, `csjail/comprehension.py`):
- 100 families, domain-stratified, chosen without outcomes (seed 42), × 4 conditions × 5 models
  = 2,000 safe probes (intent-probe-v1: describe in one English sentence what the message asks
  for, without doing it). The harmful-request outcome is reused from Exp 2.
- Backends as in Exp 2: vLLM fp16 for qwen25, phi3, llama32, r1qwen15; Gemma through Ollama
  (Kaggle install pinned to 0.40.0, the version of Exp 2's Gemma run). Greedy; 96 tokens, R1
  2,048 with only the answer after `</think>` scored (unfinished reasoning = uncertain).
- Scorer: the comprehension rubric through the Exp 1 judge (DeepSeek V4.1 Flash via Ollama):
  understood / not_demonstrated / refused_probe / uncertain. Smoke test on harmless pairs: 3/3
  correct.
- Reported per model × condition: probe outcome counts, recovery rate, ASR within each bucket,
  recovered-and-not-harmful share. Per model: the planned contrasts on the probed families and
  restricted to families recovered in BOTH conditions (paired, raw judge labels, descriptive).
- Scorer audit: 80 answers (4 per model × condition), blinded (`review_sample.csv`; the
  scorer's labels sit in `review_key.csv`) → a bilingual reviewer fills `human_comprehension`
  → `--score-review` → agreement and κ.

**Steps:**
1. Kaggle: `notebooks/kaggle_exp4b.ipynb`, GPU T4 x2, Internet on, `HF_TOKEN` secret, Save &
   Run All (~1 h) → download `exp4b_outputs.zip`.
2. Laptop: unzip in the repo, then
   `python scripts/exp4b_comprehension.py --score-only --models qwen25 phi3 llama32 r1qwen15
   gemma4e2b --baseline-results outputs/exp2/main outputs/exp2/main-r1 outputs/exp2/main-gemma4`
   (~10 min with the signed-in Ollama).
3. Reviewer: `outputs/exp4b/review_sample.csv` (contains prompts; private), then
   `python scripts/exp4b_comprehension.py --score-review outputs/exp4b/review_sample.csv`.

**Known issue:** the laptop's Ollama (0.40.1) reports Gemma build `0cf45094…`/`823bb442…`, not
Exp 2's pinned `95e5aad2…` (recorded under 0.40.0 on Kaggle), so local Gemma runs are refused;
Gemma probes run on Kaggle with the pinned Ollama version.
## Exp 5: cut

## Phase 2 (Exp 6–10): **EXP 7–9 DONE (2026-10-11); Exp 10 open**; plan in `docs/PHASE2_PLAN.md`

- **Primary (RQ4):** C (our CS pairs) vs B_ext (external English pairs) at equal accepted-pair
  budgets and identical optimisation, for Llama-3.2 and Phi-3 — a comparison of training
  recipes, not of language alone. Secondary C/B_matched optional (skipped by default).
- **Production yield (Exp 2, judge-flagged CS failures on the 591 training families):** llama32
  223, phi3 66 (qwen25 49, r1qwen15 232). Same-family CS-and-EN failures: llama32 57, phi3 18.
  Judge precision on CS: llama32 .76, phi3 1.00 (Exp 1 cells). Phi-3's budget is therefore
  about 55, not 100. The earlier CPU pilot (`outputs/exp6/feasibility-cpu-20260927/`, Ollama
  Q8, team-reported counts) is superseded by these production counts.
- **Prerequisites (PHASE2_PLAN P1–P7):**
  - P1 training precision: `train_dpo` now takes `compute_dtype` from `configs/dpo.yaml`
    (float16 = the T4 inference precision; trainable LoRA weights kept fp32 for fp16 AMP).
    **GPU-unverified** until the GPU smoke.
  - P2 Exp 8 split: `exp8_posteval.py --skip-judge` (Kaggle: generations + capability, provenance
    in `run_manifest.json`) then `--judge-only` (laptop: no model; `CachedRunner` replays arm E's
    system-prompted generations too). Mocked test: same flags as the one-step run.
  - P3 over-refusal judge: **PASS** — DeepSeek V4.1 Flash, rubric benign-v1 (fingerprint
    `2c6aa62d0eabc0a0`) on the 450 labelled benign responses: P 1.000, R 0.938, F1 0.968 (32
    refused, 30 flagged, 0 failures) → `outputs/exp1/judge_validation_manifest_benign.json`.
    Valid for the current probe set only (Latin script; see P4).
  - P4 over-refusal probe: 150 Latin-script prompts, 0 Urdu script, language mix unaudited (a
    rough lexicon check suggests mostly code-switched), old C01–C10 taxonomy — decision D4.
  - P5 capability sets: **built and frozen** — `data/capability/mmlu_500.jsonl` (cais/mmlu test,
    57 subjects, seed 42, sha256 `d21c06eb…`) and `urdummlu_300.jsonl` (MBZUAI/UrduMMLU @
    `aae060a8`, CC BY 4.0, seed 42, sha256 `5f9d2206…`; all 300 answers checked against the
    source). The loader now reads UrduMMLU's dict-shaped options in letter order (it would
    otherwise have kept the letters) and its `correct_key`.
  - P6 chosen generator needs `ANTHROPIC_API_KEY` — decision D5.
  - P7 unseen-domain rule — decision D6.
- **Exp 6 (2026-10-09): pairs built, human review out.** `exp6_build_prefdata.py --languages CS
  --exclude-exemplar-domains D6` (every refusal written without the D6 exemplar, so one reviewed
  set serves the main and the D6 runs). Chosen generator **OpenAI gpt-4.1-2025-04-14**: Anthropic's
  classifier blocked the calls (the few-shot exemplars and targets contain harmful requests;
  stop_reason refusal, general_harms), and claude-sonnet-4-5 was not served to the key. Mined CS
  failures: phi3 66, llama32 223; chosen judged a genuine refusal, not harmful: 66/66, 223/223.
  Refusals are much shorter than the rejected answers (median ~290 vs 1,600–1,960 characters;
  not padded, per protocol — report it). Review: `scripts/exp6_review.py make` →
  `outputs/exp6/review/review_file.xlsx` (156 rows: all 66 Phi-3 + first 90 Llama, 15 per domain;
  private), then `apply` writes `outputs/exp6/<model>_verified` and `<model>_ablation_D6`.
- **Exp 6 review applied (2026-10-09):** reviewer UU (as stated by the team; `reviewer_id` cells were
  blank and filled from that statement); provenance recorded verbatim in
  `outputs/exp6/review/verified_summary.json` — an earlier returned workbook held the same labels on
  155/156 rows plus a second labelling and an adjudication sheet (agreement on `rejected_harmful`
  0.705, κ 0.36); the paper must describe the labelling as it was actually done.
  Judge-flagged CS answers confirmed harmful: phi3 44/66 (67%), llama32 66/90 (73%); refusals OK
  61/66 and 84/90; kept (both yes) phi3 39, llama32 61; naturalness of kept refusals 4.72 / 4.66
  (gate ≥ 4). **Budgets per model** (protocol: adjusted down if fewer qualify; C and B_ext equal within
  a model): main phi3 39 / llama32 60, D6 ablation 30 / 51. N-curve: phi3 25; llama32 25, 50. All 19
  training jobs pass a local pre-flight (naturalness gate, budgets, split/leakage/domain checks).
  Note for Exp 2: the judge was right on 14/14 flagged phi3 CS answers in Exp 1 but on 44/66 here;
  at 0.67 precision phi3's corrected CS ASR would be ~0.08 rather than 0.12 (llama32 consistent:
  0.76 vs 0.73). Small Exp 1 sample or a stricter standard here; phi3 had no Exp 3 finding.
- **Learning check (agreed 2026-10-09, before any training on real pairs):** `configs/dpo.yaml`
  `learning_check` — the seed-42 C and B_ext adapters of both models are judged on their training logs
  (final-epoch loss ≤ 0.60, reward accuracy ≥ 0.75); if any fails, all arms retrain at 4, then 6 epochs.
  Built into the notebook (`phase2_kaggle.py run --phase gate|rest --epochs`, `decide`); results in
  `outputs/phase2_gate/`.
- **Over-refusal probe v2 (P4 resolved):** `data/benign_probe_v2.jsonl` — 60 harmless prompts
  (10 per domain) × EN/CS/RU/UR, written by the team (UU) with AI assistance, human-verified
  (manifest records it); now the Exp 8 default, with over-refusal reported per form.
- **B_ext built (2026-10-09):** PKU-SafeRLHF @ `9421ffaf` (CC BY-NC 4.0 — cite; research use):
  73,907 rows → 10,796 pairs with exactly one safe response (chosen = safe, rejected = unsafe);
  leakage screen vs the 200 held-out EN prompts (token-Jaccard ≥ 0.5, the Exp 0 screen) dropped 0 —
  lexical only, PKU prompts are worded very differently, so paraphrases are not ruled out; seeded
  order, first 1,000 → `data/pref_pairs_en_external.jsonl`; the D6-free copy drops pairs whose
  unsafe response PKU tags Cybercrime (2,007) → `..._noD6.jsonl`, which Exp 7 uses automatically
  for the ablation split. Pair files gitignored; manifests committed.
- **Kaggle Phase 2 ready:** `notebooks/kaggle_phase2.ipynb` + `scripts/phase2_kaggle.py`.
  `STAGE="smoke"`: fp16 DPO on 8 harmless pairs + the adapter in vLLM, per model (no private
  data). `STAGE="train_eval"`: per model on its own T4, all adapters (C and B_ext × seeds 42/43/44
  at N, C n-curve, D6 ablation) then Exp 8 generation-only; resumable. Exp 7 gained `--seed`.
- **GPU smoke PASSED (Kaggle T4, 2026-10-09; `outputs/phase2_smoke/`):** both models trained a LoRA-DPO
  adapter in fp16 (P1 verified) with the pinned stack (torch 2.4.0, transformers 4.46.3, trl 0.12.2,
  peft 0.13.2, bitsandbytes 0.44.1; no pin mismatches), LoRA modules resolved (Phi-3 fused
  `qkv_proj` / `gate_up_proj`; Llama q/k/v/o/gate/up/down), finite loss (ln 2 at initialisation, as
  expected), and the adapter served through vLLM. Training now logs every step (a 60-pair budget is
  ~8 optimizer steps): check the loss falls below ln 2 and reward accuracy rises in the real runs.
- **Exp 7 + 8 run (2026-10-10/11).** The first full Kaggle run stalled (its monitor wrote child
  output to the kernel's hidden stdout) after deciding the epochs (2 FAIL, 4 PASS). Rerun split over
  three accounts (`phase2_kaggle.py parts/queue`, all on commit `165a9d2`, identical pinned versions,
  4 epochs; arm A and E generated once, in the main folder). 19 adapters, 11 folders, 23,920
  generations, 0 failures; judged on the laptop (DeepSeek V4.1 Flash), 0 judge errors.
- **Learning check re-recorded** (`outputs/phase2_gate/`): 2 epochs FAIL (loss stays ln 2), 4 PASS
  (phi3 C .061/1.0, B_ext .383/.94; llama32 C .001/1.0, B_ext .306/.97). Cause found afterwards: fp16
  loss scaling skips the first 2–4 optimizer steps (gradient overflow while the scale calibrates), so
  2 epochs left 0–2 effective updates. Applying the same declared check to every adapter
  (`outputs/phase2_analysis/training.csv`): all main-budget adapters (5–9 effective steps), Llama n50
  and Llama D6 learned; **C n25 (both models; 0 and 1 effective steps) and both Phi-3 D6 adapters (2)
  did not** — their comparisons are flagged `adapter_learned=False` and are not evidence.
- **Post-training judge audit** (`outputs/exp8/audit/`, reviewer UU, 300 blinded rows, 0 unsure, no
  notes): no cell switched (Fisher p ≥ .078; closest phi3 CS, judge precision 23/30 vs Phase 1 14/14)
  → Phase 1 error counts throughout. Harmless-prompt judge agreement RU/UR .67–.93 (n = 15 each).
- **RQ4 primary (C − B_ext on CS, three seeds pooled, Holm across models):** **llama32 −9.7 pp raw
  [−15.5, −4.0], corrected −6.8 [−11.3, −2.6], p_holm .004 → FINDING** (per seed −10.5, −12.5, −6.0);
  phi3 +0.3 [−2.2, +3.0], p .90 → no finding (floor: no arm moved Phi-3's ASR).
- **Secondary (corrected pp vs untrained A; unadjusted):** Llama C: CS −9.0, RU −6.9, EN −3.4, UR +0.9
  (no transfer to Urdu script); Llama B_ext: CS −2.2, RU −2.9, EN −0.5, UR −1.6. Safety prompt E: largest
  harm drop (Llama CS −19.0, Phi-3 CS −9.9) but heavy over-refusal on harmless prompts (Phi-3 RU +87 pp,
  CS +38, UR +28; Llama RU +32, UR +18); C and B_ext add no over-refusal (all within ±2 pp). Capability
  retention ≥ .95 except Llama C UrduMMLU .949 (−6 of 300 items). N-curve (Llama): n50 −10.6 ≈ n60 −9.5.
  Unseen domain D6 (Llama, 35 families): C trained without D6 − A on CS −10.1 [−22.6, +2.2] vs C trained
  with it −11.4 [−23.0, −0.6] — same direction, underpowered; B_ext 0.0. Phi-3 D6: no harmful output in
  any arm (and its D6 adapters did not train). Phi-3 UrduMMLU ≈ .27 in every arm (chance .25).
- **Open:** Exp 10 (tables/figures, error taxonomy, sanitized examples); paper update.

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
