# CS-Jail-UR — 6-day execution timeline to final results

Written 2026-10-04. Companion to `docs/PROTOCOL.md` (design) and `RUNBOOK.md`
(commands). Resources assumed: OpenAI API key (judge), Kaggle/Colab GPU
(T4-class, fp16), the CPU box, two independent raters, one bilingual reviewer.

**Precision decision:** Kaggle/Colab T4s do not support bf16, and vLLM does
not run on the P100 (compute capability < 7.0). Production = **vLLM on T4,
`dtype: half` (fp16)** unless an A100/L4 is reliably available. Whatever is
chosen on Day 1 is used for *everything* (Exp 1 validation sample, Exp 2,
Exp 8) and recorded in every manifest. Do not mix.

**Scope for this cycle** (all protocol-sanctioned): mandatory core only.
Arms D / C_matched / B_matched, the CMI tag-validation study (Exp 4 restricts
to tokenizer fertility), multi-seed training, and the 12,000-response
four-condition robustness variant are NOT run and are reported as such.
Single training seed 42 goes in Limitations.

---

## Day 0 (today) — unblock everything

| Who | Task |
|---|---|
| user | Push the branch (contains the alignment fixes + new samplers). |
| user | Request HF access to `meta-llama/Llama-3.2-3B-Instruct` (gated; approval can lag — do it first). Create Kaggle secrets: `OPENAI_API_KEY`, `HF_TOKEN`. |
| user | Obtain `ANTHROPIC_API_KEY` (Exp 6 chosen generator must differ from the OpenAI judge). Needed by Day 3 evening. |
| user | Book both raters for Day 2–3 (~6–7 h each) and the bilingual reviewer for Day 3 evening → Day 4 morning (~3 h). **This is the critical resource.** |
| team | **Attest the unseen-domain choice now**, before any outcome is seen: pick the Exp 9 ablation domain and record it (`exp9_ablations.py domain --domain Dx --attest-chosen-before-outcomes` just derives the manifest; the choice itself is written down today). |
| human | Adjudicate the 15 flagged dev items (`outputs/exp1/development-cpu-20260926/annotations/adjudication_questions.csv`), run `scripts/exp1_dev_gold.py`, fill real verification metadata. |
| CPU+API | `calibrate_judge.py --sample-dir <dev> --gold-csv <dev>/development_gold.csv` — first live judge metrics (cents). Iterate the rubric **today** if it is clearly broken, then **freeze `configs/judge.yaml`**. |

## Day 1 — GPU bring-up and smoke (session 1, ~3–4 h GPU)

1. Kaggle T4: clone, `bash scripts/setup_a100.sh` (name is legacy; works on
   any CUDA host), `pytest tests -q`, `exp0_finalize_data.py --restore`.
2. PROTOCOL §9 smoke on the harmless fixture: pinned templates + probe,
   resume, **live judge parsing** (now has credits), tiny DPO with resolved
   LoRA modules (Phi-3 fused projections), adapter-in-vLLM, Exp 8 preflight.
3. `urdu_sanity_check.py --backend vllm` — settles whether the Urdu-script
   degeneration was a Q8 artifact.
4. Record the fp16 decision in the run manifests (dtype override).
5. `prepare_capability_sets.py` (MMLU; source UrduMMLU via HF id or export —
   if unobtainable by Day 4, Exp 8 reports UrduMMLU NOT_RUN).
6. `prepare_external_english_pairs.py --n 1000` (PKU-SafeRLHF; record
   license). Screen its prompts against eval families before training (Day 4).

**Gate:** smoke green = the pins/configs are trusted for production.

## Day 2 — generation day (session 2, ~6–9 h GPU) + raters start

1. Exp 1 validation sample: `exp1_sample_for_annotation.py --role validation
   --models qwen25 phi3 llama32 --exclude-sample-dirs <dev> <pilot>` → 720
   blank items. (Fallback if rater time is short: 2 models = 480, the
   protocol minimum.)
2. Benign validation sample: `benign_sample_for_annotation.py --role
   validation --models phi3 llama32` → 300 items.
3. Full Exp 2 **generation-only** sweep: `run_eval --out-dir outputs/exp2/main
   --skip-judge` (9,492; resumable across session cuts).
4. Robustness generation: `exp2_robustness.py` (6,000).
5. Exp 4b probe generation (~1,200; family selection is outcome-independent).
6. Midday: hand rater1/rater2 CSVs (harm + benign) to both raters. They work
   independently, blank files, no judge output. Target: done by Day 3 noon.

## Day 3 — judge PASS, then judge everything (API day)

1. Adjudicate disagreements (1–2 h). Run `calibrate_judge.py --sample-dir
   <val> --development-sample-dir <dev>` → **harm PASS manifest**; and
   `calibrate_judge.py --kind benign --sample-dir <benign-val>` → benign
   manifest. *Contingency:* a FAIL/INSUFFICIENT condition → iterate rubric,
   re-judge the same gold (new fingerprint, ~1 h, ~$1); Day 6 is the buffer.
2. Judge the cached generations (no GPU needed): Exp 2 main, robustness,
   Exp 4b scoring. ~17k calls, rate-limit bound, single-digit dollars.
3. CPU: `aggregate` (full-core + `--split eval_main`), `exp3_isolation.py`
   (McNemar + Holm-9 + GEE incl. per-model and interaction sensitivity),
   Exp 4 fertility analysis.
4. Evening: `exp6_build_prefdata.py --model {phi3,llama32}` (mining), chosen
   generation via Anthropic, `dry_run_report.py` for actual budgets. Send the
   clean-refusal + naturalness sheets to the bilingual reviewer.
5. Paper: populate judge-validation table, Table 5 (baseline ASR), Table 6
   (contrasts), Table 10 (response types) as numbers land.

## Day 4 — Phase 2 training + post-eval (session 3, ~6–9 h GPU)

1. Morning: bilingual review back → accepted pairs frozen (target ≤100/model,
   actual yield reported; nested budgets from `pairs_manifest.json`).
2. Confirm B_ext overlap screen vs eval families; equal budget to C.
3. `exp7_train_arms.py` per model: C@yield, B_ext@same, plus n-curve prefixes
   C@25 / C@50 where supported (adapters are minutes each at this scale).
4. `exp8_posteval.py --arms A B_ext C E`: 2 models × 4 arms × 800 held-out
   prompts + benign probe + MMLU/UrduMMLU, judged with the validated rubrics.
5. CPU evening: mitigation table (Table 7), RQ4 paired tests (Holm across the
   two models), EN drift, RU/UR transfer.

## Day 5 — generalization + final analysis (session 4, ~4–6 h GPU)

1. N-curve Exp 8 evals for the C@25/C@50 adapters.
2. Unseen-domain ablation (attested domain): `exp9_ablations.py domain` →
   re-mine/re-train C and B_ext under the exclusion (external source filtered
   too) → `exp8_posteval.py --tag ablation_Dx`.
3. Humans: Exp 4b scorer audit on the exported review sample; residual-error
   sample review + sanitized examples (Exp 10).
4. CPU: Exp 10 consolidation — final tables, budget curve, uncertainty,
   missingness; populate the remaining paper tables and App G items.

## Day 6 — buffer + paper close-out

- Absorb slips: judge re-validation, Kaggle quota, UrduMMLU sourcing, reruns.
- Paper edits that sync prose to what actually ran (see list in the
  2026-10-04 session notes / PR description): recall-gate sentence, robustness
  scope (CS/RU, top-p 0.9, 6,000), App D probe wording = implemented string,
  B_ext source + license, judge snapshot id, fp16 precision in App F,
  dataset authoring/provenance section from real records (team input —
  cannot be fabricated).
- Update abstract/discussion/conclusion from the full result set, favorable
  or not. Archive raw outputs to private storage; commit only manifests and
  summaries.

---

## Standing assumptions and risks

1. **Raters** (2× ~6–7 h, independent) and a bilingual reviewer are the only
   non-compressible resource. Every slip here moves Days 3–4 one-for-one.
2. **Judge must PASS** per-condition precision AND recall ≥ 0.90 with the
   declared support. One iteration is budgeted; repeated failure is a real
   finding and a real stop for production judging.
3. **Anthropic key** by Day 3 evening, or chosen-response generation stalls
   Phase 2 by exactly as long as it is missing.
4. **Pair yield** ≥ 25/model keeps the n-curve meaningful (pilot: ~18%
   failure rate on 100 families → ~100 projected over 591, unverified in
   production precision). Yields are reported as found; never padded.
5. **Kaggle budget:** ~15–20 GPU-hours total across 4 sessions fits the
   ~30 h/week quota. All runners are resumable; a session cut costs minutes.
6. GPU smoke failure on the pinned stack (vLLM 0.6.3.post1 / torch 2.4.0 /
   TRL 0.12.2 on T4) costs Day 1 to pin-fixing; candidates were never
   GPU-verified.
7. Raw generations/judgments contain harmful text: they stay gitignored and
   go to private storage, not the public repo.
