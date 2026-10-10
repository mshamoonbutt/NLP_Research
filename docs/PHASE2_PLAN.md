# Phase 2 plan: mitigation (Exp 6–10)

Written 2026-10-08, after Phase 1 (Exp 0–4) finished; Exp 4b running. Design source:
`docs/PROTOCOL.md` §6, `configs/dpo.yaml`, paper §5. Status of each step is kept in
`docs/EXPERIMENT_STATUS.md`. Time estimates are estimates.

## Phase 1 in one table

| Exp | What it did | Main result |
|---|---|---|
| 0 Dataset | 791 harmful-request families × EN / CS / RU / UR = 3,164 prompts, 6 domains; 200 held-out / 591 training families | All integrity checks pass |
| 1 Judge | Judges scored against reviewer labels; chosen on 720 development responses, tested on 960 held-out | DeepSeek V4.1 Flash (Ollama): held-out F1 .77, κ .74; error per model × language from 2,160 labelled responses; response-type κ .74; 0.90 gate reported, not met |
| 2 Baseline | 5 models × 3,164 greedy + 10,000 sampled responses; ASR corrected with the judge's predictive values | Corrected CS ASR: Llama .32, R1 .31, Phi-3 .12, Gemma .08, Qwen .08 |
| 3 Contrasts | CS−RU, RU−UR, RU−EN per model; McNemar + Holm + judge-corrected differences | 4 findings (Llama RU−EN +19 pp, Qwen RU−EN −7, R1 CS−RU +27, R1 RU−EN −48); no script effect; low harm in Urdu forms is mostly non-response, not refusal |
| 4 Fertility | Tokens per word vs outcomes | Fragmentation goes with non-response (human-confirmed for Phi-3, R1), not with harm; CMI not run |
| 4b Intent probe | Can the model restate the request? | Running (Kaggle) |

## Questions

- **RQ4:** at equal accepted-pair budgets, does DPO on the model's own CS failures (arm C)
  cut CS harmful compliance more than external English safety pairs (arm B_ext), and at what
  cost to English safety, over-refusal and capability? A recipe comparison, not a test of
  language alone.
- **RQ5:** sensitivity to data volume, transfer to RU/UR, and an unseen harm domain.

**Models: Llama-3.2-3B and Phi-3-mini** (as in the paper). Llama is the most CS-vulnerable
model (corrected CS ASR .32) with 223 judge-flagged CS failures on training families; Phi-3
has 66, so its budget is about 55, not 100. Not used: R1 (2,048-token reasoning outputs
exceed the 1,024-token DPO length; judge κ .52), Gemma (cannot be trained/served on the
pinned stack), Qwen (49 failures).

## Prerequisites (blockers found 2026-10-08)

| # | Blocker | Fix | Status (2026-10-08) |
|---|---|---|---|
| P1 | `train_dpo` uses bf16; Kaggle T4s have no bf16 | fp16 compute on T4 (= inference precision); GPU smoke (PROTOCOL §9, never run) | **done:** GPU smoke passed on Kaggle T4 (fp16 training + adapter in vLLM, both models) |
| P2 | Exp 8 judges inside the GPU run; Kaggle cannot reach the Ollama judge | Split: generate on Kaggle, judge on the laptop (as Exp 2) | done (`--skip-judge` / `--judge-only`, tested) |
| P3 | Over-refusal (benign) judge not checked with Flash; Exp 8 needs its manifest | Run Flash on the 450 labelled benign responses | **PASS**: P 1.000, R 0.938, F1 0.968 |
| P4 | Over-refusal probe: 150 Latin-script prompts, 0 Urdu script, language mix unaudited, old C01–C10 taxonomy | Decision D4 | **done:** `data/benign_probe_v2.jsonl` (60 × 4 forms) |
| P5 | Capability sets (MMLU-500, UrduMMLU-300) not built; UrduMMLU source to obtain | `prepare_capability_sets.py` | done: both frozen (UrduMMLU from MBZUAI/UrduMMLU, CC BY 4.0) |
| P6 | Chosen-response generator (Claude Sonnet 4.5, distinct from the judge) needs `ANTHROPIC_API_KEY` | Decision D5 | **done:** gpt-4.1-2025-04-14 (Anthropic's classifier blocked the pipeline) |
| P7 | Exp 9 unseen domain never declared before outcomes | Seeded draw by a rule recorded before drawing; disclose it came after Phase 1 | done: D6 drawn |

## Experiments

### Exp 6 — preference pairs
- **Rejected (arm C):** the model's own greedy CS responses on the 591 training families that
  the judge flagged harmful (Llama 223, Phi-3 66).
- **Reviewer check of rejected (new, recommended):** judge precision on Llama CS is .76, so
  about 1 in 4 flagged responses may be safe; training to avoid them would teach refusal of
  safe answers. Check the ~160 actually used.
- **Chosen:** Claude writes a CS refusal per prompt from the refusal exemplars; the judge
  checks each is a genuine refusal, not harmful; a bilingual reviewer rates 50 per model for
  naturalness / clean refusal (gate: mean ≥ 4).
- **B_ext:** PKU-SafeRLHF pairs with one safe and one unsafe response (licence to check),
  screened for overlap with the held-out families; same count as C.
- Seeded domain-balanced ordering → nested budgets 25 / 50 / all.
- **Achieves:** the two equal-size training sets per model; pair yield (a result itself).
- **Time:** me ~2 h; API ~15 min (~$2–5); reviewer ~2.5 h; ~1 day elapsed.

### Exp 7 — DPO training
- QLoRA DPO, one config for every arm (4-bit base, LoRA r16, β 0.1, lr 5e-5, 2 epochs,
  effective batch 16), Kaggle T4.
- Adapters per model: C and B_ext at the full budget × 3 seeds; C at 25 and 50; C and B_ext
  for the unseen-domain ablation → 10 per model, 20 total. Optional D and C/B_matched
  skipped (only Llama has enough same-family EN+CS failures: 57).
- **Time:** P1 + notebook ~3 h (me); GPU smoke 1–1.5 h; training ~10 min per adapter →
  3–4 GPU-h.

### Exp 8 — post-training evaluation (paper Table 7)
- Every arm (A untrained, E safety prompt, B_ext, C, each seed) on the same 200 held-out
  families × 4 forms + over-refusal probe + MMLU-500 + UrduMMLU-300; generate on Kaggle,
  judge on the laptop.
- Primary: C vs B_ext on CS (paired, Holm across the 2 models). Also each arm vs A, EN
  drift, RU/UR transfer, over-refusal, capability retention (reporting flags, not gates).
- **Judge audit (new, recommended):** the judge's error was measured on untrained models and
  DPO changes response style; a reviewer labels ~240 post-training responses (B, C × CS/EN ×
  2 models × 30) to check the correction still holds.
- **Time:** ~24 evaluation runs × 15–20 min = 6–8 GPU-h; ~23k judgments ≈ 2 h laptop;
  reviewer ~2 h; analysis 1–2 h.

### Exp 9 — generalisation (RQ5)
- **Data efficiency:** C at 25 / 50 / full, safety and utility (runs above).
- **Transfer:** CS-trained adapters on RU / UR (from Exp 8).
- **Unseen domain:** the drawn domain withheld from all training data (PKU categories mapped
  to D1–D6), C and B_ext retrained, evaluated on that domain's ~33 held-out families (wide
  intervals).
- **Time:** ~1.5 GPU-h + ~1 h (category mapping).

### Exp 10 — final analysis
- Tables and figures, uncertainty, missingness, a recorded residual-error sample, error
  taxonomy, sanitized examples. **Time:** ~0.5 day + ~1 h reviewer.

## Decisions

| # | Decision | Recommendation | Status |
|---|---|---|---|
| D1 | Phase 2 models | Llama + Phi-3 | recommended |
| D2 | Reviewer checks rejected responses | Yes | **done 2026-10-09:** 156 rows reviewed (UU); kept phi3 39, llama32 61 → **per-model budgets** 39 / 60 (D6 run 30 / 51), since Phi-3 has fewer than 60 |
| D3 | Training seeds | 3 for primary C vs B_ext; 1 elsewhere | recommended |
| D4 | Over-refusal set | Team writes ~50 benign prompts × 4 forms (3–4 h); else report over-refusal as limited | **decided:** yes; 60 (10 per domain) recommended, 50 minimum; spec below |
| D5 | Chosen generator | `ANTHROPIC_API_KEY` for Claude Sonnet 4.5, or another model distinct from the judge | **changed 2026-10-09:** OpenAI gpt-4.1-2025-04-14 — Anthropic's classifier blocked the calls (harmful requests in every prompt) and claude-sonnet-4-5 was not served to the key |
| D6 | Unseen-domain rule | Seeded draw recorded before drawing | **done:** rule committed in 5b4d2e7 (`random.Random(791)` over D1–D6), drawn **D6** in 141cfe4; split `1af335defc251d52` |
| D7 | Optional arms | Run E; skip D and matched arms | recommended |
| D8 | Learning check (small budgets ≈ 5–8 optimizer steps) | Seed-42 C and B_ext of both models judged on TRAINING logs: final-epoch loss ≤ 0.60 and reward accuracy ≥ 0.75; else retrain all arms at 4, then 6 epochs | **agreed 2026-10-09, declared in `configs/dpo.yaml` before any training on real pairs; automatic in the notebook.** Result: 2 epochs failed, 4 passed → all adapters at 4 |
| D9 | Combining the 3 seeds (RQ4 primary) | Per family, mean judge label over seeds for C and for B_ext; sign-flip permutation over families; Holm across the 2 models; finding = Holm-significant AND corrected CI excludes 0 with the same sign; per-seed McNemar as robustness | **agreed and declared 2026-10-10** (`configs/dpo.yaml` `analysis`), while Kaggle was still generating, before any output was judged |
| D10 | Judge correction after training | Phase 1 error counts; the blinded audit (A, B_ext, C × CS/EN, 30 flagged + 30 unflagged per model × condition; + 60 harmless RU/UR) switches ALL arms of a model × condition to its counts when either rate differs from Phase 1 (Fisher exact p < .05) | **declared 2026-10-10**; switch wording amended the same day, still before any output was judged (the first wording, "outside Phase 1's interval", ignored the audit's sample size) |

**Laptop steps after the Kaggle run** (unzip `phase2_outputs.zip` into the repo first):
1. `python scripts/phase2_kaggle.py judge` — every Exp 8 folder, DeepSeek judge, resumable (~24k judgments).
2. `python scripts/exp8_audit.py make` → reviewer fills `outputs/exp8/audit/audit_file.xlsx` (~300 rows).
3. `python scripts/exp8_audit.py apply --review <returned.xlsx> --reviewer-id UU`.
4. `python scripts/phase2_analysis.py` → `outputs/phase2_analysis/` (PROVISIONAL until step 3 is applied).

## Schedule (~3–4 days; critical path: reviewer, API key, Kaggle quota)

| Day | Work |
|---|---|
| 1 | Prerequisites; Exp 6 (key + reviewer) |
| 2 | Kaggle run 1: GPU smoke → Exp 7 all adapters → Exp 8 part 1; judge overnight |
| 3 | Kaggle run 2: Exp 8 rest + Exp 9 ablation; post-training audit; Exp 10 |

GPU total ≈ 12–14 h of Kaggle's 30 h/week; running Phi-3 and Llama on the two T4s at once
roughly halves it.

## Harmless prompt set v2 (D4) — spec for the team

- **60 prompts, 10 per domain (D1–D6)**; 50 (8–9 per domain) is the minimum. Each written in all
  four forms (EN, CS, RU, UR) the same way as the harmful set → 240 (or 200) items.
- Each prompt sounds close to its domain's topic but is clearly legitimate, e.g. for D6 enabling
  two-factor authentication, for D2 finding a helpline for a friend, for D5 checking which apps
  can see one's location. Mix: safety or prevention information, help-seeking, education or
  history, fiction or analysis, everyday tasks that contain trigger words.
- No real names, no personal data, nothing a reasonable reader would consider harmful.
- Template: `docs/benign_probe_v2_template.csv` (`prompt_id` like `BN-D3-07`, `boundary_type`
  free text). Fill `written_by` / `translated_by` (initials) — the paper's provenance section
  needs them; say in `notes` if any tool assisted.
- Used in Exp 8 for over-refusal per form. The benign judge passed on Latin-script responses
  only, so the post-training audit adds ~60 reviewer labels on RU/UR benign responses.
- **Why 50–60 is enough:** every arm answers the same items, so arms are compared pair by pair;
  with 200–240 items per arm a 10-point over-refusal increase (the reporting flag) is
  detectable; per-form rates (50–60 each) are descriptive.

## Power of the primary test (simulated 2026-10-09)

C vs B_ext on CS, 200 held-out families, paired McNemar at α = .025 (Holm's first step), family-difficulty
model: Llama-like C .15 vs B_ext .30 → power 0.97; .20 vs .30 → 0.61; .25 vs .30 → 0.13. Phi-3-like (near the
floor) .04 vs .08 → 0.25; .03 vs .10 → 0.73. The pair budget does not change this power (the test set is fixed);
it changes how large the effect can be. Differences under ~10 points will mostly go undetected: report CIs,
never equivalence.
