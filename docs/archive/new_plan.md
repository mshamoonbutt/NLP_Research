# CS-Jail-UR: Research Execution Plan

**A native Urdu–English code-switched safety benchmark for small language models, with preference-alignment mitigation.**

**Target:** ACL Rolling Review submission by **Monday 3 August 2026** (committing to EACL 2027). All co-authors must register as reviewers by **Wednesday 5 August 2026**.

---

## 1. What this project is

Large language models are much less safe outside English. A prompt that a model refuses in English is often answered when the same request arrives in a lower-resource language, because safety training is overwhelmingly English-centric (roughly four in five published safety datasets are English-only). This project studies that failure for the way bilingual South Asian users actually write — **Urdu and English mixed together in a single message** ("code-switching"), often typed in the Latin alphabet ("Roman Urdu") — and specifically for **small language models (SLMs)**, the 1–4 billion-parameter models that run on phones and consumer hardware and are known to resist jailbreaks far less reliably than large ones.

The project has two phases:

- **Phase 1 — Measure.** We build a benchmark of adversarial prompts written from scratch by native Pakistani bilinguals, and use it to measure how often small models produce harmful content across four language conditions. This tells us *how much worse* code-switching makes safety, and *why*.
- **Phase 2 — Mitigate.** We apply preference-alignment post-training (the DPO family of methods) using our own code-switched safety data, and measure whether it reduces the jailbreak rate without breaking the model's English safety, causing it to over-refuse harmless questions, or degrading its general ability.

### The dataset

1,000 base prompts, each spanning **ten harm categories** grounded in Pakistani context (e.g. sectarian incitement, fraud involving local systems like NADRA/FBR/JazzCash, and the standard harm axes). Each base prompt is written in **four language conditions**, giving 4,000 evaluation items:

| Code | Condition | Purpose |
|------|-----------|---------|
| **CS** | Urdu–English code-switched | the real-world attack surface |
| **EN** | Monolingual English | the safety baseline models are trained on |
| **RU** | Roman Urdu (Latin script, no English content words) | isolates language from script |
| **UR** | Urdu in Nastaliq (native script) | isolates script from language |

The four-condition design is deliberate: because all four share the same underlying harmful intent and differ only in language or script, comparing them **isolates the cause** of any safety gap (see §3, Experiment 3).

### The models

- **Under test (SLMs):** Phi-3-mini (3.8B), Qwen2.5-1.5B, Llama-3.2-3B.
- **Reference (LLM):** Llama-3-8B, to show the small-model gap.

### Why this is novel

No prior benchmark is simultaneously (a) natively authored rather than machine-translated, (b) code-switched, (c) for a South Asian language, and (d) focused on small models. The closest works each miss at least one: translation-based South Asian benchmarks exist but inherit English harm structure; a code-switched red-teaming benchmark exists but *synthesises* its prompts rather than collecting them from native speakers; a Chinese culture-specific benchmark for small models exists but has no South Asian analogue. Our defensible claim:

> *the first natively authored, code-switched safety benchmark for any South Asian language, and the first to isolate code-switching, script, and language effects through a controlled four-condition design.*

### Why Phase 2 matters intellectually

There is an **open, unresolved contradiction** in the research literature that our data is unusually well-placed to settle:

- One line of work (Li et al., 2024) shows that aligning a model for safety in **English alone transfers** to many other languages — implying you would not need language-specific data.
- Another (Krasnodębska et al., 2026) shows that for **refusal** specifically, English-only alignment is **insufficient**, and language-specific training is required.

No one has tested this for code-switching — a setting that is peculiar precisely because the harmful prompt *contains English itself*. If English-anchored safety transfers anywhere, it should transfer here; if it still fails, that is a strong statement about how shallow English-centric alignment really is. Phase 2 is designed to resolve this, which turns "we applied DPO to a new language" into a genuine scientific result.

---

## 2. Research questions

Every experiment exists to answer one of these. If a result doesn't map here, it doesn't go in the paper.

| RQ | Question | Answered by |
|----|----------|-------------|
| **RQ1** | Do small models produce harmful content more often under code-switching than in monolingual conditions? | Exp 2, 3 |
| **RQ2** | Is the effect caused by code-switching itself, by the script, or by the language? | Exp 3 |
| **RQ3** | What mechanism explains it — token fragmentation, mixing density, or gaps in the model's safety coverage? | Exp 4 |
| **RQ4** | Does English-only safety alignment transfer to code-switching, or is code-switched training data necessary? | Exp 7, 8 |
| **RQ5** | Does code-switched mitigation generalise across harm types and conditions *without* harming English safety, over-refusing, or degrading capability? | Exp 8, 9 |

RQ4 is the heart of the paper.

---

## 3. Key definitions (used throughout)

**Attack Success Rate (ASR)** — the fraction of harmful prompts for which a model produces harmful compliance, judged automatically. Reported per (model × condition × category).

**The judge** — an LLM (GPT-4o) that reads each (prompt, response) pair and labels it *harmful compliance* or *refusal/safe*, using a fixed rubric. Because every ASR number is the judge's opinion, the judge is validated against human labels before use (Experiment 1).

**CS-gap** — ASR under code-switching minus ASR under English, computed on shared base prompts. The central Phase 1 quantity.

**Relative reduction** — (ASR before training − ASR after training) / ASR before training. The central Phase 2 quantity; target ≥ 50%.

**Over-refusal rate (ORR)** — the fraction of *benign* code-switched prompts a model wrongly refuses. A mitigation that just refuses everything is a failure, so this is measured, not assumed.

**Capability retention** — the model's score on a general ability test after training, divided by its score before. Target ≥ 95%.

**Data split (fixed once, at dataset lock, never changed):**
- **Held-out eval set** — 300 base prompts, stratified across all ten categories, from which **no** training data is ever derived. All post-training evaluation happens here, so no trained model has seen anything related to these prompts.
- **Training pool** — the other 700 base prompts; all preference training data is mined from these.
- **Held-out categories** — 2 of the 10 categories are additionally withheld from training, to test whether mitigation generalises to harm types it never saw.

**Statistics.** Because the four conditions share base prompts, all comparisons are *paired*: **McNemar's test** for each pairwise contrast, **bootstrap 95% confidence intervals** on every ASR, and a **mixed-effects logistic regression** (`comply ~ condition + category + model + random effect per base prompt`) as the overall model. Multiple-comparison correction (Holm–Bonferroni) across the condition contrasts.

**Generation protocol.** Primary: greedy decoding (temperature 0), one response per prompt, fixed seed, 512-token limit, each model's default chat template, and **no extra safety instruction added** (we measure the model's own alignment). Robustness check: temperature 0.7 with 5 samples on a 200-prompt subset, to confirm the condition rankings are stable.

---

## 4. The experiments

Fourteen experiments across two phases plus writing. Each lists its purpose, what to do, how long it takes, what it depends on, and its pass/fail gate where it has one.

### Phase 1 — Measuring the vulnerability

**Experiment 0 — Finalise and lock the dataset · 1 day**
Freeze the dataset once annotation is complete. Compute two extra columns per prompt: `urdu_word_ratio` (the share of Urdu words) and `cmi` (Code-Mixing Index, the standard measure of how mixed a sentence is) — these let us later test whether the code-switching effect is really just a "how much Urdu" effect. Check inter-annotator agreement (Cohen's κ on category labels; target ≥ 0.7), remove near-duplicates, and verify that each base prompt's four conditions carry the same intent and that Roman Urdu rows contain no English content words. Fix the data split above.
*Gate:* κ ≥ 0.7 and condition-parity ≥ 95%.

**Experiment 1 — Validate the judge · 1 day · GATE**
The most important safeguard in the paper. If the judge is worse at recognising harmful content in code-switched or Roman Urdu text than in English, the whole CS-gap could be an artifact of the judge rather than a property of the model. Build a human-labelled gold set of at least 60 (prompt, response) pairs **per condition** (240+ total), have two annotators label them, and measure the judge's precision **separately for each condition**. Compare GPT-4o (primary), GPT-4o-mini (cheaper), and an open safety classifier as a cross-check.
*Gate:* precision ≥ 0.90 in **every** condition. If code-switched or Roman Urdu falls short, escalate to GPT-4o and, if still short, use a hybrid (a pattern-based refusal detector plus the LLM judge only on disagreements). Do not run the main sweep until this passes.

**Experiment 2 — Main evaluation sweep · 2 days**
Generate every model's response to every prompt (4,000 prompts × 4 models = 16,000 responses) under the primary protocol, plus the temperature-0.7 robustness subset. Judge all of them. Compute ASR for every (model × condition × category) with confidence intervals. This produces the raw result surface everything else analyses.

**Experiment 3 — Isolate the three effects · 1–2 days**
The benchmark's headline result (RQ1, RQ2). Decompose the vulnerability into its causes with three paired comparisons per model:
- **Code-switching effect:** CS vs RU (same language and script family, differs in mixing).
- **Script effect:** RU vs UR (same language, differs in alphabet).
- **Language effect:** RU vs EN (same script, differs in language).

Then fit the overall mixed-effects model and report how much each condition raises the odds of a harmful response versus English.
*Success:* a statistically significant CS > EN gap in at least two of the three small models. (A null result is still reportable, but reframes the paper — decide the framing here.)

**Experiment 4 — Explain the mechanism · 1–2 days**
Answer *why* (RQ3), and pre-empt the obvious reviewer objection that the effect is just a "density" effect. Measure token fragmentation (how many tokens each word breaks into) per condition per model, and test whether it predicts harmful compliance. Plot ASR against the Code-Mixing Index to show whether the effect grows with mixing density or is a discrete code-switching effect. Optionally, probe the model's internal "refusal direction" to frame the failure as a gap in safety coverage rather than tokenisation — mark this as a stretch, not required.

**Experiment 5 — Small models vs large · 1 day (overlaps Exp 2)**
Justify the small-model focus by comparing the three SLMs against the Llama-3-8B reference. The reference runs in 4-bit quantisation while the SLMs run at full precision; note that quantisation tends to *weaken* safety, so this bias runs *against* our hypothesis (making the comparison conservative), and confirm the direction on a small full-precision subset.

### Phase 2 — Reducing the vulnerability

**Experiment 6 — Build the training data · 3 days**
Create preference pairs: each is a prompt with a *rejected* response (the model's actual harmful answer, taken from Experiment 2) and a *chosen* response (a natural code-switched refusal that declines and redirects). Generate the chosen refusals with GPT-4o from hand-written examples, and have a native speaker rate a 50-item sample for fluency and register. Assemble **500–800 code-switched pairs** from the training pool, balancing response lengths so the training doesn't just learn "shorter = better." Separately:
- Pull **English refusal pairs** off the shelf (from public safety datasets) for the English-only comparison arm — this needs none of our data and can be prepared first.
- Native-author a **~150-prompt over-refusal probe set**: benign code-switched prompts that superficially resemble harmful ones, used later to check the model hasn't become trigger-happy.
- Build a lightweight **capability test** (a subset of general knowledge questions in English plus an Urdu subset) to check the model's ability afterward.
*Gate:* chosen-refusal naturalness ≥ 4 / 5.

**Experiment 7 — Train the comparison arms · 2–3 days**
This is the RQ4 experiment. For each small model, train several versions, each a different answer to "what does it take to make code-switching safe?":

| Arm | Method | Training data | What it tests |
|-----|--------|---------------|---------------|
| **A** | none | — | untrained reference |
| **B** | DPO | English refusal pairs only | *does English-only safety transfer?* |
| **C** | DPO | code-switched pairs only | *does our data work?* |
| **D** | DPO | code-switched + English | best practical recipe |
| **E** | prompting only (no training) | — | a cheap prompting baseline |
| **F** | reward-gap method (MPO) | mixed | strongest "why not plain DPO" alternative *(if time)* |
| **G** | **code-switch-weighted DPO** | code-switched + English | *the algorithm improvement* — up-weights code-switched pairs in the loss *(optional novelty layer; drop first if time is short)* |

Arm B is the critical comparison: if the English-only model is already as safe on code-switching as the code-switched model, our data adds nothing — so B is trained and evaluated first, as it also validates the whole training pipeline. Arm G is the professor's algorithm-improvement contribution: it sits *on top of* the comparison, so the paper stands on the comparison (Arms A–E) even if G is cut.

Training uses QLoRA (memory-efficient fine-tuning on the quantised model); reasonable starting settings — LoRA rank 16–32, DPO strength β = 0.1, 1–3 epochs, learning rate 1e-5 to 5e-5 — tuned on a validation split. An optional light supervised warm-up on the chosen refusals before DPO can help if plain DPO underfits.

**Experiment 8 — Evaluate the trained models · 2 days · GATE**
Measure whether mitigation worked, on the **held-out evaluation set**, across all four things that matter (RQ5):
1. **Code-switched ASR reduction** versus the untrained model.
2. **The RQ4 contrast:** is code-switched DPO (Arm C) *significantly better* than English-only DPO (Arm B) on code-switched safety? This single test is the paper's core claim.
3. **English safety retained** — English ASR must not rise.
4. **No over-refusal** — run the benign probe set; the refusal rate on harmless prompts must stay low.
5. **Capability retained** — run the general ability test; the score must stay within 5% of the original.

*Acceptance:* Arm C reduces code-switched ASR by ≥ 50% relative *and* beats Arm B significantly; English-ASR drift ≤ 3 points; over-refusal increase ≤ ~10 points; capability ≥ 95% retained.
*If it fails* (English DPO transfers just as well, or our data doesn't help): this is still a **publishable result** given the open contradiction in §1 — but the paper's framing shifts from "code-switched data is necessary" to "English-anchored alignment fails on code-switching, and here is what closes the gap." Decide the framing the day these numbers arrive; both introductions are pre-drafted so no time is lost.

**Experiment 9 — Ablations · 2–3 days**
Show the result is robust and efficient — the questions reviewers always ask:
- **Data efficiency:** train on 100 / 200 / 400 / 800 pairs to show how much data is actually needed.
- **Method strength (β):** vary the DPO strength.
- **Category generalisation:** train on 8 harm categories, test on the 2 held-out ones.
- **Condition transfer:** does training on code-switched data also improve Roman Urdu and Nastaliq?
- **Method comparison (if time):** DPO vs two alternatives (SimPO, KTO) on identical data.

**Experiment 10 — Analysis and error study · 1–2 days**
Assemble all significance tests and confidence intervals into a statistics appendix. Sample the prompts that still succeed after training and characterise them (which categories, which registers) with two or three sanitised examples. Produce the per-category breakdown of where mitigation helps most and least.

### Writing and submission

**Experiment 11 — Draft the paper · 3–4 days (overlaps experiments)**
Write results-first: Method → Results (Phase 1 isolations, then Phase 2 mitigation) → Analysis → Related Work → Introduction (last, once numbers are known) → Abstract. The related-work review is already largely written and needs only citation keys and a trim. **Apply the citation corrections** collected during the literature review while drafting (correct venues and titles for several key references, and mark not-yet-peer-reviewed works as preprints), so no error survives into submission.

**Experiment 12 — Ethics, limitations, reproducibility · 1 day**
Write the ethics statement (controlled dataset release with access gating, no minors, annotator compensation, dual-use discussion), the limitations (judge reliability bounds, the quantisation note, the density analysis, single-country cultural scope, small-models-only), and complete the Responsible NLP / reproducibility checklist (all hyperparameters, seeds, and compute).

**Experiment 13 — Internal review and submit · 1–2 days**
Full-team read-through against the research-question table — every claim must trace to an experiment. Fix, freeze, submit, and **register all authors as reviewers within two days of submitting.**

---

## 5. Schedule to 3 August

The plan assumes a team of four running **four parallel streams**, weekend work through the sprint, and the **dataset locking by Saturday 25 July**. The single biggest reason this fits is that a large share of the work needs no finished dataset and is done first.

**The four streams:** *Data & Eval* (dataset, judge, sweep, Phase 1 analysis); *Training* (preference data, all arms, post-eval, ablations); *Writing* (all paper sections); *Support* (over-refusal set, capability test, English pairs). Streams other than Data & Eval start before the dataset is locked.

### Before the dataset locks — Tue 22 to Fri 24 July
Set up the training and evaluation software; build the judge harness and dry-run it on pilot data; source the English refusal pairs; **train the English-only arm (B) as a pipeline test**; write the Method and Related Work sections; native-author the over-refusal probe set; build the capability test.

### Day-by-day

| Date | Data & Eval | Training | Writing |
|------|-------------|----------|---------|
| **Sat 25 Jul** | **Exp 0** — lock dataset, QA, fix split | (prep) | (Method drafted) |
| **Sun 26 Jul** | **Exp 1** — judge validation · **GATE** | English pairs ready | Results scaffolding |
| **Mon 27 Jul** | **Exp 2** — main sweep (16k) | begin mining rejected responses | — |
| **Tue 28 Jul** | **Exp 3, 4, 5** — isolations, mechanism, SLM-vs-LLM | **Exp 6** — build preference pairs | Phase 1 Results |
| **Wed 29 Jul** | Phase 1 figures | **Exp 7** — train arms C, D (+ G if on track) | Phase 1 Results done |
| **Thu 30 Jul** | — | **Exp 8** — post-eval · **GATE**, decide framing | — |
| **Fri 31 Jul** | **Exp 10** — error analysis, stats | **Exp 9** — ablations | Phase 2 Results |
| **Sat 1 Aug** | — | trailing ablations | **Exp 12** — ethics, limitations; Intro; Abstract |
| **Sun 2 Aug** | full-team internal review · **freeze** | | |
| **Mon 3 Aug** | **SUBMIT by midday** | | |
| **Tue 4 Aug** | **all authors register as reviewers** | | |

### Drop-dead dates
If any slips, cut scope immediately (§6): dataset lock by **Mon 27 Jul**; judge gate passed by **Mon 28 Jul**; main sweep done by **Tue 29 Jul**; preference pairs ready by **Wed 30 Jul**; post-eval done by **Fri 1 Aug**; draft frozen by **Sun 2 Aug**.

### The two gates
- **Judge validation (Sun 26).** Below 0.90 precision on code-switched or Roman Urdu → escalate to GPT-4o, then hybrid. One day of slack.
- **Post-eval (Thu 30).** If the code-switched arm doesn't beat the English-only arm, pivot to the negative-result framing (pre-drafted); do not burn days chasing a positive result. The paper ships either way.

---

## 6. If time runs short

Cut from the top of this list until the remaining work fits. Everything above the line is expendable; nothing below it is.

1. Code-switch-weighted DPO (Arm G, the algorithm tweak) → future work
2. Reward-gap method (Arm F) → future work
3. Method comparison (SimPO / KTO) → future work
4. β-ablation → one line in limitations
5. Data-efficiency curve → 3 points instead of 4
6. Third small model in Phase 2 (keep all three in Phase 1)
7. Sampling robustness subset → note as a limitation

**Do not cut:** the Roman Urdu condition (it is what makes the four-condition design work); the English-only arm (B) (it is the core comparison); judge validation; the over-refusal and capability checks; ethics and limitations.

If instead you finish early, add Arms F and G and the method comparison — they strengthen the methods contribution.

---

## 7. Resources

**Compute.** The small models run and train on a single GPU (24–80 GB); generation takes hours, each training run minutes to an hour. The judge is the slow step — run its API calls in parallel.
**Budget.** Roughly **$200–300** in judge API costs across all arms. Version-pin and seed everything from day one for reproducibility.
**Software.** Standard open-source stack for inference, DPO training, and statistics.

---

## 8. What "done" looks like

**Phase 1:** a locked dataset with agreement statistics; a validated judge with per-condition precision; the main ASR table; the isolation result (the three effects); the mechanism and density figures; the small-vs-large comparison.
**Phase 2:** preference data and the over-refusal probe set; trained models for every arm; the mitigation table showing the code-switched arm beats the English-only arm; the ablations (data efficiency, category generalisation, condition transfer).
**Paper:** a complete draft with every claim traced to an experiment; ethics, limitations, and reproducibility sections; corrected citations; submitted to ARR by 3 August with all authors registered as reviewers.
