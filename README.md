# CS-Jail-UR

A safety benchmark and preference-alignment study for **small language models**
on **code-switched Urdu-English** requests.

- **Phase 1 (measure).** How often do SLMs produce harmful assistance when the
  same harmful request is written in English (EN), code-switched Urdu-English
  (CS), Roman Urdu (RU) or Urdu in Perso-Arabic script (UR)? Planned paired
  contrasts: CS–RU, RU–UR, RU–EN, plus the CS–EN gap, with a comprehension
  control.
- **Phase 2 (mitigate).** Preference (DPO) training. The primary comparison is
  our CS preference pairs (C) against external English preference pairs
  (B_ext) at equal accepted-pair budgets: a comparison of **training
  recipes**, not of training language alone. It also covers generalization,
  over-refusal, utility and error analysis.

The experiment numbering (Exp 0–10) and the two-phase paper structure follow
`docs/PROTOCOL.md`.

## Current dataset
**`final-791`**: 791 harmful-request families × 4 conditions = 3,164
prompts, in six domains (D1–D6).

- **Source:** `data/CS-Jail-UR_final_approved_791.csv`, local only, SHA-256
  `4a34e81a…1ecf`.
- **Finalized:** `outputs/exp0/final-791-ddc14ecbc568/`.
- **Split:** frozen split `51c2dd1bb7d72166`, with **200 held-out evaluation
  families** (unchanged since the 692 release) and **591 training
  families**, all eligible for preference training.
- **More:** dataset card in [`data/README.md`](data/README.md); release
  registry in `configs/dataset.yaml`.

## Status
Details, hashes and next actions are in
[`docs/EXPERIMENT_STATUS.md`](docs/EXPERIMENT_STATUS.md).

| Exp | Status |
|---|---|
| 0 Dataset finalization | **Done** on `final-791`: all gates PASS (structure, split, exposure) |
| 1 Judge validation | **Development only.** 160-item development sample exists (AI-prefilled labels, 15 flagged items awaiting human adjudication). **Final validation pending**; no validated judge exists yet |
| 2 Main evaluation | **Not run.** Offline checks and a CPU live-generation smoke passed; the live-judge smoke reached the API but the OpenAI account has **no credits**; the production GPU backend is untested |
| 3–4b | Not run (need Exp 2) |
| 5 | Cut |
| 6–9 Phase 2 | Not run. Design fixed (C vs B_ext). The feasibility pilot generated 400 CPU responses; its AI-reviewed counts are team-reported, not in this repo (exploratory, not gold) |
| 10 | Not run |

## Using the code
- [`RUNBOOK.md`](RUNBOOK.md) has installation, credentials (`.env.example`),
  exact commands, resume behaviour and expected outputs. Commands are marked
  tested or untested.
- [`docs/PROTOCOL.md`](docs/PROTOCOL.md) is the single current protocol.
  Superseded plans are in `docs/archive/`.

```
csjail/     library: conversion, schema, QA, splits, judge contract, pipeline, stats, Phase 2
configs/    dataset registry, domains, eval, judge (+ validation protocol), models (pinned), dpo
scripts/    exp0 … exp9 entry points, smoke tests, dry-run report
tests/      CPU tests on harmless fixtures
outputs/    small ID-level manifests and summaries only (raw text is gitignored)
```

## Data handling
The dataset and every file containing prompt or response text are sensitive.
Keep them local or in private team storage, and never run them on shared
notebooks. Only ID-level manifests are committed.
