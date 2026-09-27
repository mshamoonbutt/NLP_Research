# Phase 2 feasibility pilot — status: GENERATED, awaiting human review

This pilot estimates how many training pairs each Phase 2 model yields. It is
not training data and not judge-validation gold.

- **Sample.** 100 `train_pool` families, stratified by domain, from dataset
  `final-748-ac1a1b70b4a6`. Each gets CS and EN prompts on phi3 and llama32,
  for 400 responses in total.
- **Backend.** Ollama Q8_0 on CPU (see `sample_manifest.json`). Counts are
  planning estimates and may differ somewhat from the production bf16 models.

**Review (one reviewer is enough; a second is optional).** Fill `unsafe`
(`true`/`false`) in `rater1.csv` for every row, following `RATER_GUIDE.md` in
`outputs/exp1/development-cpu-20260926/`. `response_kind` and `notes` are
optional here. If a second reviewer fills `rater2.csv`, any disagreement is
counted as unresolved. Then run:

```bash
python scripts/phase2_feasibility.py --sample-dir outputs/exp6/feasibility-cpu-20260927
```

For each model it reports:

- **Arm C:** candidate pairs, i.e. families whose CS response is unsafe.
- **Matched B/C/D:** families whose CS **and** EN responses are both unsafe.

Each count comes with a Wilson CI and a projection over the 548 training
families. Counts are **before** chosen-refusal validation and deduplication,
so some pairs will be lost.

**Planning target (a budget, not a published minimum).** Roughly 100–200
distinct-family pairs per model. Decide whether to expand the training pool
based on the matched count.
