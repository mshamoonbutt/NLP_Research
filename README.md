# CS-Jail-UR

A safety benchmark and preference-alignment study for small language models
using code-switched Urdu-English. There are 692 harmful-request families in
six domains (D1–D6), each written in four aligned variants: English (EN),
code-switched (CS), Roman Urdu (RU), and Urdu in Perso-Arabic script (UR).

- **Phase 1** measures unsafe-response rates and paired contrasts across the
  four conditions. It also checks comprehension.
- **Phase 2** runs preference-alignment mitigation. Code-switched pairs are
  compared against a matched English control, with generalization, utility
  and error analysis.

**Protocol:** [`docs/PROTOCOL.md`](docs/PROTOCOL.md) is the single current
protocol. **Commands:** [`RUNBOOK.md`](RUNBOOK.md). **History:**
[`docs/archive/`](docs/archive/) holds the v0 report, earlier plans and the
Sept-7 guide (superseded in part).

## Layout

```
csjail/                    library
  convert_final.py         final seven-column CSV -> long JSONL (active)
  data.py                  schema + exact family/condition contract
  qa.py                    script / loanword / duplicate audits (IDs only)
  splits.py                groups, frozen split manifests, ablation manifest
  artifacts.py             resolves + hash-checks finalized Exp 0 artifacts
  judge.py, outcomes.py    versioned judge contract + the ONE unsafe predicate
  judge_validation.py      Exp 1 gold/adjudication/metrics/manifest guard
  models.py, pipeline.py   vLLM runner + persist-first resumable gen/judge caches
  run_eval.py              Exp 2 engine (also used by Exp 8)
  asr.py, metrics.py       null-safe ASR, bootstrap, McNemar, GEE, Holm
  aggregate.py, robustness.py
  comprehension.py         Exp 4b
  prefdata.py, chosen_gen.py, train_dpo.py   Exp 6-7
  convert_v1.py, convert_csv.py   LEGACY converters (historical reproduction only)
configs/                   domains, eval, judge (+validation protocol), models (pinned), dpo, capability
scripts/                   exp0 ... exp9 entry points, smoke tests, dry_run_report.py
tests/                     CPU tests (harmless fixtures)
data/                      fixtures, exemplars, benign probe, QA templates (dataset itself is local-only)
outputs/exp0/<version>/    frozen ID-level manifests (dataset JSONL is gitignored)
```

## Quick start

```bash
pip install -e ".[dev]"                      # CPU: Exp 0, statistics, tests
pytest tests -q
python scripts/exp0_finalize_data.py --source-csv data/CS-Jail-UR_final_692.csv
python scripts/dry_run_report.py
```

The GPU host additionally needs `pip install -e ".[train,judge]"`, plus
`OPENAI_API_KEY`, `ANTHROPIC_API_KEY` and `HF_TOKEN`. See `RUNBOOK.md`.

## Data handling

The harmful dataset, and any file holding prompt or response text, stays
local. Those files are gitignored and must not be run on shared notebooks.
Only ID-level manifests (splits, QA flags, hashes) are committed.
