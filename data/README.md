# `data/`

| File | Purpose | Tracked? |
|---|---|---|
| `CS-Jail-UR_approved_748.csv` | Current dataset (748 families), seven columns: `prompt_id, domain_id, domain_name, EN, CS, RU, UR` | **no**, local only |
| `CS-Jail-UR_final_692.csv` | Previous version: the same 692 families without the 56 X1 additions | **no**, local only |
| `csjail_fixture.jsonl` | Harmless 12-family fixture in the current long schema (built from `tests/fixtures/final_fixture.csv`) | yes |
| `overrefusal_probe.jsonl` | 150 benign boundary prompts for Exp 8. Language coverage is unaudited: there are no condition tags, and the items are mostly Roman-Urdu/English | yes |
| `refusal_exemplars.jsonl` | 8 hand-written CS refusal exemplars, D-domain tagged. The sexual-content one is outside the final taxonomy and is skipped | yes |
| `refusal_exemplars_en.jsonl` | 7 English exemplars. These are model-assisted renderings and need bilingual review | yes |
| `judge_calibration/*.jsonl` | Legacy English 30+30 set, used only by `calibrate_judge.py --smoke` | yes |
| `qa/` | Templates for the optional QA ledger and duplicate decisions | yes |

## Long-format schema

The authoritative definition is `csjail/data.py::Prompt`. Unknown fields are
rejected, not dropped.

```jsonc
{
  "id": "CSJUR-V3-0001::CS",        // <family id>::<condition>
  "base_id": "CSJUR-V3-0001",       // opaque family id, never int-cast
  "condition": "CS",                 // CS | EN | RU | UR
  "prompt": "...",                   // verbatim source text (outer whitespace/CRLF normalization recorded)
  "domain_id": "D6", "domain_name": "Cyber abuse and unauthorized access",
  "harm_category": "D6",             // compatibility alias == domain_id
  "dataset_version": "final-748-ac1a1b70b4a6",
  "group_id": "CSJUR-V3-0001",       // duplicate/relative group, set in Exp 0
  "provenance": {"source_file": "...", "source_row": 1, "id_namespace": "CSJUR-V3",
                 "author": null, "model_assistance": null, ...},
  "qa": null,
  "urdu_word_ratio": null, "cmi": null,                           // validated only
  "urdu_word_ratio_heuristic": 0.31, "cmi_heuristic": 28.0,       // unvalidated diagnostics
  "feature_method": "...", "feature_validation_status": "unvalidated-heuristic"
}
```

Exactly one row is required per family and condition. Legacy v0/v1 files, with
H/C categories or the SM condition, load only with
`load_dataset(..., allow_legacy=True)`, for historical reproduction.
