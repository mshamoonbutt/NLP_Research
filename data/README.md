# CS-Jail-UR dataset card (`data/`)

## Current release: `final-791`

| | |
|---|---|
| File | `data/CS-Jail-UR_final_approved_791.csv` (**local only**, gitignored) |
| SHA-256 | `4a34e81a38932fef769e563bbfe735606eadb4c473d24898ae249bd8480a1ecf` |
| Finalized as | `outputs/exp0/final-791-ddc14ecbc568/` (split `51c2dd1bb7d72166`) |
| Families / rows | 791 families × 4 conditions = 3,164 prompts |
| Registry | `configs/dataset.yaml` (Exp 0 checks the checksum before use) |

Lineage (no existing family's text, domain or ID changed at any step):

| Release | Families | SHA-256 (prefix) | Status | Change |
|---|---:|---|---|---|
| `final-692` | 692 | `cc98267a…` | superseded | first seven-column release; defined the 200-family eval set |
| `approved-748` | 748 | `d6ca1734…` | superseded | + 56 `CSJUR-X1` |
| `final-791` | 791 | `4a34e81a…` | **active** | + 3 `CSJUR-X1`, + 40 `CSJUR-X2`; three metadata columns added |

## Schema (ten columns)

| Column | Meaning |
|---|---|
| `row_number` | Display/order metadata only. It is **never** an identifier or split key. |
| `prompt_id` | Opaque string family ID, never cast to a number. Namespaces: 490 numeric, 18 `CSJUR-R`, 100 `CSJUR-V3`, 84 `CSJUR-V4`, 59 `CSJUR-X1`, 40 `CSJUR-X2`. Namespaces show batches, not authorship. |
| `domain_id`, `domain_name` | One of D1–D6, validated against `configs/domains.yaml`. |
| `EN`, `CS`, `RU`, `UR` | The same request in English; code-switched Urdu-English; Roman Urdu (Urdu in Latin script); and Urdu in Perso-Arabic script. |
| `evaluation_stratum` | `harmful` for all rows. It describes the prompt population (harmful requests), **not** an observed harmful model response. |
| `approval_status` | `approved` for all rows. This is supplied review metadata. It is **not** evidence of independent inter-annotator agreement, and it says nothing about judge accuracy. |

The seven content columns are required. The three metadata columns are
preserved verbatim per family (`provenance.source_metadata` in the finalized
JSONL). Any stratum or approval value other than the ones above fails
conversion.

| Domain | Name | Families |
|---|---|---:|
| D1 | Hate, harassment and coercion | 122 |
| D2 | Self-harm and suicide | 123 |
| D3 | Dangerous acts, violence and criminal facilitation | 137 |
| D4 | Deception, fraud and intellectual-property abuse | 128 |
| D5 | Privacy and personal-information abuse | 137 |
| D6 | Cyber abuse and unauthorized access | 144 |

## Provenance and review
- Per-batch authorship and any model assistance are **not recorded** in the
  release; the finalized records keep `author` and `model_assistance` as
  `null`. Document them from real records before any provenance claim in the
  paper, and do not claim "all written from scratch by native speakers"
  without records.
- `approval_status=approved` is the team's supplied approval. No
  independent-agreement statistic accompanies this release; none is invented
  and none is a gate.
- Exp 0 (`FINDINGS.md`) reports only structural facts:
  - no blank cells
  - no equal variants within a family
  - no exact duplicates across families
  - 0 near-duplicate candidates at token-Jaccard ≥ 0.92
  - 69 UR items containing Latin letters (review flag)
  - 2 RU items with possible English clauses (review flag)

  These checks do not certify semantic equivalence across variants or item
  harmfulness.

## Splits, groups and exposure
- **Evaluation:** 200 held-out families, stratified across all six domains
  (D1–D6 = 35/31/33/34/32/35). Membership has been **frozen since 692**; each
  later release was an append-only extension.
- **Training pool:** 591 families (87/92/104/94/105/109).
- **Grouping:** all four language variants, and any duplicate/relative group,
  always share a split.
- **New-family screen:** new families that loosely resemble an evaluation
  family are quarantined until a reviewer clears them
  (`data/qa/duplicate_decisions.csv`).
- **Exposure:** the Exp 1 judge-development sample (20 families) and the
  Phase 2 feasibility pilot (100 families; 11 overlap) are all training-pool
  families.
  - Exp 0 fails if any exposed family touches evaluation.
  - Exposed families remain eligible for training.
  - Reserved exclusions: none.
  - Eligible training families: **591**.
- **Unseen-domain ablation:** kept separate from the main split (Exp 9,
  explicit and retrained); it never silently removes domains from the main
  pool.

## Future training-only extensions
Extra training prompts go into a **separately versioned extension**
(`scripts/exp0_finalize_data.py --training-extension-csv X --extension-name
NAME`).

- The core release, its evaluation membership and its wording stay frozen.
- Extension families are train-only; any that resemble an eval family are
  quarantined.
- The extension release is named `<core>+ext-NAME-<hash>` and does not move
  `LATEST`.
- Example: 791 core + 250 extension families = 1,041 in total, of which up to
  841 are non-evaluation. The held-out test set stays at 200.

## Licence
The repository declares `CC-BY-NC-4.0` (`pyproject.toml`). The licence and
access policy for a **public dataset release** is not finalized; an earlier
plan proposed CC BY-NC 4.0 with gated access for the most explicit items.
Treat it as unknown until the team decides.

## Handling
The release contains harmful requests. Keep the CSV, the finalized JSONL and
all generated responses local or in a private location. Only ID-level
manifests are committed.

## Other files here
| File | Purpose | Tracked? |
|---|---|---|
| `CS-Jail-UR_approved_748.csv`, `CS-Jail-UR_final_692.csv` | superseded releases (lineage only) | no |
| `csjail_fixture.jsonl` | harmless 12-family fixture in the current long schema | yes |
| `overrefusal_probe.jsonl` | 150 benign boundary prompts (Exp 8); language coverage unaudited | yes |
| `refusal_exemplars.jsonl`, `refusal_exemplars_en.jsonl` | few-shot refusal exemplars (English ones model-assisted, pending review) | yes |
| `judge_calibration/*.jsonl` | legacy English 30+30 smoke set only | yes |
| `qa/` | templates: QA ledger, duplicate decisions | yes |
| `capability/mmlu_500.jsonl` (+ `.manifest.json`) | Exp 8 capability: 500 items of MMLU (`cais/mmlu`, config all, test split, seed 42; MIT licence), frozen | yes |
| `capability/urdummlu_300.jsonl` (+ `.manifest.json`) | Exp 8 capability: 300 items of UrduMMLU (`MBZUAI/UrduMMLU` @ `aae060a8`, file `urdummlu.json`, sha256 `b2187d0c…`, seed 42; CC BY 4.0 — cite the authors), frozen | yes |
| `pref_pairs_en_external.jsonl` | B_ext source, built by `scripts/prepare_external_english_pairs.py` | no |

## Finalized long-format record (`dataset_final.jsonl`)
```jsonc
{
  "id": "CSJUR-X2-0001::CS",        // <family id>::<condition>
  "base_id": "CSJUR-X2-0001",
  "condition": "CS",                 // CS | EN | RU | UR
  "prompt": "...",                   // verbatim (outer-whitespace trims recorded)
  "domain_id": "D5", "domain_name": "...", "harm_category": "D5",
  "dataset_version": "final-791-ddc14ecbc568",
  "group_id": "CSJUR-X2-0001",
  "provenance": {"source_file": "CS-Jail-UR_final_approved_791.csv", "source_row": 1,
                 "id_namespace": "CSJUR-X2", "author": null, "model_assistance": null,
                 "source_metadata": {"row_number": "...", "evaluation_stratum": "harmful",
                                     "approval_status": "approved"}},
  "cmi": null, "urdu_word_ratio": null,                      // validated fields: none yet
  "cmi_heuristic": 28.0, "urdu_word_ratio_heuristic": 0.31,  // unvalidated diagnostics
  "feature_validation_status": "unvalidated-heuristic"
}
```
