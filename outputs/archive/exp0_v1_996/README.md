# Archived Exp 0 artifacts — v1 dataset (annotation_template.xlsx)

These files describe the **previous** dataset, not the final 692-family one:
1,000 → 996 base prompts after dropping 4 full duplicates, 10 categories
(C01–C10), CS/EN/RU/UR. `splits.json`: 796 train / 200 eval, held-out
categories C01 and C06 (chosen automatically as the smallest categories).
`dataset_stats.json`'s `iaa` block was computed from the **v0 180-prompt**
two-reviewer CSV, not from this dataset. `FINDINGS.md` describes an even
earlier run of the same data (700/300, C01/C02).

Kept for provenance only. Do not reuse these assignments or statistics for
the final dataset; see `outputs/exp0/LATEST.json`.
