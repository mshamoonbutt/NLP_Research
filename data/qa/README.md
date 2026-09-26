# Dataset QA records (optional, evidence-only)

Nothing here is required to run the pipeline, and nothing may be filled with
invented values. Keep records keyed by family ID (and dataset version).

- `qa_ledger_template.csv` — per-family review record: harmful eligibility,
  four-way semantic equivalence (requested act, target, intent, constraints,
  framing, information supplied), language-condition validity (incl. the RU
  loanword policy), CS naturalness, duplicate decision, resolution, status
  (`open` / `resolved` / `wontfix`), notes. Import with
  `exp0_finalize_data.py --qa-ledger <file>`.
- Independent double annotation (if any) goes in a separate file with
  `rater1_<field>` / `rater2_<field>` columns, labelled BEFORE adjudication;
  pass it with `--independent-annotations`. Agreement is reported
  descriptively (raw + Cohen's kappa; NA when undefined). It is not a gate.
- `duplicate_decisions_template.csv` — reviewer decisions for Exp 0's
  token-Jaccard near-duplicate candidates: `duplicate`, `related` (both keep a
  shared group so they never straddle train/eval) or `distinct`. Save as
  `data/qa/duplicate_decisions.csv`; unresolved candidates are grouped
  conservatively.
- A family judged not harmful-eligible, or a substantive replacement that
  changes the requested act, is handled by issuing a new dataset version with
  lineage -- never by silently editing or deleting rows.
