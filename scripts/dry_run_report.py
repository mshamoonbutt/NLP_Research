#!/usr/bin/env python3
"""Pre-production dry-run report (CPU; no model, no API).

    python scripts/dry_run_report.py [--exp0-dir ...] [--results outputs/exp2/main]

Summarizes: finalized dataset + split counts, open review flags, judge
validation status, planned generation volumes per experiment, and Phase-2
pair budgets -- actual mined counts if Exp 2 results exist, otherwise an
explicitly ILLUSTRATIVE table (not a forecast). Writes
outputs/dry_run_report.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

from csjail.artifacts import resolve_exp0  # noqa: E402
from csjail.models import phase2_models  # noqa: E402
from csjail.splits import EVAL, TRAIN  # noqa: E402
from csjail.utils.io import write_text_lf  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def judge_status(path: Path) -> str:
    if not path.exists():
        return "NOT_RUN"
    return json.loads(path.read_text(encoding="utf-8")).get("status", "?")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp0-dir", default=None)
    ap.add_argument("--results", nargs="*", default=[])
    args = ap.parse_args(argv)
    art = resolve_exp0(args.exp0_dir)
    ecfg = yaml.safe_load((ROOT / "configs" / "eval.yaml").read_text(encoding="utf-8"))
    dcfg = yaml.safe_load((ROOT / "configs" / "dpo.yaml").read_text(encoding="utf-8"))
    counts = art.split["meta"]["counts"]
    n_fam = art.manifest["n_families"]
    models, conds = ecfg["models"], ecfg["conditions"]
    rc = ecfg["robustness"]
    flags = json.loads((art.dir / "qa_review_flags.json").read_text(encoding="utf-8"))
    qa = json.loads((art.dir / "qa_summary.json").read_text(encoding="utf-8"))
    p2 = phase2_models()
    n_train, n_eval = counts[TRAIN]["total"], counts[EVAL]["total"]

    rep = {
        "dataset_version": art.dataset_version, "split_id": art.split_id,
        "families": n_fam, "rows": art.manifest["n_rows"], "split_counts": counts,
        "exp0_gates": art.manifest["gates"],
        "open_review_flags": {
            "script": len(flags["script"]),
            "ru_possible_english_clause": len(flags["ru_possible_english_clause"]),
            "equal_variants_within_family": len(flags["equal_variants_within_family"]),
            "exact_duplicates_across_families": len(flags["exact_duplicates_across_families"]),
            "ru_items_with_technical_loans_info": flags["ru_loanword_inventory"][
                "n_ru_items_with_technical_loans"],
            "qa_ledger": qa["qa_ledger"] and qa["qa_ledger"]["status_counts"],
            "independent_agreement": "available" if qa["agreement"] else "none supplied",
        },
        "judge_validation": {
            "harm": judge_status(ROOT / "outputs/exp1/judge_validation_manifest.json"),
            "benign": judge_status(ROOT / "outputs/exp1/judge_validation_manifest_benign.json")},
        "planned_generations": {
            "exp1_gold": f"{2} models x {len(conds)} conditions x 60 families = {2 * len(conds) * 60} (minimum; top-up if support short)",
            "exp2_primary": f"{len(models)} x {len(conds)} x {n_fam} = {len(models) * len(conds) * n_fam}",
            "exp2_robustness": f"{rc['n_families']} families x {len(rc['conditions'])} conditions "
                               f"({'/'.join(rc['conditions'])}) x {rc['n']} draws x {len(models)} models = "
                               f"{rc['n_families'] * len(rc['conditions']) * rc['n'] * len(models)} "
                               f"(all four conditions would be "
                               f"{rc['n_families'] * 4 * rc['n'] * len(models)})",
            "exp4b_probes": f"100 families x {len(conds)} x {len(models)} = {100 * len(conds) * len(models)} "
                            "(harmful-attempt responses reused from Exp 2)",
            "exp8_per_arm_per_model": f"{n_eval} eval families x {len(conds)} = {n_eval * len(conds)} "
                                      "+ 150 benign probes + capability MCQs",
        },
        "phase2_models": p2,
        "pair_budget": None,
    }
    if args.results:
        from csjail.aggregate import load_results
        from csjail.prefdata import mine_rejected, supported_budgets

        recs = load_results(args.results)
        actual = {}
        for m in p2:
            mine = {}
            for lang in ("CS", "EN"):
                mined, _ = mine_rejected([r for r in recs if r["model"] == m], art.split,
                                         model=m, condition=lang)
                mine[lang] = len(mined)
            fam_cs = {x["base_id"] for x in mine_rejected([r for r in recs if r["model"] == m],
                                                          art.split, model=m, condition="CS")[0]}
            fam_en = {x["base_id"] for x in mine_rejected([r for r in recs if r["model"] == m],
                                                          art.split, model=m, condition="EN")[0]}
            actual[m] = {"raw_unsafe_candidates": mine,
                         "matched_upper_bound_before_chosen_validation": len(fam_cs & fam_en),
                         "budgets_if_all_validate": supported_budgets(len(fam_cs & fam_en),
                                                                     dcfg["prefdata"]["n_curve"])}
        rep["pair_budget"] = {"kind": "actual (before chosen validation)", "per_model": actual}
    else:
        rep["pair_budget"] = {
            "kind": "ILLUSTRATIVE ONLY - not a forecast; replace with actual counts after Exp 2",
            "candidate_training_families": n_train,
            "raw_cs_candidates_at_rate": {f"{int(r * 100)}%": round(n_train * r)
                                          for r in (0.02, 0.05, 0.10, 0.20, 0.30, 0.45)},
            "note": "matched B/C additionally needs an unsafe EN response for the same family, "
                    "so matched N <= min(CS, EN) candidates; the v0 study saw 0.6-2.8% EN and "
                    "1.8-7.7% CS unsafe rates on a different dataset/rubric",
            "target_pairs_cap": dcfg["prefdata"]["target_pairs"],
        }
    out = ROOT / "outputs" / "dry_run_report.json"
    write_text_lf(out, json.dumps(rep, ensure_ascii=False, indent=2))
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
