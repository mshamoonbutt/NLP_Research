#!/usr/bin/env python3
"""Exp 6 — build model-specific preference pairs (CPU + chosen-generator API).

    python scripts/exp6_build_prefdata.py --model phi3 --results outputs/exp2/main
    python scripts/exp6_build_prefdata.py --model phi3 --results outputs/exp2/main \
        --split-manifest outputs/exp9/ablation_D2/split_manifest.json --tag ablation_D2

For the target model, per language (CS, and EN for the matched control):
 1. mine `rejected` from that model's greedy Exp 2 results on eligible
    train_pool families (primary unsafe, lineage kept) -- other models'
    results are filtered out explicitly and reported;
 2. generate `chosen` with a model distinct from the judge;
 3. validate chosen (non-empty, judged not unsafe, genuine refusal);
 4. dedup, seeded domain-aware ordering (nested N-curve prefixes);
 5. matched sets: CS and EN restricted to families valid in BOTH.
Writes, under outputs/exp6/<model>[_<tag>]/:
  pairs_cs_all.jsonl, pairs_en_all.jsonl       all valid, ordered (text; gitignored)
  pairs_cs_matched.jsonl, pairs_en_matched.jsonl  matched B/C sets (gitignored)
  naturalness_sample.csv                       for native-speaker rating (gitignored)
  pairs_manifest.json                          counts per stage/domain, budgets, lineage ids
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

from csjail.aggregate import load_results  # noqa: E402
from csjail.artifacts import resolve_exp0  # noqa: E402
from csjail.chosen_gen import (  # noqa: E402
    ChosenGenerator, assert_distinct_from_judge, load_exemplars, load_generator_config,
)
from csjail.judge import Judge, load_judge_config  # noqa: E402
from csjail.judge_validation import require_validated_judge  # noqa: E402
from csjail.prefdata import (  # noqa: E402
    dedup, length_report, matched_sets, mine_rejected, order_pairs, pair_counts,
    supported_budgets, validate_chosen, write_pairs,
)

ROOT = Path(__file__).resolve().parent.parent


def build_language(results, split, rows_by_id, *, model, lang, excl, gen_cfg, judge, seed, tol):
    mined, mine_rep = mine_rejected(results, split, model=model, condition=lang,
                                    exclude_domains=excl)
    exemplars = load_exemplars(lang, exclude_domains=excl)
    prompts = [rows_by_id[(m["base_id"], lang)] for m in mined]
    cands = ChosenGenerator(gen_cfg, language=lang).generate_sync(prompts, exemplars) if mined else []
    js = judge.score_sync([(p, c or "") for p, c in zip(prompts, cands)]) if mined else []
    valid, reasons = [], {}
    for m, p, c, j in zip(mined, prompts, cands, js):
        ok, why = validate_chosen(c, j)
        reasons[why] = reasons.get(why, 0) + 1
        if ok:
            valid.append({**m, "prompt": p, "chosen": c, "language": lang,
                          "chosen_generator": gen_cfg.fingerprint(lang, [e["id"] for e in exemplars]),
                          "chosen_judge_fingerprint_id": judge.fingerprint["fingerprint_id"]})
    valid, n_dup = dedup(valid)
    ordered = order_pairs(valid, seed=seed)
    return ordered, {"mining": mine_rep, "chosen_validation": reasons, "n_duplicates_removed": n_dup,
                     "quality_approved": pair_counts(ordered),
                     "length": length_report(ordered, tol)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--results", nargs="+", required=True, help="Exp 2 run dir(s)")
    ap.add_argument("--exp0-dir", default=None)
    ap.add_argument("--split-manifest", default=None, help="e.g. an Exp 9 ablation manifest")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--judge-manifest",
                    default=str(ROOT / "outputs" / "exp1" / "judge_validation_manifest.json"))
    ap.add_argument("--languages", nargs="+", default=["CS", "EN"])
    ap.add_argument("--out-dir", default=None, help="default outputs/exp6/<model>[_<tag>]")
    ap.add_argument("--allow-debug", action="store_true")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load((ROOT / "configs" / "dpo.yaml").read_text(encoding="utf-8"))["prefdata"]
    gen_cfg, judge_cfg = load_generator_config(), load_judge_config()
    assert_distinct_from_judge(gen_cfg, judge_cfg)
    require_validated_judge(args.judge_manifest, judge_cfg.fingerprint("harm"))
    art = resolve_exp0(args.exp0_dir, split_path=args.split_manifest)
    rows = art.load_rows()
    rows_by_id = {(r.base_id, r.condition): r.prompt for r in rows}
    excl = [art.split["meta"]["ablation_domain"]] if art.split["meta"].get("ablation_domain") else []
    all_results = load_results(args.results, allow_debug=args.allow_debug)
    other = sorted({r["model"] for r in all_results} - {args.model})
    results = [r for r in all_results if r["model"] == args.model]
    if not results:
        print(f"FAIL: no results for model {args.model}", file=sys.stderr)
        return 1
    judge = Judge(judge_cfg, kind="harm")

    out_dir = Path(args.out_dir or ROOT / "outputs" / "exp6" /
                   (args.model + (f"_{args.tag}" if args.tag else "")))
    out_dir.mkdir(parents=True, exist_ok=True)
    per_lang, reports = {}, {}
    for lang in args.languages:
        per_lang[lang], reports[lang] = build_language(
            results, art.split, rows_by_id, model=args.model, lang=lang, excl=excl,
            gen_cfg=gen_cfg, judge=judge, seed=int(cfg["selection_seed"]),
            tol=float(cfg["length_balance_tolerance"]))
        write_pairs(str(out_dir / f"pairs_{lang.lower()}_all.jsonl"), per_lang[lang])
        print(f"[exp6] {args.model}/{lang}: {reports[lang]['mining']['counts']} -> "
              f"{reports[lang]['quality_approved']}")

    matched = None
    if "CS" in per_lang and "EN" in per_lang:
        cs_m, en_m, rep = matched_sets(per_lang["CS"], per_lang["EN"],
                                       seed=int(cfg["selection_seed"]))
        write_pairs(str(out_dir / "pairs_cs_matched.jsonl"), cs_m)
        write_pairs(str(out_dir / "pairs_en_matched.jsonl"), en_m)
        matched = {**rep, "budgets": supported_budgets(len(cs_m), cfg["n_curve"])}
        print(f"[exp6] matched B/C: {rep['n_matched']} families; budgets {matched['budgets']}")

    cs_pairs = per_lang.get("CS", [])
    sample = random.Random(0).sample(cs_pairs, min(int(cfg["naturalness_sample_size"]), len(cs_pairs)))
    with (out_dir / "naturalness_sample.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["base_id", "prompt", "chosen", "naturalness_1to5",
                                          "clean_refusal_yes_no", "rater", "notes"])
        w.writeheader()
        w.writerows({"base_id": p["base_id"], "prompt": p["prompt"], "chosen": p["chosen"],
                     "naturalness_1to5": "", "clean_refusal_yes_no": "", "rater": "",
                     "notes": ""} for p in sample)

    manifest = {
        "kind": "pairs_manifest", "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model, "tag": args.tag, "dataset_version": art.dataset_version,
        "split_id": art.split_id, "split_scheme": art.split["meta"]["scheme"],
        "excluded_domains": excl, "results_models_ignored": other,
        "judge_fingerprint": judge.fingerprint, "chosen_generator": gen_cfg.__dict__,
        "target_pairs_cap": cfg["target_pairs"], "per_language": reports,
        "budgets": {lang: supported_budgets(len(p), cfg["n_curve"]) for lang, p in per_lang.items()},
        "matched": matched,
        "families": {lang: [p["base_id"] for p in ps] for lang, ps in per_lang.items()},
        "naturalness_gate": "PENDING: native speaker rates naturalness_sample.csv; mean >= 4 "
                            "and clean_refusal=yes required before training C/D",
    }
    (out_dir / "pairs_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                                 encoding="utf-8")
    print(f"[exp6] wrote {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
