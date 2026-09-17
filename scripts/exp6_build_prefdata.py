#!/usr/bin/env python3
"""Exp 6 — build DPO preference pairs (mostly CPU; `chosen` generation calls
the Anthropic API).

Pipeline:
  1. Mine `rejected` — harmful (judge label==1) CS completions from the
     Exp 2 judgment files, restricted to the training pool (excludes eval
     holdout + held-out categories, so RQ4 stays uncontaminated).
  2. Generate `chosen` — a natural code-switched refusal for each mined
     prompt, via a model DISTINCT from the judge (Anthropic by default;
     see csjail/chosen_gen.py — this is what keeps the pipeline
     non-circular).
  3. Clean-refusal filter — re-judge each (prompt, chosen) pair with the
     SAME judge used everywhere else; drop any candidate the judge would
     call PARTIAL/hedged (label 2) or that failed to refuse at all
     (label 1). This is the "clean refusal" criterion (guide Exp 6 rec #8):
     hedged teacher refusals are exactly what arXiv 2602.11157 found
     increases downstream jailbreak success.
  4. Assemble — dedup, length-balance, cap at `target_pairs` (250, not 800
     — see configs/dpo.yaml prefdata comment for the arithmetic).
  5. Export a 50-sample naturalness-rating CSV for a native speaker
     (1-5 scale, gated at mean >= 4 — see --export-naturalness-sample).

Usage:
    python scripts/exp6_build_prefdata.py \
        --judgments "results/baseline_*/*_CS.jsonl" \
        --model phi3 \
        --splits outputs/exp0/splits.json \
        --out outputs/pref_pairs_cs_phi3.jsonl
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

from csjail.chosen_gen import (  # noqa: E402
    ChosenGenerator,
    assert_distinct_from_judge,
    is_clean_refusal,
    load_exemplars,
    load_generator_config,
)
from csjail.judge import Judge, load_judge_config  # noqa: E402
from csjail.prefdata import assemble_pairs, mine_rejected, write_pairs  # noqa: E402
from csjail.utils.io import read_jsonl  # noqa: E402

DPO_CFG = Path(__file__).resolve().parent.parent / "configs" / "dpo.yaml"


def _load_judgment_rows(pattern: str) -> list[dict]:
    """Read one or more run_eval.py JSONL outputs, keep only prompt_eval rows,
    and rename judge_label -> label to match csjail.prefdata's schema."""
    rows: list[dict] = []
    for path in sorted(glob.glob(pattern)):
        for r in read_jsonl(path):
            if r.get("kind") != "prompt_eval":
                continue
            rows.append({**r, "label": r.get("judge_label")})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judgments", required=True,
                    help="glob pattern over run_eval.py JSONL outputs for "
                         "condition=CS, e.g. 'results/baseline_*/phi3_CS.jsonl'")
    ap.add_argument("--splits", default="outputs/exp0/splits.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--exemplars", default="data/refusal_exemplars.jsonl")
    ap.add_argument("--naturalness-sample-out", default=None,
                    help="if set, also write an N-row CSV for a native "
                         "speaker to rate naturalness 1-5")
    ap.add_argument("--naturalness-sample-size", type=int, default=50)
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(DPO_CFG.read_text(encoding="utf-8"))
    target_pairs = cfg["prefdata"]["target_pairs"]
    length_tol = cfg["prefdata"]["length_balance_tolerance"]

    gen_cfg = load_generator_config()
    judge_cfg = load_judge_config()
    assert_distinct_from_judge(gen_cfg, judge_cfg)

    assignments = json.loads(Path(args.splits).read_text(encoding="utf-8"))["assignments"]
    judgment_rows = _load_judgment_rows(args.judgments)
    mined = mine_rejected(judgment_rows, assignments, condition="CS")
    print(f"[exp6] mined {len(mined)} harmful CS completions from the training pool")
    if not mined:
        print("FAIL: nothing mined — check --judgments glob and splits.json",
              file=sys.stderr)
        return 1

    exemplars = load_exemplars(args.exemplars)
    generator = ChosenGenerator(gen_cfg)
    candidates = generator.generate_sync([m["prompt"] for m in mined], exemplars)

    judge = Judge(judge_cfg)
    judgments = judge.score_sync(
        list(zip([m["prompt"] for m in mined], candidates, strict=True)))

    chosen_by_base_id: dict[str, str] = {}
    n_rejected_by_filter = 0
    for m, cand, j in zip(mined, candidates, judgments, strict=True):
        if is_clean_refusal(j.label):
            chosen_by_base_id[m["base_id"]] = cand
        else:
            n_rejected_by_filter += 1
    print(f"[exp6] clean-refusal filter: kept {len(chosen_by_base_id)}, "
          f"dropped {n_rejected_by_filter} (hedged/partial/non-refusal)")

    pairs, report = assemble_pairs(
        mined, chosen_by_base_id,
        target_pairs=target_pairs, length_tolerance=length_tol,
    )
    print(f"[exp6] {report}")
    write_pairs(args.out, pairs)
    print(f"[exp6] wrote {len(pairs)} pairs -> {args.out}")

    if args.naturalness_sample_out:
        import random

        sample = random.sample(pairs, min(args.naturalness_sample_size, len(pairs)))
        out_path = Path(args.naturalness_sample_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["base_id", "prompt", "chosen",
                                              "naturalness_1to5", "rater"])
            w.writeheader()
            for p in sample:
                w.writerow({"base_id": p["base_id"], "prompt": p["prompt"],
                           "chosen": p["chosen"], "naturalness_1to5": "",
                           "rater": ""})
        print(f"[exp6] wrote {len(sample)}-row naturalness sample -> {out_path}")
        print("[exp6] NEXT: hand this to a native speaker, gate at mean >= 4 "
              "before training Arm C/D")
    return 0


if __name__ == "__main__":
    sys.exit(main())
