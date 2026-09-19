#!/usr/bin/env python3
"""Exp 1 — sample & generate the judge-validation gold set (GPU).

Step 1 of the Exp 1 hard gate (see judge precision check in
scripts/calibrate_judge.py --calibration-csv). Draws >=60 prompts PER
CONDITION (>=240 total), stratified by harm_category, generates responses
with >=2 SLMs (judge precision may vary by model — a model that produces
more degenerate/repetitive output is harder to label consistently, so
validating on one model and judging three is an untested extrapolation),
and exports (prompt, response) pairs to a CSV for two human annotators to
fill in `gold` (0=refusal, 1=full_comply, 2=partial).

This does NOT run the judge. It only produces the annotation CSV. Once
annotators return it, run:
    python scripts/calibrate_judge.py --calibration-csv outputs/exp1/annotation_gold.csv

Usage:
    python scripts/exp1_sample_for_annotation.py \
        --models qwen25 phi3 --n-per-condition 60
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from csjail.data import CONDITIONS, load_dataset  # noqa: E402
from csjail.models import SamplingConfig, SLMRunner, resolve  # noqa: E402
from csjail.splits import proportional_stratified_indices  # noqa: E402

CSV_FIELDS = ["id", "model", "condition", "harm_category", "base_id",
              "prompt", "response", "gold", "annotator"]


def _stratified_sample(rows, condition: str, n: int, seed: int):
    """Proportional-by-category sample of `n` rows in `condition`."""
    cond_rows = [r for r in rows if r.condition == condition]
    idx = proportional_stratified_indices(
        [r.harm_category for r in cond_rows], n, seed=seed)
    return [cond_rows[i] for i in idx]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="outputs/exp0/dataset_with_features.jsonl")
    ap.add_argument("--models", nargs="+", default=["qwen25", "phi3"],
                    help="draw the gold set from >=2 architecturally "
                         "different models, not one")
    ap.add_argument("--n-per-condition", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="outputs/exp1/annotation_gold_candidates.csv")
    args = ap.parse_args(argv)

    if len(args.models) < 2:
        print("FAIL: Exp 1 requires >=2 models for the gold set "
              "(judge precision may vary by model)", file=sys.stderr)
        return 1

    rows = load_dataset(args.dataset)
    out_rows = []
    for model in args.models:
        spec = resolve(model)
        runner = SLMRunner(spec)
        try:
            for cond in CONDITIONS:
                sample = _stratified_sample(rows, cond, args.n_per_condition,
                                            args.seed)
                outs = runner.generate(
                    [r.prompt for r in sample],
                    SamplingConfig(temperature=0.0, max_tokens=512),
                )
                for r, out in zip(sample, outs, strict=True):
                    out_rows.append({
                        "id": f"{model}-{r.id}",
                        "model": model,
                        "condition": cond,
                        "harm_category": r.harm_category,
                        "base_id": r.base_id,
                        "prompt": r.prompt,
                        "response": out,
                        "gold": "",
                        "annotator": "",
                    })
                print(f"[exp1] {model} / {cond}: {len(sample)} sampled "
                      f"(target {args.n_per_condition})")
        finally:
            runner.shutdown()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(out_rows)

    print(f"[exp1] wrote {len(out_rows)} rows -> {out_path}")
    print(f"[exp1] {len(args.models)} models x {len(CONDITIONS)} conditions x "
          f"~{args.n_per_condition} = target {len(args.models) * len(CONDITIONS) * args.n_per_condition}")
    print("[exp1] NEXT: hand this CSV to two human annotators to fill `gold` "
          "(0=refusal, 1=full_comply, 2=partial), then run "
          "scripts/calibrate_judge.py --calibration-csv <returned CSV>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
