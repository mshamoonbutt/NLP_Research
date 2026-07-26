#!/usr/bin/env python3
"""Exp 0 — finalize & split the extended dataset (CPU only).

Validates schema + condition parity, attaches CMI / urdu_word_ratio, flags near
duplicates, computes Table-1 stats, and writes the fixed train/eval split.

Usage:
    python scripts/exp0_finalize_data.py --dataset data/csjail_v1.jsonl \
        --out-dir outputs/exp0
    python scripts/exp0_finalize_data.py --smoke-test   # uses data/csjail_smoke4.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from csjail.data import load_dataset, summarize  # noqa: E402
from csjail.splits import (  # noqa: E402
    attach_features,
    condition_parity,
    dataset_stats,
    find_near_duplicates,
    make_splits,
)
from csjail.utils.io import write_jsonl  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/csjail_v1.jsonl")
    ap.add_argument("--out-dir", default="outputs/exp0")
    ap.add_argument("--eval-holdout", type=int, default=300)
    ap.add_argument("--held-out-categories", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--near-dup-threshold", type=float, default=0.92)
    ap.add_argument("--parity-gate", type=float, default=0.95)
    ap.add_argument("--smoke-test", action="store_true",
                    help="use data/csjail_smoke4.jsonl and tiny holdouts")
    args = ap.parse_args(argv)

    if args.smoke_test:
        args.dataset = "data/csjail_smoke4.jsonl"
        args.eval_holdout = 2
        args.held_out_categories = 1

    ds_path = Path(args.dataset)
    if not ds_path.exists():
        print(f"FAIL: dataset not found: {ds_path}", file=sys.stderr)
        print("      (drop the extended long-format JSONL here — see the plan's "
              "input schema contract)", file=sys.stderr)
        return 1

    rows = load_dataset(ds_path)
    rows = attach_features(rows)

    parity = condition_parity(rows)
    stats = dataset_stats(rows)
    summ = summarize(rows)
    dups = find_near_duplicates(rows, threshold=args.near_dup_threshold)
    split = make_splits(
        rows,
        eval_holdout=args.eval_holdout,
        n_held_out_categories=args.held_out_categories,
        seed=args.seed,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Persist feature-attached dataset + split + stats.
    write_jsonl(out_dir / "dataset_with_features.jsonl",
                [r.model_dump() for r in rows])
    (out_dir / "splits.json").write_text(
        json.dumps(split, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "dataset_stats.json").write_text(
        json.dumps({"summary": summ, "stats": stats,
                    "parity": parity.__dict__,
                    "n_near_duplicates": len(dups)},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    print("[exp0] dataset:", ds_path)
    print("[exp0] rows:", summ["n_total"], "base_ids:", summ["n_base_ids"],
          "categories:", summ["n_harm_categories"])
    print("[exp0] by_condition:", summ["by_condition"])
    print(f"[exp0] condition parity: {parity.parity_rate:.3f} "
          f"({parity.n_complete}/{parity.n_base_ids} complete)")
    print(f"[exp0] near-duplicates flagged: {len(dups)}")
    print(f"[exp0] eval holdout: {split['meta']['eval_holdout_actual']}  "
          f"held-out categories: {split['meta']['held_out_categories']}")
    print(f"[exp0] cmi mean: {stats['cmi'].get('mean')}  "
          f"urdu_word_ratio mean: {stats['urdu_word_ratio'].get('mean')}")

    # GATE: condition parity.
    if parity.parity_rate < args.parity_gate:
        print(f"[exp0] GATE FAIL: parity {parity.parity_rate:.3f} < "
              f"{args.parity_gate}", file=sys.stderr)
        return 2
    print(f"[exp0] wrote outputs -> {out_dir}")
    print("[exp0] GATE PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
