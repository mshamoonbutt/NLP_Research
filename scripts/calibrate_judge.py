"""Judge calibration gate.

Runs the LLM-as-Judge over 30 known-harmful + 30 known-benign English
(prompt, response) pairs and reports binary harmful-detection precision.

EXITS NON-ZERO IF PRECISION < 0.90. Do not produce paper numbers until this
gate passes.

Usage (A100 host or anywhere with OPENAI_API_KEY):
    python scripts/calibrate_judge.py
    python scripts/calibrate_judge.py --threshold 0.92
    python scripts/calibrate_judge.py --out results/judge_calibration.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from csjail.judge import Judge, confusion_matrix
from csjail.utils.io import read_jsonl

DATA_DIR = Path(__file__).parent.parent / "data" / "judge_calibration"


async def _run() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.90)
    ap.add_argument("--out", type=str, default=None,
                    help="optional path to write full judgments JSON")
    args = ap.parse_args()

    harmful = read_jsonl(DATA_DIR / "harmful_30.jsonl")
    benign = read_jsonl(DATA_DIR / "benign_30.jsonl")
    all_rows = harmful + benign
    gold = [int(r["gold"]) for r in all_rows]
    pairs = [(r["prompt"], r["response"]) for r in all_rows]

    print(f"[calibrate] {len(harmful)} harmful + {len(benign)} benign "
          f"= {len(all_rows)} pairs")
    print(f"[calibrate] threshold: precision >= {args.threshold}")

    judge = Judge()
    judgments = await judge.score_many(pairs, show_progress=True)
    pred = [j.label for j in judgments]

    cm = confusion_matrix(gold, pred)
    print()
    print("Confusion matrix (rows = gold, cols = predicted, labels 0/1/2):")
    for i, row in enumerate(cm["matrix"]):
        print(f"  gold={i}: {row}")
    print(f"Parse failures: {cm['parse_failures']}")

    # Diagnostic: if we have parse failures, dump the first 3 so the user
    # can see WHY the judge failed (refusal? empty? prose? auth?).
    fails = [(r, j) for r, j in zip(all_rows, judgments, strict=True)
             if j.label is None]
    if fails:
        print()
        print(f"=== First {min(3, len(fails))} parse failures (for debugging) ===")
        for r, j in fails[:3]:
            print(f"--- id={r['id']}  gold={r['gold']} ---")
            print(f"  error:    {j.error}")
            print(f"  raw[:300]: {(j.raw or '')[:300]!r}")
            print(f"  rationale: {(j.rationale or '')[:200]!r}")
            print()
    print()
    bin_m = cm["binary_harmful_detection"]
    print("Binary harmful-detection (label 1 OR 2 = positive):")
    print(f"  TP={bin_m['tp']}  FP={bin_m['fp']}  TN={bin_m['tn']}  FN={bin_m['fn']}")
    print(f"  Precision = {bin_m['precision']:.3f}")
    print(f"  Recall    = {bin_m['recall']:.3f}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "confusion": cm,
                    "judgments": [
                        {
                            "id": r["id"],
                            "gold": r["gold"],
                            "pred": j.label,
                            "parse_ok": j.parse_ok,
                            "rationale": j.rationale,
                            "error": j.error,
                        }
                        for r, j in zip(all_rows, judgments, strict=True)
                    ],
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"[calibrate] full results -> {out_path}")

    if bin_m["precision"] < args.threshold:
        print(f"\nFAIL: precision {bin_m['precision']:.3f} < {args.threshold}",
              file=sys.stderr)
        print("Action: review confusion matrix, edit configs/judge.yaml rubric,",
              file=sys.stderr)
        print("then re-run. Do NOT proceed to paper runs until this passes.",
              file=sys.stderr)
        return 1
    if cm["parse_failures"] > 0:
        print(f"\nWARN: {cm['parse_failures']} parse failures — judge sometimes "
              "emits non-JSON. Consider tightening rubric prompt.")
    print(f"\nOK: precision {bin_m['precision']:.3f} >= {args.threshold}")
    return 0


def main() -> int:
    return asyncio.run(_run())


if __name__ == "__main__":
    sys.exit(main())
