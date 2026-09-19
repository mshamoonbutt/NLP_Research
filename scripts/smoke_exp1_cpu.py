#!/usr/bin/env python3
"""CPU smoke of the Exp 1 pipeline — NO model, NO API.

Proves the plumbing on the REAL dataset: stratified per-condition sampling
from >=2 (stub) models -> annotation CSV -> per-condition judge precision
AND recall. Every GPU/API call (SLMRunner.generate, the OpenAI judge) is
replaced by a deterministic stub, so this runs anywhere and needs no keys.
Mirrors scripts/smoke_pipeline_cpu.py's approach for Exp 2/3/6.

Exercises the real code paths Exp 1 depends on:
  - csjail.splits.proportional_stratified_indices (via _stratified_sample,
    imported straight from scripts/exp1_sample_for_annotation.py)
  - csjail.judge.precision_by_condition (the actual Exp 1 GATE metric that
    scripts/calibrate_judge.py --calibration-csv wraps)

The stub judge has THREE deliberately different behaviours across
conditions, each proving a different thing about the gate:
  - EN: the stub SLM never complies, so EN has zero actual-harmful gold
    examples. Precision/recall are UNDEFINED there, not 0.0 -- proves
    confusion_matrix's None-on-empty-denominator fix, and that the gate
    does not spuriously fail a condition the judge did nothing wrong on.
  - modelB/RU: the stub judge hallucinates "harmful" on some genuine
    refusals (a false positive) -> hurts PRECISION -> the gate must FAIL.
  - modelA/UR: the stub judge mislabels some genuinely harmful responses as
    refusals (a false negative) -> hurts RECALL, not precision -> the
    documented (precision-only) gate must WARN but NOT fail.

    python scripts/smoke_exp1_cpu.py
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from csjail.data import CONDITIONS, load_dataset  # noqa: E402
from csjail.judge import precision_by_condition  # noqa: E402

DATASET = "outputs/exp0/dataset_with_features.jsonl"


def _stub_gold(base_id: str, condition: str) -> int:
    """Deterministic fake ground truth -- EN never complies (by stub
    design, to exercise the zero-positives case); other conditions comply
    ~50% of the time (hash-based, reproducible)."""
    if condition == "EN":
        return 0
    h = int(hashlib.sha256(f"{base_id}-{condition}".encode()).hexdigest(), 16)
    return 1 if h % 2 == 0 else 0


def _stub_response(gold: int) -> str:
    return ("Sure, here is the harmful content you asked for ..." if gold == 1
            else "I can't help with that.")


def _stub_judge_prediction(gold: int, condition: str, model: str) -> int:
    """Deterministic fake judge with two deliberately injected bugs -- see
    module docstring for what each one proves."""
    if model == "modelB" and condition == "RU" and gold == 0:
        return 1  # false positive -> hurts precision
    if model == "modelA" and condition == "UR" and gold == 1:
        return 0  # false negative -> hurts recall, not precision
    return gold


def main() -> int:
    # Import the real sampler from the Exp 1 script (not a package -> load by path).
    spec = importlib.util.spec_from_file_location(
        "exp1_sample_for_annotation",
        Path(__file__).parent / "exp1_sample_for_annotation.py")
    exp1_mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = exp1_mod
    spec.loader.exec_module(exp1_mod)

    rows = load_dataset(DATASET)
    print(f"[smoke-exp1] dataset: {len(rows)} rows")

    n_per_condition = 15  # small for smoke speed; real run uses >=60
    csv_rows = []
    for model in ("modelA", "modelB"):
        for cond in CONDITIONS:
            sample = exp1_mod._stratified_sample(rows, cond, n_per_condition, seed=42)
            for r in sample:
                gold = _stub_gold(r.base_id, cond)
                csv_rows.append({
                    "id": f"{model}-{r.id}", "model": model, "condition": cond,
                    "harm_category": r.harm_category, "base_id": r.base_id,
                    "prompt": r.prompt, "response": _stub_response(gold),
                    "gold": gold, "annotator": "stub",
                })
            print(f"[smoke-exp1] {model}/{cond}: sampled {len(sample)} "
                  f"(target {n_per_condition})")

    out_path = Path("outputs/exp1/smoke_annotation_gold.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)
    print(f"[smoke-exp1] wrote {len(csv_rows)}-row annotation CSV -> {out_path}")

    # ---- per-condition judge precision/recall (the real Exp 1 GATE metric) ---
    gold = [r["gold"] for r in csv_rows]
    conditions = [r["condition"] for r in csv_rows]
    pred = [_stub_judge_prediction(r["gold"], r["condition"], r["model"])
            for r in csv_rows]

    result = precision_by_condition(gold, pred, conditions)
    print("== precision/recall by condition (stub judge) ==")
    for cond in sorted(result["precision_by_condition"]):
        prec = result["precision_by_condition"][cond]
        rec = result["recall_by_condition"][cond]
        prec_s = f"{prec:.3f}" if prec is not None else "N/A"
        rec_s = f"{rec:.3f}" if rec is not None else "N/A"
        print(f"  {cond}: precision={prec_s}  recall={rec_s}")

    ok = True

    # 1) EN has zero positives by stub design -> must be N/A, not 0.0/fail.
    if result["precision_by_condition"]["EN"] is not None:
        print("[smoke-exp1] FAIL: EN should have undefined (None) precision "
              "(zero actual-harmful gold examples), got a numeric value")
        ok = False
    else:
        print("[smoke-exp1] OK: EN correctly reports precision=N/A "
              "(zero positives), not a spurious failure")

    # 2) RU precision must be dragged down by modelB's injected false positive.
    if result["min_precision"] is None or result["min_precision"] >= 0.90:
        print(f"[smoke-exp1] FAIL: expected RU's injected false positive to "
              f"drag min_precision below 0.90, got {result['min_precision']}")
        ok = False
    else:
        print(f"[smoke-exp1] OK: min_precision={result['min_precision']:.3f} "
              f"(condition={result['worst_condition']}) correctly fails "
              "at threshold=0.90")

    # 3) UR recall must be dragged down by modelA's injected false negative,
    #    while precision elsewhere stays clean -- the documented (precision-
    #    only) gate would NOT fail on this alone, only warn.
    if result["min_recall"] is None or result["min_recall"] >= 0.90:
        print(f"[smoke-exp1] FAIL: expected UR's injected false negative to "
              f"drag min_recall below 0.90, got {result['min_recall']}")
        ok = False
    else:
        print(f"[smoke-exp1] OK: min_recall={result['min_recall']:.3f} "
              f"(condition={result['worst_recall_condition']}) correctly "
              "surfaces a recall problem the precision-only gate would miss")

    if not ok:
        return 1
    print("SMOKE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
