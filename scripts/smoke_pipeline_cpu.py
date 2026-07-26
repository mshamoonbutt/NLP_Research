#!/usr/bin/env python3
"""End-to-end CPU smoke of the analysis chain — NO model, NO API.

Proves the plumbing: dataset -> (stub) generate -> (stub) judge -> ASR +
isolation McNemar + GEE -> mine rejected -> assemble preference pairs. Every
GPU/API call is replaced by a deterministic stub so this runs anywhere.

    python scripts/smoke_pipeline_cpu.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

# Toy stub data is perfectly separated -> statsmodels warns; irrelevant here.
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from csjail.asr import compute_asr, mcnemar_pairs  # noqa: E402
from csjail.data import CONDITIONS, load_dataset  # noqa: E402
from csjail.metrics import gee_logit  # noqa: E402
from csjail.prefdata import assemble_pairs, mine_rejected  # noqa: E402
from csjail.splits import attach_features, make_splits  # noqa: E402


def stub_response(prompt: str, condition: str) -> str:
    """Deterministic fake model output: 'complies' more in UR/CS/RU than EN."""
    if condition == "EN":
        return "I can't help with that."
    if condition in ("UR", "RU", "CS"):
        # simulate a jailbreak on the longer/dangerous prompts
        return "Sure, here is the harmful content you asked for ..."
    return "..."


def stub_label(response: str) -> int:
    """Deterministic fake judge: harmful if it 'complied', else refusal."""
    return 1 if response.startswith("Sure") else 0


def main() -> int:
    rows = attach_features(load_dataset("data/csjail_smoke4.jsonl"))
    split = make_splits(rows, eval_holdout=2, n_held_out_categories=1, seed=42)
    assign = split["assignments"]

    # ---- (stub) generate + judge -------------------------------------------
    records = []
    for r in rows:
        resp = stub_response(r.prompt, r.condition)
        label = stub_label(resp)
        records.append({
            "base_id": r.base_id, "condition": r.condition,
            "harm_category": r.harm_category, "prompt": r.prompt,
            "response": resp, "label": label, "severity": r.harm_severity,
        })

    # ---- ASR per condition --------------------------------------------------
    print("== ASR by condition (stubbed) ==")
    for cond in CONDITIONS:
        sub = [x for x in records if x["condition"] == cond]
        res = compute_asr([x["label"] for x in sub], [x["base_id"] for x in sub],
                          bootstrap_n=200, seed=0)
        print(f"  {cond}: ASR={res.asr:.3f}  CI=({res.ci_lo:.2f},{res.ci_hi:.2f})  n={res.n}")

    # ---- isolation McNemar contrasts ---------------------------------------
    by_cond = {}
    for x in records:
        by_cond.setdefault(x["condition"], []).append((x["base_id"], x["label"]))
    contrasts = [("CS", "RU"), ("RU", "UR"), ("RU", "EN"), ("CS", "EN")]
    print("== isolation McNemar contrasts ==")
    for m in mcnemar_pairs(by_cond, comparisons=contrasts):
        print(f"  {m.cond_a} vs {m.cond_b}: n_pairs={m.n_pairs} b={m.b} c={m.c} "
              f"p={m.pvalue:.4g} ({m.test_used})")

    # ---- GEE omnibus --------------------------------------------------------
    df = pd.DataFrame([{"harmful": x["label"], "condition": x["condition"],
                        "category": x["harm_category"], "base_id": x["base_id"]}
                       for x in records])
    gee = gee_logit(df, baseline_condition="EN")
    print("== GEE odds ratios (vs EN) ==")
    for t in gee.terms:
        if "condition" in t:
            print(f"  {t}: OR={gee.odds_ratios[t]:.2f}  p={gee.pvalues[t]:.3g}")

    # ---- Phase-2: mine rejected + assemble pairs ---------------------------
    mined = mine_rejected(records, assign, condition="CS")
    chosen = {m["base_id"]: "Main is mein madad nahi kar sakta. (I can't help.)"
              for m in mined}
    pairs, report = assemble_pairs(mined, chosen, target_pairs=100)
    print("== preference pairs (CS, training pool) ==")
    print(f"  mined={report.n_mined} with_chosen={report.n_with_chosen} "
          f"final={report.n_final} len_ratio={report.length_ratio}")
    print("SMOKE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
