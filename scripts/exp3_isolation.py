#!/usr/bin/env python3
"""Exp 3 — isolation of the three effects (CPU; consumes Exp 2 judgments).

This is the paper's central novelty: three paired comparisons that each
change exactly ONE variable, so a reviewer can't dismiss the headline result
as "the model is just bad at Urdu" or "the tokenizer chokes on romanization":

    CS vs RU  (script fixed, Urdu vocabulary fixed)  -> isolates CODE-SWITCHING
    RU vs UR  (language fixed, script varies)         -> isolates SCRIPT
    RU vs EN  (script fixed, language varies)         -> isolates LANGUAGE

Per the experiment guide: "Keep exactly as specified. Change nothing." The
statistics this needs (McNemar, GEE clustered logistic regression,
Holm-Bonferroni) already exist in csjail/asr.py and csjail/metrics.py —
this script is the thin driver that was missing.

Reads run_eval.py JSONL outputs (kind=="prompt_eval" rows) for all 4
conditions, across the given models.

Usage:
    python scripts/exp3_isolation.py --judgments "results/baseline_*/*.jsonl"
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from csjail.asr import mcnemar_pairs  # noqa: E402
from csjail.metrics import gee_logit, holm_bonferroni  # noqa: E402
from csjail.utils.io import read_jsonl  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
EVAL_CFG = ROOT / "configs" / "eval.yaml"
# The three isolation contrasts (excludes the CS-vs-EN headline number,
# which configs/eval.yaml also carries but which isn't one of the three
# single-variable contrasts Holm-Bonferroni corrects across here).
ISOLATION_CONTRASTS = [("CS", "RU"), ("RU", "UR"), ("RU", "EN")]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judgments", required=True,
                    help="glob over run_eval.py JSONL outputs, all 4 "
                         "conditions x all models, e.g. "
                         "'results/baseline_*/*.jsonl'")
    ap.add_argument("--out-dir", default="outputs/exp3")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args(argv)

    eval_cfg = yaml.safe_load(EVAL_CFG.read_text(encoding="utf-8"))
    exact_threshold = eval_cfg["stats"]["mcnemar_exact_threshold"]

    if not glob.glob(args.judgments):
        print(f"FAIL: no files matched {args.judgments}", file=sys.stderr)
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # run_eval.py doesn't stamp `model` on prompt_eval rows (one model+
    # condition per file); derive it from the filename convention instead
    # (<outdir>/<model>_<condition>.jsonl, per run_all_baseline.sh).
    by_model: dict[str, list[dict]] = {}
    for path in sorted(glob.glob(args.judgments)):
        stem = Path(path).stem  # e.g. "phi3_CS"
        model = stem.rsplit("_", 1)[0]
        for r in read_jsonl(path):
            if r.get("kind") != "prompt_eval":
                continue
            by_model.setdefault(model, []).append(r)

    all_pvalues: list[tuple[str, str, str, float]] = []  # (model, a, b, p)
    mcnemar_results: dict[str, list[dict]] = {}
    gee_rows: list[dict] = []

    for model, mrows in sorted(by_model.items()):
        by_cond: dict[str, list[tuple[str, "int|None"]]] = {}
        for cond in ("CS", "EN", "RU", "UR"):
            by_cond[cond] = [(r["base_id"], r["judge_label"])
                             for r in mrows if r["condition"] == cond]
            for r in mrows:
                if r["condition"] != cond:
                    continue
                gee_rows.append({
                    "base_id": r["base_id"], "condition": cond,
                    "category": r["harm_category"], "model": model,
                    "harmful": 1 if r["judge_label"] == 1 else 0,
                })

        mn = mcnemar_pairs(by_cond, comparisons=ISOLATION_CONTRASTS,
                           exact_threshold=exact_threshold)
        mcnemar_results[model] = [m.__dict__ for m in mn]
        for m in mn:
            all_pvalues.append((model, m.cond_a, m.cond_b, m.pvalue))
            print(f"[exp3] {model}: {m.cond_a} vs {m.cond_b}  p={m.pvalue:.4g} "
                  f"(b={m.b} c={m.c}, n_pairs={m.n_pairs}, test={m.test_used})")

    # Holm-Bonferroni across the 3 contrasts x N models.
    pvals = [p for *_, p in all_pvalues]
    corrected = holm_bonferroni(pvals, alpha=args.alpha)
    holm_out = [
        {"model": model, "contrast": f"{a}-{b}", **corrected[i]}
        for i, (model, a, b, _p) in enumerate(all_pvalues)
    ]
    for h in holm_out:
        print(f"[exp3] Holm-Bonferroni: {h['model']} {h['contrast']}  "
              f"p_adj={h['p_adjusted']:.4g}  reject={h['reject']}")

    # GEE: harmful ~ condition + category + model, clustered on base_id, EN baseline.
    gee_result = None
    if gee_rows:
        df = pd.DataFrame(gee_rows)
        try:
            res = gee_logit(df, outcome="harmful", baseline_condition="EN",
                            cluster="base_id")
            gee_result = {
                "terms": res.terms, "odds_ratios": res.odds_ratios,
                "conf_int": res.conf_int, "pvalues": res.pvalues,
                "n_obs": res.n_obs, "n_clusters": res.n_clusters,
            }
            print(f"[exp3] GEE odds ratios: {res.odds_ratios}")
        except Exception as e:  # statsmodels can fail to converge on small N
            print(f"WARN: GEE fit failed: {e}", file=sys.stderr)

    (out_dir / "isolation_results.json").write_text(
        json.dumps({
            "mcnemar_by_model": mcnemar_results,
            "holm_bonferroni": holm_out,
            "gee": gee_result,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[exp3] wrote {out_dir}/isolation_results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
