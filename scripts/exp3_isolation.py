#!/usr/bin/env python3
"""Exp 3 — the planned paired contrasts (CPU; consumes Exp 2 results).

    CS vs RU   code-switching contrast (Latin script; lexical composition differs)
    RU vs UR   script contrast (Roman vs Perso-Arabic script Urdu)
    RU vs EN   language contrast (Latin script; Urdu vs English)
    CS vs EN   headline descriptive gap (confirmatory only if declared in eval.yaml)

These are informative, semantically matched comparisons -- NOT perfect
single-variable interventions: word choice, register, loanwords, translation
and comprehension can still differ between variants.

Per model: paired McNemar on shared families (missing pairs counted, never
imputed; duplicate family entries rejected) + the paired difference with a
domain-stratified family-bootstrap CI. Holm correction over the DECLARED
family (planned contrasts x models). GEE (unsafe ~ condition + domain +
model, clustered on family, EN baseline) drops missing outcomes and reports
how many. Model identity is read from the records, never from filenames.

    python scripts/exp3_isolation.py --results outputs/exp2/main
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

from csjail.aggregate import IncompatibleRunsError, load_results, outcome_maps  # noqa: E402
from csjail.asr import mcnemar_pairs  # noqa: E402
from csjail.metrics import gee_logit, holm_bonferroni  # noqa: E402
from csjail.outcomes import primary_unsafe  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True, help="Exp 2 run dir(s)")
    ap.add_argument("--arm", default="A")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--allow-debug", action="store_true")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load((ROOT / "configs" / "eval.yaml").read_text(encoding="utf-8"))
    st, ct = cfg["stats"], cfg["contrasts"]
    planned = [tuple(c) for c in ct["planned"]]
    descriptive = [tuple(c) for c in ct["descriptive"]]
    if ct.get("cs_en_confirmatory"):
        planned, descriptive = planned + descriptive, []

    try:
        recs = [r for r in load_results(args.results, allow_debug=args.allow_debug)
                if r.get("arm", "A") == args.arm]
        maps = outcome_maps(recs)
    except IncompatibleRunsError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1
    fam_dom = {r["base_id"]: r["domain_id"] for r in recs}

    tests, results = [], {}
    for (model, arm), by_cond in sorted(maps.items()):
        res = {}
        for label, contrasts in (("planned", planned), ("descriptive", descriptive)):
            out = mcnemar_pairs(by_cond, comparisons=contrasts,
                                exact_threshold=st["mcnemar_exact_threshold"], strata=fam_dom,
                                bootstrap_n=st["bootstrap_n"], seed=st["bootstrap_seed"])
            res[label] = [m.as_dict() for m in out]
            if label == "planned":
                tests += [(model, m) for m in out]
            for m in out:
                d = "NA" if m.diff is None else f"{m.diff:+.3f} [{m.diff_ci_lo:+.3f},{m.diff_ci_hi:+.3f}]"
                print(f"[exp3] {model} {m.cond_a}-{m.cond_b} ({label}): diff={d} "
                      f"b={m.a_unsafe_b_safe} c={m.a_safe_b_unsafe} complete={m.n_complete}/"
                      f"{m.n_shared} p={m.pvalue if m.pvalue is None else f'{m.pvalue:.4g}'}")
        results[model] = res

    holm = holm_bonferroni([m.pvalue for _, m in tests], alpha=args.alpha)
    holm_out = [{"model": model, "contrast": f"{m.cond_a}-{m.cond_b}", **h}
                for (model, m), h in zip(tests, holm)]
    for h in holm_out:
        print(f"[exp3] Holm ({h['family_size']} tests): {h['model']} {h['contrast']} "
              f"p_adj={h['p_adjusted']} reject={h['reject']}")

    def _gee(df, formula=None):
        try:
            g = gee_logit(df, outcome="unsafe", baseline_condition="EN", cluster="base_id",
                          formula=formula)
            return {"formula": g.formula, "odds_ratios": g.odds_ratios, "conf_int": g.conf_int,
                    "pvalues": g.pvalues, "n_obs": g.n_obs, "n_clusters": g.n_clusters,
                    "n_dropped_missing": g.n_dropped_missing}
        except Exception as e:  # statsmodels may fail on degenerate data; report it
            print(f"WARN: GEE failed: {e}", file=sys.stderr)
            return {"error": f"{type(e).__name__}: {e}"}

    gee = gee_by_model = gee_interaction = None
    try:
        import pandas as pd

        df = pd.DataFrame([{"base_id": r["base_id"], "condition": r["condition"],
                            "domain": r["domain_id"], "model": r["model"],
                            "unsafe": primary_unsafe(r)} for r in recs])
        df["unsafe"] = df["unsafe"].astype("float")  # None -> NaN (dropped, not safe)
        # Pooled additive model, per-model models, and a pooled sensitivity model
        # with condition-by-model interactions (paper §4.3): one common language
        # effect is not assumed.
        gee = _gee(df)
        gee_by_model = {m: _gee(df[df["model"] == m]) for m in sorted(df["model"].unique())}
        if df["model"].nunique() > 1:
            gee_interaction = _gee(
                df, formula="unsafe ~ C(condition, Treatment('EN')) * C(model) + C(domain)")
        if "n_obs" in gee:
            print(f"[exp3] GEE n_obs={gee['n_obs']} dropped_missing={gee['n_dropped_missing']}")
    except Exception as e:
        gee = gee or {"error": f"{type(e).__name__}: {e}"}
        print(f"WARN: GEE failed: {e}", file=sys.stderr)

    out_dir = Path(args.out_dir or Path(args.results[0]) / "exp3")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "isolation_results.json").write_text(json.dumps({
        "arm": args.arm, "planned_contrasts": planned, "descriptive_contrasts": descriptive,
        "mcnemar_by_model": results, "holm": holm_out, "gee": gee,
        "gee_by_model": gee_by_model, "gee_interaction_sensitivity": gee_interaction,
        "wording_note": "paired, semantically matched contrasts; not perfect single-variable "
                        "causal interventions",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[exp3] wrote {out_dir / 'isolation_results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
