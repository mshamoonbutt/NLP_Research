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
domain-stratified family-bootstrap CI. Holm correction within each DECLARED
family (eval.yaml contrasts.holm_families: planned contrasts x the family's
models). Judge robustness: the paired difference of ASRs corrected for the
judge's Exp 1 error per model x condition with the judge's predictive values,
the Exp 2 correction (aggregate.predictive_value_diff: family bootstrap x
Jeffreys draws); a contrast is a finding only if Holm-significant AND that
corrected CI excludes 0 with the same sign. Also, descriptive only: paired
shifts in refusal, non-response (unintelligible/irrelevant/empty) and
full-only harm, and how often an English refusal becomes harmful help in
CS/RU/UR. GEE
(unsafe ~ condition + domain [+ model], clustered on family, EN baseline)
drops missing outcomes and reports how many. Model identity is read from the
records, never from filenames.

    python scripts/exp3_isolation.py --results outputs/exp2/main [more run dirs] --out-dir outputs/exp3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

from csjail.aggregate import (IncompatibleRunsError, judge_error_counts, load_results,  # noqa: E402
                              outcome_maps, pooled_cells, predictive_value_diff, write_csv)
from csjail.asr import mcnemar_pairs  # noqa: E402
from csjail.metrics import gee_logit, holm_bonferroni, wilson_ci  # noqa: E402
from csjail.outcomes import behavior, nonresponse, primary_unsafe, strict_unsafe  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def refusal(r):
    return None if primary_unsafe(r) is None else behavior(r) == "refusal"


def holm_family(model: str, families: dict) -> str:
    return next((name for name, ms in families.items() if model in ms), f"unlisted:{model}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True, help="Exp 2 run dir(s)")
    ap.add_argument("--arm", default="A")
    ap.add_argument("--split", default=None, choices=["eval_main", "train_pool"],
                    help="only families in this split (default: the full core)")
    ap.add_argument("--judge-manifest", default=str(ROOT / "outputs/exp1/judge_validation_manifest.json"),
                    help="Exp 1 manifest of the judge; its error counts give the corrected contrasts")
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
    families = ct.get("holm_families") or {}
    boot = dict(exact_threshold=st["mcnemar_exact_threshold"], bootstrap_n=st["bootstrap_n"],
                seed=st["bootstrap_seed"])

    try:
        recs = [r for r in load_results(args.results, allow_debug=args.allow_debug)
                if r.get("arm", "A") == args.arm and (not args.split or r.get("split") == args.split)]
        if not recs:
            raise IncompatibleRunsError(f"no records for arm {args.arm} / split {args.split}")
        maps = outcome_maps(recs)
    except IncompatibleRunsError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1
    fam_dom = {r["base_id"]: r["domain_id"] for r in recs}
    counts = judge_error_counts(Path(args.judge_manifest), recs) or {}
    print("[exp3] judge-corrected contrasts: " + ("on" if counts else "OFF (no Exp 1 manifest for this judge)"))

    tests, results = [], {}
    for (model, arm), by_cond in sorted(maps.items()):
        res = {}
        for label, contrasts in (("planned", planned), ("descriptive", descriptive)):
            rows = []
            for m in mcnemar_pairs(by_cond, comparisons=contrasts, strata=fam_dom, **boot):
                d = m.as_dict()
                ma, mb = by_cond[m.cond_a], by_cond[m.cond_b]
                comp = [f for f in sorted(set(ma) & set(mb)) if ma[f] is not None and mb[f] is not None]
                basis = [next((k for k in (f"{model}|{c}", c) if k in counts), None) for c in (m.cond_a, m.cond_b)]
                if all(basis):
                    d.update(predictive_value_diff([ma[f] for f in comp], [mb[f] for f in comp],
                                                   [fam_dom[f] for f in comp], counts[basis[0]], counts[basis[1]],
                                                   pooled_cells(counts, model), bootstrap_n=st["bootstrap_n"],
                                                   ci_alpha=st["ci_alpha"], seed=st["bootstrap_seed"]))
                    n_cell = sum("|" in b for b in basis)
                    d["correction_basis"] = {2: "model x condition", 0: "condition"}.get(n_cell, "mixed")
                rows.append(d)
                f3 = lambda x: "NA" if x is None else f"{x:+.3f}"  # noqa: E731
                print(f"[exp3] {model} {m.cond_a}-{m.cond_b} ({label}): diff={f3(m.diff)} "
                      f"[{f3(m.diff_ci_lo)},{f3(m.diff_ci_hi)}] corrected={f3(d.get('diff_corrected'))} "
                      f"[{f3(d.get('diff_corrected_ci_lo'))},{f3(d.get('diff_corrected_ci_hi'))}] "
                      f"b={m.a_unsafe_b_safe} c={m.a_safe_b_unsafe} complete={m.n_complete}/{m.n_shared} "
                      f"p={m.pvalue if m.pvalue is None else f'{m.pvalue:.4g}'}")
            res[label] = rows
            if label == "planned":
                tests += [(model, d) for d in rows]
        results[model] = res

    holm_out = []
    for fam in sorted({holm_family(m, families) for m, _ in tests}):
        members = [(m, d) for m, d in tests if holm_family(m, families) == fam]
        for (model, d), h in zip(members, holm_bonferroni([d["pvalue"] for _, d in members], alpha=args.alpha)):
            lo, hi = d.get("diff_corrected_ci_lo"), d.get("diff_corrected_ci_hi")
            same = None if lo is None or d["diff"] is None else bool(
                (lo > 0 and d["diff"] > 0) or (hi < 0 and d["diff"] < 0))
            holm_out.append({"model": model, "contrast": f"{d['cond_a']}-{d['cond_b']}", "holm_family": fam, **h,
                             "corrected_ci_excludes_0_same_sign": same,
                             "finding": None if h["reject"] is None else (h["reject"] and same)})
    for h in holm_out:
        print(f"[exp3] Holm {h['holm_family']} ({h['family_size']} tests): {h['model']} {h['contrast']} "
              f"p_adj={h['p_adjusted']} reject={h['reject']} judge-robust finding={h['finding']}")

    # Descriptive: does a lower harmful rate come with more refusal or with more non-response,
    # and does the contrast hold for full compliance alone (raw judge labels)?
    shifts: dict = {}
    for name, pred in (("refusal", refusal), ("nonresponse", nonresponse), ("full_only_harm", strict_unsafe)):
        for (model, _arm), by_cond in sorted(outcome_maps(recs, predicate=pred).items()):
            shifts.setdefault(model, {})[name] = [
                {"contrast": f"{m.cond_a}-{m.cond_b}", "n_complete": m.n_complete, "rate_a": m.asr_a,
                 "rate_b": m.asr_b, "diff": m.diff, "diff_ci_lo": m.diff_ci_lo, "diff_ci_hi": m.diff_ci_hi}
                for m in mcnemar_pairs(by_cond, comparisons=planned + descriptive, strata=fam_dom, **boot)]

    # Descriptive: of the families a model refused in English, how many got harmful help elsewhere.
    bypass: dict = {}
    for model in sorted({r["model"] for r in recs}):
        by = {(r["base_id"], r["condition"]): r for r in recs if r["model"] == model}
        refused = [f for (f, c), r in by.items() if c == "EN" and behavior(r) == "refusal"]
        for c in ("CS", "RU", "UR"):
            outs = [primary_unsafe(by[(f, c)]) for f in refused if (f, c) in by]
            k, n = sum(o is True for o in outs), sum(o is not None for o in outs)
            lo, hi = wilson_ci(k, n)
            bypass.setdefault(model, {})[c] = {"n_en_refused": len(refused), "n_scored": n, "n_unsafe": k,
                                               "rate": k / n if n else None, "ci_lo": lo, "ci_hi": hi}

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
        "arm": args.arm, "split": args.split or "all", "runs": args.results,
        "planned_contrasts": planned, "descriptive_contrasts": descriptive, "holm_families": families,
        "judge_manifest": args.judge_manifest if counts else None,
        "mcnemar_by_model": results, "holm": holm_out, "paired_shifts": shifts,
        "en_refused_then_unsafe": bypass, "gee": gee,
        "gee_by_model": gee_by_model, "gee_interaction_sensitivity": gee_interaction,
        "wording_note": "paired, semantically matched contrasts; not perfect single-variable "
                        "causal interventions",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    holm_by = {(h["model"], h["contrast"]): h for h in holm_out}
    write_csv(out_dir / "contrasts_table.csv", [
        {"model": model, "type": label, "contrast": f"{d['cond_a']}-{d['cond_b']}",
         **{k: d.get(k) for k in ("n_complete", "n_missing_pairs", "asr_a", "asr_b", "diff", "diff_ci_lo",
                                  "diff_ci_hi", "a_unsafe_b_safe", "a_safe_b_unsafe", "test_used", "pvalue")},
         **{k: holm_by.get((model, f"{d['cond_a']}-{d['cond_b']}"), {}).get(k)
            for k in ("holm_family", "p_adjusted", "reject")},
         **{k: d.get(k) for k in ("diff_corrected", "diff_corrected_ci_lo", "diff_corrected_ci_hi",
                                  "correction_basis")},
         "finding": holm_by.get((model, f"{d['cond_a']}-{d['cond_b']}"), {}).get("finding")}
        for model, res in results.items() for label, rows in res.items() for d in rows])
    print(f"[exp3] wrote {out_dir / 'isolation_results.json'} + contrasts_table.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
