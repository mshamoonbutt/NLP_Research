#!/usr/bin/env python3
"""Exp 4 — tokenizer fertility and harmful compliance (CPU; consumes Exp 2 results).

Fertility F_m(x) = T_m(x) / W(x): tokens of the raw prompt under model m's pinned
tokenizer (no chat template, no special tokens) per whitespace-delimited word (paper
App E). It needs no language tagger. The Urdu-share / CMI analyses need a validated
token-level tagger (csjail/features.py is an unvalidated heuristic), so they are NOT run.

Per model:
  * fertility by condition (mean, median, quartiles) and its ratio to EN;
  * tokenizer check: n_prompt_tokens recorded at generation minus T_m(x) should be one
    constant (the chat-template overhead) for every prompt; the share of prompts at the
    most common value is reported, which also verifies a mirror tokenizer against the run;
  * GEE (logit, exchangeable, clustered on family; z = standardized within condition),
    exploratory:
        outcome ~ z(fertility) + C(condition, EN) + C(domain) + z(log words)
    for outcome = harmful compliance and non-response (unintelligible / irrelevant /
    empty), pooled over conditions and per condition. Conditions where every response has
    the same outcome are left out (no information, and they make the fit separate). With
    condition in the model, the fertility term compares prompts WITHIN a
    condition: an association, not a mechanism. Holm across the pooled fits (models x 2).
  * human-label check: the same pooled fits on the reviewer-labelled responses (Exp 1
    development + held-out for the original models, gold-audit-05 for the added ones),
    once with the judge's labels and once with the reviewer's on identical rows, so a
    fertility association that exists only in the judge's labels shows up as such.

    python scripts/exp4_features.py --results outputs/exp2/main --out-dir outputs/exp4
    # a gated tokenizer without HF_TOKEN: --tokenizer llama32=unsloth/Llama-3.2-3B-Instruct
    # tokenizers for models without results yet (fertility tables only): --models r1qwen15 gemma4e2b
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import urllib.request
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

from csjail.aggregate import IncompatibleRunsError, load_results, write_csv  # noqa: E402
from csjail.artifacts import resolve_exp0, sha256_file  # noqa: E402
from csjail.judge_validation import parse_bool  # noqa: E402
from csjail.metrics import gee_logit, holm_bonferroni  # noqa: E402
from csjail.outcomes import nonresponse, primary_unsafe  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TOK_DIR = ROOT / ".cache" / "tokenizers"     # gitignored
EXP1 = ROOT / "outputs" / "exp1"
OUTCOMES = {"unsafe": primary_unsafe, "nonresp": nonresponse}


def fetch_tokenizer(repo: str, revision: str | None) -> Path:
    rev = revision or "main"
    path = TOK_DIR / repo.replace("/", "--") / rev / "tokenizer.json"
    if not path.exists():
        hdr = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"} if os.environ.get("HF_TOKEN") else {}
        req = urllib.request.Request(f"https://huggingface.co/{repo}/resolve/{rev}/tokenizer.json", headers=hdr)
        with urllib.request.urlopen(req, timeout=300) as r:
            data = r.read()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return path


def labelled_pairs(recs: list[dict]) -> dict:
    """{(model, row_id): (judge record, reviewer record)} for every reviewer-labelled response:
    Exp 1 development + held-out (original models; the judge's prediction on the very response
    the reviewer saw) and gold-audit-05 (added models; judged in their Exp 2 runs). Reviewer
    labels are wrapped as judge-shaped records so the same outcome functions apply."""
    import csv
    fps = {r.get("judge_fingerprint_id") for r in recs} - {None}
    fp = fps.pop() if len(fps) == 1 else None
    def lab(unsafe, kind):
        return {"judge_status": "ok", "judge_unsafe": unsafe, "judge_response_kind": kind}
    out = {}
    for d in ("rubric-dev-01", "heldout-960"):
        pred_path = EXP1 / d / f"judge_predictions_harm_{fp}.jsonl"
        if not pred_path.exists():
            continue
        preds = {}
        for line in pred_path.open(encoding="utf-8"):
            q = json.loads(line)
            if q.get("judge_status") == "ok" or q["item_id"] not in preds:
                preds[q["item_id"]] = q
        for r in csv.DictReader((EXP1 / d / "final_labels.csv").open(encoding="utf-8")):
            _, model, row_id = r["item_id"].split(":", 2)
            out[(model, row_id)] = (preds.get(r["item_id"]), lab(r["unsafe"] == "true", r["response_kind"]))
    ga5 = EXP1 / "gold-audit-05"
    if (ga5 / "reviewer_returned.csv").exists():
        with (ga5 / "AUDIT_KEY_do_not_share.csv").open(encoding="utf-8") as f:
            key = {k["audit_id"]: k["item_id"] for k in csv.DictReader(f)}
        by_gen = {r["gen_key"]: r for r in recs}
        with (ga5 / "reviewer_returned.csv").open(encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                g = by_gen.get(key[r["audit_id"]])
                if g is not None:
                    out[(g["model"], g["row_id"])] = (g, lab(parse_bool(r["unsafe"]), r["response_kind"]))
    return out


def fit(sub, y: str, formula: str) -> dict:
    try:
        g = gee_logit(sub, outcome=y, cluster="base_id", formula=formula)
        return {"formula": g.formula, "or_per_sd": g.odds_ratios["z_fert"], "ci": g.conf_int["z_fert"],
                "p": g.pvalues["z_fert"], "n_obs": g.n_obs, "n_events": int(sub[y].sum()),
                "n_dropped_missing": g.n_dropped_missing}
    except Exception as e:  # report degenerate fits, never hide them
        return {"error": f"{type(e).__name__}: {e}"}


def informative(df, cols: list[str]) -> list[str]:
    """Conditions where every column in `cols` has both outcomes. A condition where every
    response has the same outcome says nothing about the within-condition slope and makes the
    fit separate, so it is left out."""
    ok = None
    for y in cols:
        rate = df.groupby("condition")[y].mean()
        s = set(rate[(rate > 0) & (rate < 1)].index)
        ok = s if ok is None else ok & s
    return sorted(ok or [])


def pooled_formula(y: str, keep: list[str], domain: bool = True) -> str:
    base = "EN" if "EN" in keep else keep[0]
    return (f"{y} ~ z_fert + C(condition, Treatment('{base}'))" + (" + C(domain)" if domain else "")
            + " + z_logw")


def quartiles(xs: list[float]) -> dict:
    q = statistics.quantiles(xs, n=4)
    return {"n": len(xs), "mean": statistics.fmean(xs), "median": q[1], "q25": q[0], "q75": q[2]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True, help="Exp 2 run dir(s)")
    ap.add_argument("--models", nargs="*", default=[], help="also tokenize these (no results needed)")
    ap.add_argument("--tokenizer", nargs="*", default=[],
                    help="model=repo[@revision] overrides, e.g. an ungated mirror of a gated tokenizer")
    ap.add_argument("--exp0-dir", default=None)
    ap.add_argument("--out-dir", default=str(ROOT / "outputs" / "exp4"))
    ap.add_argument("--allow-debug", action="store_true")
    args = ap.parse_args(argv)
    reg = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text(encoding="utf-8"))["models"]
    override = dict(t.split("=", 1) for t in args.tokenizer)

    try:
        recs = [r for r in load_results(args.results, allow_debug=args.allow_debug) if r.get("arm", "A") == "A"
                and r.get("sample_index", 0) == 0]
    except IncompatibleRunsError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1
    by_key = {(r["model"], r["row_id"]): r for r in recs}
    prompts = resolve_exp0(args.exp0_dir).load_rows()
    models = sorted({r["model"] for r in recs} | set(args.models))

    from tokenizers import Tokenizer

    pairs = labelled_pairs(recs)

    rows, out = [], {"runs": args.results, "definition": "tokens of the raw prompt (no chat template, no "
                     "special tokens) / whitespace words", "models": {}}
    for model in models:
        repo, _, rev = override.get(model, f"{reg[model]['hf_id']}@{reg[model].get('revision') or ''}").partition("@")
        path = fetch_tokenizer(repo, rev or None)
        tok = Tokenizer.from_file(str(path))
        mrows = []
        for p in prompts:
            words = len(p.prompt.split())
            n_tok = len(tok.encode(p.prompt, add_special_tokens=False).ids)
            r = by_key.get((model, p.id))
            mrows.append({"model": model, "row_id": p.id, "base_id": p.base_id, "condition": p.condition,
                          "domain": p.domain_id, "n_tokens": n_tok, "n_words": words,
                          "fertility": n_tok / words if words else None,
                          "unsafe": None if r is None else primary_unsafe(r),
                          "nonresp": None if r is None else nonresponse(r),
                          **({f"{y}_{src}": OUTCOMES[y](rec) for y in OUTCOMES
                              for src, rec in zip(("judge", "human"), pairs[(model, p.id)])}
                             if pairs.get((model, p.id), (None,))[0] else {}),
                          "n_prompt_tokens_run": None if r is None else r.get("n_prompt_tokens")})
        rows += mrows
        conds = sorted({x["condition"] for x in mrows})
        fert = {c: quartiles([x["fertility"] for x in mrows if x["condition"] == c and x["fertility"]]) for c in conds}
        for c in conds:
            fert[c]["ratio_to_EN"] = fert[c]["mean"] / fert["EN"]["mean"]
        resid = Counter(x["n_prompt_tokens_run"] - x["n_tokens"] for x in mrows if x["n_prompt_tokens_run"] is not None)
        check = None
        if resid:
            mode, k = resid.most_common(1)[0]
            n = sum(resid.values())
            check = {"n_compared": n, "template_overhead_mode": mode, "share_at_mode": k / n,
                     "share_within_1": sum(v for d, v in resid.items() if abs(d - mode) <= 1) / n}
        res = {"tokenizer": {"repo": repo, "revision": rev or "main", "tokenizer_json_sha256": sha256_file(path)},
               "fertility_by_condition": fert, "tokenizer_check_vs_run": check, "gee": {}}
        scored = [x for x in mrows if x["unsafe"] is not None]
        if scored:
            import pandas as pd

            df = pd.DataFrame(mrows)
            df = df[df["fertility"].notna()].copy()
            df["logw"] = df["n_words"].map(math.log)
            # Standardized WITHIN condition: "one SD more fragmented than other prompts in the
            # same condition", the comparison the condition-adjusted term actually makes.
            for col, src in (("z_fert", "fertility"), ("z_logw", "logw")):
                df[col] = df.groupby("condition")[src].transform(lambda s: (s - s.mean()) / s.std())

            for y in OUTCOMES:
                df[y] = df[y].astype("float")
                keep = informative(df, [y])
                sub = df[df["condition"].isin(keep)]
                res["gee"][y] = {"conditions": keep,
                                 "pooled": fit(sub, y, pooled_formula(y, keep)) if keep
                                 else {"error": "no condition has both outcomes"},
                                 "by_condition": {c: fit(sub[sub["condition"] == c], y,
                                                         f"{y} ~ z_fert + C(domain) + z_logw") for c in keep}}
            # Human-label check on identical rows; no domain term (few events per cell).
            lab = df[df["unsafe_human"].notna()].copy() if "unsafe_human" in df else df.iloc[0:0]
            res["human_label_check"] = {"n_labelled": len(lab)}
            for y in OUTCOMES if len(lab) else ():
                cols = [f"{y}_judge", f"{y}_human"]
                for c in cols:
                    lab[c] = lab[c].astype("float")
                keep = informative(lab, cols)
                sub = lab[lab["condition"].isin(keep)]
                res["human_label_check"][y] = {"conditions": keep, **{
                    f"{c.rsplit('_', 1)[1]}_labels": fit(sub, c, pooled_formula(c, keep, domain=False)) if keep
                    else {"error": "no condition has both outcomes under both labels"} for c in cols}}
        out["models"][model] = res
        f2 = lambda v: f"{v:.2f}"  # noqa: E731
        print(f"[exp4] {model}: fertility " + " ".join(f"{c} {f2(fert[c]['mean'])}" for c in conds)
              + (f" | tokenizer check: {check['share_at_mode']:.3f} at overhead {check['template_overhead_mode']}"
                 if check else " | no run to check against"))
        for y, gs in res["gee"].items():
            for name, g in [("pooled", gs["pooled"])] + list(gs["by_condition"].items()):
                print(f"[exp4]   {y} {name}: " + (f"GEE failed: {g['error']}" if "error" in g else
                      f"OR per within-condition SD {g['or_per_sd']:.2f} [{g['ci'][0]:.2f},{g['ci'][1]:.2f}] "
                      f"p={g['p']:.3g} (n={g['n_obs']}, events {g['n_events']})"))

        for y, h in res.get("human_label_check", {}).items():
            if y != "n_labelled":
                print(f"[exp4]   human-label check {y} (n labelled {res['human_label_check']['n_labelled']}): "
                      + " | ".join(f"{k}: " + (g["error"] if "error" in g else
                                               f"OR {g['or_per_sd']:.2f} [{g['ci'][0]:.2f},{g['ci'][1]:.2f}] "
                                               f"events {g['n_events']}")
                                   for k, g in h.items() if k != "conditions"))

    # Holm across the pooled fits (models x outcomes); per-condition fits stay descriptive.
    pooled = [(m, y, g["pooled"]) for m, r in out["models"].items() for y, g in r["gee"].items()
              if "p" in g["pooled"]]
    for (m, y, g), h in zip(pooled, holm_bonferroni([g["p"] for _, _, g in pooled])):
        g["p_holm"], g["holm_family_size"] = h["p_adjusted"], h["family_size"]
        print(f"[exp4] Holm ({h['family_size']} pooled fits): {m} {y} OR {g['or_per_sd']:.2f} "
              f"p_adj={h['p_adjusted']:.3g}")

    od = Path(args.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    counts = ("model", "row_id", "base_id", "condition", "domain", "n_tokens", "n_words", "fertility",
              "n_prompt_tokens_run")   # counts only: no outcomes or labels per item
    write_csv(od / "fertility_by_prompt.csv", [{k: x[k] for k in counts} for x in rows])
    (od / "features_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[exp4] wrote {od / 'features_results.json'} + fertility_by_prompt.csv (counts only, no text)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
