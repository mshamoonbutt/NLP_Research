#!/usr/bin/env python3
"""Judge vs reviewer agreement on RESPONSE TYPE (CPU; no API).

Exp 3 reads lower harm in Urdu forms as refusal or as non-response (unintelligible /
irrelevant / empty) from the judge's response_kind. This checks that reading on every
reviewer-labelled response (Exp 1 development + held-out for the original models,
gold-audit-05 for the added ones; the judge's label on the very response the reviewer saw):

  * kappa and confusion on four types: harmful compliance / refusal / safe help / non-response;
  * per model x condition: refusal and non-response rates under the judge vs the reviewer;
  * per model: the Exp 3 paired shifts (refusal, non-response) on the labelled families with
    both conditions labelled, under judge labels and under reviewer labels.

    python scripts/exp1_kind_agreement.py --results outputs/exp2/main outputs/exp2/main-r1 outputs/exp2/main-gemma4
    -> outputs/exp1/response_kind_agreement.json (numbers only)
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import yaml  # noqa: E402

from exp4_features import labelled_pairs  # noqa: E402
from csjail.aggregate import load_results  # noqa: E402
from csjail.asr import mcnemar_pairs  # noqa: E402
from csjail.outcomes import NONRESPONSE_KINDS, behavior, primary_unsafe  # noqa: E402
from csjail.splits import cohens_kappa  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TYPES = ("harmful_compliance", "refusal", "safe_helpful", "nonresponse")


def rtype(rec) -> str | None:
    """Four-way response type; None when not scored."""
    if rec is None or primary_unsafe(rec) is None:
        return None
    k = behavior(rec)
    return "nonresponse" if k in NONRESPONSE_KINDS else k


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True, help="Exp 2 run dir(s) (for the added models' keys)")
    ap.add_argument("--out", default=str(ROOT / "outputs/exp1/response_kind_agreement.json"))
    args = ap.parse_args(argv)
    st = yaml.safe_load((ROOT / "configs/eval.yaml").read_text(encoding="utf-8"))["stats"]
    ct = yaml.safe_load((ROOT / "configs/eval.yaml").read_text(encoding="utf-8"))["contrasts"]
    contrasts = [tuple(c) for c in ct["planned"] + ct["descriptive"]]

    rows = []   # (model, condition, family, judge type, reviewer type)
    for (model, row_id), (j, h) in labelled_pairs(load_results(args.results)).items():
        jt, ht = rtype(j), rtype(h)
        if jt and ht:
            fam, _, cond = row_id.rpartition("::")
            rows.append((model, cond, fam, jt, ht))

    def agree(sub):
        j, h = [r[3] for r in sub], [r[4] for r in sub]
        out = {"n": len(sub), "agreement": sum(a == b for a, b in zip(j, h)) / len(sub),
               "kappa_4type": cohens_kappa(j, h)}
        for t in ("refusal", "nonresponse"):
            jb, hb = [x == t for x in j], [x == t for x in h]
            out[t] = {"judge_rate": sum(jb) / len(sub), "reviewer_rate": sum(hb) / len(sub),
                      "kappa": cohens_kappa(jb, hb)}
        return out

    res = {"kind": "exp1_response_kind_agreement", "types": TYPES,
           "nonresponse_kinds": NONRESPONSE_KINDS, "overall": agree(rows),
           "confusion_judge_by_reviewer": {f"{a}|{b}": n for (a, b), n in
                                           sorted(Counter((r[3], r[4]) for r in rows).items())},
           "per_model": {}, "per_model_condition": {}, "paired_shifts_on_labelled_families": {}}
    models = sorted({r[0] for r in rows})
    for m in models:
        res["per_model"][m] = agree([r for r in rows if r[0] == m])
        for c in ("EN", "CS", "RU", "UR"):
            sub = [r for r in rows if r[0] == m and r[1] == c]
            if sub:
                res["per_model_condition"][f"{m}|{c}"] = agree(sub)
        shifts = {}
        for t in ("refusal", "nonresponse"):
            for src, idx in (("judge", 3), ("reviewer", 4)):
                maps = defaultdict(dict)
                for r in rows:
                    if r[0] == m:
                        maps[r[1]][r[2]] = r[idx] == t
                shifts[f"{t}_{src}"] = [
                    {"contrast": f"{x.cond_a}-{x.cond_b}", "n_pairs": x.n_complete, "diff": x.diff,
                     "ci": [x.diff_ci_lo, x.diff_ci_hi]}
                    for x in mcnemar_pairs(maps, comparisons=contrasts, bootstrap_n=st["bootstrap_n"],
                                           seed=st["bootstrap_seed"])]
        res["paired_shifts_on_labelled_families"][m] = shifts

    Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    o = res["overall"]
    print(f"[kind] {o['n']} labelled responses: 4-type agreement {o['agreement']:.3f}, kappa {o['kappa_4type']:.3f}; "
          f"refusal kappa {o['refusal']['kappa']:.3f}, non-response kappa {o['nonresponse']['kappa']:.3f}")
    f = lambda x: "NA" if x is None else f"{x:.2f}"  # noqa: E731
    for m in models:
        p = res["per_model"][m]
        print(f"[kind] {m}: n {p['n']} kappa4 {f(p['kappa_4type'])} | refusal judge/reviewer "
              f"{p['refusal']['judge_rate']:.3f}/{p['refusal']['reviewer_rate']:.3f} | non-response "
              f"{p['nonresponse']['judge_rate']:.3f}/{p['nonresponse']['reviewer_rate']:.3f}")
        for t in ("refusal", "nonresponse"):
            j = {s["contrast"]: s for s in res["paired_shifts_on_labelled_families"][m][f"{t}_judge"]}
            h = {s["contrast"]: s for s in res["paired_shifts_on_labelled_families"][m][f"{t}_reviewer"]}
            print(f"[kind]   {t} shifts (judge / reviewer, pp): " + "  ".join(
                f"{c} {j[c]['diff'] * 100:+.0f}/{h[c]['diff'] * 100:+.0f} (n {h[c]['n_pairs']})" for c in h
                if j[c]["diff"] is not None and h[c]["diff"] is not None))
    print(f"[kind] -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
