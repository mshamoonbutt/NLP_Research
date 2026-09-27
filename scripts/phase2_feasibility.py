#!/usr/bin/env python3
"""Phase 2 feasibility: count candidate training pairs per model and arm
from a human-reviewed pilot sample (exp1_sample_for_annotation --role feasibility).

    python scripts/phase2_feasibility.py --sample-dir outputs/exp6/feasibility-cpu-20260927

Reads items.csv and the reviewed rater1.csv (and rater2.csv if also filled;
disagreements count as unresolved). A family is a candidate for
  C (CS pairs)           if its CS response is unsafe
  matched B / D          if BOTH its CS and EN responses are unsafe
Counts are BEFORE chosen-refusal validation and dedup (some will be lost),
and come from whatever backend generated the pilot (see sample_manifest).
Rates get Wilson CIs and are projected to the full train_pool size.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from csjail.artifacts import resolve_exp0  # noqa: E402
from csjail.judge_validation import parse_bool, read_csv  # noqa: E402
from csjail.metrics import wilson_ci  # noqa: E402
from csjail.prefdata import supported_budgets  # noqa: E402
from csjail.splits import trainable_families  # noqa: E402


def labels(sample_dir: Path) -> tuple[dict, int]:
    """item_id -> unsafe (True/False/None). Uses rater2 only where filled."""
    r1 = {r["item_id"]: parse_bool(r.get("unsafe")) for r in read_csv(sample_dir / "rater1.csv")}
    r2 = {r["item_id"]: parse_bool(r.get("unsafe")) for r in read_csv(sample_dir / "rater2.csv")}
    out, disagree = {}, 0
    for i, a in r1.items():
        b = r2.get(i)
        if a is not None and b is not None and a != b:
            out[i], disagree = None, disagree + 1
        else:
            out[i] = a if a is not None else b
    return out, disagree


def report(sample_dir: Path, pool_size: int) -> dict:
    items = read_csv(sample_dir / "items.csv")
    lab, disagree = labels(sample_dir)
    by = defaultdict(dict)   # (model, family) -> {condition: unsafe}
    dom = {}
    for it in items:
        by[(it["model"], it["base_id"])][it["condition"]] = lab.get(it["item_id"])
        dom[it["base_id"]] = it["domain_id"]
    out = {"n_items": len(items), "n_unlabelled": sum(v is None for v in lab.values()),
           "n_rater_disagreements": disagree, "train_pool_size": pool_size, "models": {}}
    for model in sorted({m for m, _ in by}):
        fams = {f: c for (m, f), c in by.items() if m == model}
        cs = [f for f, c in fams.items() if c.get("CS") is True]
        cs_known = [f for f, c in fams.items() if c.get("CS") is not None]
        matched = [f for f, c in fams.items() if c.get("CS") is True and c.get("EN") is True]
        both_known = [f for f, c in fams.items() if c.get("CS") is not None and c.get("EN") is not None]

        def arm(hits, known):
            lo, hi = wilson_ci(len(hits), len(known))
            rate = len(hits) / len(known) if known else None
            return {"pilot_count": len(hits), "reviewed_families": len(known), "rate": rate,
                    "rate_ci95": [lo, hi],
                    "projected_train_pool": None if rate is None else
                    [round(lo * pool_size), round(rate * pool_size), round(hi * pool_size)],
                    "by_domain": dict(sorted(Counter(dom[f] for f in hits).items()))}
        out["models"][model] = {
            "n_families": len(fams),
            "C_cs_pairs": arm(cs, cs_known),
            "matched_B_C_D": arm(matched, both_known),
            "EN_unsafe": sum(1 for c in fams.values() if c.get("EN") is True),
            "pilot_budgets_if_all_validate": supported_budgets(len(matched)),
        }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-dir", required=True)
    ap.add_argument("--exp0-dir", default=None)
    args = ap.parse_args(argv)
    sd = Path(args.sample_dir)
    rep = report(sd, len(trainable_families(resolve_exp0(args.exp0_dir).split)))
    (sd / "feasibility_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    for m, r in rep["models"].items():
        for name in ("C_cs_pairs", "matched_B_C_D"):
            a = r[name]
            print(f"{m:8} {name:14} pilot {a['pilot_count']}/{a['reviewed_families']}  "
                  f"projected over {rep['train_pool_size']} train families (lo/mid/hi): "
                  f"{a['projected_train_pool']}")
    if rep["n_unlabelled"] or rep["n_rater_disagreements"]:
        print(f"note: {rep['n_unlabelled']} items unlabelled/unresolved "
              f"({rep['n_rater_disagreements']} rater disagreements)")
    print(f"-> {sd / 'feasibility_report.json'}  (counts are before chosen-refusal validation)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
