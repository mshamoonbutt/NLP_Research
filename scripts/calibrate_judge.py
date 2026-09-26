#!/usr/bin/env python3
"""Exp 1b — validate the judge and freeze it.

    python scripts/calibrate_judge.py --sample-dir outputs/exp1/validation-<ts> \
        [--development-sample-dir outputs/exp1/development-<ts>] [--kind harm]

Reads items.csv, rater1.csv, rater2.csv, adjudication.csv from the sample dir,
builds adjudicated gold (agreement reported BEFORE adjudication), runs the
judge configured in configs/judge.yaml (or re-scores saved predictions with
--predictions-from), and computes per-condition precision/recall/F1 with
Wilson CIs, support, confusion matrices, failure rates and per-model
breakdowns. Writes a judge validation manifest; Exp 2 refuses to judge unless
that manifest is PASS for the identical judge fingerprint.

Exit codes: 0 PASS, 1 FAIL, 2 INSUFFICIENT_EVIDENCE, 3 bad input.

`--smoke` runs the legacy English-only 30+30 check: a plumbing smoke test,
never a certification, and it never writes a manifest.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from csjail.judge import Judge, load_judge_config  # noqa: E402
from csjail.judge_validation import (  # noqa: E402
    FAIL, INSUFFICIENT, PASS, ValidationInputError, build_gold, evaluate, read_csv,
    write_manifest,
)
from csjail.utils.io import read_jsonl, sha256_file, write_jsonl  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
EXIT = {PASS: 0, FAIL: 1, INSUFFICIENT: 2}


def _smoke() -> int:
    d = ROOT / "data" / "judge_calibration"
    rows = read_jsonl(d / "harmful_30.jsonl") + read_jsonl(d / "benign_30.jsonl")
    js = Judge().score_sync([(r["prompt"], r["response"]) for r in rows])
    ok = sum(1 for r, j in zip(rows, js) if j.ok and j.unsafe == (int(r["gold"]) in (1, 2)))
    fails = sum(1 for j in js if not j.ok)
    print(f"[smoke] {ok}/{len(rows)} agree with legacy gold, {fails} judge failures.")
    print("[smoke] SMOKE ONLY: English-only, not the Exp 1 gate; no manifest written.")
    return 0 if fails == 0 else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-dir", help="Exp 1 validation sample dir")
    ap.add_argument("--kind", choices=["harm", "benign"], default="harm")
    ap.add_argument("--development-sample-dir", default=None,
                    help="rubric-development sample; must share no family with validation")
    ap.add_argument("--predictions-from", default=None,
                    help="re-score saved judge predictions (JSONL) instead of calling the API")
    ap.add_argument("--no-gate-on-recall", action="store_true",
                    help="precision-only gate (must be declared before observing results)")
    ap.add_argument("--manifest-out", default=None,
                    help="default outputs/exp1/judge_validation_manifest[_benign].json")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)

    if args.smoke:
        return _smoke()
    if not args.sample_dir:
        ap.error("--sample-dir is required (or --smoke)")
    sd = Path(args.sample_dir)
    try:
        sman = json.loads((sd / "sample_manifest.json").read_text(encoding="utf-8"))
        role = sman.get("role")
        if role not in ("validation", "development"):
            raise ValidationInputError(f"unknown sample role {role!r}")
        if role == "validation" and args.development_sample_dir:
            dman = json.loads((Path(args.development_sample_dir) / "sample_manifest.json")
                              .read_text(encoding="utf-8"))
            overlap = sorted(set(dman["families"]) & set(sman["families"]))
            if overlap:
                raise ValidationInputError(f"{len(overlap)} families shared with the rubric-"
                                           f"development sample, e.g. {overlap[:3]}")
        items = read_csv(sd / "items.csv")
        gold, gold_report = build_gold(items, read_csv(sd / "rater1.csv"),
                                       read_csv(sd / "rater2.csv"),
                                       read_csv(sd / "adjudication.csv"), kind=args.kind)
    except (ValidationInputError, FileNotFoundError, KeyError) as e:
        print(f"FAIL (input): {e}", file=sys.stderr)
        return 3

    cfg = load_judge_config()
    fp = cfg.fingerprint(args.kind)
    if args.predictions_from:
        saved = {r["item_id"]: r for r in read_jsonl(args.predictions_from)}
        if any(r.get("judge_fingerprint_id") != fp["fingerprint_id"] for r in saved.values()):
            print("FAIL (input): saved predictions were made by a different judge/rubric",
                  file=sys.stderr)
            return 3
    else:
        js = Judge(cfg, kind=args.kind).score_sync([(it["prompt"], it["response"]) for it in items])
        saved = {it["item_id"]: {"item_id": it["item_id"],
                                 "judge_fingerprint_id": fp["fingerprint_id"],
                                 **j.as_record_fields()} for it, j in zip(items, js)}
        write_jsonl(sd / f"judge_predictions_{args.kind}_{fp['fingerprint_id']}.jsonl",
                    list(saved.values()))
    key = "judge_refused" if args.kind == "benign" else "judge_unsafe"
    preds = {i: {"status": r.get("judge_status"), "value": r.get(key),
                 "response_kind": r.get("judge_response_kind")} for i, r in saved.items()}

    result = evaluate(items, gold, preds, cfg.validation, kind=args.kind,
                      gold_report=gold_report, gate_on_recall=not args.no_gate_on_recall)
    if role == "development":
        # Rubric-development feedback only: same metrics, never a validation manifest.
        dev = sd / f"development_report_{args.kind}_{fp['fingerprint_id']}.json"
        dev.write_text(json.dumps({"kind": "judge_development_report", "not_a_gate": True,
                                   "judge_fingerprint": fp, "gold_report": gold_report,
                                   "result": result}, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        _print_result(fp, gold_report, result)
        print(f"[calibrate] DEVELOPMENT sample: metrics are rubric-iteration feedback, not a "
              f"gate; no validation manifest written -> {dev}")
        return 0
    out = Path(args.manifest_out or ROOT / "outputs" / "exp1" /
               f"judge_validation_manifest{'' if args.kind == 'harm' else '_benign'}.json")
    write_manifest(out, fingerprint=fp, result=result,
                   sample_manifest={k: sman[k] for k in ("role", "sample_kind", "dataset_version",
                                                         "split_id", "seed", "models", "n_items",
                                                         "families", "items_sha256")},
                   gold_sha256={n: sha256_file(sd / n) for n in
                                ("items.csv", "rater1.csv", "rater2.csv", "adjudication.csv")},
                   gold_report=gold_report)

    _print_result(fp, gold_report, result)
    print(f"[calibrate] STATUS: {result['status']}  manifest -> {out}")
    return EXIT[result["status"]]


def _print_result(fp: dict, gold_report: dict, result: dict) -> None:
    print(f"[calibrate] judge {fp['model']} rubric {fp['rubric_version']} "
          f"({fp['fingerprint_id']})")
    print(f"[calibrate] pre-adjudication agreement: {gold_report['pre_adjudication_agreement']}")
    for c, m in result["per_condition"].items():
        s = result["condition_status"].get(c, {"status": "n/a"})
        f = lambda x: "NA" if x is None else f"{x:.3f}"  # noqa: E731
        print(f"  {c}: P={f(m['precision'])} R={f(m['recall'])} F1={f(m['f1'])} "
              f"support={m['support']} failures={m['judge_failures']} -> {s['status']} "
              f"{s.get('reason', '')}")
    for r in result["global_reasons"]:
        print(f"  ! {r}")


if __name__ == "__main__":
    sys.exit(main())
