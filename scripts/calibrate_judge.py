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

Exit codes: 0 PASS, 1 FAIL, 2 INSUFFICIENT_EVIDENCE, 3 bad input, 4 incomplete
(API failures such as an exhausted budget; finished predictions are kept in
judge_predictions_<kind>_<fingerprint>.jsonl and a re-run judges only the rest).

`--smoke` runs the legacy English-only 30+30 check: a plumbing smoke test,
never a certification, and it never writes a manifest.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from csjail.artifacts import sha256_text  # noqa: E402
from csjail.judge import Judge, load_judge_config  # noqa: E402
from csjail.judge_validation import (  # noqa: E402
    FAIL, INSUFFICIENT, PASS, ValidationInputError, build_gold, evaluate, read_csv,
    write_manifest,
)
from csjail.outcomes import STATUS_OK  # noqa: E402
from csjail.pipeline import JsonlCache  # noqa: E402
from csjail.utils.io import read_jsonl, sha256_file  # noqa: E402

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


def gold_from_csv(path: str) -> tuple[dict, dict]:
    """Provisional development gold (gold_unsafe, gold_response_kind, gold_status)."""
    from csjail.judge_validation import parse_bool

    gold, statuses = {}, {}
    for r in read_csv(path):
        statuses[r["gold_status"]] = statuses.get(r["gold_status"], 0) + 1
        if r["gold_status"] == "pending_disagreement":
            continue
        gold[r["item_id"]] = {"value": parse_bool(r["gold_unsafe"]),
                              "response_kind": r["gold_response_kind"],
                              "resolution": r["gold_status"]}
    report = {"source": path, "gold_status_counts": statuses, "n_items": sum(statuses.values()),
              "n_unlabeled": 0, "n_unresolved_disagreements": 0, "n_disagreements": None,
              "pre_adjudication_agreement": "not applicable (provisional development gold)"}
    return gold, report


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
    ap.add_argument("--predict-only", action="store_true",
                    help="judge and cache predictions, then stop: no metrics, no verdict, no "
                         "manifest (two-phase validation: labels are collected afterwards)")
    ap.add_argument("--judge-model", default=None,
                    help="DEVELOPMENT ONLY: score with this model instead of the configured one "
                         "(same rubric), to compare judge models on the development sample; "
                         "refused for validation samples")
    ap.add_argument("--judge-provider", default=None,
                    help="DEVELOPMENT ONLY, with --judge-model: an OpenAI-compatible provider "
                         "from csjail.judge.OPENAI_COMPATIBLE (e.g. deepseek)")
    ap.add_argument("--judge-concurrency", type=int, default=None,
                    help="DEVELOPMENT ONLY: parallel requests for the comparison judge")
    ap.add_argument("--judge-max-tokens", type=int, default=None,
                    help="DEVELOPMENT ONLY: output budget for the comparison judge; reasoning models "
                         "(e.g. gpt-oss) spend it on thinking first, so 300 leaves no answer")
    ap.add_argument("--manifest-out", default=None,
                    help="default outputs/exp1/judge_validation_manifest[_benign].json")
    ap.add_argument("--gold-csv", default=None,
                    help="DEVELOPMENT only: provisional gold file (e.g. development_gold.csv "
                         "from scripts/exp1_dev_gold.py); pending-disagreement items are skipped")
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
        # heldout: a labelled test set for judges already chosen -- scored like a
        # development sample (feedback report, never a manifest), reported separately.
        if role not in ("validation", "development", "heldout"):
            raise ValidationInputError(f"unknown sample role {role!r}")
        if role == "validation" and args.development_sample_dir:
            dman = json.loads((Path(args.development_sample_dir) / "sample_manifest.json")
                              .read_text(encoding="utf-8"))
            overlap = sorted(set(dman["families"]) & set(sman["families"]))
            if overlap:
                raise ValidationInputError(f"{len(overlap)} families shared with the rubric-"
                                           f"development sample, e.g. {overlap[:3]}")
        items = read_csv(sd / "items.csv")
        if args.gold_csv:
            if role == "validation":
                raise ValidationInputError("--gold-csv is for development feedback only; final "
                                           "validation needs two independent rater files")
            gold, gold_report = gold_from_csv(args.gold_csv)
        else:
            gold, gold_report = build_gold(items, read_csv(sd / "rater1.csv"),
                                           read_csv(sd / "rater2.csv"),
                                           read_csv(sd / "adjudication.csv"), kind=args.kind)
    except (ValidationInputError, FileNotFoundError, KeyError) as e:
        print(f"FAIL (input): {e}", file=sys.stderr)
        return 3

    cfg = load_judge_config()
    if args.judge_model or args.judge_provider or args.judge_concurrency or args.judge_max_tokens:
        if role == "validation" or not args.judge_model:
            print("FAIL (input): --judge-model/--judge-provider/--judge-concurrency are for "
                  "development samples only (and need --judge-model); the validation judge is "
                  "the configured one", file=sys.stderr)
            return 3
        import dataclasses
        cfg = dataclasses.replace(cfg, model=args.judge_model, model_snapshot=None,
                                  provider=args.judge_provider or cfg.provider,
                                  concurrency=args.judge_concurrency or cfg.concurrency,
                                  max_tokens=args.judge_max_tokens or cfg.max_tokens)
    fp = cfg.fingerprint(args.kind)
    if args.predictions_from:
        saved = {r["item_id"]: r for r in read_jsonl(args.predictions_from)}
        if any(r.get("judge_fingerprint_id") != fp["fingerprint_id"] for r in saved.values()):
            print("FAIL (input): saved predictions were made by a different judge/rubric",
                  file=sys.stderr)
            return 3
    else:
        # Resumable: each chunk is appended as soon as it is judged, and a re-run
        # reuses every OK prediction for the identical prompt+response.
        cache = JsonlCache(sd / f"judge_predictions_{args.kind}_{fp['fingerprint_id']}.jsonl",
                           "item_id")
        pair = {it["item_id"]: sha256_text(it["prompt"] + "\x00" + it["response"]) for it in items}
        todo = [it for it in items
                if cache.records.get(it["item_id"], {}).get("judge_status") != STATUS_OK
                or cache.records[it["item_id"]].get("pair_sha256") != pair[it["item_id"]]]
        print(f"[calibrate] {len(items) - len(todo)} cached predictions reused, "
              f"{len(todo)} to judge")
        try:
            judge = Judge(cfg, kind=args.kind) if todo else None
        except RuntimeError as e:                 # missing API key or client package
            print(f"FAIL (input): {e}", file=sys.stderr)
            return 3
        t0 = time.monotonic()
        for start in range(0, len(todo), 64):
            chunk = todo[start:start + 64]
            js = judge.score_sync([(it["prompt"], it["response"]) for it in chunk])
            now = datetime.now(timezone.utc).isoformat()
            cache.append({"item_id": it["item_id"], "judge_fingerprint_id": fp["fingerprint_id"],
                          "pair_sha256": pair[it["item_id"]], "created_utc": now,
                          **j.as_record_fields()}
                         for it, j in zip(chunk, js, strict=True))
            done, mins = start + len(chunk), (time.monotonic() - t0) / 60
            rate = f"{done / mins:.1f}" if mins > 0.05 else "n/a"
            print(f"[calibrate] {done}/{len(todo)} judged, {rate} per minute, "
                  f"failures so far {sum(1 for i in todo[:done] if cache.records[i['item_id']].get('judge_status') != STATUS_OK)}",
                  flush=True)
        saved = {it["item_id"]: cache.records[it["item_id"]] for it in items}
        api_failed = [i for i, r in saved.items() if r.get("judge_status") == "api_error"]
        if api_failed:
            print(f"[calibrate] INCOMPLETE: {len(api_failed)} items not judged because the API "
                  f"failed (e.g. {saved[api_failed[0]].get('judge_error', '')[:160]!r}). The "
                  f"{len(items) - len(api_failed)} finished predictions are saved; re-run the "
                  "same command to judge only the rest. No verdict or manifest written.",
                  file=sys.stderr)
            return 4
        if args.predict_only:
            flagged = sum(1 for r in saved.values() if r.get("judge_unsafe" if args.kind == "harm"
                                                            else "judge_refused"))
            print(f"[calibrate] PREDICT-ONLY: {len(saved)} predictions cached, {flagged} flagged; "
                  "no metrics, verdict or manifest written")
            return 0
    key = "judge_refused" if args.kind == "benign" else "judge_unsafe"
    preds = {i: {"status": r.get("judge_status"), "value": r.get(key),
                 "response_kind": r.get("judge_response_kind")} for i, r in saved.items()}

    result = evaluate(items, gold, preds, cfg.validation, kind=args.kind,
                      gold_report=gold_report, gate_on_recall=not args.no_gate_on_recall)
    if role in ("development", "heldout"):
        # Feedback only: same metrics, never a validation manifest.
        dev = sd / f"{role}_report_{args.kind}_{fp['fingerprint_id']}.json"
        dev.write_text(json.dumps({"kind": "judge_development_report", "not_a_gate": True,
                                   "judge_fingerprint": fp, "gold_report": gold_report,
                                   "result": result}, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        _print_result(fp, gold_report, result)
        print(f"[calibrate] {role.upper()} sample: metrics are feedback, not a gate; no "
              f"validation manifest written -> {dev}")
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
