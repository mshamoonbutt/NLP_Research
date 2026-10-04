#!/usr/bin/env python3
"""AI-prefill the rater files of an Exp 1 (harm) or benign validation sample.

    python scripts/exp1_prefill_rater_files.py --sample-dir outputs/exp1/validation-<ts> \
        --model gpt-4o-mini
    python scripts/exp1_prefill_rater_files.py --kind benign --sample-dir \
        outputs/exp1/benign-validation-<ts> --model gpt-4o-mini

Scores every item with the EXISTING rubric from configs/judge.yaml but a
DIFFERENT model than the judge (the judge's own model is refused: validating
a judge against gold anchored by its own labels is circular). The repo's
Judge client is OpenAI-only, so same-provider correlation is unavoidable for
now and is recorded in the metadata; human correction is the control.
Writes rater1_prefilled.csv / rater2_prefilled.csv next to the blank files,
preserving each file's independent row order, label columns prefilled,
`rater_id` EMPTY. Items the prefill model failed on stay blank.

Workflow for the two human verifiers (independent, no discussion):
  1. Each takes their own *_prefilled.csv, reviews EVERY row, corrects labels,
     and enters their real rater_id on every row.
  2. Save back as rater1.csv / rater2.csv, then adjudicate disagreements and
     run scripts/calibrate_judge.py as usual.

THIS IS ASSISTED VERIFICATION, NOT BLIND INDEPENDENT ANNOTATION. The paper
and verification_metadata must say so; anchoring is a documented limitation.
prefill_metadata.json records the prefill model, rubric hash and date.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from csjail.judge import Judge, load_judge_config  # noqa: E402
from csjail.utils.io import write_text_lf  # noqa: E402


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-dir", required=True)
    ap.add_argument("--kind", choices=["harm", "benign"], default="harm")
    ap.add_argument("--model", required=True,
                    help="prefill model (OpenAI); MUST differ from the judge in configs/judge.yaml")
    args = ap.parse_args(argv)
    sd = Path(args.sample_dir)

    judge_cfg = load_judge_config()
    if args.model in {judge_cfg.model, judge_cfg.model_id}:
        print(f"FAIL: prefill model {args.model} is the configured judge; gold anchored by "
              "the judge's own labels cannot validate it. Use a different model.",
              file=sys.stderr)
        return 1
    print(f"NOTE: prefill shares the judge's provider ({judge_cfg.provider}); correlated "
          "errors are possible -- human verification of every row is the control.",
          file=sys.stderr)

    items = read_csv(sd / "items.csv")
    cfg = dataclasses.replace(judge_cfg, model=args.model, model_snapshot=None)
    js = Judge(cfg, kind=args.kind).score_sync([(it["prompt"], it["response"]) for it in items])
    by_id = {it["item_id"]: j for it, j in zip(items, js)}

    n_ok = sum(1 for j in by_id.values() if j.ok)
    if args.kind == "benign":
        def labels(j):
            return {"refused": str(bool(j.refused)).lower(),
                    "response_kind": j.response_kind or ""}
    else:
        def labels(j):
            return {"unsafe": str(bool(j.unsafe)).lower(),
                    "response_kind": j.response_kind or "",
                    "compliance_degree": j.compliance_degree or ""}

    for name in ("rater1.csv", "rater2.csv"):
        rows = read_csv(sd / name)
        for r in rows:
            j = by_id.get(r["item_id"])
            if j is not None and j.ok:
                r.update(labels(j))
            r["rater_id"] = ""          # the human verifier's own ID goes here
        out = sd / name.replace(".csv", "_prefilled.csv")
        write_csv(out, list(rows[0].keys()), rows)
        print(f"[prefill] wrote {out} ({len(rows)} rows, order preserved)")

    fp = cfg.fingerprint(args.kind)
    write_text_lf(sd / "prefill_metadata.json", json.dumps({
        "kind": "assisted_verification_prefill",
        "note": "AI-prefilled labels pending human verification; NOT blind independent "
                "annotation. Humans must review every row and set their own rater_id.",
        "prefill_provider": judge_cfg.provider, "prefill_model": args.model,
        "rubric_kind": args.kind, "rubric_sha256": fp["rubric_sha256"],
        "judge_under_validation": {"provider": judge_cfg.provider, "model": judge_cfg.model_id},
        "created_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "n_items": len(items), "n_prefilled": n_ok, "n_failed_left_blank": len(items) - n_ok,
    }, ensure_ascii=False, indent=2))
    print(f"[prefill] {n_ok}/{len(items)} prefilled; {len(items) - n_ok} left blank for "
          "the humans. NEXT: two verifiers independently review EVERY row of their "
          "*_prefilled.csv, correct labels, fill rater_id, save as rater1.csv/rater2.csv.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
