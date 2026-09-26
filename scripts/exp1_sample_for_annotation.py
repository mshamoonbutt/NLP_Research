#!/usr/bin/env python3
"""Exp 1a — draw and generate the judge-validation (or rubric-development) sample.

Unit: `--n-per-model-condition` families x each condition x each model. The
default 60 x 4 conditions x 2 models = 480 responses (60 per model per
condition is a starting MINIMUM, not proof of adequate precision; top up per
configs/judge.yaml `validation` if support is short).

- Families come from the TRAINING POOL of the frozen split (never eval_main),
  stratified by domain, identical across conditions and models (paired).
- `--role development` vs `--role validation`: rubric development and final
  validation must use disjoint families; pass earlier samples via
  `--exclude-sample-dirs` and the script refuses any overlap.
- `--sample-kind challenge` marks a deliberately enriched diagnostic set; it
  is reported separately and never used for the PASS gate.
- Output (sample dir): items.csv (with text; gitignored), rater1.csv,
  rater2.csv (same items, independently shuffled, NO judge output),
  adjudication.csv template, sample_manifest.json (ids + hashes only).

    python scripts/exp1_sample_for_annotation.py --models qwen25 phi3 --role validation
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

from csjail.artifacts import resolve_exp0, sha256_json, sha256_text  # noqa: E402
from csjail.data import CONDITIONS, family_domains  # noqa: E402
from csjail.judge_validation import read_csv  # noqa: E402
from csjail.splits import proportional_stratified_indices, trainable_families  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RATER_FIELDS = ["item_id", "rater_id", "prompt", "response", "unsafe", "response_kind",
                "compliance_degree", "notes"]
ADJ_FIELDS = ["item_id", "gold_unsafe", "gold_response_kind", "gold_compliance_degree",
              "resolution", "adjudicator", "notes"]


def sample_families(rows, split: dict, n: int, seed: int, exclude: set[str]) -> list[str]:
    pool = sorted(trainable_families(split) - exclude)
    dom = family_domains(rows)
    idx = proportional_stratified_indices([dom[f] for f in pool], n, seed=seed)
    return [pool[i] for i in idx]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp0-dir", default=None)
    ap.add_argument("--models", nargs="+", default=["qwen25", "phi3"])
    ap.add_argument("--n-per-model-condition", type=int, default=60)
    ap.add_argument("--role", choices=["validation", "development"], required=True)
    ap.add_argument("--sample-kind", choices=["representative", "challenge"],
                    default="representative")
    ap.add_argument("--exclude-sample-dirs", nargs="*", default=[],
                    help="earlier sample dirs whose families must not be reused")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)

    if len(set(args.models)) < 2:
        print("FAIL: draw the gold set from >=2 target models", file=sys.stderr)
        return 1
    art = resolve_exp0(args.exp0_dir)
    rows = art.load_rows()
    exclude: set[str] = set()
    for d in args.exclude_sample_dirs:
        exclude |= set(json.loads((Path(d) / "sample_manifest.json").read_text(
            encoding="utf-8"))["families"])
    fams = sample_families(rows, art.split, args.n_per_model_condition, args.seed, exclude)
    if len(fams) < args.n_per_model_condition:
        print(f"WARN: only {len(fams)} eligible families", file=sys.stderr)
    by_key = {(r.base_id, r.condition): r for r in rows}

    eval_cfg = yaml.safe_load((ROOT / "configs" / "eval.yaml").read_text(encoding="utf-8"))
    from csjail.models import SamplingConfig, SLMRunner, resolve  # GPU

    sampling = SamplingConfig(**eval_cfg["sampling"])
    items, model_prov = [], {}
    for model in args.models:
        runner = SLMRunner(resolve(model))
        try:
            model_prov[model] = runner.provenance()
            for cond in CONDITIONS:
                sub = [by_key[(f, cond)] for f in fams]
                outs = runner.generate_text([r.prompt for r in sub], sampling)
                for r, out in zip(sub, outs, strict=True):
                    items.append({
                        "item_id": f"{args.role}:{model}:{r.id}", "role": args.role,
                        "sample_kind": args.sample_kind, "model": model,
                        "model_revision": runner.spec.revision, "base_id": r.base_id,
                        "domain_id": r.domain_id, "condition": cond, "row_id": r.id,
                        "response_id": sha256_text(out)[:16], "prompt": r.prompt,
                        "response": out})
        finally:
            runner.shutdown()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_dir or ROOT / "outputs" / "exp1" / f"{args.role}-{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "items.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(items[0].keys()))
        w.writeheader(); w.writerows(items)
    for k, name in ((1, "rater1.csv"), (2, "rater2.csv")):
        order = items[:]
        random.Random(args.seed + k).shuffle(order)   # independent order, no predictions
        with (out_dir / name).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=RATER_FIELDS)
            w.writeheader()
            w.writerows({"item_id": it["item_id"], "rater_id": "", "prompt": it["prompt"],
                         "response": it["response"], "unsafe": "", "response_kind": "",
                         "compliance_degree": "", "notes": ""} for it in order)
    with (out_dir / "adjudication.csv").open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=ADJ_FIELDS).writeheader()
    manifest = {
        "kind": "exp1_sample_manifest", "role": args.role, "sample_kind": args.sample_kind,
        "created_utc": stamp, "dataset_version": art.dataset_version, "split_id": art.split_id,
        "source_pool": "train_pool", "seed": args.seed, "models": args.models,
        "n_per_model_condition": args.n_per_model_condition, "n_items": len(items),
        "families": fams, "excluded_families_from": args.exclude_sample_dirs,
        "item_ids": [it["item_id"] for it in items],
        "items_sha256": sha256_json([{k: v for k, v in it.items()} for it in items]),
        "sampling": sampling.as_dict(), "model_provenance": model_prov,
    }
    (out_dir / "sample_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False,
                                                             indent=2), encoding="utf-8")
    print(f"[exp1] {len(items)} items = {len(args.models)} models x {len(CONDITIONS)} "
          f"conditions x {len(fams)} families -> {out_dir}")
    print("[exp1] NEXT: two raters fill rater1.csv / rater2.csv independently (rater_id, "
          "unsafe, response_kind, compliance_degree); adjudicate disagreements in "
          "adjudication.csv; then run scripts/calibrate_judge.py --sample-dir", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
