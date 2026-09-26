#!/usr/bin/env python3
"""Exp 7 — train one comparison arm for one model (GPU only).

    python scripts/exp7_train_arms.py --model phi3 --arm C  --budget all
    python scripts/exp7_train_arms.py --model phi3 --arm B  --budget all     # matched English
    python scripts/exp7_train_arms.py --model phi3 --arm B_ext --budget all  # external English
    python scripts/exp7_train_arms.py --model phi3 --arm C --budget 50        # N-curve point
    python scripts/exp7_train_arms.py --model phi3 --arm C --pairs-dir outputs/exp6/phi3_ablation_D2 \
        --split-manifest outputs/exp9/ablation_D2/split_manifest.json --tag ablation_D2

Enforced at this boundary (again, independently of Exp 6):
  - every family is in the split's train_pool and shares no group with
    eval/holdout families; no pair from an ablation-excluded domain;
  - rejected answers come from THIS model;
  - B and C use the matched sets with the SAME families and count;
  - the budget is a prefix of the seeded ordering (nested N-curve) and must be
    <= available pairs -- never padded or duplicated;
  - the naturalness gate for C/D must be recorded as passed (--naturalness-csv).
Adapters go to outputs/models/<arm>_<model>[_n<budget>][_<tag>]/ with
training_manifest.json (lineage, split id, pair file hashes, modules).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

from csjail.artifacts import resolve_exp0, sha256_json  # noqa: E402
from csjail.models import resolve  # noqa: E402
from csjail.prefdata import (  # noqa: E402
    PairBuildError, assert_pairs_trainable, load_external_english, pair_counts, read_pairs,
)

ROOT = Path(__file__).resolve().parent.parent


def naturalness_gate(path: str | None) -> dict:
    if not path or not Path(path).exists():
        return {"status": "MISSING"}
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f) if (r.get("naturalness_1to5") or "").strip()]
    if not rows:
        return {"status": "UNRATED"}
    mean = sum(float(r["naturalness_1to5"]) for r in rows) / len(rows)
    clean = [r for r in rows if (r.get("clean_refusal_yes_no") or "").strip().lower()]
    clean_rate = (sum(1 for r in clean if r["clean_refusal_yes_no"].strip().lower() == "yes")
                  / len(clean)) if clean else None
    ok = mean >= 4.0 and clean_rate is not None and clean_rate == 1.0
    return {"status": "PASS" if ok else "FAIL", "mean": mean, "n_rated": len(rows),
            "clean_refusal_rate": clean_rate}


def take_budget(pairs: list[dict], budget: str) -> list[dict]:
    pairs = sorted(pairs, key=lambda p: p.get("order_rank", 0))
    if budget == "all":
        return pairs
    n = int(budget)
    if n > len(pairs):
        raise PairBuildError(f"budget {n} > {len(pairs)} available pairs; not padding")
    return pairs[:n]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--arm", required=True, choices=["B", "C", "B_ext", "D"])
    ap.add_argument("--budget", default="all", help="all | integer (prefix of seeded ordering)")
    ap.add_argument("--pairs-dir", default=None, help="default outputs/exp6/<model>[_<tag>]")
    ap.add_argument("--exp0-dir", default=None)
    ap.add_argument("--split-manifest", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--naturalness-csv", default=None,
                    help="rated naturalness_sample.csv (required for C/D)")
    ap.add_argument("--out-root", default=str(ROOT / "outputs" / "models"))
    args = ap.parse_args(argv)

    cfg = yaml.safe_load((ROOT / "configs" / "dpo.yaml").read_text(encoding="utf-8"))
    spec = resolve(args.model)
    art = resolve_exp0(args.exp0_dir, split_path=args.split_manifest)
    split = art.split
    excl = [split["meta"]["ablation_domain"]] if split["meta"].get("ablation_domain") else []
    pdir = Path(args.pairs_dir or ROOT / "outputs" / "exp6" /
                (args.model + (f"_{args.tag}" if args.tag else "")))
    pman = json.loads((pdir / "pairs_manifest.json").read_text(encoding="utf-8"))
    if pman["model"] != args.model or pman["split_id"] != art.split_id:
        print(f"FAIL: pairs in {pdir} were built for model={pman['model']} "
              f"split={pman['split_id']}, not {args.model}/{art.split_id}", file=sys.stderr)
        return 1

    nat = {"status": "not_required"}
    if args.arm in ("C", "D"):
        nat = naturalness_gate(args.naturalness_csv)
        if nat["status"] != "PASS":
            print(f"FAIL: chosen naturalness gate is {nat}; rate naturalness_sample.csv first",
                  file=sys.stderr)
            return 1

    try:
        if args.arm == "C":
            pairs = take_budget(read_pairs(str(pdir / "pairs_cs_matched.jsonl")), args.budget)
        elif args.arm == "B":
            pairs = take_budget(read_pairs(str(pdir / "pairs_en_matched.jsonl")), args.budget)
            c_fams = [p["base_id"] for p in take_budget(
                read_pairs(str(pdir / "pairs_cs_matched.jsonl")), args.budget)]
            if [p["base_id"] for p in pairs] != c_fams:
                raise PairBuildError("matched B does not use the same families/order as C")
        elif args.arm == "B_ext":
            src = cfg["prefdata"]["external_english_pairs"]
            n_c = len(take_budget(read_pairs(str(pdir / "pairs_cs_matched.jsonl")), args.budget))
            pairs = load_external_english(str(ROOT / src), limit=n_c)
            if len(pairs) < n_c:
                raise PairBuildError(f"only {len(pairs)} external English pairs for budget {n_c}")
        else:  # D
            cs = take_budget(read_pairs(str(pdir / "pairs_cs_matched.jsonl")), args.budget)
            en = take_budget(read_pairs(str(pdir / "pairs_en_matched.jsonl")), args.budget)
            if cfg.get("d_budget", "matched") == "matched":
                half = len(cs) // 2
                pairs = cs[:half] + en[half:len(cs)]   # same total N as C, disjoint families
            else:
                pairs = cs + en
        assert_pairs_trainable(pairs, split, model=args.model, exclude_domains=excl)
    except (PairBuildError, ValueError, FileNotFoundError) as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1

    name = f"{args.arm}_{args.model}" + (f"_n{args.budget}" if args.budget != "all" else "") \
        + (f"_{args.tag}" if args.tag else "")
    out_dir = Path(args.out_root) / name
    print(f"[exp7] arm={args.arm} model={args.model} n={len(pairs)} -> {out_dir}")
    from csjail.train_dpo import train_dpo  # lazy (GPU deps)

    train_dpo(spec.hf_id, pairs, cfg["dpo"], str(out_dir), revision=spec.revision,
              manifest_extra={
                  "arm": args.arm, "model_key": args.model, "budget": args.budget,
                  "split_id": art.split_id, "split_scheme": split["meta"]["scheme"],
                  "dataset_version": art.dataset_version, "excluded_domains": excl,
                  "pairs_dir": str(pdir), "pairs_manifest_created": pman["created_utc"],
                  "pair_families": [p.get("base_id") for p in pairs],
                  "pair_lineage_sha256": sha256_json([p.get("rejected_lineage") for p in pairs]),
                  "pair_counts": pair_counts(pairs), "naturalness_gate": nat,
                  "d_budget": cfg.get("d_budget") if args.arm == "D" else None})
    print(f"[exp7] DONE -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
