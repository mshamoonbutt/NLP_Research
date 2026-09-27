#!/usr/bin/env python3
"""Exp 9 — ablation set-up (CPU). Training/evaluation reuse Exp 6-8 scripts.

Unseen-domain ablation (NOT free: it needs fresh training runs)
    python scripts/exp9_ablations.py domain --domain D2 --attest-chosen-before-outcomes
  -> outputs/exp9/ablation_D2/split_manifest.json: the domain's train_pool
     families become `domain_supplementary` (withheld from ALL supervision:
     pairs, exemplars, development data); eval_main is unchanged, so the
     domain's frozen held-out subset is evaluated. Then, from the ORIGINAL base:
       exp6 --split-manifest <that> --tag ablation_D2
       exp7 --split-manifest <that> --tag ablation_D2 (arms needed)
       exp8 --split-manifest <that> --tag ablation_D2
     Report the domain's eval_main subset as the primary unseen-domain result;
     domain_supplementary items only as a separately identified extra set.
     An adapter trained under the main manifest must never be relabelled.

Data-efficiency N-curve (nested prefixes of one seeded ordering)
    python scripts/exp9_ablations.py ncurve --model phi3
  -> prints the runnable budgets from pairs_manifest.json ({50,100,200,all}
     up to the available count) and the exact exp7/exp8 commands.

Condition transfer (CS-trained model on RU/UR) is read from Exp 8 results.
The beta sweep is cut.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from csjail.artifacts import resolve_exp0  # noqa: E402
from csjail.splits import make_domain_ablation  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def cmd_domain(args) -> int:
    art = resolve_exp0(args.exp0_dir)
    try:
        abl = make_domain_ablation(art.split, args.domain,
                                   attested_before_outcomes=args.attest_chosen_before_outcomes)
    except ValueError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1
    out = ROOT / "outputs" / "exp9" / f"ablation_{args.domain}"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "split_manifest.json"
    if path.exists() and not args.overwrite:
        print(f"FAIL: {path} exists (frozen); pass --overwrite deliberately", file=sys.stderr)
        return 1
    path.write_text(json.dumps(abl, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[exp9] {args.domain} ablation split {abl['meta']['split_id']}: {abl['meta']['counts']}")
    print(f"[exp9] -> {path}\n[exp9] next: exp6/exp7/exp8 with --split-manifest {path} "
          f"--tag ablation_{args.domain} (fresh adapters from the base model)")
    return 0


def cmd_ncurve(args) -> int:
    pdir = ROOT / "outputs" / "exp6" / (args.model + (f"_{args.tag}" if args.tag else ""))
    man = json.loads((pdir / "pairs_manifest.json").read_text(encoding="utf-8"))
    cs = (man.get("budgets") or {}).get("CS") or {}
    print(f"[exp9] {args.model}: validated CS pairs = {cs.get('n_available')}; "
          f"status = {cs.get('status')} (primary design: C vs B_ext at equal N)")
    for b in cs.get("runnable", []):
        if b == "all":
            continue
        print(f"python scripts/exp7_train_arms.py --model {args.model} --arm C --budget {b} "
              "--naturalness-csv <rated csv>")
        print(f"python scripts/exp7_train_arms.py --model {args.model} --arm B_ext --budget {b}")
    print("then evaluate each budget with scripts/exp8_posteval.py --tag n<budget> "
          "(adapters are named <arm>_<model>_n<budget>)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("domain")
    d.add_argument("--domain", required=True)
    d.add_argument("--attest-chosen-before-outcomes", action="store_true")
    d.add_argument("--exp0-dir", default=None)
    d.add_argument("--overwrite", action="store_true")
    n = sub.add_parser("ncurve")
    n.add_argument("--model", required=True)
    n.add_argument("--tag", default=None)
    args = ap.parse_args(argv)
    return cmd_domain(args) if args.cmd == "domain" else cmd_ncurve(args)


if __name__ == "__main__":
    sys.exit(main())
