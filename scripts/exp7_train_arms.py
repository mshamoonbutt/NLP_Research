#!/usr/bin/env python3
"""Exp 7 — train the comparison arms (GPU only).

Arms (from configs/dpo.yaml): B=English-only DPO, C=CS-only DPO, D=CS+English.
A (untrained) and E (prompt-only) are not trained here. Train Arm B FIRST — it
validates the whole train->eval path on off-the-shelf data.

    # on the 4080, after: pip install -e ".[train]"
    python scripts/exp7_train_arms.py --model qwen25 --arm B \
        --en-pairs outputs/pref_pairs_en.jsonl
    python scripts/exp7_train_arms.py --model qwen25 --arm C \
        --cs-pairs outputs/pref_pairs_cs.jsonl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

from csjail.models import resolve  # noqa: E402
from csjail.utils.io import read_jsonl, write_jsonl  # noqa: E402

DPO_CFG = Path(__file__).resolve().parent.parent / "configs" / "dpo.yaml"


def _merge_pairs(cs_path: str, en_path: str, out_path: str) -> str:
    rows = read_jsonl(cs_path) + read_jsonl(en_path)
    write_jsonl(out_path, rows)
    return out_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model key, e.g. qwen25")
    ap.add_argument("--arm", required=True, choices=["B", "C", "D"])
    ap.add_argument("--cs-pairs", default="outputs/pref_pairs_cs.jsonl")
    ap.add_argument("--en-pairs", default="outputs/pref_pairs_en.jsonl")
    ap.add_argument("--out-root", default="outputs/models")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(DPO_CFG.read_text(encoding="utf-8"))
    dpo_cfg = cfg["dpo"]
    base_hf_id = resolve(args.model).hf_id

    if args.arm == "B":
        pairs = args.en_pairs
    elif args.arm == "C":
        pairs = args.cs_pairs
    else:  # D = CS + English
        out_dir = Path(args.out_root) / f"D_{args.model}"
        out_dir.mkdir(parents=True, exist_ok=True)
        pairs = _merge_pairs(args.cs_pairs, args.en_pairs,
                             str(out_dir / "pairs_merged.jsonl"))

    if not Path(pairs).exists():
        print(f"FAIL: preference pairs not found: {pairs}", file=sys.stderr)
        return 1

    out_dir = str(Path(args.out_root) / f"{args.arm}_{args.model}")
    print(f"[exp7] arm={args.arm} model={args.model} base={base_hf_id}")
    print(f"[exp7] pairs={pairs} -> {out_dir}")

    from csjail.train_dpo import train_dpo  # lazy (GPU deps)

    adapter = train_dpo(base_hf_id, pairs, dpo_cfg, out_dir)
    print(f"[exp7] DONE adapter -> {adapter}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
