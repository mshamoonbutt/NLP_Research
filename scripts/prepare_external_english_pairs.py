#!/usr/bin/env python3
"""Build the external English preference set for arm B_ext (GPU/any host with
`datasets` + network; NOT yet executed in this repo).

    python scripts/prepare_external_english_pairs.py --n 1000

Default source: PKU-Alignment/PKU-SafeRLHF (check its dataset card for the
licence and terms before use). A row becomes a pair only when exactly one of
its two responses is labelled safe: chosen = the safe response, rejected =
the unsafe one. Rows are shuffled with a fixed seed and the first --n kept, so
B_ext at budget N (Exp 7) uses the first N pairs. These pairs are OFF-POLICY
(someone else's responses) and from a different source: B_ext vs C compares
training recipes, not training language alone.
Output: data/pref_pairs_en_external.jsonl (+ .manifest.json).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from csjail.utils.io import sha256_file, write_jsonl, write_text_lf  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def to_pairs(rows, *, source: str) -> list[dict]:
    out = []
    for i, r in enumerate(rows):
        s0, s1 = r.get("is_response_0_safe"), r.get("is_response_1_safe")
        if s0 is None or s1 is None or bool(s0) == bool(s1) or not r.get("prompt"):
            continue
        safe, unsafe = ("response_0", "response_1") if s0 else ("response_1", "response_0")
        if not (r.get(safe) or "").strip() or not (r.get(unsafe) or "").strip():
            continue
        out.append({"prompt": r["prompt"], "chosen": r[safe], "rejected": r[unsafe],
                    "source": "external", "source_dataset": source, "source_row": i,
                    "base_id": None, "domain_id": None})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="PKU-Alignment/PKU-SafeRLHF")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(ROOT / "data" / "pref_pairs_en_external.jsonl"))
    args = ap.parse_args(argv)
    try:
        from datasets import load_dataset
    except ImportError:
        print("FAIL: pip install -e '.[train]' (needs `datasets`)", file=sys.stderr)
        return 1
    ds = load_dataset(args.dataset, split=args.split)
    pairs = to_pairs(ds, source=f"{args.dataset}:{args.split}")
    random.Random(args.seed).shuffle(pairs)
    pairs = pairs[: args.n]
    write_jsonl(args.out, pairs)
    write_text_lf(Path(args.out).with_suffix(".manifest.json"), json.dumps({
        "source": args.dataset, "split": args.split, "seed": args.seed, "n": len(pairs),
        "rule": "exactly one response labelled safe; chosen=safe, rejected=unsafe",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sha256": sha256_file(args.out)}, indent=2))
    print(f"[ext-en] {len(pairs)} pairs -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
