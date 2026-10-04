#!/usr/bin/env python3
"""Build the fixed capability subsets for Exp 8 (configs/capability.yaml).

    python scripts/prepare_capability_sets.py                 # MMLU only
    python scripts/prepare_capability_sets.py --urdummlu-dataset <hf-id>
    python scripts/prepare_capability_sets.py --urdummlu-file <local.jsonl|csv>

Writes data/capability/mmlu_500.jsonl and data/capability/urdummlu_300.jsonl:
one JSONL row per item {id, question, choices, answer_idx, subject}, a fixed
seeded sample drawn ONCE and then frozen (the paper preserves item
identifiers across arms; re-running refuses to overwrite). A manifest records
source, split, seed and output hashes. Needs `datasets` (pip install -e
'.[train]') and network for the HF sources.

UrduMMLU's distribution channel is the authors' repository
(github.com/mbzuai-nlp/UrduMMLU); pass whichever HF id or exported file you
obtained it as. Items are used verbatim -- no translation or editing.
"""
from __future__ import annotations

import argparse
import csv as csvmod
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from csjail.utils.io import sha256_file, write_jsonl, write_text_lf  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "capability"
LETTERS = "ABCDEFGHIJ"


def normalize(r: dict, i: int, *, source: str) -> dict:
    """Map common MCQ schemas to {id, question, choices, answer_idx, subject}."""
    q = r.get("question") or r.get("Question")
    choices = r.get("choices") or r.get("options") or r.get("Options")
    if choices is None and all(k in r for k in ("A", "B", "C", "D")):
        choices = [r["A"], r["B"], r["C"], r["D"]]
    ans = r.get("answer") if r.get("answer") is not None else r.get("Answer")
    if q is None or choices is None or ans is None:
        raise KeyError(f"unrecognized schema; row keys: {sorted(r.keys())[:15]}")
    choices = list(choices)
    if isinstance(ans, int) or (isinstance(ans, str) and ans.strip().isdigit()):
        idx = int(ans)
    else:
        idx = LETTERS.index(str(ans).strip().upper()[0])
    if not 0 <= idx < len(choices):
        raise ValueError(f"row {i}: answer index {idx} out of range for {len(choices)} choices")
    subject = r.get("subject") or r.get("Subject") or r.get("category")
    return {"id": f"{source}:{subject or 'na'}:{i}", "question": q, "choices": choices,
            "answer_idx": idx, "subject": subject}


def build(rows, n: int, seed: int, *, source: str) -> list[dict]:
    items = [normalize(dict(r), i, source=source) for i, r in enumerate(rows)]
    if len(items) > n:
        items = random.Random(seed).sample(items, n)
        items.sort(key=lambda x: x["id"])
    return items


def write_frozen(path: Path, items: list[dict], manifest: dict) -> bool:
    if path.exists():
        print(f"[capability] {path} already exists; NOT overwritten (frozen subset). "
              "Delete it manually only before any arm has been evaluated.", file=sys.stderr)
        return False
    write_jsonl(path, items)
    manifest.update({"n_items": len(items), "sha256": sha256_file(path),
                     "item_ids_first5": [it["id"] for it in items[:5]]})
    write_text_lf(path.with_suffix(".manifest.json"),
                  json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"[capability] wrote {path} ({len(items)} items, sha256 {manifest['sha256'][:12]})")
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mmlu-dataset", default="cais/mmlu")
    ap.add_argument("--mmlu-config", default="all")
    ap.add_argument("--mmlu-split", default="test")
    ap.add_argument("--mmlu-n", type=int, default=500)
    ap.add_argument("--urdummlu-dataset", default=None, help="HF id of UrduMMLU, if available")
    ap.add_argument("--urdummlu-file", default=None, help="local JSONL/CSV export of UrduMMLU")
    ap.add_argument("--urdummlu-split", default="test")
    ap.add_argument("--urdummlu-n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset
    except ImportError:
        print("FAIL: pip install -e '.[train]' (needs `datasets`)", file=sys.stderr)
        return 1

    ds = load_dataset(args.mmlu_dataset, args.mmlu_config, split=args.mmlu_split)
    items = build(ds, args.mmlu_n, args.seed, source="mmlu")
    write_frozen(OUT_DIR / "mmlu_500.jsonl", items, {
        "source_dataset": args.mmlu_dataset, "config": args.mmlu_config,
        "split": args.mmlu_split, "seed": args.seed, "source_rows": len(ds)})

    if args.urdummlu_file:
        p = Path(args.urdummlu_file)
        if p.suffix.lower() == ".csv":
            rows = list(csvmod.DictReader(p.open(encoding="utf-8-sig")))
        else:
            rows = [json.loads(line) for line in p.open(encoding="utf-8") if line.strip()]
        items = build(rows, args.urdummlu_n, args.seed, source="urdummlu")
        write_frozen(OUT_DIR / "urdummlu_300.jsonl", items, {
            "source_file": str(p), "source_sha256": sha256_file(p), "seed": args.seed,
            "source_rows": len(rows)})
    elif args.urdummlu_dataset:
        ds = load_dataset(args.urdummlu_dataset, split=args.urdummlu_split)
        items = build(ds, args.urdummlu_n, args.seed, source="urdummlu")
        write_frozen(OUT_DIR / "urdummlu_300.jsonl", items, {
            "source_dataset": args.urdummlu_dataset, "split": args.urdummlu_split,
            "seed": args.seed, "source_rows": len(ds)})
    else:
        print("[capability] UrduMMLU skipped: pass --urdummlu-dataset or --urdummlu-file "
              "(see github.com/mbzuai-nlp/UrduMMLU). Exp 8 reports its block NOT_RUN "
              "until data/capability/urdummlu_300.jsonl exists.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
