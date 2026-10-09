#!/usr/bin/env python3
"""Build the external English preference set for arm B_ext (needs `datasets` + network).

    python scripts/prepare_external_english_pairs.py --n 1000 --ablation-domain D6

Source: PKU-Alignment/PKU-SafeRLHF at a pinned revision (licence CC BY-NC 4.0: research use,
cite it). A row becomes a pair only when exactly one of its two responses is labelled safe:
chosen = the safe response, rejected = the unsafe one. Then:
  * leakage screen: pairs whose prompt reaches token-Jaccard >= 0.5 with the English prompt of
    any held-out (eval_main) family are dropped (the Exp 0 screen and threshold; lexical only);
  * seeded shuffle, first --n kept -> data/pref_pairs_en_external.jsonl, so B_ext at budget N
    uses the first N pairs;
  * --ablation-domain D6: the same order without pairs whose unsafe response PKU tags with that
    domain's categories (D6 -> Cybercrime) -> data/pref_pairs_en_external_noD6.jsonl, so the
    unseen-domain run withholds the domain from B_ext as from C.
These pairs are OFF-POLICY (other models' responses) and from a different source: B_ext vs C
compares training recipes, not training language alone. Manifests record every count.
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
REVISION = "9421ffafec3fa40a1f1a7d567b4d525079477ecb"   # PKU-SafeRLHF main, 2024-10-18
DOMAIN_TO_PKU = {"D6": ("Cybercrime",)}                  # our domain -> PKU harm categories


def to_pairs(rows, *, source: str) -> list[dict]:
    out = []
    for i, r in enumerate(rows):
        s0, s1 = r.get("is_response_0_safe"), r.get("is_response_1_safe")
        if s0 is None or s1 is None or bool(s0) == bool(s1) or not r.get("prompt"):
            continue
        safe, unsafe = ("response_0", "response_1") if s0 else ("response_1", "response_0")
        if not (r.get(safe) or "").strip() or not (r.get(unsafe) or "").strip():
            continue
        cats = sorted(k for k, v in (r.get(f"{unsafe}_harm_category") or {}).items() if v)
        out.append({"prompt": r["prompt"], "chosen": r[safe], "rejected": r[unsafe],
                    "source": "external", "source_dataset": source, "source_row": i,
                    "rejected_harm_categories": cats, "rejected_severity": r.get(f"{unsafe}_severity_level"),
                    "base_id": None, "domain_id": None})
    return out


def heldout_overlap(pairs: list[dict], heldout_en: list[str], threshold: float) -> set[int]:
    """Indices of pairs whose prompt reaches `threshold` token-Jaccard with a held-out prompt."""
    from csjail.qa import normalize_for_dup

    held = [frozenset(normalize_for_dup(t).split()) for t in heldout_en]
    hit = set()
    for i, p in enumerate(pairs):
        a = frozenset(normalize_for_dup(p["prompt"]).split())
        if a and any(len(a & b) / len(a | b) >= threshold for b in held if b):
            hit.add(i)
    return hit


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="PKU-Alignment/PKU-SafeRLHF")
    ap.add_argument("--revision", default=REVISION)
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--screen-threshold", type=float, default=0.5)
    ap.add_argument("--ablation-domain", default=None, choices=sorted(DOMAIN_TO_PKU))
    ap.add_argument("--exp0-dir", default=None)
    ap.add_argument("--out", default=str(ROOT / "data" / "pref_pairs_en_external.jsonl"))
    args = ap.parse_args(argv)
    try:
        from datasets import load_dataset
    except ImportError:
        print("FAIL: pip install -e '.[train]' (needs `datasets`)", file=sys.stderr)
        return 1
    from csjail.artifacts import resolve_exp0
    from csjail.splits import eval_families

    art = resolve_exp0(args.exp0_dir)
    ev = eval_families(art.split)
    heldout_en = [r.prompt for r in art.load_rows() if r.base_id in ev and r.condition == "EN"]
    ds = load_dataset(args.dataset, split=args.split, revision=args.revision)
    pairs = to_pairs(ds, source=f"{args.dataset}@{args.revision[:12]}:{args.split}")
    hit = heldout_overlap(pairs, heldout_en, args.screen_threshold)
    screened = [p for i, p in enumerate(pairs) if i not in hit]
    random.Random(args.seed).shuffle(screened)
    base = {"source": args.dataset, "revision": args.revision, "licence": "CC BY-NC 4.0", "split": args.split,
            "seed": args.seed, "rows": len(ds), "pairs_exactly_one_safe": len(pairs),
            "rule": "exactly one response labelled safe; chosen=safe, rejected=unsafe",
            "leakage_screen": {"method": "token-Jaccard vs held-out EN prompts (csjail.qa.normalize_for_dup)",
                               "threshold": args.screen_threshold, "n_heldout_prompts": len(heldout_en),
                               "dropped": len(hit)},
            "created_utc": datetime.now(timezone.utc).isoformat()}
    outs = [(Path(args.out), screened[: args.n], {})]
    if args.ablation_domain:
        cats = set(DOMAIN_TO_PKU[args.ablation_domain])
        kept = [p for p in screened if not cats & set(p["rejected_harm_categories"])]
        outs.append((Path(args.out).with_name(f"{Path(args.out).stem}_no{args.ablation_domain}.jsonl"),
                     kept[: args.n], {"ablation_domain": args.ablation_domain, "excluded_pku_categories": sorted(cats),
                                      "dropped_for_domain": len(screened) - len(kept)}))
    for path, ps, extra in outs:
        write_jsonl(path, ps)
        write_text_lf(path.with_suffix(".manifest.json"), json.dumps(
            {**base, **extra, "n": len(ps), "sha256": sha256_file(path)}, indent=2) + "\n")
        print(f"[ext-en] {len(ps)} pairs -> {path}")
    print(f"[ext-en] {len(pairs)} exactly-one-safe pairs; {len(hit)} dropped by the held-out screen")
    return 0


if __name__ == "__main__":
    sys.exit(main())
