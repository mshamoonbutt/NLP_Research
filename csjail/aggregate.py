"""Combine per-run JSONL outputs into a publication-ready CSV.

Usage:
    python -m csjail.aggregate results/*.jsonl --out results/headline_table.csv
    python -m csjail.aggregate results/*.jsonl --out tab.csv --with-mcnemar
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import pandas as pd

from csjail.asr import mcnemar_pairs
from csjail.utils.io import iter_jsonl


def _load_one_run(path: Path) -> tuple[dict, dict, list[dict]]:
    """Returns (provenance_header, summary_record, per_prompt_rows)."""
    rows = list(iter_jsonl(path))
    if len(rows) < 2:
        raise ValueError(f"{path}: too few lines (need provenance + summary)")
    header = rows[0]
    summary = rows[1]
    prompts = rows[2:]
    if header.get("kind") != "provenance":
        raise ValueError(f"{path}: line 1 is not a provenance header")
    if summary.get("kind") != "summary":
        raise ValueError(f"{path}: line 2 is not a summary record")
    return header, summary, prompts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="results/*.jsonl paths or globs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--with-mcnemar", action="store_true",
                    help="also write headline_mcnemar.csv next to --out")
    args = ap.parse_args(argv)

    # Expand globs (Windows shells often don't)
    paths: list[Path] = []
    for pat in args.inputs:
        matches = [Path(p) for p in glob.glob(pat)]
        if not matches and Path(pat).exists():
            matches = [Path(pat)]
        paths.extend(matches)
    paths = sorted(set(paths))
    if not paths:
        print("[aggregate] no input files matched", file=sys.stderr)
        return 1
    print(f"[aggregate] reading {len(paths)} run files")

    # Headline table: one row per (model, condition, harm_category)
    headline_rows: list[dict] = []
    per_prompt_by_run: dict[tuple[str, str], list[dict]] = {}
    for p in paths:
        header, summary, prompts = _load_one_run(p)
        model = header["model"]["key"]
        cond = header["args"]["condition"]
        per_prompt_by_run[(model, cond)] = prompts

        overall_row = {
            "model": model,
            "condition": cond,
            "harm_category": "ALL",
            "n": summary["n"],
            "asr": summary["asr"],
            "ci_lo": summary["ci_lo"],
            "ci_hi": summary["ci_hi"],
            "n_full_comply": summary["n_full_comply"],
            "n_partial": summary["n_partial"],
            "n_refuse": summary["n_refuse"],
            "n_parse_fail": summary["n_parse_fail"],
            "source_file": p.name,
        }
        headline_rows.append(overall_row)
        for c in summary.get("per_harm_category", []):
            headline_rows.append({
                "model": model,
                "condition": cond,
                "harm_category": c["harm_category"],
                "n": c["n"],
                "asr": c["asr"],
                "ci_lo": c["ci_lo"],
                "ci_hi": c["ci_hi"],
                "n_full_comply": c["n_full_comply"],
                "n_partial": c["n_partial"],
                "n_refuse": c["n_refuse"],
                "n_parse_fail": c["n_parse_fail"],
                "source_file": p.name,
            })

    df = pd.DataFrame(headline_rows)
    df = df.sort_values(["model", "harm_category", "condition"]).reset_index(drop=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, float_format="%.4f")
    print(f"[aggregate] headline -> {out_path}  ({len(df)} rows)")

    # McNemar pairwise table — isolation contrasts (extended CS/EN/RU/UR design):
    #   CS-RU (code-switching), RU-UR (script), RU-EN (language), CS-EN (headline).
    if args.with_mcnemar:
        mn_rows: list[dict] = []
        comparisons = [("CS", "RU"), ("RU", "UR"), ("RU", "EN"), ("CS", "EN")]
        models = sorted({m for (m, _c) in per_prompt_by_run})
        for model in models:
            judgments_by_cond: dict[str, list[tuple[str, int]]] = {}
            for (m, cond), prompts in per_prompt_by_run.items():
                if m != model:
                    continue
                judgments_by_cond[cond] = [
                    (r["base_id"], r["judge_label"]) for r in prompts
                ]
            results = mcnemar_pairs(judgments_by_cond, comparisons=comparisons)
            for r in results:
                mn_rows.append({
                    "model": model,
                    "cond_a": r.cond_a,
                    "cond_b": r.cond_b,
                    "n_pairs": r.n_pairs,
                    "b_only_a_hit": r.b,
                    "c_only_b_hit": r.c,
                    "statistic": r.statistic,
                    "pvalue": r.pvalue,
                    "test_used": r.test_used,
                })
        mn_path = out_path.with_name(out_path.stem + "_mcnemar.csv")
        pd.DataFrame(mn_rows).to_csv(mn_path, index=False, float_format="%.6g")
        print(f"[aggregate] mcnemar -> {mn_path}  ({len(mn_rows)} rows)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
