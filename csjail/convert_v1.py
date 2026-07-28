"""Convert the extended wide annotation template to the long JSONL the pipeline
expects (the CS/EN/RU/UR 10-category design).

The extended template (`annotation_template.xlsx` / `.csv`) is WIDE — one row per
base prompt carrying all four condition texts:

    prompt_id, category (1..10), category_name, author,
    base_cs_prompt, en_translation, ur_translation, roman_urdu_translation,
    harm_severity_score (optional, sparsely annotated)

The eval pipeline is LONG — one row per (prompt x condition). This module bridges
them, emitting EXACTLY FOUR long rows per base prompt: CS, EN, UR, RU.

Mapping rules:
- category c (1..10)          -> harm_category  f"C{c:02d}"  (C01..C10)
- base_id                     -> f"C{c:02d}-{prompt_id:04d}"
- base_cs_prompt              -> condition CS
- en_translation             -> condition EN
- ur_translation             -> condition UR
- roman_urdu_translation     -> condition RU
- harm_severity_score        -> harm_severity (int 1..3) when present, else absent
  (NEVER imputed — 82% of the template is unrated and severity is descriptive).
- cs_style / cs_authenticity  -> not in this template; left absent (the schema
  now accepts CS rows without a style label).

category_name and author are preserved under `_meta` for provenance.

Usage:
    python -m csjail.convert_v1 --input annotation_template.xlsx \\
        --out data/csjail_v1.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from csjail.utils.io import write_jsonl

# Wide-template columns and the condition each maps to.
CONDITION_COLUMNS: dict[str, str] = {
    "base_cs_prompt": "CS",
    "en_translation": "EN",
    "ur_translation": "UR",
    "roman_urdu_translation": "RU",
}
REQUIRED_COLS = ["prompt_id", "category", *CONDITION_COLUMNS.keys()]


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in (".xlsx", ".xlsm", ".xls"):
        return pd.read_excel(path, sheet_name=0)
    return pd.read_csv(path)


def _validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"input missing required columns: {missing} (have {list(df.columns)})"
        )


def _clean(x: Any) -> str:
    return str(x).strip()


def _severity(row: pd.Series) -> int | None:
    v = row.get("harm_severity_score")
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    try:
        iv = int(round(float(v)))
    except (TypeError, ValueError):
        return None
    return iv if 1 <= iv <= 3 else None


def _expand_row(row: pd.Series) -> list[dict[str, Any]]:
    cat = int(row["category"])
    if not 1 <= cat <= 99:
        raise ValueError(f"prompt_id={row.get('prompt_id')}: invalid category {cat}")
    pid = int(row["prompt_id"])
    base_id = f"C{cat:02d}-{pid:04d}"
    harm_cat = f"C{cat:02d}"
    sev = _severity(row)

    meta = {
        "prompt_id": pid,
        "category_name": row.get("category_name"),
        "author": row.get("author"),
    }
    common = {"harm_category": harm_cat, "base_id": base_id, "_meta": meta}
    if sev is not None:
        common["harm_severity"] = sev

    out: list[dict[str, Any]] = []
    for col, cond in CONDITION_COLUMNS.items():
        text = _clean(row[col])
        if not text or text.lower() == "nan":
            raise ValueError(f"{base_id}: empty {cond} text (column {col!r})")
        out.append({
            "id": f"{base_id}-{cond}",
            "condition": cond,
            "prompt": text,
            **common,
        })
    return out


def convert(
    input_path: str | Path,
    *,
    drop_duplicates: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read the wide template, return (long_rows, report).

    drop_duplicates=True removes base prompts whose text is identical across ALL
    four conditions to an earlier prompt (keeping the first / lowest prompt_id).
    Partial matches (e.g. only the EN translation coincides) are NOT removed.
    """
    path = Path(input_path)
    df = _read_table(path)
    _validate_columns(df)
    n_in = len(df)

    dup_pid = df["prompt_id"].duplicated().sum()
    if dup_pid:
        raise ValueError(f"duplicate prompt_id values in input: {dup_pid}")

    out: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        out.extend(_expand_row(r))

    # Drop full-duplicate base prompts (identical in all 4 conditions).
    removed_base_ids: list[str] = []
    if drop_duplicates:
        by_base: dict[str, dict[str, str]] = {}
        order: list[str] = []
        for r in out:
            if r["base_id"] not in by_base:
                order.append(r["base_id"])
            by_base.setdefault(r["base_id"], {})[r["condition"]] = r["prompt"]
        seen_sig: dict[tuple, str] = {}
        drop: set[str] = set()
        for b in order:  # preserves input order -> keeps first occurrence
            c = by_base[b]
            sig = (c.get("CS"), c.get("EN"), c.get("UR"), c.get("RU"))
            if sig in seen_sig:
                drop.add(b)
            else:
                seen_sig[sig] = b
        if drop:
            removed_base_ids = sorted(drop)
            out = [r for r in out if r["base_id"] not in drop]

    # Data-quality probe: byte-identical texts across conditions (esp. CS==RU,
    # which dilutes the core CS-vs-RU code-switching contrast).
    by_base: dict[str, dict[str, str]] = {}
    for r in out:
        by_base.setdefault(r["base_id"], {})[r["condition"]] = r["prompt"]
    identical = {"CS==RU": 0, "CS==EN": 0, "RU==EN": 0, "CS==UR": 0}
    for conds in by_base.values():
        if conds.get("CS") == conds.get("RU"):
            identical["CS==RU"] += 1
        if conds.get("CS") == conds.get("EN"):
            identical["CS==EN"] += 1
        if conds.get("RU") == conds.get("EN"):
            identical["RU==EN"] += 1
        if conds.get("CS") == conds.get("UR"):
            identical["CS==UR"] += 1

    n_sev = sum(1 for r in out if "harm_severity" in r)
    report = {
        "input_path": str(path),
        "n_base_prompts_input": n_in,
        "n_base_prompts_kept": n_in - len(removed_base_ids),
        "n_full_duplicates_removed": len(removed_base_ids),
        "removed_base_ids": removed_base_ids,
        "n_long_rows": len(out),
        "by_condition": pd.Series([r["condition"] for r in out]).value_counts().to_dict(),
        "by_harm_category": pd.Series([r["harm_category"] for r in out]).value_counts().sort_index().to_dict(),
        "n_rows_with_severity": n_sev,
        "n_rows_without_severity": len(out) - n_sev,
        "identical_condition_pairs_by_base_id": identical,
    }
    return out, report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="path to the wide annotation template (.xlsx or .csv)")
    ap.add_argument("--out", required=True, help="output long JSONL path")
    ap.add_argument("--drop-duplicates", action="store_true",
                    help="remove base prompts identical in all 4 conditions "
                         "(keeps the first occurrence)")
    args = ap.parse_args(argv)

    path = Path(args.input)
    if not path.exists():
        print(f"FAIL: input not found: {path}", file=sys.stderr)
        return 1

    rows, report = convert(path, drop_duplicates=args.drop_duplicates)
    write_jsonl(args.out, rows)
    print("[convert_v1] report:")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["identical_condition_pairs_by_base_id"]["CS==RU"]:
        n = report["identical_condition_pairs_by_base_id"]["CS==RU"]
        print(f"[convert_v1] NOTE: {n} base prompts have CS == RU (identical text). "
              "These dilute the CS-vs-RU isolation contrast; see Exp0 CMI report.",
              file=sys.stderr)
    print(f"[convert_v1] wrote {len(rows)} long rows -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
