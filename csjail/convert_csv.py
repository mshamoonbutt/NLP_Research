"""Convert the wide annotation CSV to the long JSONL the pipeline expects.

The annotator template is WIDE — one row per base prompt with EN, UR, and CS
translations all in the same row, plus two reviewers' authenticity and harm
severity scores. The eval pipeline is LONG — one JSONL row per
(prompt × condition). This module bridges them.

Mapping rules:
- category 1..5  -> H1..H5
- style A or B   -> condition CS  (Urdu-English code-switched, main attack)
- style C        -> condition SM  (Nastaliq + Roman script-mixed, RQ5 only)
- base_id        -> f"H{cat}-{prompt_id:03d}"
- Per row, emit EXACTLY THREE long rows: EN, UR, and (CS or SM).
- cs_authenticity / harm_severity = round(mean(reviewer1, reviewer2)) in [1,3].
- Min-authenticity filter: rows whose mean authenticity < `min_authenticity`
  are dropped (default 2.0; spec says discard any prompt rated 1 by both,
  which corresponds to mean <= 1).

Usage:
    python -m csjail.convert_csv \\
        --csv "annotation_template_final - annotation_template.csv.csv" \\
        --out data/csjail_v0.jsonl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from csjail.utils.io import write_jsonl


REQUIRED_COLS = [
    "prompt_id", "category", "style",
    "base_cs_prompt", "en_translation", "ur_translation",
    "authenticity_score1", "authenticity_score2",
    "harm_severity_score1", "harm_severity_score2",
]


def _validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")


def _round_int(x: float, lo: int = 1, hi: int = 3) -> int:
    v = int(round(float(x)))
    return max(lo, min(hi, v))


def _expand_row(row: pd.Series) -> list[dict[str, Any]]:
    """Expand one wide row into 2 or 3 long rows (EN, UR, and CS or SM)."""
    cat = int(row["category"])
    if not 1 <= cat <= 5:
        raise ValueError(f"row prompt_id={row.get('prompt_id')}: invalid category {cat}")
    pid = int(row["prompt_id"])
    base_id = f"H{cat}-{pid:03d}"
    harm_cat = f"H{cat}"
    style = str(row["style"]).strip().upper()
    if style not in ("A", "B", "C"):
        raise ValueError(f"row prompt_id={pid}: invalid style {style!r}")
    cs_cond = "SM" if style == "C" else "CS"

    auth_avg = (float(row["authenticity_score1"])
                + float(row["authenticity_score2"])) / 2
    sev = _round_int(
        (float(row["harm_severity_score1"])
         + float(row["harm_severity_score2"])) / 2
    )
    cs_auth = _round_int(auth_avg)

    en_text = str(row["en_translation"]).strip()
    ur_text = str(row["ur_translation"]).strip()
    cs_text = str(row["base_cs_prompt"]).strip()

    common_meta = {
        "harm_category": harm_cat,
        "base_id": base_id,
        "harm_severity": sev,
        # Provenance fields are passed through but ignored by Prompt schema.
        "_meta": {
            "prompt_id": pid,
            "sub_type": row.get("sub_type"),
            "sub_type_name": row.get("sub_type_name"),
            "author": row.get("author"),
            "date_written": row.get("date_written"),
            "authenticity_avg": auth_avg,
            "authenticity_raw": [
                int(row["authenticity_score1"]),
                int(row["authenticity_score2"]),
            ],
            "harm_severity_raw": [
                int(row["harm_severity_score1"]),
                int(row["harm_severity_score2"]),
            ],
        },
    }

    rows: list[dict[str, Any]] = []
    rows.append({
        "id": f"{base_id}-EN",
        "condition": "EN",
        "prompt": en_text,
        **common_meta,
    })
    rows.append({
        "id": f"{base_id}-UR",
        "condition": "UR",
        "prompt": ur_text,
        **common_meta,
    })
    rows.append({
        "id": f"{base_id}-{cs_cond}-{style}",
        "condition": cs_cond,
        "cs_style": style,
        "cs_authenticity": cs_auth,
        "prompt": cs_text,
        **common_meta,
    })
    return rows


def convert(
    csv_path: str | Path,
    *,
    min_authenticity: float = 2.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read the wide CSV, return (long_rows, report).

    `min_authenticity` filters by MEAN of the two reviewer scores. Default 2.0
    keeps prompts where at least one reviewer rated 2 and the other >= 1
    (i.e. drops only the "both said 1" rows). Set to 2.5 for stricter.
    """
    df = pd.read_csv(csv_path)
    _validate_columns(df)
    n_in = len(df)

    df["auth_avg"] = (df["authenticity_score1"].astype(float)
                      + df["authenticity_score2"].astype(float)) / 2
    dropped = df[df["auth_avg"] < min_authenticity]
    df = df[df["auth_avg"] >= min_authenticity].reset_index(drop=True)
    n_kept = len(df)

    out: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        out.extend(_expand_row(r))

    report = {
        "csv_path": str(csv_path),
        "n_base_prompts_input": n_in,
        "n_base_prompts_kept": n_kept,
        "n_dropped_low_authenticity": int(len(dropped)),
        "min_authenticity_threshold": min_authenticity,
        "n_long_rows": len(out),
        "by_condition": pd.Series([r["condition"] for r in out]).value_counts().to_dict(),
        "by_harm_category": pd.Series([r["harm_category"] for r in out]).value_counts().to_dict(),
        "by_cs_style": pd.Series([r.get("cs_style") for r in out if r.get("cs_style")]).value_counts().to_dict(),
    }
    return out, report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="path to the wide annotation CSV")
    ap.add_argument("--out", required=True, help="output JSONL path")
    ap.add_argument("--min-authenticity", type=float, default=2.0,
                    help="drop base rows where mean reviewer authenticity < this")
    args = ap.parse_args(argv)

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"FAIL: csv not found: {csv_path}", file=sys.stderr)
        return 1

    rows, report = convert(csv_path, min_authenticity=args.min_authenticity)
    write_jsonl(args.out, rows)
    print("[convert] report:")
    import json as _json
    print(_json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[convert] wrote {len(rows)} long rows -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
