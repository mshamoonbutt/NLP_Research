"""Quick summary of one or more result JSONL files.

Usage:
    python scripts/peek.py results/baseline_<timestamp>/*.jsonl
    python scripts/peek.py results/baseline_<timestamp>/phi3_EN.jsonl
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python scripts/peek.py <result.jsonl> [more.jsonl ...]",
              file=sys.stderr)
        return 1

    # Expand globs ourselves (Windows shells don't always)
    paths: list[Path] = []
    for arg in sys.argv[1:]:
        matches = sorted(glob.glob(arg))
        if matches:
            paths.extend(Path(m) for m in matches)
        elif Path(arg).exists():
            paths.append(Path(arg))
        else:
            print(f"[warn] no match for {arg}", file=sys.stderr)

    if not paths:
        return 1

    header = (f"{'file':<22s}  {'ASR':>5s}  {'hit':>5s}  "
              f"{'95% CI':>15s}  {'part':>4s}  {'ref':>4s}  {'pf':>3s}")
    print(header)
    print("-" * len(header))
    for path in paths:
        try:
            with path.open(encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
            if len(rows) < 2:
                print(f"{path.name:<22s}  (no summary line)")
                continue
            s = rows[1]
            name = path.stem
            print(f"{name:<22s}  {s['asr']:>5.3f}  "
                  f"{s['n_full_comply']:>2d}/{s['n']:<3d}  "
                  f"[{s['ci_lo']:.3f},{s['ci_hi']:.3f}]  "
                  f"{s['n_partial']:>4d}  {s['n_refuse']:>4d}  "
                  f"{s['n_parse_fail']:>3d}")
        except Exception as e:
            print(f"{path.name:<22s}  [error: {e}]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
