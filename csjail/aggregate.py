"""Load, check and summarize evaluation results.

    python -m csjail.aggregate outputs/exp2/main [more run dirs] [--allow-debug]

Joins are refused when runs are incompatible: different dataset versions,
split ids or judge fingerprints, duplicate (model, arm, row, sample, sampling)
keys, or debug runs (unless --allow-debug). Every table row reports
n_planned / n_scored / n_missing, primary unsafe ASR (full + partial) with a
domain-stratified family-bootstrap CI and missing-data bounds; when the judge's
Exp 1 manifest matches the runs, also ASR corrected for that judge's measured
error per language (asr.corrected_asr, condition totals only); strict
(full-only) ASR, behaviour rates over n_planned, and for condition totals an
equal-weight six-domain macro ASR.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

import yaml

from csjail.artifacts import sha256_json
from csjail.asr import behavior_rates, compute_asr, corrected_asr, family_map, macro_asr
from csjail.outcomes import behavior, primary_unsafe, strict_unsafe
from csjail.utils.io import read_jsonl

ROOT = Path(__file__).resolve().parent.parent


class IncompatibleRunsError(Exception):
    pass


def load_results(run_dirs: Iterable[str | Path], *, allow_debug: bool = False) -> list[dict]:
    recs: list[dict] = []
    for d in run_dirs:
        d = Path(d)
        paths = sorted(d.glob("results*.jsonl")) if d.is_dir() else [d]
        if not paths:
            raise IncompatibleRunsError(f"no results*.jsonl in {d}")
        for path in paths:
            for r in read_jsonl(path):
                if r.get("kind") == "result":
                    r.setdefault("_source", str(path))
                    recs.append(r)
    check_compatible(recs, allow_debug=allow_debug)
    return recs


def result_key(r: dict) -> tuple:
    return (r["model"], r.get("arm", "A"), r["row_id"], r.get("sample_index", 0),
            sha256_json(r.get("sampling")))


def check_compatible(recs: list[dict], *, allow_debug: bool = False) -> None:
    if not recs:
        raise IncompatibleRunsError("no result records")
    dbg = sorted({r["_source"] for r in recs if r.get("run_debug")})
    if dbg and not allow_debug:
        raise IncompatibleRunsError(f"debug runs cannot enter paper tables: {dbg[:3]}")
    for field in ("dataset_version", "split_id"):
        vals = {r.get(field) for r in recs}
        if len(vals) > 1:
            raise IncompatibleRunsError(f"mixed {field}: {sorted(map(str, vals))}")
    fps = {r.get("judge_fingerprint_id") for r in recs if r.get("judge_fingerprint_id")}
    if len(fps) > 1:
        raise IncompatibleRunsError(f"mixed judge fingerprints: {sorted(fps)}")
    seen: dict[tuple, str] = {}
    for r in recs:
        k = result_key(r)
        if k in seen:
            raise IncompatibleRunsError(f"duplicate result {k[:4]} in {seen[k]} and {r['_source']}")
        seen[k] = r["_source"]


def outcome_maps(recs: list[dict], *, predicate=primary_unsafe
                 ) -> dict[tuple[str, str], dict[str, dict[str, Optional[bool]]]]:
    """{(model, arm): {condition: {family: outcome}}} for single-sample runs.
    Multi-sample records must be aggregated first (csjail.robustness)."""
    multi = [r for r in recs if r.get("sample_index", 0) != 0 or
             int((r.get("sampling") or {}).get("n", 1)) > 1]
    if multi:
        raise IncompatibleRunsError("multi-sample records: aggregate draws per family "
                                    "(csjail.robustness) before paired tests")
    grouped: dict[tuple, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in recs:
        grouped[(r["model"], r.get("arm", "A"))][r["condition"]].append(
            (r["base_id"], predicate(r)))
    return {k: {c: family_map(v, what=f"{k}/{c}") for c, v in conds.items()}
            for k, conds in grouped.items()}


def judge_error_counts(manifest_path: Path, recs: list[dict]) -> Optional[dict]:
    """tp/fn/tn/fp of the judge that scored `recs`, from its Exp 1 manifest, keyed
    "model|condition" (preferred: the judge's false-alarm rate differs by model) and
    "condition" (fallback); None when there is no manifest for exactly that judge."""
    if not manifest_path.exists():
        return None
    man = json.loads(manifest_path.read_text(encoding="utf-8"))
    fps = {r.get("judge_fingerprint_id") for r in recs if r.get("judge_fingerprint_id")}
    per = (man.get("result") or {}).get("per_language")
    if fps != {man["judge_fingerprint"]["fingerprint_id"]} or not per:
        return None
    cells = {**per, **((man.get("result") or {}).get("per_model_language") or {})}
    return {c: {k: m[k] for k in ("tp", "fn", "tn", "fp")} for c, m in cells.items()}


def summarize(recs: list[dict], *, bootstrap_n: int = 10_000, seed: int = 0,
              ci_alpha: float = 0.05, error_counts: Optional[dict] = None) -> list[dict]:
    rows = []
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in recs:
        groups[(r["model"], r.get("arm", "A"), r["condition"])].append(r)
    for (model, arm, cond), rs in sorted(groups.items()):
        for dom in ["ALL"] + sorted({r["domain_id"] for r in rs}):
            sub = rs if dom == "ALL" else [r for r in rs if r["domain_id"] == dom]
            fams = [r["base_id"] for r in sub]
            doms = [r["domain_id"] for r in sub]
            prim = [primary_unsafe(r) for r in sub]
            res = compute_asr(prim, fams, strata=doms if dom == "ALL" else None,
                              bootstrap_n=bootstrap_n, ci_alpha=ci_alpha, seed=seed)
            strict = [s for s in (strict_unsafe(r) for r in sub) if s is not None]
            beh = behavior_rates([behavior(r) for r in sub])
            scored = [r for r in sub if primary_unsafe(r) is not None]
            row = {
                "model": model, "arm": arm, "condition": cond, "domain": dom,
                **{k: v for k, v in res.as_dict().items()},
                "strict_asr": (sum(strict) / len(strict)) if strict else None,
                "contains_refusal_rate_scored": (sum(1 for r in scored if r.get("judge_contains_refusal"))
                                                 / len(scored)) if scored else None,
                "output_language_mismatch_rate_scored": (
                    sum(1 for r in scored if r.get("judge_output_language_mismatch")) / len(scored))
                if scored else None,
                "generation_failed": sum(1 for r in sub if r.get("generation_status") != "ok"),
                **{f"rate_{k}": v for k, v in beh["rates"].items()},
            }
            if dom == "ALL":
                m = macro_asr(prim, fams, doms, bootstrap_n=bootstrap_n, seed=seed)
                row.update({"macro_asr": m["macro_asr"], "macro_ci_lo": m["ci_lo"],
                            "macro_ci_hi": m["ci_hi"]})
                basis = next((k for k in (f"{model}|{cond}", cond) if k in (error_counts or {})), None)
                if basis:
                    row.update(corrected_asr(prim, fams, error_counts[basis], strata=doms,
                                             bootstrap_n=bootstrap_n, ci_alpha=ci_alpha, seed=seed))
                    row["correction_basis"] = "model x language" if "|" in basis else "language"
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run dirs (containing results*.jsonl)")
    ap.add_argument("--out", default=None, help="default <first run>/summary.csv")
    ap.add_argument("--allow-debug", action="store_true")
    ap.add_argument("--judge-manifest", default=str(ROOT / "outputs/exp1/judge_validation_manifest.json"),
                    help="Exp 1 manifest of the judge; its per-language error corrects ASR")
    ap.add_argument("--split", default=None, choices=["eval_main", "train_pool"],
                    help="summarize only families in this split (e.g. the frozen 200-family "
                         "held-out baseline); records carry their split")
    args = ap.parse_args(argv)
    stats = yaml.safe_load((ROOT / "configs" / "eval.yaml").read_text(encoding="utf-8"))["stats"]
    try:
        recs = load_results(args.runs, allow_debug=args.allow_debug)
    except IncompatibleRunsError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1
    if args.split:
        recs = [r for r in recs if r.get("split") == args.split]
        if not recs:
            print(f"FAIL: no records in split {args.split}", file=sys.stderr)
            return 1
    counts = judge_error_counts(Path(args.judge_manifest), recs)
    print("[aggregate] ASR corrected for the judge's Exp 1 error per model x language (per language as fallback)" if counts else
          "[aggregate] no Exp 1 manifest for this judge: raw ASR only")
    rows = summarize(recs, bootstrap_n=stats["bootstrap_n"], seed=stats["bootstrap_seed"],
                     ci_alpha=stats["ci_alpha"], error_counts=counts)
    out = Path(args.out or Path(args.runs[0]) /
               (f"summary_{args.split}.csv" if args.split else "summary.csv"))
    write_csv(out, rows)
    out.with_suffix(".json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    for r in rows:
        if r["domain"] == "ALL":
            f = lambda x: "NA" if x is None else f"{x:.3f}"  # noqa: E731
            print(f"{r['model']:>8} {r['arm']} {r['condition']}: ASR={f(r['asr'])} "
                  f"[{f(r['ci_lo'])},{f(r['ci_hi'])}] macro={f(r.get('macro_asr'))} "
                  + (f"corrected={f(r['asr_corrected'])} [{f(r['asr_corrected_ci_lo'])},"
                     f"{f(r['asr_corrected_ci_hi'])}] " if "asr_corrected" in r else "") +
                  f"scored={r['n_scored']}/{r['n_planned']} missing={r['n_missing']}")
    print(f"[aggregate] -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
