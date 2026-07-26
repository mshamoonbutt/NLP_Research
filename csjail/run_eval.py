"""End-to-end eval: one (model x condition) -> one JSONL of judgments.

Each call writes a single file at --out with:
  Line 1: provenance header (model rev, judge, dataset hash, args, git sha, ...)
  Lines 2..N: one JSON object per prompt with prompt, response, judgment, ...

Run this once per (model, condition). `scripts/run_all_baseline.sh` calls it
3 SLMs * 4 conditions = 12 times.

Usage:
    python -m csjail.run_eval --model qwen25 --condition EN \\
        --dataset data/csjail_v0.jsonl \\
        --out results/qwen25_EN.jsonl
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from csjail.asr import aggregate_by_category, compute_asr
from csjail.data import filter_prompts, load_dataset, summarize
from csjail.judge import Judge, load_judge_config
from csjail.models import SamplingConfig, resolve, runtime_info
from csjail.utils.io import sha256_file, write_jsonl
from csjail.utils.seeding import set_global_seed


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
            timeout=2,
        )
        return out.stdout.strip() if out.returncode == 0 else "not-a-repo"
    except Exception:
        return "unknown"


def _provenance(args, model_spec, dataset_path: Path, n_prompts: int) -> dict[str, Any]:
    judge_cfg = load_judge_config()
    return {
        "kind": "provenance",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
        "model": {
            "key": model_spec.key,
            "hf_id": model_spec.hf_id,
            "dtype": model_spec.dtype,
            "max_model_len": model_spec.max_model_len,
        },
        "judge": {
            "provider": judge_cfg.provider,
            "model": judge_cfg.model,
            "temperature": judge_cfg.temperature,
        },
        "dataset": {
            "path": str(dataset_path),
            "sha256": sha256_file(dataset_path),
            "n_prompts_for_this_condition": n_prompts,
        },
        "runtime": runtime_info(),
        "git_sha": _git_sha(),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model key (configs/models.yaml)")
    ap.add_argument("--condition", required=True,
                    choices=["CS", "EN", "RU", "UR", "SM"])
    ap.add_argument("--dataset", required=True, type=str)
    ap.add_argument("--out", required=True, type=str)
    # Categories are dynamic (extended design has 10, e.g. C01..C10); validated
    # against the dataset rather than a fixed choices list.
    ap.add_argument("--harm-category", default=None)
    ap.add_argument("--min-cs-authenticity", type=int, default=None)
    ap.add_argument("--max-prompts", type=int, default=None,
                    help="cap N prompts (debug)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bootstrap-n", type=int, default=10_000)
    ap.add_argument("--system", default=None, help="optional system prompt")
    ap.add_argument("--skip-judge", action="store_true",
                    help="generate responses only; write labels=null (debug)")
    args = ap.parse_args(argv)

    set_global_seed(args.seed)
    dataset_path = Path(args.dataset)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) Load + filter dataset
    rows = load_dataset(dataset_path)
    summary = summarize(rows)
    print(f"[eval] dataset summary: {json.dumps(summary, ensure_ascii=False)}")
    filt = filter_prompts(
        rows,
        condition=args.condition,
        harm_category=args.harm_category,
        min_cs_authenticity=args.min_cs_authenticity,
    )
    if args.max_prompts:
        filt = filt[: args.max_prompts]
    if not filt:
        print(f"[eval] FAIL: 0 prompts after filtering", file=sys.stderr)
        return 2
    print(f"[eval] {len(filt)} prompts for model={args.model} condition={args.condition}")

    # 2) Load model
    spec = resolve(args.model)
    print(f"[eval] loading {spec.hf_id} ...")
    t0 = time.time()
    from csjail.models import SLMRunner

    runner = SLMRunner(spec)
    print(f"[eval] model load: {time.time() - t0:.1f}s")

    # 3) Generate
    sampling = SamplingConfig(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    t0 = time.time()
    responses = runner.generate(
        [r.prompt for r in filt],
        sampling=sampling,
        system=args.system,
        show_progress=True,
    )
    gen_dt = time.time() - t0
    print(f"[eval] generate: {gen_dt:.1f}s ({gen_dt / len(filt):.2f}s/prompt)")
    runner.shutdown()

    # 4) Judge
    if args.skip_judge:
        judgments = [None] * len(filt)
        print("[eval] --skip-judge: writing labels=null")
    else:
        judge = Judge()
        pairs = list(zip(
            [r.prompt for r in filt], responses, strict=True,
        ))
        t0 = time.time()
        judgments = judge.score_sync(pairs, show_progress=True)
        print(f"[eval] judge: {time.time() - t0:.1f}s")

    # 5) Assemble per-prompt records
    records: list[dict[str, Any]] = []
    for prompt_obj, resp, j in zip(filt, responses, judgments, strict=True):
        records.append({
            "kind": "prompt_eval",
            "id": prompt_obj.id,
            "base_id": prompt_obj.base_id,
            "harm_category": prompt_obj.harm_category,
            "condition": prompt_obj.condition,
            "cs_style": prompt_obj.cs_style,
            "cs_authenticity": prompt_obj.cs_authenticity,
            "harm_severity": prompt_obj.harm_severity,
            "prompt": prompt_obj.prompt,
            "response": resp,
            "judge_label": (j.label if j is not None else None),
            "judge_parse_ok": (j.parse_ok if j is not None else None),
            "judge_rationale": (j.rationale if j is not None else None),
            "judge_error": (j.error if j is not None else None),
        })

    # 6) Inline summary stats per harm category
    if not args.skip_judge:
        cat_rows = [
            {
                "base_id": r["base_id"],
                "harm_category": r["harm_category"],
                "label": r["judge_label"],
                "severity": r["harm_severity"],
            }
            for r in records
        ]
        per_cat = aggregate_by_category(cat_rows, bootstrap_n=args.bootstrap_n,
                                        seed=args.seed)
        overall = compute_asr(
            [r["judge_label"] for r in records],
            [r["base_id"] for r in records],
            bootstrap_n=args.bootstrap_n, seed=args.seed,
        )
        summary_record = {
            "kind": "summary",
            "n": overall.n,
            "asr": overall.asr,
            "ci_lo": overall.ci_lo,
            "ci_hi": overall.ci_hi,
            "n_full_comply": overall.n_full_comply,
            "n_partial": overall.n_partial,
            "n_refuse": overall.n_refuse,
            "n_parse_fail": overall.n_parse_fail,
            "per_harm_category": per_cat,
        }
        print(f"[eval] OVERALL ASR for {args.model}/{args.condition}: "
              f"{overall.asr:.3f} (95% CI {overall.ci_lo:.3f}-{overall.ci_hi:.3f}, "
              f"n={overall.n})")
    else:
        summary_record = {"kind": "summary", "note": "skipped: --skip-judge"}

    # 7) Write JSONL (header, summary, then per-prompt records)
    header = _provenance(args, spec, dataset_path, len(filt))
    write_jsonl(out_path, [header, summary_record, *records])
    print(f"[eval] wrote {len(records) + 2} lines -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
