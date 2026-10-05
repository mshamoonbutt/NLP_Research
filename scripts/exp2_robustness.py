#!/usr/bin/env python3
"""Exp 2 robustness — sampled decoding on a fixed family subset (GPU + judge).

Default scope (configs/eval.yaml `robustness`): 200 families, identical across
models, CS and RU only, 5 draws at T=0.7, top_p 0.9 -> 200 x 2 x 5 x 3 =
6,000 extra responses. This is a recommended BUDGET scope; it is not a claim
that all-condition robustness (12,000 responses) was run.

    python scripts/exp2_robustness.py --greedy-results outputs/exp2/main
    python scripts/exp2_robustness.py --summarize-only --greedy-results outputs/exp2/main
    python scripts/exp2_robustness.py --skip-judge --greedy-results outputs/exp2/main
        (generation only, before the judge PASSes; re-run without the flag later
         to judge the cached generations -- nothing is regenerated)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from csjail.aggregate import load_results  # noqa: E402
from csjail.artifacts import resolve_exp0  # noqa: E402
from csjail.data import filter_prompts  # noqa: E402
from csjail.robustness import select_families, summarize  # noqa: E402
from csjail.run_eval import (  # noqa: E402
    DEFAULT_JUDGE_MANIFEST, evaluate_system, git_sha, load_eval_config, prepare_judge,
)
from csjail.utils.io import read_jsonl, write_jsonl  # noqa: E402


def main(argv=None) -> int:
    cfg = load_eval_config()
    rc = cfg["robustness"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=cfg["models"])
    ap.add_argument("--conditions", nargs="+", default=rc["conditions"])
    ap.add_argument("--greedy-results", required=True, help="Exp 2 primary run dir")
    ap.add_argument("--out-dir", default="outputs/exp2/robustness")
    ap.add_argument("--exp0-dir", default=None)
    ap.add_argument("--judge-manifest", default=str(DEFAULT_JUDGE_MANIFEST))
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--judge-only", action="store_true",
                    help="judge cached generations without loading any model (no GPU); model "
                         "provenance is replayed from the --greedy-results run manifest")
    ap.add_argument("--skip-judge", action="store_true",
                    help="generation only; judge the cached generations in a later run")
    ap.add_argument("--allow-unvalidated-judge", action="store_true", help="DEBUG ONLY")
    ap.add_argument("--allow-debug", action="store_true")
    args = ap.parse_args(argv)

    art = resolve_exp0(args.exp0_dir)
    rows = art.load_rows()
    fams = select_families(rows, art.split, n=rc["n_families"], pool=rc["family_pool"],
                           seed=rc["subset_seed"])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sampling = {k: rc[k] for k in ("temperature", "top_p", "max_tokens", "n", "seed")}
    k = int(rc["n"])

    if not args.summarize_only:
        judge, debug = None, False
        if not args.skip_judge:
            try:
                judge, _man, debug = prepare_judge("harm", args.judge_manifest,
                                                   args.allow_unvalidated_judge)
            except RuntimeError as e:         # missing API key or client package
                print(f"[robust] FAIL (input): {e}", file=sys.stderr)
                return 3
        from csjail.models import CachedRunner, NotCached, SLMRunner, resolve

        if args.judge_only:
            greedy_models = json.loads((Path(args.greedy_results) / "run_manifest.json")
                                       .read_text(encoding="utf-8"))["models"]
        results = []
        for model in args.models:
            runner = (CachedRunner(greedy_models[model]) if args.judge_only
                      else SLMRunner(resolve(model), require_pinned=True))
            try:
                for cond in args.conditions:
                    sub = filter_prompts(rows, condition=cond, base_ids=set(fams))
                    results += evaluate_system(runner=runner, rows=sub, arm="A",
                                               sampling=sampling, system=None, out_dir=out_dir,
                                               judge=judge, split=art.split,
                                               chunk_size=int(cfg["inference"]["chunk_size"]),
                                               skip_judge=args.skip_judge)
            except NotCached as e:
                print(f"[robust] FAIL (judge-only): {e}", file=sys.stderr)
                return 1
            finally:
                runner.shutdown()
        for r in results:
            r["run_debug"] = debug
        write_jsonl(out_dir / "results.jsonl", results)
        (out_dir / "subset_manifest.json").write_text(json.dumps({
            "families": fams, "n_families": len(fams), "conditions": args.conditions,
            "models": args.models, "sampling": sampling, "pool": rc["family_pool"],
            "subset_seed": rc["subset_seed"], "split_id": art.split_id,
            "dataset_version": art.dataset_version, "git_sha": git_sha(),
            "planned_responses": len(fams) * len(args.conditions) * k * len(args.models),
        }, indent=2), encoding="utf-8")
        if args.skip_judge:
            print(f"[robust] GENERATION ONLY: {len(results)} records -> {out_dir}; re-run "
                  "without --skip-judge after the judge PASSes (generations are reused)")
            return 0

    sampled = [r for r in read_jsonl(out_dir / "results.jsonl") if r.get("kind") == "result"]
    n_api_failed = sum(1 for r in sampled if r.get("judge_status") == "api_error")
    if n_api_failed:
        print(f"[robust] INCOMPLETE: {n_api_failed} judgments failed at the API (e.g. an exhausted "
              f"budget). Finished judgments are cached in {out_dir}/judgments.jsonl; re-run the "
              "same command to judge only the rest.", file=sys.stderr)
        return 4
    if any(r.get("run_debug") for r in sampled) and not args.allow_debug:
        print("FAIL: sampled results come from a debug run", file=sys.stderr)
        return 1
    greedy = [r for r in load_results([args.greedy_results], allow_debug=args.allow_debug)
              if r["base_id"] in set(fams) and r.get("arm", "A") == "A"]
    summ = summarize(sampled, greedy, k=k, contrast=tuple(args.conditions[:2]))
    (out_dir / "robustness_summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    for key, v in summ["per_model_condition"].items():
        print(f"[robust] {key}: per-draw={v['per_draw_asr']} ASR@{k}={v[f'asr_at_{k}']} "
              f"(unresolved {v[f'asr_at_{k}_n_unresolved']}) greedy={v['greedy_asr_same_families']}")
    for key, v in summ["contrast"].items():
        print(f"[robust] {key}: direction consistent with greedy = "
              f"{v['direction_consistent_with_greedy']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
