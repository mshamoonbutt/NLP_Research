"""Exp 2 — primary evaluation sweep (and the shared evaluation engine).

    python -m csjail.run_eval --out-dir outputs/exp2/main
    python -m csjail.run_eval --models phi3 --conditions CS RU --max-families 10 \
        --out-dir outputs/exp2/debug --allow-unvalidated-judge --allow-unpinned-models

- Inputs: the FINALIZED Exp 0 artifact (dataset + frozen split manifest,
  hash-checked) and configs/eval.yaml (single settings source; CLI overrides
  are recorded).
- Each model is loaded ONCE and generates all requested conditions.
- Stage 1 generation is persisted incrementally (out_dir/generations.jsonl)
  before stage 2 judging (out_dir/judgments.jsonl); re-running resumes.
- Judging requires a PASS judge-validation manifest for the exact current
  judge fingerprint. `--allow-unvalidated-judge` is a debug bypass that marks
  the run `debug: true`; aggregation rejects debug runs by default.
- results.jsonl: one record per (model, arm, row, sample) with model identity,
  family, domain, condition, split + split id, dataset version, prompt and
  response hashes, response, finish reason, token counts, generation status,
  judge status and validated judge fields.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml

from csjail.artifacts import resolve_exp0, sha256_json
from csjail.data import CONDITIONS, filter_prompts
from csjail.pipeline import JsonlCache, merge_results, run_generation, run_judging
from csjail.utils.io import sha256_file, write_jsonl

ROOT = Path(__file__).resolve().parent.parent
EVAL_CFG = ROOT / "configs" / "eval.yaml"
DEFAULT_JUDGE_MANIFEST = ROOT / "outputs" / "exp1" / "judge_validation_manifest.json"


def git_sha() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                             text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else "not-a-repo"
    except Exception:
        return "unknown"


def load_eval_config(path: str | Path = EVAL_CFG) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def select_families(split: dict, which: str) -> Optional[set[str]]:
    if which == "all":
        return None
    return {f for f, a in split["assignments"].items() if a["split"] == which}


def prepare_judge(kind: str, manifest_path: Optional[str], allow_unvalidated: bool):
    """Return (judge, validation_manifest_or_None, debug_flag)."""
    from csjail.judge import Judge, load_judge_config
    from csjail.judge_validation import UnvalidatedJudgeError, require_validated_judge

    cfg = load_judge_config()
    fp = cfg.fingerprint(kind)
    try:
        man = require_validated_judge(manifest_path, fp)
        debug = False
    except UnvalidatedJudgeError as e:
        if not allow_unvalidated:
            raise
        print(f"[eval] WARNING (debug run): {e}", file=sys.stderr)
        man, debug = None, True
    return Judge(cfg, kind=kind), man, debug


def evaluate_system(*, runner, rows, arm: str, sampling: dict, system: Optional[str],
                    out_dir: Path, judge, split: dict, chunk_size: int,
                    skip_judge: bool = False) -> list[dict]:
    """Generate (cached) + judge (cached) one model/arm over `rows`."""
    from csjail.models import ModelIdentity, SamplingConfig

    identity = ModelIdentity.from_runner(runner, arm=arm, system=system).as_dict()
    sc = SamplingConfig(**sampling)
    gens = run_generation(
        lambda ps: runner.generate(ps, sc, system=system, show_progress=True),
        rows, identity=identity, sampling=sc.as_dict(),
        cache=JsonlCache(out_dir / "generations.jsonl", "gen_key"),
        split_lookup=split["assignments"], split_id=split["meta"]["split_id"],
        chunk_size=chunk_size)
    judgments, fp_id = {}, None
    if not skip_judge:
        fp_id = judge.fingerprint["fingerprint_id"]
        judgments = run_judging(judge, gens, {r.id: r.prompt for r in rows},
                                cache=JsonlCache(out_dir / "judgments.jsonl", "judge_key"))
    return merge_results(gens, judgments, judge_fingerprint_id=fp_id)


def main(argv: list[str] | None = None) -> int:
    cfg = load_eval_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=cfg["models"])
    ap.add_argument("--conditions", nargs="+", default=cfg["conditions"], choices=list(CONDITIONS))
    ap.add_argument("--families", default="all", choices=["all", "eval_main", "train_pool"],
                    help="Phase 1 descriptive sweep uses all families")
    ap.add_argument("--exp0-dir", default=None)
    ap.add_argument("--split-manifest", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--judge-manifest", default=str(DEFAULT_JUDGE_MANIFEST))
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-families", type=int, default=None, help="debug cap")
    ap.add_argument("--skip-judge", action="store_true")
    ap.add_argument("--allow-unvalidated-judge", action="store_true", help="DEBUG ONLY")
    ap.add_argument("--allow-unpinned-models", action="store_true", help="DEBUG ONLY")
    ap.add_argument("--allow-fallback-template", action="store_true", help="DEBUG ONLY")
    args = ap.parse_args(argv)

    sampling = dict(cfg["sampling"])
    overrides = {k: v for k, v in (("temperature", args.temperature),
                                   ("max_tokens", args.max_tokens), ("seed", args.seed))
                 if v is not None}
    sampling.update(overrides)

    art = resolve_exp0(args.exp0_dir, split_path=args.split_manifest)
    rows = art.load_rows()
    fams = select_families(art.split, args.families)
    if args.max_families:
        pool = sorted(fams if fams is not None else {r.base_id for r in rows})
        fams = set(pool[: args.max_families])
    rows = [r for r in rows if r.condition in args.conditions
            and (fams is None or r.base_id in fams)]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    judge, judge_man, debug = (None, None, False)
    if not args.skip_judge:
        judge, judge_man, debug = prepare_judge("harm", args.judge_manifest,
                                                args.allow_unvalidated_judge)
    debug = debug or args.allow_unpinned_models or args.allow_fallback_template \
        or bool(args.max_families) or args.skip_judge

    from csjail.models import SLMRunner, resolve

    all_results, model_prov = [], {}
    for model in args.models:
        runner = SLMRunner(resolve(model), require_pinned=not args.allow_unpinned_models,
                           allow_fallback_template=args.allow_fallback_template)
        try:
            model_prov[model] = runner.provenance()
            for cond in args.conditions:  # one load, all conditions
                sub = filter_prompts(rows, condition=cond)
                all_results += evaluate_system(
                    runner=runner, rows=sub, arm="A", sampling=sampling, system=None,
                    out_dir=out_dir, judge=judge, split=art.split,
                    chunk_size=int(cfg["inference"]["chunk_size"]), skip_judge=args.skip_judge)
        finally:
            runner.shutdown()

    for r in all_results:
        r["run_debug"] = debug
    write_jsonl(out_dir / "results.jsonl", all_results)
    manifest: dict[str, Any] = {
        "kind": "eval_run_manifest", "experiment": "exp2_primary",
        "created_utc": datetime.now(timezone.utc).isoformat(), "git_sha": git_sha(),
        "debug": debug, "args": vars(args), "sampling": sampling, "sampling_overrides": overrides,
        "eval_config_sha256": sha256_file(EVAL_CFG),
        "dataset_version": art.dataset_version, "dataset_manifest": art.manifest,
        "split_id": art.split_id, "split_scheme": art.split["meta"]["scheme"],
        "families_selector": args.families, "n_rows": len(rows),
        "judge_fingerprint": judge.fingerprint if judge else None,
        "judge_validation_manifest_sha256": (sha256_file(args.judge_manifest)
                                             if judge_man else None),
        "models": model_prov,
        "results_sha256": sha256_json([r.get("gen_key") for r in all_results]),
        "planned_primary_responses": len(rows) * len(args.models),
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                               encoding="utf-8")
    print(f"[eval] {len(all_results)} result records -> {out_dir / 'results.jsonl'}"
          + ("  [DEBUG RUN]" if debug else ""))
    print(f"[eval] summarize: python -m csjail.aggregate {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
