#!/usr/bin/env python3
"""Exp 1a — draw and generate the judge-validation (or rubric-development) sample.

Unit: `--n-per-model-condition` families x each condition x each model. The
default 60 x 4 conditions x 2 models = 480 responses (60 per model per
condition is a starting MINIMUM, not proof of adequate precision; top up per
configs/judge.yaml `validation` if support is short).

- Families come from the TRAINING POOL of the frozen split (never eval_main),
  stratified by domain, identical across conditions and models (paired).
- `--role development` vs `--role validation`: rubric development and final
  validation must use disjoint families; pass earlier samples via
  `--exclude-sample-dirs` and the script refuses any overlap.
- `--sample-kind challenge` marks a deliberately enriched diagnostic set; it
  is reported separately and never used for the PASS gate.
- `--backend vllm` (production, GPU) or `--backend ollama` (CPU fallback,
  quantized; development only unless --allow-nonproduction-backend).
- Generation is persisted incrementally (generations.jsonl) and the run is
  resumable: re-run with the same --out-dir. The plan (families, models,
  settings) is frozen in sample_plan.json on the first run.
- Output (sample dir): items.csv (with text; gitignored), rater1.csv,
  rater2.csv (same items, independently shuffled, NO judge output),
  adjudication.csv template, sample_manifest.json (ids + hashes only).

    python scripts/exp1_sample_for_annotation.py --models qwen25 phi3 --role validation
    python scripts/exp1_sample_for_annotation.py --role development --backend ollama \
        --n-per-model-condition 20 --out-dir outputs/exp1/development-cpu
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

from csjail.artifacts import resolve_exp0, sha256_json  # noqa: E402
from csjail.data import CONDITIONS, family_domains  # noqa: E402
from csjail.pipeline import JsonlCache, run_generation  # noqa: E402
from csjail.splits import proportional_stratified_indices, trainable_families  # noqa: E402
from csjail.utils.io import write_text_lf  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RATER_FIELDS = ["item_id", "rater_id", "prompt", "response", "unsafe", "response_kind",
                "compliance_degree", "notes"]
ADJ_FIELDS = ["item_id", "gold_unsafe", "gold_response_kind", "gold_compliance_degree",
              "resolution", "adjudicator", "notes"]


def sample_families(rows, split: dict, n: int, seed: int, exclude: set[str]) -> list[str]:
    pool = sorted(trainable_families(split) - exclude)
    dom = family_domains(rows)
    idx = proportional_stratified_indices([dom[f] for f in pool], n, seed=seed)
    return [pool[i] for i in idx]


def make_runner(backend: str, model: str):
    from csjail.models import resolve

    if backend == "ollama":
        from csjail.ollama_backend import OllamaRunner

        return OllamaRunner(resolve(model))
    from csjail.models import SLMRunner

    return SLMRunner(resolve(model), require_pinned=True)


def _csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp0-dir", default=None)
    ap.add_argument("--models", nargs="+", default=["qwen25", "phi3"])
    ap.add_argument("--n-per-model-condition", type=int, default=60)
    ap.add_argument("--role", choices=["validation", "development"], required=True)
    ap.add_argument("--sample-kind", choices=["representative", "challenge"],
                    default="representative")
    ap.add_argument("--exclude-sample-dirs", nargs="*", default=[],
                    help="earlier sample dirs whose families must not be reused")
    ap.add_argument("--backend", choices=["vllm", "ollama"], default="vllm")
    ap.add_argument("--allow-nonproduction-backend", action="store_true",
                    help="permit --backend ollama for a VALIDATION sample (not recommended)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunk-size", type=int, default=8, help="generations persisted per chunk")
    ap.add_argument("--out-dir", default=None, help="re-use to resume an interrupted run")
    args = ap.parse_args(argv)

    if len(set(args.models)) < 2:
        print("FAIL: draw the gold set from >=2 target models", file=sys.stderr)
        return 1
    if args.role == "validation" and args.backend != "vllm" and not args.allow_nonproduction_backend:
        print("FAIL: the validation sample must come from the production backend (vllm, pinned "
              "bf16 weights); --backend ollama is quantized and for development only",
              file=sys.stderr)
        return 1

    art = resolve_exp0(args.exp0_dir)
    rows = art.load_rows()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_dir or ROOT / "outputs" / "exp1" / f"{args.role}-{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    exclude: set[str] = set()
    for d in args.exclude_sample_dirs:
        exclude |= set(json.loads((Path(d) / "sample_manifest.json").read_text(
            encoding="utf-8"))["families"])
    eval_cfg = yaml.safe_load((ROOT / "configs" / "eval.yaml").read_text(encoding="utf-8"))
    plan = {
        "role": args.role, "sample_kind": args.sample_kind, "models": args.models,
        "n_per_model_condition": args.n_per_model_condition, "seed": args.seed,
        "backend": args.backend, "dataset_version": art.dataset_version,
        "split_id": art.split_id, "sampling": eval_cfg["sampling"],
        "excluded_families_from": args.exclude_sample_dirs,
        "families": sample_families(rows, art.split, args.n_per_model_condition, args.seed, exclude),
    }
    plan_path = out_dir / "sample_plan.json"
    if plan_path.exists():
        old = json.loads(plan_path.read_text(encoding="utf-8"))
        if old != plan:
            diff = sorted(k for k in plan if old.get(k) != plan.get(k))
            print(f"FAIL: {out_dir} holds a different sample plan (differs in {diff}); "
                  "use a new --out-dir", file=sys.stderr)
            return 1
        print(f"[exp1] resuming {out_dir}")
    else:
        write_text_lf(plan_path, json.dumps(plan, ensure_ascii=False, indent=2))
    fams = plan["families"]
    if len(fams) < args.n_per_model_condition:
        print(f"WARN: only {len(fams)} eligible families", file=sys.stderr)

    from csjail.models import ModelIdentity, SamplingConfig

    sampling = SamplingConfig(**eval_cfg["sampling"])
    fam_set = set(fams)
    sub = sorted((r for r in rows if r.base_id in fam_set), key=lambda r: (r.condition, r.base_id))
    cache = JsonlCache(out_dir / "generations.jsonl", "gen_key")
    gens_by_model, model_prov = {}, {}
    for model in args.models:
        runner = make_runner(args.backend, model)
        try:
            model_prov[model] = runner.provenance()
            ident = ModelIdentity.from_runner(runner, arm="A", system=None).as_dict()
            ident["extra"] = {k: model_prov[model].get(k) for k in
                              ("backend", "ollama_tag", "ollama_digest", "quantization")}
            done = [0]

            def gen_fn(prompts, _r=runner, _m=model):
                out = _r.generate(prompts, sampling, show_progress=False)
                done[0] += len(prompts)
                print(f"[exp1] {_m}: +{len(prompts)} generated this run "
                      f"({done[0]} so far) {datetime.now().strftime('%H:%M:%S')}", flush=True)
                return out

            gens_by_model[model] = run_generation(
                gen_fn, sub, identity=ident, sampling=sampling.as_dict(), cache=cache,
                split_lookup=art.split["assignments"], split_id=art.split_id,
                chunk_size=args.chunk_size)
        finally:
            runner.shutdown()

    failed = [g for gs in gens_by_model.values() for g in gs if g["generation_status"] != "ok"]
    if failed:
        print(f"[exp1] {len(failed)} generations failed (e.g. {failed[0].get('generation_error')}); "
              f"re-run the same command to retry. Rater files NOT written.", file=sys.stderr)
        return 2

    prompts = {r.id: r.prompt for r in sub}
    items = []
    for model, gens in gens_by_model.items():
        for g in gens:
            items.append({
                "item_id": f"{args.role}:{model}:{g['row_id']}", "role": args.role,
                "sample_kind": args.sample_kind, "model": model, "backend": args.backend,
                "base_id": g["base_id"], "domain_id": g["domain_id"], "condition": g["condition"],
                "row_id": g["row_id"], "gen_key": g["gen_key"],
                "response_id": g["response_sha256"][:16], "finish_reason": g.get("finish_reason"),
                "prompt": prompts[g["row_id"]], "response": g["response"]})
    _csv(out_dir / "items.csv", list(items[0].keys()), items)
    for k, name in ((1, "rater1.csv"), (2, "rater2.csv")):
        order = items[:]
        random.Random(args.seed + k).shuffle(order)   # independent order, no predictions
        _csv(out_dir / name, RATER_FIELDS,
             [{"item_id": it["item_id"], "rater_id": "", "prompt": it["prompt"],
               "response": it["response"], "unsafe": "", "response_kind": "",
               "compliance_degree": "", "notes": ""} for it in order])
    _csv(out_dir / "adjudication.csv", ADJ_FIELDS, [])
    finish = {}
    for it in items:
        finish[it["finish_reason"]] = finish.get(it["finish_reason"], 0) + 1
    manifest = {
        "kind": "exp1_sample_manifest", "created_utc": stamp, **{k: plan[k] for k in (
            "role", "sample_kind", "dataset_version", "split_id", "seed", "models",
            "n_per_model_condition", "families", "excluded_families_from")},
        "source_pool": "train_pool", "generation_backend": args.backend,
        "n_items": len(items), "item_ids": [it["item_id"] for it in items],
        "items_sha256": sha256_json(items), "sampling": sampling.as_dict(),
        "finish_reasons": finish, "model_provenance": model_prov,
    }
    write_text_lf(out_dir / "sample_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"[exp1] {len(items)} items = {len(args.models)} models x {len(CONDITIONS)} "
          f"conditions x {len(fams)} families -> {out_dir}  (finish reasons: {finish})")
    print("[exp1] NEXT: two raters fill rater1.csv / rater2.csv independently (rater_id, "
          "unsafe, response_kind, compliance_degree); adjudicate disagreements in "
          "adjudication.csv; then run scripts/calibrate_judge.py --sample-dir", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
