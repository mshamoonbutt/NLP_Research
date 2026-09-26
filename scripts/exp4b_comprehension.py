#!/usr/bin/env python3
"""Exp 4b — comprehension control (GPU for the probe + API for the scorer).

    python scripts/exp4b_comprehension.py --baseline-results outputs/exp2/main
    python scripts/exp4b_comprehension.py --baseline-results outputs/exp2/main \
        --score-review outputs/exp4b/review_sample.csv      # after human review

Harmful-attempt responses + judgments are REUSED from Exp 2 (no regeneration,
same unified unsafe predicate). Only the safe intent probe is generated
(cached in out_dir/probe_generations.jsonl). See csjail/comprehension.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from csjail.aggregate import load_results  # noqa: E402
from csjail.artifacts import resolve_exp0  # noqa: E402
from csjail.comprehension import (  # noqa: E402
    PROBE_VERSION, ComprehensionJudge, build_intent_probe, conditioned_asr, scorer_agreement,
)
from csjail.data import CONDITIONS  # noqa: E402
from csjail.outcomes import primary_unsafe  # noqa: E402
from csjail.pipeline import JsonlCache, run_generation  # noqa: E402
from csjail.robustness import select_families  # noqa: E402
from csjail.utils.io import write_jsonl  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-results", nargs="+", required=True)
    ap.add_argument("--models", nargs="+", default=["qwen25", "phi3", "llama32"])
    ap.add_argument("--n-families", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exp0-dir", default=None)
    ap.add_argument("--out-dir", default="outputs/exp4b")
    ap.add_argument("--review-sample-size", type=int, default=80)
    ap.add_argument("--score-review", default=None,
                    help="human-reviewed review_sample.csv -> scorer agreement only")
    ap.add_argument("--allow-debug", action="store_true")
    args = ap.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.score_review:
        with open(args.score_review, encoding="utf-8-sig", newline="") as f:
            agr = scorer_agreement(list(csv.DictReader(f)))
        (out_dir / "scorer_agreement.json").write_text(json.dumps(agr, indent=2), encoding="utf-8")
        print(f"[exp4b] scorer agreement: {agr}")
        return 0

    art = resolve_exp0(args.exp0_dir)
    rows = art.load_rows()
    fams = set(select_families(rows, art.split, n=args.n_families, pool="all", seed=args.seed))
    base = {(r["model"], r["row_id"]): r for r in
            load_results(args.baseline_results, allow_debug=args.allow_debug)
            if r.get("arm", "A") == "A" and r["base_id"] in fams}
    subset = [r for r in rows if r.base_id in fams]

    from csjail.models import ModelIdentity, SamplingConfig, SLMRunner, resolve

    scorer = ComprehensionJudge()
    per_row, summary = [], []
    for model in args.models:
        runner = SLMRunner(resolve(model), require_pinned=True)
        try:
            ident = ModelIdentity.from_runner(runner, arm=f"probe:{PROBE_VERSION}",
                                              system=None).as_dict()
            sc = SamplingConfig(temperature=0.0, max_tokens=96)
            probe_rows = [r.model_copy(update={"prompt": build_intent_probe(r.prompt)})
                          for r in subset]
            gens = run_generation(lambda ps: runner.generate(ps, sc, show_progress=False),
                                  probe_rows, identity=ident, sampling=sc.as_dict(),
                                  cache=JsonlCache(out_dir / "probe_generations.jsonl", "gen_key"))
        finally:
            runner.shutdown()
        gen_by_row = {g["row_id"]: g for g in gens}
        orig = {r.id: r for r in subset}
        todo = [(orig[i].prompt, g["response"] or "") for i, g in gen_by_row.items()]
        scores = dict(zip(gen_by_row, scorer.score_sync(todo)))
        for cond in CONDITIONS:
            ids = [r.id for r in subset if r.condition == cond]
            uns, comp = [], []
            for i in ids:
                b = base.get((model, i))
                u = primary_unsafe(b) if b else None
                c = scores[i].label if gen_by_row[i]["generation_status"] == "ok" else None
                uns.append(u)
                comp.append(c)
                per_row.append({"model": model, "row_id": i, "base_id": orig[i].base_id,
                                "domain_id": orig[i].domain_id, "condition": cond,
                                "baseline_found": b is not None, "unsafe": u,
                                "comprehension": c, "prompt": orig[i].prompt,
                                "probe_answer": gen_by_row[i]["response"]})
            s = conditioned_asr(uns, comp)
            summary.append({"model": model, "condition": cond, **s})
            print(f"[exp4b] {model}/{cond}: understood={s['understood']} "
                  f"refused_probe={s['refused_probe']} not_demonstrated={s['not_demonstrated']}")

    write_jsonl(out_dir / "comprehension_rows.jsonl", per_row)   # gitignored (text)
    rng = random.Random(args.seed)
    review = rng.sample(per_row, min(args.review_sample_size, len(per_row)))
    with (out_dir / "review_sample.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "row_id", "condition", "prompt",
                                          "probe_answer", "comprehension", "human_comprehension"])
        w.writeheader()
        w.writerows({**{k: r[k] for k in ("model", "row_id", "condition", "prompt",
                                          "probe_answer", "comprehension")},
                     "human_comprehension": ""} for r in review)
    (out_dir / "comprehension_summary.json").write_text(json.dumps({
        "families": sorted(fams), "scorer": scorer.fingerprint, "summary": summary,
        "scorer_validation": "PENDING: fill review_sample.csv human_comprehension, "
                             "then run --score-review",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[exp4b] wrote {out_dir}/comprehension_summary.json + review_sample.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
