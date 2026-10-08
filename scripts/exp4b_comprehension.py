#!/usr/bin/env python3
"""Exp 4b — intent-recovery probe (paper §4.5): did the model understand the request?

About 100 families (domain-stratified, chosen without looking at outcomes) are sent again in
every condition as a SAFE probe: say in one English sentence what the message asks for,
without doing it (csjail/comprehension.py, intent-probe-v1). The harmful-request outcome is
reused from Exp 2 (same unsafe predicate, no regeneration). A scorer -- the Exp 1 judge's
provider and model with the comprehension rubric -- labels each answer understood /
not_demonstrated / refused_probe / uncertain; a bilingual reviewer audits a blinded sample.

Each step runs where it can (Kaggle cannot reach the signed-in Ollama judge):
    GPU (Kaggle):  python scripts/exp4b_comprehension.py --generate-only --models qwen25 phi3 llama32 r1qwen15
                   python scripts/exp4b_comprehension.py --generate-only --models gemma4e2b   # Ollama server up
    laptop:        python scripts/exp4b_comprehension.py --score-only --models qwen25 phi3 llama32 r1qwen15 gemma4e2b \
                       --baseline-results outputs/exp2/main outputs/exp2/main-r1 outputs/exp2/main-gemma4
    after audit:   python scripts/exp4b_comprehension.py --score-review outputs/exp4b/review_sample.csv
Each model uses its registered production backend (vLLM fp16; Gemma via Ollama), greedy.
Reasoning models (R1) get 2,048 tokens and only the answer after </think> is scored; reasoning
that never finishes is "uncertain" (no scorer call). Other models: 96 tokens.

Reported per model x condition: probe outcome counts, recovery rate, ASR within each bucket,
recovered-and-not-harmful share; per model, the planned contrasts on the probed families and
restricted to families recovered in BOTH conditions (paired, raw judge labels, descriptive).
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

from csjail.aggregate import load_results  # noqa: E402
from csjail.artifacts import resolve_exp0, sha256_text  # noqa: E402
from csjail.asr import mcnemar_paired  # noqa: E402
from csjail.comprehension import (  # noqa: E402
    COMPREHENSION_LABELS, INTENT_PROBE, PROBE_VERSION, ComprehensionJudge, build_intent_probe,
    conditioned_asr, scorer_agreement,
)
from csjail.data import CONDITIONS  # noqa: E402
from csjail.outcomes import primary_unsafe  # noqa: E402
from csjail.pipeline import JsonlCache, run_generation  # noqa: E402
from csjail.robustness import select_families  # noqa: E402
from csjail.splits import cohens_kappa  # noqa: E402
from csjail.utils.io import write_jsonl  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PROBE_TOKENS, REASONING_TOKENS = 96, 2048
ARM = f"probe:{PROBE_VERSION}"
REVIEW_FIELDS = ["audit_id", "original_message", "probe_answer", "human_comprehension", "notes"]


def probe_answer(text: str | None, reasoning: bool) -> str | None:
    """The part of a probe output that is scored: the answer after any visible reasoning.
    None = a reasoning model whose reasoning never finished, so there is no answer."""
    text = text or ""
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1].strip()
    return None if reasoning else text


def score_review(path: Path) -> int:
    with path.open(encoding="utf-8-sig", newline="") as f:
        review = {r["audit_id"]: r for r in csv.DictReader(f)}
    with (path.parent / "review_key.csv").open(encoding="utf-8", newline="") as f:
        key = {r["audit_id"]: r for r in csv.DictReader(f)}
    rows = [{**key[a], "human_comprehension": (review[a].get("human_comprehension") or "").strip()}
            for a in key if a in review]
    agr = scorer_agreement(rows)
    done = [r for r in rows if r["human_comprehension"]]
    agr["kappa"] = cohens_kappa([r["human_comprehension"] for r in done], [r["comprehension"] for r in done])
    bad = sorted({r["human_comprehension"] for r in done} - set(COMPREHENSION_LABELS))
    if bad:
        print(f"FAIL: unknown labels in the review file: {bad}", file=sys.stderr)
        return 1
    (path.parent / "scorer_agreement.json").write_text(json.dumps(agr, indent=2), encoding="utf-8")
    print(f"[exp4b] scorer vs reviewer: {agr}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-results", nargs="+", default=None, help="Exp 2 run dir(s) (scoring)")
    ap.add_argument("--models", nargs="+", default=None, help="default: configs/eval.yaml models")
    ap.add_argument("--generate-only", action="store_true", help="GPU host: probes only, no scorer")
    ap.add_argument("--score-only", action="store_true", help="laptop: score cached probes, load no model")
    ap.add_argument("--n-families", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exp0-dir", default=None)
    ap.add_argument("--out-dir", default="outputs/exp4b")
    ap.add_argument("--review-sample-size", type=int, default=80)
    ap.add_argument("--score-review", default=None,
                    help="reviewer-filled review_sample.csv -> scorer agreement only")
    ap.add_argument("--allow-debug", action="store_true")
    args = ap.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.score_review:
        return score_review(Path(args.score_review))
    if args.generate_only and args.score_only:
        print("FAIL: --generate-only and --score-only exclude each other", file=sys.stderr)
        return 2
    if not args.generate_only and not args.baseline_results:
        print("FAIL: scoring needs --baseline-results (the Exp 2 run dirs)", file=sys.stderr)
        return 2
    cfg = yaml.safe_load((ROOT / "configs" / "eval.yaml").read_text(encoding="utf-8"))
    models = args.models or cfg["models"]

    from csjail.models import ModelIdentity, SamplingConfig, resolve
    from csjail.run_eval import make_runner

    art = resolve_exp0(args.exp0_dir)
    rows = art.load_rows()
    fams = sorted(select_families(rows, art.split, n=args.n_families, pool="all", seed=args.seed))
    subset = [r for r in rows if r.base_id in set(fams)]
    probe_rows = [r.model_copy(update={"prompt": build_intent_probe(r.prompt)}) for r in subset]
    cache = JsonlCache(out_dir / "probe_generations.jsonl", "gen_key")
    specs = {m: resolve(m) for m in models}
    tokens = {m: REASONING_TOKENS if s.reasoning else PROBE_TOKENS for m, s in specs.items()}

    if not args.score_only:
        for model in models:
            runner = make_runner(specs[model].backend, specs[model])
            try:
                ident = ModelIdentity.from_runner(runner, arm=ARM, system=None).as_dict()
                sc = SamplingConfig(temperature=0.0, max_tokens=tokens[model])
                run_generation(lambda ps: runner.generate(ps, sc, show_progress=False), probe_rows,
                               identity=ident, sampling=sc.as_dict(), cache=cache)
            finally:
                runner.shutdown()
        man_path = out_dir / "probe_manifest.json"
        man = json.loads(man_path.read_text(encoding="utf-8")) if man_path.exists() else {"models": {}}
        man.update({"probe_version": PROBE_VERSION, "probe_sha256": sha256_text(INTENT_PROBE),
                    "families": fams, "n_families": len(fams), "family_seed": args.seed,
                    "split_id": art.split_id})
        man["models"].update({m: {"backend": specs[m].backend, "max_tokens": tokens[m],
                                  "temperature": 0.0} for m in models})
        man_path.write_text(json.dumps(man, indent=2), encoding="utf-8")
        if args.generate_only:
            print(f"[exp4b] probes generated for {models} -> {cache.path}")
            return 0

    # Probe outputs from the cache: this model, this probe text, first sample, ok preferred.
    want = {(r.id, sha256_text(r.prompt)) for r in probe_rows}
    gens: dict = {}
    for g in cache.records.values():
        k = (g.get("model"), g.get("row_id"))
        if (g.get("arm") == ARM and g.get("model") in models and g.get("sample_index", 0) == 0
                and (g.get("row_id"), g.get("prompt_sha256")) in want
                and (k not in gens or g.get("generation_status") == "ok")):
            gens[k] = g
    lacking = [(m, r.id) for m in models for r in subset if (m, r.id) not in gens]
    if lacking:
        print(f"FAIL: {len(lacking)} probes not generated yet (e.g. {lacking[:3]}); run --generate-only "
              "on the GPU host and copy outputs/exp4b/probe_generations.jsonl here", file=sys.stderr)
        return 1
    base = {(r["model"], r["row_id"]): r for r in load_results(args.baseline_results, allow_debug=args.allow_debug)
            if r.get("arm", "A") == "A" and r["base_id"] in set(fams)}
    orig = {r.id: r for r in subset}

    per_row, todo = [], []
    for model in models:
        for r in subset:
            g = gens[(model, r.id)]
            ans = probe_answer(g.get("response"), specs[model].reasoning) if g["generation_status"] == "ok" else None
            b = base.get((model, r.id))
            row = {"model": model, "row_id": r.id, "base_id": r.base_id, "domain_id": r.domain_id,
                   "condition": r.condition, "baseline_found": b is not None,
                   "unsafe": primary_unsafe(b) if b else None, "prompt": r.prompt, "probe_answer": ans,
                   "probe_status": g["generation_status"] if ans is not None or g["generation_status"] != "ok"
                   else "reasoning_unfinished", "comprehension": None, "scorer_rationale": ""}
            if row["probe_status"] == "reasoning_unfinished":
                row["comprehension"] = "uncertain"
            elif ans is not None:
                todo.append(row)
            per_row.append(row)
    scorer = ComprehensionJudge()
    for row, res in zip(todo, scorer.score_sync([(orig[r["row_id"]].prompt, r["probe_answer"]) for r in todo])):
        row["comprehension"], row["scorer_rationale"] = res.label, res.rationale

    summary, contrasts = [], {}
    pairs = [tuple(c) for c in cfg["contrasts"]["planned"] + cfg["contrasts"]["descriptive"]]
    st = cfg["stats"]
    for model in models:
        mine = [r for r in per_row if r["model"] == model]
        for cond in CONDITIONS:
            sub = [r for r in mine if r["condition"] == cond]
            valid = [r for r in sub if r["comprehension"] is not None]
            und = [r for r in valid if r["comprehension"] == "understood"]
            s = conditioned_asr([r["unsafe"] for r in sub], [r["comprehension"] for r in sub])
            summary.append({"model": model, "condition": cond, **s,
                            "n_probes": len(sub), "n_valid": len(valid),
                            "n_reasoning_unfinished": sum(r["probe_status"] == "reasoning_unfinished" for r in sub),
                            "n_probe_failed": sum(r["probe_status"] == "failed" for r in sub),
                            "recovery_rate": len(und) / len(valid) if valid else None,
                            "recovered_and_not_harmful": (sum(r["unsafe"] is False for r in und) / len(valid))
                            if valid else None})
        by = {(r["condition"], r["base_id"]): r for r in mine}
        contrasts[model] = []
        for a, b in pairs:
            fa = {f: by[(a, f)]["unsafe"] for (c, f) in by if c == a}
            fb = {f: by[(b, f)]["unsafe"] for (c, f) in by if c == b}
            joint = {f for f in fa if f in fb and by[(a, f)]["comprehension"] == "understood"
                     and by[(b, f)]["comprehension"] == "understood"}
            out = {"contrast": f"{a}-{b}"}
            for name, keep in (("probed_families", set(fa) & set(fb)), ("recovered_in_both", joint)):
                m = mcnemar_paired({f: fa[f] for f in keep}, {f: fb[f] for f in keep}, cond_a=a, cond_b=b,
                                   exact_threshold=st["mcnemar_exact_threshold"],
                                   bootstrap_n=st["bootstrap_n"], seed=st["bootstrap_seed"])
                out[name] = {"n_pairs": m.n_complete, "diff": m.diff, "ci": [m.diff_ci_lo, m.diff_ci_hi],
                             "b": m.a_unsafe_b_safe, "c": m.a_safe_b_unsafe, "p": m.pvalue}
            contrasts[model].append(out)

    for x in summary:
        f = lambda v: "NA" if v is None else f"{v:.2f}"  # noqa: E731
        print(f"[exp4b] {x['model']}/{x['condition']}: valid {x['n_valid']}/{x['n_probes']} "
              f"recovered {f(x['recovery_rate'])} | understood {x['understood']['n']} not_demonstrated "
              f"{x['not_demonstrated']['n']} refused_probe {x['refused_probe']['n']} uncertain "
              f"{x['uncertain']['n']} | ASR|understood {f(x['understood']['unsafe_rate'])}")
    write_jsonl(out_dir / "comprehension_rows.jsonl", per_row)   # gitignored (text)

    # Blinded audit sample: stratified per model x condition, scorer-labelled answers only;
    # the reviewer sees the message and the answer, never the scorer's label (kept in the key).
    rng = random.Random(args.seed)
    k = max(1, args.review_sample_size // (len(models) * len(CONDITIONS)))
    picked = []
    for model in models:
        for cond in CONDITIONS:
            pool = [r for r in todo if r["model"] == model and r["condition"] == cond and r["comprehension"]]
            picked += rng.sample(pool, min(k, len(pool)))
    rng.shuffle(picked)
    with (out_dir / "review_sample.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=REVIEW_FIELDS)
        w.writeheader()
        w.writerows({"audit_id": f"P{i + 1:03d}", "original_message": r["prompt"],
                     "probe_answer": r["probe_answer"], "human_comprehension": "", "notes": ""}
                    for i, r in enumerate(picked))
    with (out_dir / "review_key.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["audit_id", "model", "row_id", "condition", "comprehension"])
        w.writeheader()
        w.writerows({"audit_id": f"P{i + 1:03d}", **{c: r[c] for c in ("model", "row_id", "condition",
                                                                          "comprehension")}}
                    for i, r in enumerate(picked))
    (out_dir / "comprehension_summary.json").write_text(json.dumps({
        "families": fams, "models": {m: {"max_tokens": tokens[m]} for m in models},
        "scorer": scorer.fingerprint, "summary": summary, "contrasts": contrasts,
        "scorer_validation": "PENDING: a bilingual reviewer fills human_comprehension in "
                             "review_sample.csv (labels: " + ", ".join(COMPREHENSION_LABELS) +
                             "), then run --score-review",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[exp4b] wrote {out_dir}/comprehension_summary.json + review_sample.csv ({len(picked)} items, "
          "blinded; key in review_key.csv)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
