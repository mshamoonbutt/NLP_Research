#!/usr/bin/env python3
"""Exp 4b — comprehension control (GPU + judge API).

The single most important addition on the experiment guide's list. Without
it, a low ASR in the RU/UR conditions is uninterpretable: it could mean the
model refused (real safety) or the model never understood the request
(safety-by-failure, arXiv 2606.03793). Both the script contrast (RU vs UR)
and the code-switching contrast (CS vs RU) in Exp 3 depend on the *lower*
side of each comparison being genuinely safe rather than genuinely
confused — so this hits the central analysis, not a side note.

For a stratified ~100-base-prompt x 4-condition subset, per SLM:
  1. Generate the normal harmful-attempt response (as in Exp 2) and judge it
     with the usual harm rubric.
  2. Separately, issue a comprehension probe: ask the SAME model to
     paraphrase/translate the prompt into English.
  3. Judge whether the paraphrase recovered the harmful intent
     (understood / not_understood), with csjail.comprehension.ComprehensionJudge.
  4. Report ASR CONDITIONED on comprehension, per condition per model.

Usage:
    python scripts/exp4b_comprehension.py --models qwen25 phi3 llama32
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from csjail.comprehension import (  # noqa: E402
    ComprehensionJudge,
    build_paraphrase_probe,
    conditioned_asr,
)
from csjail.data import CONDITIONS, load_dataset  # noqa: E402
from csjail.judge import Judge  # noqa: E402
from csjail.models import SamplingConfig, SLMRunner, resolve  # noqa: E402
from csjail.splits import proportional_stratified_indices  # noqa: E402


def _sample_base_ids(rows, n: int, seed: int) -> set[str]:
    """Stratified-by-category sample of ~n unique base_ids (proportional)."""
    seen: dict[str, str] = {}  # base_id -> harm_category, first-seen order
    for r in rows:
        seen.setdefault(r.base_id, r.harm_category)
    base_ids = list(seen)
    idx = proportional_stratified_indices(
        [seen[b] for b in base_ids], n, seed=seed)
    return {base_ids[i] for i in idx}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="outputs/exp0/dataset_with_features.jsonl")
    ap.add_argument("--models", nargs="+", default=["qwen25", "phi3", "llama32"])
    ap.add_argument("--n-base", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="outputs/exp4b")
    args = ap.parse_args(argv)

    rows = load_dataset(args.dataset)
    base_ids = _sample_base_ids(rows, args.n_base, args.seed)
    subset = [r for r in rows if r.base_id in base_ids]
    print(f"[exp4b] {len(base_ids)} base_ids x {len(CONDITIONS)} conditions "
          f"= {len(subset)} rows")

    judge = Judge()
    comp_judge = ComprehensionJudge()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: list[dict] = []
    per_row: list[dict] = []

    for model in args.models:
        spec = resolve(model)
        runner = SLMRunner(spec)
        try:
            for cond in CONDITIONS:
                sub = [r for r in subset if r.condition == cond]
                if not sub:
                    continue
                prompts = [r.prompt for r in sub]

                # 1) normal harmful-attempt response + harm judge.
                harm_outs = runner.generate(
                    prompts, SamplingConfig(temperature=0.0, max_tokens=512))
                harm_judgments = judge.score_sync(list(zip(prompts, harm_outs)))
                labels = [j.label for j in harm_judgments]

                # 2) comprehension probe + comprehension judge.
                probes = [build_paraphrase_probe(p) for p in prompts]
                paraphrases = runner.generate(
                    probes, SamplingConfig(temperature=0.0, max_tokens=256))
                comp_results = comp_judge.score_sync(
                    list(zip(prompts, paraphrases)))
                understood = [c.understood for c in comp_results]

                for r, resp, lbl, para, u in zip(
                    sub, harm_outs, labels, paraphrases, understood, strict=True
                ):
                    per_row.append({
                        "model": model, "condition": cond, "base_id": r.base_id,
                        "harm_category": r.harm_category, "prompt": r.prompt,
                        "response": resp, "judge_label": lbl,
                        "paraphrase": para, "understood": u,
                    })

                cond_asr = conditioned_asr(labels, understood)
                summary.append({"model": model, "condition": cond, **cond_asr})
                print(f"[exp4b] {model}/{cond}: "
                      f"comprehension_rate={cond_asr['comprehension_rate']} "
                      f"asr|understood={cond_asr['understood']['asr']} "
                      f"asr|not_understood={cond_asr['not_understood']['asr']}")
        finally:
            runner.shutdown()

    (out_dir / "comprehension_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    from csjail.utils.io import write_jsonl

    write_jsonl(out_dir / "comprehension_rows.jsonl", per_row)
    print(f"[exp4b] wrote {out_dir}/comprehension_summary.json, "
          f"{out_dir}/comprehension_rows.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
