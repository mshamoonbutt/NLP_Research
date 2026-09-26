#!/usr/bin/env python3
"""CPU smoke of the whole analysis chain on the harmless fixture -- no model,
no API. Every GPU/API call is replaced by a deterministic stub:

 fixture CSV -> convert_final -> groups + frozen split -> (stub) persist-first
 generation (greedy + 5-draw sampled) -> (stub) judge with the real contract ->
 aggregate (denominators, micro/macro ASR) -> paired McNemar + Holm + GEE ->
 robustness ASR@5 -> mining + chosen validation + matched B/C + training-
 boundary checks.

    python scripts/smoke_pipeline_cpu.py
"""
from __future__ import annotations

import hashlib
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from csjail.aggregate import outcome_maps, summarize  # noqa: E402
from csjail.asr import mcnemar_pairs  # noqa: E402
from csjail.convert_final import convert  # noqa: E402
from csjail.data import Prompt, filter_prompts  # noqa: E402
from csjail.judge import parse_harm, rule_judgment  # noqa: E402
from csjail.metrics import holm_bonferroni  # noqa: E402
from csjail.models import GenOutput  # noqa: E402
from csjail.outcomes import primary_unsafe  # noqa: E402
from csjail.pipeline import JsonlCache, merge_results, run_generation, run_judging  # noqa: E402
from csjail.prefdata import (  # noqa: E402
    assert_pairs_trainable, matched_sets, mine_rejected, order_pairs, validate_chosen,
)
from csjail.qa import exact_duplicate_groups, token_jaccard_candidates  # noqa: E402
from csjail.robustness import summarize as robust_summary  # noqa: E402
from csjail.splits import build_groups, make_splits  # noqa: E402

FIXTURE = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "final_fixture.csv"


def _h(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest(), 16)


class StubJudge:
    fingerprint = {"fingerprint_id": "stub-judge", "rubric_kind": "harm"}

    def score_sync(self, pairs, show_progress=False):
        out = []
        for _p, resp in pairs:
            ruled = rule_judgment(resp, "harm")
            if ruled:
                out.append(ruled)
            elif resp.startswith("UNSAFE"):
                out.append(parse_harm('{"unsafe": true, "response_kind": "harmful_compliance", '
                                      '"compliance_degree": "partial", "contains_refusal": true, '
                                      '"output_language_mismatch": false, "rationale": "stub"}'))
            elif resp.startswith("GARBLE"):
                out.append(parse_harm("not json at all"))           # -> missing, not safe
            else:
                out.append(parse_harm('{"unsafe": false, "response_kind": "refusal", '
                                      '"compliance_degree": "none", "contains_refusal": true, '
                                      '"output_language_mismatch": false, "rationale": "stub"}'))
        return out


def stub_generate(model: str, rows_by_prompt: dict, n: int):
    def fn(prompts):
        res = []
        for p in prompts:
            r = rows_by_prompt[p]
            draws = []
            for i in range(n):
                v = _h(model, r.id, i) % 10
                text = ("UNSAFE stub" if (r.condition != "EN" and v < 4) or v == 0
                        else "GARBLE" if v == 9 else "I can't help with that.")
                draws.append(GenOutput(text, i, "stop", 10, 5))
            res.append(draws)
        return res
    return fn


def main() -> int:
    long_rows, report = convert(FIXTURE)
    rows = [Prompt.model_validate(r) for r in long_rows]
    print(f"== converted: {report['n_families']} families / {report['n_long_rows']} rows")
    groups, grep = build_groups(rows, exact_duplicates=exact_duplicate_groups(rows),
                                near_duplicate_candidates=token_jaccard_candidates(rows))
    split = make_splits(rows, fam_to_group=groups, eval_size=4, seed=42,
                        dataset_version=rows[0].dataset_version)
    print(f"== split {split['meta']['split_id']}: {split['meta']['counts']}")
    by_prompt = {r.prompt: r for r in rows}
    judge = StubJudge()

    with tempfile.TemporaryDirectory() as td:
        results = {}
        for label, n in (("greedy", 1), ("sampled", 5)):
            recs = []
            for model in ("modelA", "modelB"):
                for cond in ("CS", "EN", "RU", "UR"):
                    sub = filter_prompts(rows, condition=cond)
                    ident = {"model_key": model, "arm": "A"}
                    samp = {"temperature": 0.0 if n == 1 else 0.7, "n": n}
                    gens = run_generation(stub_generate(model, by_prompt, n), sub, identity=ident,
                                          sampling=samp,
                                          cache=JsonlCache(Path(td) / label / "gen.jsonl", "gen_key"),
                                          split_lookup=split["assignments"],
                                          split_id=split["meta"]["split_id"], log=lambda s: None)
                    js = run_judging(judge, gens, {r.id: r.prompt for r in sub},
                                     cache=JsonlCache(Path(td) / label / "judge.jsonl", "judge_key"),
                                     log=lambda s: None)
                    recs += merge_results(gens, js, judge_fingerprint_id="stub-judge")
            results[label] = recs

    greedy = results["greedy"]
    print("== aggregate (greedy) ==")
    for r in summarize(greedy, bootstrap_n=200):
        if r["domain"] == "ALL":
            print(f"  {r['model']} {r['condition']}: ASR={r['asr']} scored={r['n_scored']}/"
                  f"{r['n_planned']} missing={r['n_missing']} bounds=({r['bound_lo']:.2f},"
                  f"{r['bound_hi']:.2f}) macro={r['macro_asr']}")
    maps = outcome_maps(greedy)
    tests = []
    for (model, _arm), by_cond in sorted(maps.items()):
        for m in mcnemar_pairs(by_cond, comparisons=[("CS", "RU"), ("RU", "UR"), ("RU", "EN")],
                               bootstrap_n=200):
            tests.append(m)
            print(f"  {model} {m.cond_a}-{m.cond_b}: complete={m.n_complete}/{m.n_shared} "
                  f"diff={m.diff} p={m.pvalue}")
    print(f"== Holm over {len(tests)} tests: {[h['reject'] for h in holm_bonferroni([t.pvalue for t in tests])]}")

    rob = robust_summary(results["sampled"], greedy, k=5, bootstrap_n=200)
    print(f"== robustness ASR@5: {[(k, v['asr_at_5'], v['asr_at_5_n_unresolved']) for k, v in rob['per_model_condition'].items()][:4]}")

    mined, mrep = mine_rejected([r for r in greedy if r["model"] == "modelA"], split,
                                model="modelA", condition="CS")
    en_mined, _ = mine_rejected([r for r in greedy if r["model"] == "modelA"], split,
                                model="modelA", condition="EN")
    ok_j = StubJudge().score_sync([("", "I can't help with that.")])[0]
    cs_pairs = [{**m, "prompt": "p", "chosen": "Mein is mein madad nahi kar sakta, maafi."}
                for m in mined if validate_chosen("Mein is mein madad nahi kar sakta, maafi.", ok_j)[0]]
    en_pairs = [{**m, "prompt": "p", "chosen": "I can't help with that request, sorry."}
                for m in en_mined]
    c, b, rep = matched_sets(order_pairs(cs_pairs, seed=1), order_pairs(en_pairs, seed=1), seed=1)
    assert_pairs_trainable(c, split, model="modelA")
    print(f"== mining {mrep['counts']}; matched B/C = {rep['n_matched']}")
    assert all(primary_unsafe(r) is None for r in greedy if r["response"] == "GARBLE")
    print("SMOKE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
