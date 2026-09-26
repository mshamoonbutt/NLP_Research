"""Grouping, frozen splits, append-only extension, ablation (checks 4-5)."""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from csjail.qa import exact_duplicate_groups, token_jaccard_candidates
from csjail.splits import (
    DOMAIN_HOLDOUT, DOMAIN_SUPPLEMENTARY, EVAL, QUARANTINE, TRAIN, assert_trainable,
    build_groups, cohens_kappa, extend_split, make_domain_ablation, make_splits,
    pairwise_agreement, proportional_stratified_indices, verify_split,
)
from tests.conftest import ROOT, family


def many(n_per_dom: int = 10) -> list:
    rows = []
    for d in ("D1", "D2", "D3", "D4", "D5", "D6"):
        for i in range(n_per_dom):
            rows += family(f"{d}-{i:03d}", d, tag=f"topic{d}{i}")
    return rows


def test_main_split_counts_and_all_domains_trainable():
    rows = many()
    s = make_splits(rows, eval_size=12, seed=42)
    c = s["meta"]["counts"]
    assert c[EVAL]["total"] == 12 and c[TRAIN]["total"] == 48
    assert set(c[TRAIN]) - {"total"} == {"D1", "D2", "D3", "D4", "D5", "D6"}
    assert s["meta"]["holdout_domains"] == [] and DOMAIN_HOLDOUT not in c


def test_split_deterministic_and_order_independent():
    rows = many()
    a = make_splits(rows, eval_size=12, seed=7)
    b = make_splits(list(reversed(rows)), eval_size=12, seed=7)
    assert a["assignments"] == b["assignments"]
    assert a["meta"]["split_id"] == b["meta"]["split_id"]


def test_split_reproducible_across_processes():
    code = ("import sys; sys.path.insert(0, r'%s'); from tests.test_splits import many; "
            "from csjail.splits import make_splits; "
            "print(make_splits(many(), eval_size=12, seed=3)['meta']['split_id'])" % ROOT)
    ids = set()
    for seed in ("0", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             env=env, cwd=ROOT)
        assert out.returncode == 0, out.stderr
        ids.add(out.stdout.strip())
    assert len(ids) == 1


def test_domain_holdout_requires_explicit_domains_and_is_exclusive():
    rows = many()
    with pytest.raises(ValueError):
        make_splits(rows, scheme="domain_holdout")
    with pytest.raises(ValueError):
        make_splits(rows, holdout_domains=["D2"])  # scheme main forbids it
    s = make_splits(rows, eval_size=12, scheme="domain_holdout", holdout_domains=["D2"])
    a = s["assignments"]
    assert all(v["split"] == DOMAIN_HOLDOUT for v in a.values() if v["domain_id"] == "D2")
    assert not any(v["domain_id"] == "D2" for v in a.values() if v["split"] != DOMAIN_HOLDOUT)


def test_exact_and_unresolved_near_duplicates_never_straddle():
    rows = many()
    # make D1-000 an exact duplicate of D1-001 in CS, and D1-002 near-identical to D1-003
    rows = [r.model_copy(update={"prompt": "same text"}) if r.condition == "CS"
            and r.base_id in ("D1-000", "D1-001") else r for r in rows]
    rows = [r.model_copy(update={"prompt": "a b c d e f g h i j k l m n o p q r s t u v w x y z"})
            if r.condition == "EN" and r.base_id in ("D1-002", "D1-003") else r for r in rows]
    exact = exact_duplicate_groups(rows)
    cands = token_jaccard_candidates(rows)
    groups, rep = build_groups(rows, exact_duplicates=exact, near_duplicate_candidates=cands)
    assert groups["D1-000"] == groups["D1-001"]
    assert groups["D1-002"] == groups["D1-003"]
    assert rep["candidates_unresolved_grouped_conservatively"] >= 1
    for seed in range(20):
        s = make_splits(rows, fam_to_group=groups, eval_size=12, seed=seed)
        a = s["assignments"]
        assert a["D1-000"]["split"] == a["D1-001"]["split"]
        assert a["D1-002"]["split"] == a["D1-003"]["split"]
        assert verify_split(s, rows) == []


def test_distinct_decision_separates_candidates():
    rows = many()
    rows = [r.model_copy(update={"prompt": "a b c d e f g h i j k l m n o p q r s t u v w x y z"})
            if r.condition == "EN" and r.base_id in ("D1-002", "D1-003") else r for r in rows]
    cands = token_jaccard_candidates(rows)
    groups, _ = build_groups(rows, exact_duplicates=[], near_duplicate_candidates=cands,
                             decisions={("D1-002", "D1-003"): "distinct"})
    assert groups["D1-002"] != groups["D1-003"]


def test_extension_is_append_only_and_quarantines_eval_relatives():
    rows = many()
    frozen = make_splits(rows, eval_size=12, seed=1)
    eval_fam = sorted(f for f, a in frozen["assignments"].items() if a["split"] == EVAL)[0]
    new = rows + family("NEW-1", "D3") + family("NEW-2", "D4")
    groups = {f: frozen["assignments"].get(f, {}).get("group_id", f) for f in {r.base_id for r in new}}
    groups["NEW-2"] = frozen["assignments"][eval_fam]["group_id"]   # relative of an eval family
    ext = extend_split(frozen, new, fam_to_group=groups)
    for f, a in frozen["assignments"].items():
        assert ext["assignments"][f]["split"] == a["split"]
    assert ext["assignments"]["NEW-1"]["split"] == TRAIN
    assert ext["assignments"]["NEW-2"]["split"] == QUARANTINE


def test_domain_ablation_manifest():
    rows = many()
    main = make_splits(rows, eval_size=12, seed=1)
    with pytest.raises(ValueError):
        make_domain_ablation(main, "D2", attested_before_outcomes=False)
    abl = make_domain_ablation(main, "D2", attested_before_outcomes=True)
    for f, a in abl["assignments"].items():
        m = main["assignments"][f]
        if m["domain_id"] == "D2" and m["split"] == TRAIN:
            assert a["split"] == DOMAIN_SUPPLEMENTARY
        else:
            assert a["split"] == m["split"]
    assert abl["meta"]["parent_split_id"] == main["meta"]["split_id"]
    assert abl["meta"]["split_id"] != main["meta"]["split_id"]


def test_assert_trainable_blocks_eval_and_relatives():
    rows = many()
    groups = {r.base_id: r.base_id for r in rows}
    groups["D1-000"] = groups["D1-001"] = "grp:D1-000"
    s = make_splits(rows, fam_to_group=groups, eval_size=12, seed=2)
    ev = next(f for f, a in s["assignments"].items() if a["split"] == EVAL)
    with pytest.raises(ValueError, match="not in the training pool"):
        assert_trainable(s, [ev])
    tr = [f for f, a in s["assignments"].items() if a["split"] == TRAIN]
    assert_trainable(s, tr)


def test_stratified_indices_exact_and_deterministic():
    keys = ["a"] * 50 + ["b"] * 30 + ["c"] * 20
    i1 = proportional_stratified_indices(keys, 10, seed=5)
    assert len(i1) == 10 and i1 == proportional_stratified_indices(keys, 10, seed=5)
    assert sum(keys[i] == "a" for i in i1) == 5


def test_kappa_undefined_is_none_not_one():
    assert cohens_kappa([1, 1, 1], [1, 1, 1]) is None
    assert cohens_kappa([], []) is None
    assert cohens_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == pytest.approx(1.0)


def test_pairwise_agreement_descriptive(tmp_path):
    p = tmp_path / "ann.csv"
    p.write_text("family_id,rater1_equivalence,rater2_equivalence\n1,yes,yes\n2,yes,no\n3,no,no\n4,,yes\n",
                 encoding="utf-8")
    rep = pairwise_agreement(p)
    f = rep["fields"]["equivalence"]
    assert f["n"] == 3 and f["n_missing"] == 1 and f["raw_agreement"] == pytest.approx(2 / 3)
    assert "not a dataset validity gate" in rep["note"]


def test_committed_split_manifest_matches_counts():
    latest = ROOT / "outputs" / "exp0" / "LATEST.json"
    if not latest.exists():
        pytest.skip("no finalized Exp 0 artifact")
    d = ROOT / json.loads(latest.read_text())["dir"]
    man = json.loads((d / "split_manifest.json").read_text(encoding="utf-8"))
    c = man["meta"]["counts"]
    assert c[EVAL]["total"] + c[TRAIN]["total"] == man["meta"]["n_families"]
