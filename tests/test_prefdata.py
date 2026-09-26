"""Preference-pair leakage gates, chosen validation, matched B/C (checks 11-12)."""
from __future__ import annotations

import pytest

from csjail.judge import parse_harm, rule_judgment
from csjail.prefdata import (
    PairBuildError, assert_pairs_trainable, matched_sets, mine_rejected, order_pairs,
    supported_budgets, validate_chosen,
)
from csjail.splits import make_splits
from tests.conftest import family, result
from tests.test_splits import many


@pytest.fixture
def split():
    rows = many()
    groups = {r.base_id: r.base_id for r in rows}
    return make_splits(rows, fam_to_group=groups, eval_size=12, seed=4)


def fams(split, which):
    return sorted(f for f, a in split["assignments"].items() if a["split"] == which)


def test_mixed_model_input_rejected(split):
    tr = fams(split, "train_pool")[0]
    recs = [result("phi3", tr, "CS", True), result("llama32", tr, "CS", True)]
    with pytest.raises(PairBuildError, match="exactly 'phi3'"):
        mine_rejected(recs, split, model="phi3", condition="CS")


def test_mining_uses_train_pool_primary_unsafe_only(split):
    tr, ev = fams(split, "train_pool"), fams(split, "eval_main")
    recs = [result("phi3", tr[0], "CS", True, degree="partial"),      # partial -> eligible
            result("phi3", tr[1], "CS", False),
            result("phi3", tr[2], "CS", None, status="parse_error"),
            result("phi3", ev[0], "CS", True),                          # eval -> excluded
            result("phi3", tr[3], "EN", True)]                          # wrong condition
    mined, rep = mine_rejected(recs, split, model="phi3", condition="CS")
    assert [m["base_id"] for m in mined] == [tr[0]]
    assert rep["counts"]["missing_outcome"] == 1
    assert rep["counts"]["excluded_not_train_pool_or_group_relative"] == 1
    assert mined[0]["rejected_lineage"]["model"] == "phi3"


def test_ablation_domain_excluded(split):
    tr = [f for f in fams(split, "train_pool") if f.startswith("D2")][0]
    mined, rep = mine_rejected([result("phi3", tr, "CS", True, dom="D2")], split,
                               model="phi3", condition="CS", exclude_domains=["D2"])
    assert mined == [] and rep["counts"]["excluded_domain"] == 1


def refusal():
    return parse_harm('{"unsafe": false, "response_kind": "refusal", "compliance_degree": '
                      '"none", "contains_refusal": true, "output_language_mismatch": false}')


def test_validate_chosen():
    text = "Yeh mein nahi kar sakta, lekin safe alternative bata sakta hoon."
    assert validate_chosen(text, refusal()) == (True, "ok")
    assert validate_chosen("", rule_judgment("", "harm"))[0] is False
    assert validate_chosen(None, None)[0] is False
    gib = parse_harm('{"unsafe": false, "response_kind": "unintelligible", "compliance_degree": '
                     '"none", "contains_refusal": false}')
    assert validate_chosen("zzzz qqqq xxxx yyyy wwww", gib) == (False, "not_refusal:unintelligible")
    helpful = parse_harm('{"unsafe": false, "response_kind": "safe_helpful", "compliance_degree": '
                         '"none", "contains_refusal": false}')
    assert validate_chosen(text, helpful)[0] is False
    hedged = parse_harm('{"unsafe": true, "response_kind": "harmful_compliance", '
                        '"compliance_degree": "partial", "contains_refusal": true}')
    assert validate_chosen(text, hedged) == (False, "unsafe")
    assert validate_chosen(text, parse_harm("garbage")) == (False, "judge_missing")


def pairs_for(fs, dom="D1"):
    return [{"base_id": f, "domain_id": f.split("-")[0], "prompt": "p", "chosen": "c",
             "rejected": "r", "rejected_lineage": {"model": "phi3"}} for f in fs]


def test_ordering_seeded_domain_aware_and_nested(split):
    tr = fams(split, "train_pool")
    o1 = order_pairs(pairs_for(tr), seed=3)
    o2 = order_pairs(list(reversed(pairs_for(tr))), seed=3)
    assert [p["base_id"] for p in o1] == [p["base_id"] for p in o2]
    first6 = {p["domain_id"] for p in o1[:6]}
    assert first6 == {"D1", "D2", "D3", "D4", "D5", "D6"}      # not file-order biased


def test_matched_sets_same_families_and_count(split):
    tr = fams(split, "train_pool")
    c, b, rep = matched_sets(order_pairs(pairs_for(tr[:20]), seed=1),
                             order_pairs(pairs_for(tr[10:30]), seed=1), seed=1)
    assert [p["base_id"] for p in c] == [p["base_id"] for p in b]
    assert rep["n_matched"] == 10 == len(c)


def test_budgets_never_pad():
    assert supported_budgets(120)["runnable"] == [50, 100, "all"]
    assert supported_budgets(30)["status"] == "EXPLORATORY_BELOW_50"
    assert supported_budgets(0)["runnable"] == []


def test_training_boundary_rechecks(split):
    tr, ev = fams(split, "train_pool"), fams(split, "eval_main")
    assert_pairs_trainable(pairs_for(tr[:3]), split, model="phi3")
    with pytest.raises(ValueError, match="training pool"):
        assert_pairs_trainable(pairs_for([ev[0]]), split)
    with pytest.raises(PairBuildError, match="another model"):
        bad = pairs_for(tr[:1])
        bad[0]["rejected_lineage"] = {"model": "llama32"}
        assert_pairs_trainable(bad, split, model="phi3")
    d2 = [f for f in tr if f.startswith("D2")][:1]
    with pytest.raises(PairBuildError, match="excluded domain"):
        assert_pairs_trainable(pairs_for(d2), split, exclude_domains=["D2"])


def test_family_helper_used():
    assert len(family("x")) == 4
