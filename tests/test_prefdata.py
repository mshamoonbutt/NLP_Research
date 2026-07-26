"""Tests for preference-pair building (prefdata.py)."""
from __future__ import annotations

from csjail.prefdata import assemble_pairs, mine_rejected, subsample


def _assignments():
    return {
        "b1": {"split": "train", "held_out_category": False},
        "b2": {"split": "train", "held_out_category": True},   # held-out cat
        "b3": {"split": "eval", "held_out_category": False},   # eval set
        "b4": {"split": "train", "held_out_category": False},
    }


def _judgments():
    return [
        {"base_id": "b1", "condition": "CS", "prompt": "p1", "response": "harmful1", "label": 1},
        {"base_id": "b2", "condition": "CS", "prompt": "p2", "response": "harmful2", "label": 1},
        {"base_id": "b3", "condition": "CS", "prompt": "p3", "response": "harmful3", "label": 1},
        {"base_id": "b4", "condition": "CS", "prompt": "p4", "response": "refusal",  "label": 0},
        {"base_id": "b1", "condition": "EN", "prompt": "e1", "response": "harmfulEN", "label": 1},
    ]


def test_mine_rejected_filters_pool_category_condition_label():
    mined = mine_rejected(_judgments(), _assignments())
    ids = {m["base_id"] for m in mined}
    # b1 only: b2 held-out category, b3 eval, b4 not harmful, EN wrong condition
    assert ids == {"b1"}
    assert mined[0]["rejected"] == "harmful1"


def test_mine_rejected_can_include_held_out_and_eval():
    mined = mine_rejected(
        _judgments(), _assignments(),
        only_training_pool=False, exclude_held_out_categories=False,
    )
    ids = {m["base_id"] for m in mined}
    assert ids == {"b1", "b2", "b3"}  # all CS harmful regardless of split


def test_assemble_pairs_join_dedup_and_report():
    mined = [
        {"base_id": "b1", "prompt": "p", "rejected": "R", "condition": "CS", "harm_category": "C01"},
        {"base_id": "b1", "prompt": "p", "rejected": "R", "condition": "CS", "harm_category": "C01"},  # dup
        {"base_id": "b4", "prompt": "q", "rejected": "RR", "condition": "CS", "harm_category": "C02"},
    ]
    chosen = {"b1": "no, I won't help", "b4": "sorry, cannot help"}
    pairs, report = assemble_pairs(mined, chosen, target_pairs=10)
    assert report.n_mined == 3
    assert report.n_with_chosen == 3
    assert report.n_after_dedup == 2  # (p,R) collapsed
    assert report.n_final == 2
    assert all(set(p) >= {"prompt", "chosen", "rejected"} for p in pairs)


def test_assemble_pairs_skips_missing_chosen():
    mined = [{"base_id": "bX", "prompt": "p", "rejected": "R"}]
    pairs, report = assemble_pairs(mined, {}, target_pairs=10)
    assert report.n_with_chosen == 0
    assert pairs == []


def test_length_balance_flag():
    mined = [{"base_id": "b1", "prompt": "p", "rejected": "x" * 100}]
    # chosen far shorter -> ratio well below 1 -> not within tolerance
    pairs, report = assemble_pairs(mined, {"b1": "no"}, length_tolerance=0.15)
    assert report.within_tolerance is False
    assert report.length_ratio < 0.85


def test_subsample_deterministic_and_capped():
    pairs = [{"prompt": str(i), "chosen": "c", "rejected": "r"} for i in range(50)]
    a = subsample(pairs, 10, seed=42)
    b = subsample(pairs, 10, seed=42)
    assert len(a) == 10
    assert a == b
    assert subsample(pairs, 100, seed=1) == pairs  # n >= len -> all
