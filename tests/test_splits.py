"""Tests for Exp-0 finalize/split logic (splits.py)."""
from __future__ import annotations

from csjail.data import CONDITIONS, Prompt
from csjail.splits import (
    attach_features,
    condition_parity,
    dataset_stats,
    find_near_duplicates,
    make_splits,
)


def _mk(base_id: str, cat: str, cond: str, text: str) -> Prompt:
    kw = {}
    if cond in ("CS", "SM"):
        kw = {"cs_style": "A", "cs_authenticity": 3}
    return Prompt(
        id=f"{base_id}-{cond}", base_id=base_id, harm_category=cat,
        condition=cond, prompt=text, harm_severity=2, **kw,
    )


def _dataset(n_per_cat: dict[str, int]) -> list[Prompt]:
    rows: list[Prompt] = []
    for cat, n in n_per_cat.items():
        for i in range(n):
            bid = f"{cat}-{i:03d}"
            for cond in CONDITIONS:
                rows.append(_mk(bid, cat, cond, f"{cond} prompt mujhe batao {i}"))
    return rows


def test_attach_features_fills_missing():
    rows = _dataset({"C01": 2})
    assert all(r.cmi is None for r in rows)
    out = attach_features(rows)
    assert all(r.cmi is not None and r.urdu_word_ratio is not None for r in out)


def test_condition_parity_complete():
    rows = _dataset({"C01": 3, "C02": 3})
    rep = condition_parity(rows)
    assert rep.n_base_ids == 6
    assert rep.n_complete == 6
    assert rep.parity_rate == 1.0


def test_condition_parity_detects_incomplete():
    rows = _dataset({"C01": 2})
    rows = [r for r in rows if not (r.base_id == "C01-000" and r.condition == "RU")]
    rep = condition_parity(rows)
    assert rep.parity_rate < 1.0
    assert "C01-000" in rep.incomplete


def test_make_splits_exact_size_and_determinism():
    rows = _dataset({"C01": 10, "C02": 10, "C03": 10})
    s1 = make_splits(rows, eval_holdout=9, n_held_out_categories=1, seed=42)
    s2 = make_splits(rows, eval_holdout=9, n_held_out_categories=1, seed=42)
    assert s1["meta"]["eval_holdout_actual"] == 9
    assert s1["assignments"] == s2["assignments"]  # deterministic
    n_eval = sum(1 for v in s1["assignments"].values() if v["split"] == "eval")
    assert n_eval == 9


def test_make_splits_stratified_across_categories():
    rows = _dataset({"C01": 10, "C02": 10, "C03": 10})
    s = make_splits(rows, eval_holdout=9, n_held_out_categories=1, seed=1)
    # 9 eval over 3 equal categories -> ~3 each; every category represented.
    by_cat = s["meta"]["eval_by_category"]
    assert set(by_cat) == {"C01", "C02", "C03"}
    assert all(v >= 1 for v in by_cat.values())


def test_held_out_categories_marked():
    rows = _dataset({"C01": 4, "C02": 10, "C03": 10})
    s = make_splits(rows, eval_holdout=6, n_held_out_categories=1, seed=0)
    # smallest category (C01) should be the held-out one
    assert s["meta"]["held_out_categories"] == ["C01"]
    c01 = [v for k, v in s["assignments"].items() if k.startswith("C01")]
    assert all(v["held_out_category"] for v in c01)


def test_find_near_duplicates():
    rows = _dataset({"C01": 1})
    en_row = [r for r in rows if r.condition == "EN"][0]
    # add an exact-duplicate of the EN prompt under a new base_id (same condition)
    dup = _mk("C01-999", "C01", "EN", en_row.prompt)
    dups = find_near_duplicates(rows + [dup], threshold=0.9)
    assert any(dup.id in (a, b) for a, b, _ in dups)


def test_dataset_stats_shape():
    rows = attach_features(_dataset({"C01": 2, "C02": 2}))
    st = dataset_stats(rows)
    assert set(st["by_condition"]) == set(CONDITIONS)
    assert st["cmi"]["n"] == len(rows)
