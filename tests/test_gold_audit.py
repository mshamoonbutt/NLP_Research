"""Blinded gold audit: disputed/control selection and the declared scoring rule."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def load():
    spec = importlib.util.spec_from_file_location("exp1_gold_audit",
                                                  ROOT / "scripts" / "exp1_gold_audit.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["exp1_gold_audit"] = mod
    spec.loader.exec_module(mod)
    return mod


def synthetic():
    items, gold, preds = [], {}, {"a": {}, "b": {}, "c": {}}
    for n in range(60):
        iid = f"i{n:02d}"
        items.append({"item_id": iid, "model": ["m1", "m2"][n % 2], "condition": ["EN", "RU"][n % 4 // 2]})
        g = n % 5 == 0
        gold[iid] = {"value": g, "response_kind": "harmful_compliance" if g else "refusal"}
        if n < 6:            # disputed: every judge opposite to gold
            for r in preds:
                preds[r][iid] = not g
        elif n < 10:         # mixed: neither disputed nor concordant
            preds["a"][iid], preds["b"][iid], preds["c"][iid] = g, not g, g
        else:                # concordant
            for r in preds:
                preds[r][iid] = g
    return items, gold, preds


def test_selection_is_matched_disjoint_and_deterministic():
    ga = load()
    items, gold, preds = synthetic()
    d, c = ga.select_audit(items, gold, preds, seed=1, per_item=2)
    assert [x["item_id"] for x in d] == [f"i{n:02d}" for n in range(6)]
    assert len(c) == 12 and len({x["item_id"] for x in c}) == 12
    assert not {x["item_id"] for x in c} & {f"i{n:02d}" for n in range(10)}   # concordant only
    key = lambda x: (x["model"], x["condition"], gold[x["item_id"]]["value"])  # noqa: E731
    from collections import Counter
    assert Counter(key(x) for x in c) == Counter(k for x in d for k in [key(x)] * 2)
    d2, c2 = ga.select_audit(items, gold, preds, seed=1, per_item=2)
    assert [x["item_id"] for x in c2] == [x["item_id"] for x in c]


def key_and_review(flips_disputed, flips_control, n=10):
    key, rev = [], []
    for group, flips in (("disputed", flips_disputed), ("control", flips_control)):
        for j in range(n):
            aid = f"{group[0]}{j}"
            g = j % 2 == 0
            key.append({"audit_id": aid, "group": group, "gold_unsafe": str(g).lower(),
                        "judges_unsafe": str(not g if group == "disputed" else g).lower()})
            v = (not g) if j < flips else g
            rev.append({"audit_id": aid, "unsafe": str(v).lower(),
                        "response_kind": "harmful_compliance" if v else "refusal"})
    return key, rev


def test_score_applies_the_declared_rule():
    ga = load()
    res = ga.score(*key_and_review(8, 0))
    assert res["disputed"]["flip_rate"] == 0.8 and res["control"]["flip_rate"] == 0.0
    assert res["disputed"]["reviewer_agrees_with_judges"] == 8
    assert res["verdict"] == "gold_has_systematic_error"
    assert ga.score(*key_and_review(2, 0))["verdict"] == "gold_stands"
    assert ga.score(*key_and_review(4, 1))["verdict"] == "inconclusive"


def test_select_config_uses_corrected_gold_and_the_declared_rule():
    ga = load()
    items, gold = [], {}
    for n in range(80):
        iid = f"i{n:02d}"
        items.append({"item_id": iid, "condition": ["EN", "CS", "RU", "UR"][n % 4]})
        gold[iid] = {"value": n < 20}                          # 5 unsafe per condition
    # v2 also flags 20..27 (2 per condition), which the reviewer says ARE unsafe;
    # v4 flags only the original 20; v3 flags 20..27 + 28..35 (8 true FPs).
    preds = {"gpt-4o harm-v2": {f"i{n:02d}": n < 28 for n in range(80)},
             "gpt-4o harm-v3": {f"i{n:02d}": n < 36 for n in range(80)},
             "gpt-4o harm-v4": {f"i{n:02d}": n < 20 for n in range(80)}}
    corrected = {f"i{n:02d}": True for n in range(20, 28)}
    res = ga.select_config(items, gold, preds, corrected)
    t = res["table"]
    assert t["gpt-4o harm-v2"]["precision"] == 1.0 and t["gpt-4o harm-v2"]["recall"] == 1.0
    assert t["gpt-4o harm-v4"]["recall"] < 0.80                 # misses the corrected 8
    assert res["chosen"] == "gpt-4o harm-v2" and res["n_labels_changed"] == 8
    assert not res["flag_gate_unlikely"]


def test_incomplete_review_is_refused():
    ga = load()
    key, rev = key_and_review(5, 0)
    with pytest.raises(ga.ValidationInputError):
        ga.score(key, rev[:-1])
    rev[0]["response_kind"] = ""
    with pytest.raises(ga.ValidationInputError):
        ga.score(key, rev)
