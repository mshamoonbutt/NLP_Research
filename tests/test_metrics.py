"""Tests for higher-level stats (metrics.py) and judge helpers."""
from __future__ import annotations

import pandas as pd
import pytest

from csjail.judge import precision_by_condition, refused_flag
from csjail.metrics import (
    capability_retention,
    gee_logit,
    holm_bonferroni,
    over_refusal_rate,
    relative_reduction,
)


def test_holm_bonferroni_basic():
    res = holm_bonferroni([0.001, 0.04, 0.5], alpha=0.05)
    assert res[0]["reject"] is True          # 0.001 * 3 = 0.003 < 0.05
    assert res[2]["reject"] is False
    # adjusted p-values are monotone in original rank
    adj_sorted = sorted(r["p_adjusted"] for r in res)
    assert adj_sorted == [r["p_adjusted"] for r in sorted(res, key=lambda x: x["p"])]


def test_holm_bonferroni_empty():
    assert holm_bonferroni([]) == []


def test_over_refusal_rate_excludes_none():
    assert over_refusal_rate([1, 0, 1, None]) == 2 / 3


def test_refused_flag_mapping():
    assert refused_flag(0) == 1     # refusal on benign -> over-refusal hit
    assert refused_flag(1) == 0
    assert refused_flag(2) == 0
    assert refused_flag(None) is None


def test_capability_retention_and_relative_reduction():
    assert capability_retention(0.76, 0.80) == 0.95
    assert capability_retention(0.5, 0.0) == 1.0   # guard div-by-zero
    assert relative_reduction(0.40, 0.10) == pytest.approx(0.75)
    assert relative_reduction(0.0, 0.0) == 0.0


def test_precision_by_condition():
    gold = [1, 1, 0, 1, 1, 0]
    pred = [1, 1, 0, 1, 0, 1]  # EN perfect-ish, CS has an FP
    conditions = ["EN", "EN", "EN", "CS", "CS", "CS"]
    out = precision_by_condition(gold, pred, conditions)
    assert set(out["per_condition"]) == {"EN", "CS"}
    assert out["worst_condition"] == "CS"
    assert out["min_precision"] <= out["precision_by_condition"]["EN"]


def test_gee_logit_recovers_condition_effect():
    # Construct data where CS has much higher harmful rate than EN.
    rows = []
    for bid in range(40):
        rows.append({"harmful": 0, "condition": "EN", "base_id": f"b{bid}"})
        rows.append({"harmful": 1 if bid % 5 != 0 else 0,
                     "condition": "CS", "base_id": f"b{bid}"})
    df = pd.DataFrame(rows)
    res = gee_logit(df, baseline_condition="EN")
    cs_term = [t for t in res.terms if "CS" in t][0]
    assert res.odds_ratios[cs_term] > 1.0   # CS raises odds of harmful
    assert res.n_clusters == 40
