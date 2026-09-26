"""Null-safe ASR, bounds, paired tests, GEE, Phase-2 metrics (check 8)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.contingency_tables import mcnemar

from csjail.asr import behavior_rates, compute_asr, family_map, macro_asr, mcnemar_paired
from csjail.metrics import (
    absolute_pp_difference, capability_retention, gee_logit, holm_bonferroni,
    over_refusal_rate, relative_reduction, wilson_ci,
)


def test_asr_counts_and_bounds():
    out = [True, False, None, True, False]
    r = compute_asr(out, list("abcde"), bootstrap_n=200)
    assert (r.n_planned, r.n_scored, r.n_missing, r.n_unsafe) == (5, 4, 1, 2)
    assert r.asr == pytest.approx(0.5)
    assert r.bound_lo == pytest.approx(2 / 5) and r.bound_hi == pytest.approx(3 / 5)


def test_all_missing_is_na_not_zero():
    r = compute_asr([None, None], ["a", "b"], bootstrap_n=50)
    assert r.asr is None and r.ci_lo is None and r.n_missing == 2
    assert over_refusal_rate([None, None])["orr"] is None


def test_bootstrap_seeded_and_stratified():
    rng = np.random.default_rng(0)
    out = list(rng.random(120) < 0.3)
    fams = [f"f{i}" for i in range(120)]
    doms = [f"D{i % 6 + 1}" for i in range(120)]
    a = compute_asr(out, fams, strata=doms, bootstrap_n=300, seed=1)
    b = compute_asr(out, fams, strata=doms, bootstrap_n=300, seed=1)
    assert (a.ci_lo, a.ci_hi) == (b.ci_lo, b.ci_hi) and a.ci_lo < a.asr < a.ci_hi


def test_macro_asr_equal_weights():
    out = [True] * 1 + [False] * 9 + [True] * 1          # D1: 1/10, D2: 1/1
    doms = ["D1"] * 10 + ["D2"]
    m = macro_asr(out, [f"f{i}" for i in range(11)], doms, bootstrap_n=100)
    assert m["macro_asr"] == pytest.approx((0.1 + 1.0) / 2)


def test_mcnemar_matches_statsmodels_and_counts_missing():
    a = {f"f{i}": v for i, v in enumerate([True] * 8 + [False] * 10 + [True, None])}
    b = {f"f{i}": v for i, v in enumerate([False] * 6 + [True] * 2 + [False] * 8 + [True] * 2
                                          + [True, True])}
    m = mcnemar_paired(a, b, cond_a="CS", cond_b="RU", bootstrap_n=200)
    assert m.n_shared == 20 and m.n_complete == 19 and m.n_missing_pairs == 1
    ref = mcnemar(np.array([[m.both_unsafe, m.b], [m.c, m.both_safe]]), exact=True)
    assert m.pvalue == pytest.approx(ref.pvalue)
    assert m.diff == pytest.approx(m.asr_a - m.asr_b)
    assert m.diff_ci_lo <= m.diff <= m.diff_ci_hi


def test_duplicate_family_rejected():
    with pytest.raises(ValueError, match="duplicate family"):
        family_map([("f1", True), ("f1", False)])


def test_gee_drops_missing_never_coerces():
    rng = np.random.default_rng(3)
    rows = []
    for i in range(150):
        for cond, p in (("EN", 0.1), ("UR", 0.4)):
            rows.append({"base_id": f"f{i}", "condition": cond, "domain": f"D{i % 3}",
                         "unsafe": float(rng.random() < p)})
    for r in rows[:10]:
        r["unsafe"] = np.nan
    g = gee_logit(pd.DataFrame(rows), outcome="unsafe", baseline_condition="EN")
    assert g.n_dropped_missing == 10 and g.n_obs == len(rows) - 10
    assert g.odds_ratios["C(condition, Treatment('EN'))[T.UR]"] > 1


def test_phase2_metrics_na_rules():
    assert capability_retention(0.5, 0.0) is None
    assert capability_retention(None, 0.6) is None
    assert capability_retention(0.57, 0.6) == pytest.approx(0.95)
    assert relative_reduction(0.0, 0.0) is None
    assert relative_reduction(0.4, 0.1) == pytest.approx(0.75)
    assert absolute_pp_difference(0.4, 0.1) == pytest.approx(-30.0)


def test_holm_with_untestable_contrast():
    h = holm_bonferroni([0.01, None, 0.04])
    assert h[1]["p_adjusted"] is None and h[0]["family_size"] == 2
    assert h[0]["p_adjusted"] == pytest.approx(0.02) and h[2]["p_adjusted"] == pytest.approx(0.04)


def test_wilson_and_behavior_rates():
    lo, hi = wilson_ci(9, 10)
    assert 0.5 < lo < 0.9 < hi <= 1.0 and wilson_ci(0, 0) == (None, None)
    br = behavior_rates(["refusal", "refusal", "missing", "unintelligible"])
    assert br["rates"]["missing"] == 0.25 and br["denominator"] == "n_planned"
