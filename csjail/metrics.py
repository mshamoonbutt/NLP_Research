"""Higher-level statistics: GEE, Holm, Phase-2 metrics, Wilson intervals.

All functions are null-safe: undefined quantities return None (NA), never a
flattering default (no ORR 0 for an all-missing probe, no capability
retention 1.0 at zero baseline accuracy, no relative reduction at zero
baseline ASR).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass
class GEEResult:
    terms: list[str]
    odds_ratios: dict[str, float]
    conf_int: dict[str, tuple[float, float]]
    pvalues: dict[str, float]
    n_obs: int
    n_clusters: int
    n_dropped_missing: int
    formula: str
    summary_text: str


def gee_logit(df, *, outcome: str = "unsafe", baseline_condition: str = "EN",
              cluster: str = "base_id", formula: Optional[str] = None) -> GEEResult:
    """Clustered logistic regression (GEE, exchangeable) of the binary outcome
    on condition (+ domain, + model when >1), clustered on family. Rows with a
    missing outcome are DROPPED and counted -- never coerced to 0."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    n0 = len(df)
    df = df[df[outcome].notna()].copy()
    df[outcome] = df[outcome].astype(int)
    cols = set(df.columns)
    if formula is None:
        rhs = [f"C(condition, Treatment('{baseline_condition}'))"]
        for c in ("domain", "category"):
            if c in cols and df[c].nunique() > 1:
                rhs.append(f"C({c})")
                break
        if "model" in cols and df["model"].nunique() > 1:
            rhs.append("C(model)")
        formula = f"{outcome} ~ " + " + ".join(rhs)
    fit = smf.gee(formula, groups=cluster, data=df, family=sm.families.Binomial(),
                  cov_struct=sm.cov_struct.Exchangeable()).fit()
    ci = fit.conf_int()
    return GEEResult(
        terms=list(fit.params.index),
        odds_ratios={k: float(np.exp(v)) for k, v in fit.params.items()},
        conf_int={k: (float(np.exp(ci.loc[k, 0])), float(np.exp(ci.loc[k, 1])))
                  for k in fit.params.index},
        pvalues={k: float(fit.pvalues[k]) for k in fit.params.index},
        n_obs=int(fit.nobs), n_clusters=int(df[cluster].nunique()),
        n_dropped_missing=n0 - len(df), formula=formula, summary_text=str(fit.summary()),
    )


def holm_bonferroni(pvalues: Sequence[Optional[float]], *, alpha: float = 0.05) -> list[dict]:
    """Holm step-down over the declared family. A None p-value (untestable
    contrast) stays None and is excluded from m."""
    idx = [i for i, p in enumerate(pvalues) if p is not None]
    m = len(idx)
    adjusted: dict[int, float] = {}
    running = 0.0
    for rank, i in enumerate(sorted(idx, key=lambda i: pvalues[i])):
        running = max(running, min((m - rank) * pvalues[i], 1.0))
        adjusted[i] = running
    return [{"index": i, "p": pvalues[i], "p_adjusted": adjusted.get(i),
             "reject": (adjusted[i] < alpha) if i in adjusted else None,
             "family_size": m} for i in range(len(pvalues))]


def over_refusal_rate(refused: Sequence[Optional[bool]]) -> dict:
    """ORR over scored benign items; None when nothing was scored."""
    kept = [r for r in refused if r is not None]
    return {"orr": (sum(kept) / len(kept)) if kept else None, "n_planned": len(refused),
            "n_scored": len(kept), "n_missing": len(refused) - len(kept)}


def capability_retention(post: Optional[float], pre: Optional[float]) -> Optional[float]:
    if post is None or pre is None or pre <= 0:
        return None
    return post / pre


def relative_reduction(before: Optional[float], after: Optional[float]) -> Optional[float]:
    """(before - after) / before; None (NA) when before is 0 or missing --
    report absolute_pp_difference instead."""
    if before is None or after is None or before <= 0:
        return None
    return (before - after) / before


def absolute_pp_difference(before: Optional[float], after: Optional[float]) -> Optional[float]:
    if before is None or after is None:
        return None
    return (after - before) * 100.0


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[Optional[float], Optional[float]]:
    if n == 0:
        return None, None
    from scipy.stats import norm

    z = float(norm.ppf(1 - alpha / 2))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)
