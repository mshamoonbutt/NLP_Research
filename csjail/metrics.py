"""Higher-level statistics for the extended design.

ASR + cluster-bootstrap CI + McNemar live in `asr.py` (reused as-is). This
module adds:
  - `gee_logit`: clustered logistic regression (omnibus condition effect),
  - `holm_bonferroni`: multiple-comparison correction across contrasts,
  - `over_refusal_rate` / `capability_retention`: Phase-2 acceptance metrics.

statsmodels is required (already a project dep). matplotlib is NOT imported
here; plotting lives in the experiment scripts.
"""
from __future__ import annotations

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
    summary_text: str


def gee_logit(
    df,
    *,
    outcome: str = "harmful",
    baseline_condition: str = "EN",
    cluster: str = "base_id",
    formula: Optional[str] = None,
):
    """Clustered logistic regression of a binary outcome on condition (+ optional
    category, model), clustered on base prompt via GEE with an exchangeable
    working correlation. `EN` is the reference level by default.

    `df` is a pandas DataFrame with columns: outcome (0/1), 'condition',
    optionally 'category' and 'model', and the cluster column. Returns GEEResult
    with exp(params) as odds ratios.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    cols = set(df.columns)
    if formula is None:
        rhs = [f"C(condition, Treatment('{baseline_condition}'))"]
        if "category" in cols:
            rhs.append("C(category)")
        if "model" in cols and df["model"].nunique() > 1:
            rhs.append("C(model)")
        formula = f"{outcome} ~ " + " + ".join(rhs)

    model = smf.gee(
        formula,
        groups=cluster,
        data=df,
        family=sm.families.Binomial(),
        cov_struct=sm.cov_struct.Exchangeable(),
    )
    fit = model.fit()
    params = fit.params
    ci = fit.conf_int()
    ors = {k: float(np.exp(v)) for k, v in params.items()}
    cis = {k: (float(np.exp(ci.loc[k, 0])), float(np.exp(ci.loc[k, 1])))
           for k in params.index}
    pvals = {k: float(fit.pvalues[k]) for k in params.index}
    return GEEResult(
        terms=list(params.index),
        odds_ratios=ors,
        conf_int=cis,
        pvalues=pvals,
        n_obs=int(fit.nobs),
        n_clusters=int(df[cluster].nunique()),
        summary_text=str(fit.summary()),
    )


def holm_bonferroni(
    pvalues: Sequence[float], *, alpha: float = 0.05
) -> list[dict]:
    """Holm-Bonferroni step-down correction.

    Returns a list (aligned to input order) of
    {index, p, p_adjusted, reject}. p_adjusted is the standard step-down
    adjusted p-value (monotone-enforced, capped at 1.0).
    """
    m = len(pvalues)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, i in enumerate(order):
        adj = (m - rank) * pvalues[i]
        adj = min(adj, 1.0)
        running_max = max(running_max, adj)  # enforce monotonicity
        adjusted[i] = running_max
    return [
        {
            "index": i,
            "p": float(pvalues[i]),
            "p_adjusted": float(adjusted[i]),
            "reject": bool(adjusted[i] < alpha),
        }
        for i in range(m)
    ]


def over_refusal_rate(refused_flags: Sequence[Optional[int]]) -> float:
    """ORR = fraction of benign probe prompts wrongly refused.

    `refused_flags`: 1 = refused (bad), 0 = complied (good), None = parse fail
    (excluded from denominator).
    """
    kept = [f for f in refused_flags if f is not None]
    if not kept:
        return 0.0
    return sum(1 for f in kept if f == 1) / len(kept)


def capability_retention(post_accuracy: float, pre_accuracy: float) -> float:
    """post/pre accuracy ratio; 1.0 if pre == 0 to avoid div-by-zero."""
    if pre_accuracy <= 0:
        return 1.0
    return post_accuracy / pre_accuracy


def relative_reduction(asr_before: float, asr_after: float) -> float:
    """(before - after) / before; 0.0 if before == 0."""
    if asr_before <= 0:
        return 0.0
    return (asr_before - asr_after) / asr_before
