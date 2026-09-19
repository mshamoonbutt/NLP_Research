"""Exp 0 — finalize and split the dataset.

Pure-Python / CPU only. Given loaded `Prompt` rows this module:
  1. attaches `urdu_word_ratio` / `cmi` where missing (via `features.py`),
  2. checks condition parity (each base_id has all 4 active conditions),
  3. flags near-duplicate prompts within a condition (token-Jaccard),
  4. builds the fixed train/eval split with a stratified held-out eval set and
     `n` held-out categories, writable to `data/splits.json`.

The split is deterministic given `seed`, so re-running reproduces it exactly.
No embeddings/model needed — near-dup uses a lightweight token-Jaccard so it
runs anywhere; swap in sentence-embedding cosine later if desired.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from csjail.data import CONDITIONS, Prompt, pairing_coverage
from csjail.features import code_mix_stats


def attach_features(rows: list[Prompt]) -> list[Prompt]:
    """Return rows with urdu_word_ratio / cmi filled in where they were None."""
    out: list[Prompt] = []
    for r in rows:
        if r.urdu_word_ratio is None or r.cmi is None:
            st = code_mix_stats(r.prompt)
            out.append(r.model_copy(update={
                "urdu_word_ratio": r.urdu_word_ratio
                if r.urdu_word_ratio is not None else st.urdu_word_ratio,
                "cmi": r.cmi if r.cmi is not None else st.cmi,
            }))
        else:
            out.append(r)
    return out


@dataclass
class ParityReport:
    n_base_ids: int
    n_complete: int              # base_ids with all 4 active conditions
    parity_rate: float
    incomplete: list[str] = field(default_factory=list)


def condition_parity(rows: list[Prompt]) -> ParityReport:
    cov = pairing_coverage(rows)
    complete, incomplete = [], []
    for bid, counts in cov.items():
        if all(counts[c] >= 1 for c in CONDITIONS):
            complete.append(bid)
        else:
            incomplete.append(bid)
    n = len(cov)
    return ParityReport(
        n_base_ids=n,
        n_complete=len(complete),
        parity_rate=(len(complete) / n) if n else 0.0,
        incomplete=sorted(incomplete)[:50],
    )


def _tokens(text: str) -> set[str]:
    return set(text.lower().split())


def find_near_duplicates(
    rows: list[Prompt], *, threshold: float = 0.92
) -> list[tuple[str, str, float]]:
    """Flag near-duplicate prompts *within the same condition* by token-Jaccard.

    Returns (id_a, id_b, jaccard) for pairs with jaccard >= threshold. O(n^2)
    per condition — fine for a few thousand rows; upgrade to embeddings/LSH if
    it becomes a bottleneck.
    """
    by_cond: dict[str, list[Prompt]] = defaultdict(list)
    for r in rows:
        by_cond[r.condition].append(r)
    dups: list[tuple[str, str, float]] = []
    for _cond, group in by_cond.items():
        toks = [(_tokens(r.prompt), r.id) for r in group]
        for i in range(len(toks)):
            ti, idi = toks[i]
            if not ti:
                continue
            for j in range(i + 1, len(toks)):
                tj, idj = toks[j]
                if not tj:
                    continue
                inter = len(ti & tj)
                if inter == 0:
                    continue
                jac = inter / len(ti | tj)
                if jac >= threshold:
                    dups.append((idi, idj, round(jac, 3)))
    return dups


def _base_id_category(rows: list[Prompt]) -> dict[str, str]:
    """Map base_id -> harm_category (asserting consistency across conditions)."""
    out: dict[str, str] = {}
    for r in rows:
        prev = out.get(r.base_id)
        if prev is not None and prev != r.harm_category:
            raise ValueError(
                f"base_id {r.base_id} has inconsistent harm_category "
                f"({prev} vs {r.harm_category})"
            )
        out[r.base_id] = r.harm_category
    return out


def make_splits(
    rows: list[Prompt],
    *,
    eval_holdout: int = 200,
    n_held_out_categories: int = 2,
    seed: int = 42,
) -> dict:
    """Build the deterministic split.

    - Reserve `eval_holdout` base_ids as the held-out EVAL set, stratified by
      harm_category (proportional allocation, seeded).
    - Mark `n_held_out_categories` categories as held-out-from-training (chosen
      deterministically: the smallest categories, tie-broken by name, so the
      training pool keeps the most data).
    Returns {"assignments": {base_id: {"split","held_out_category"}}, "meta": {...}}.
    """
    cat_of = _base_id_category(rows)
    base_ids = sorted(cat_of)
    n_base = len(base_ids)
    if eval_holdout > n_base:
        raise ValueError(f"eval_holdout {eval_holdout} > n_base_ids {n_base}")

    by_cat: dict[str, list[str]] = defaultdict(list)
    for bid in base_ids:
        by_cat[cat_of[bid]].append(bid)
    categories = sorted(by_cat)

    rng = np.random.default_rng(seed)

    # Stratified eval sampling: proportional per category, then top up/trim to
    # hit the exact eval_holdout total.
    eval_ids: set[str] = set()
    per_cat_quota = {
        cat: int(round(eval_holdout * len(ids) / n_base))
        for cat, ids in by_cat.items()
    }
    for cat in categories:
        ids = by_cat[cat][:]
        rng.shuffle(ids)
        take = min(per_cat_quota[cat], len(ids))
        eval_ids.update(ids[:take])

    # Exact-size correction.
    all_shuffled = base_ids[:]
    rng.shuffle(all_shuffled)
    i = 0
    while len(eval_ids) < eval_holdout and i < len(all_shuffled):
        eval_ids.add(all_shuffled[i]); i += 1
    if len(eval_ids) > eval_holdout:
        drop = list(eval_ids)
        rng.shuffle(drop)
        for bid in drop[: len(eval_ids) - eval_holdout]:
            eval_ids.discard(bid)

    # Held-out categories: smallest first (preserve training data).
    cat_sizes = sorted(categories, key=lambda c: (len(by_cat[c]), c))
    held_out_cats = set(cat_sizes[:n_held_out_categories])

    assignments: dict[str, dict] = {}
    for bid in base_ids:
        assignments[bid] = {
            "split": "eval" if bid in eval_ids else "train",
            "held_out_category": cat_of[bid] in held_out_cats,
        }

    meta = {
        "seed": seed,
        "n_base_ids": n_base,
        "eval_holdout_requested": eval_holdout,
        "eval_holdout_actual": len(eval_ids),
        "held_out_categories": sorted(held_out_cats),
        "categories": categories,
        "eval_by_category": dict(Counter(cat_of[b] for b in eval_ids)),
    }
    return {"assignments": assignments, "meta": meta}


def proportional_stratified_indices(
    keys: list[str], n: int, *, seed: int = 42,
) -> list[int]:
    """Indices of a proportional-by-key stratified sample of ~n items,
    deterministic given `seed`. `keys` is a parallel array (e.g. harm_category
    per item); works on indices (always hashable) so it's agnostic to what
    the caller's items actually are (Prompt rows, base_id strings, ...).

    Shared by Exp 1's per-condition gold-set sampler and Exp 4b's
    comprehension-control subset -- both need "pick ~n items, proportional
    to each harm_category's share, seeded". `make_splits`'s own eval-holdout
    quota uses the same algorithm inline rather than this helper, since it
    was already committed with a specific RNG call sequence and re-pointing
    it here would silently change which base_ids land in eval for the same
    seed.
    """
    n_total = len(keys)
    if n_total == 0 or n <= 0:
        return []
    n = min(n, n_total)
    by_key: dict[str, list[int]] = defaultdict(list)
    for i, k in enumerate(keys):
        by_key[k].append(i)

    rng = np.random.default_rng(seed)
    picked: set[int] = set()
    for k in sorted(by_key):
        idx = by_key[k][:]
        rng.shuffle(idx)
        quota = int(round(n * len(idx) / n_total))
        picked.update(idx[:quota])

    # Exact-size correction against a fully shuffled index pool.
    all_idx = list(range(n_total))
    rng.shuffle(all_idx)
    i = 0
    while len(picked) < n and i < len(all_idx):
        picked.add(all_idx[i])
        i += 1
    if len(picked) > n:
        drop = list(picked)
        rng.shuffle(drop)
        for idx in drop[: len(picked) - n]:
            picked.discard(idx)
    return sorted(picked)


def cohens_kappa(rater_a: list[int], rater_b: list[int]) -> float:
    """Unweighted Cohen's kappa for two raters' categorical/ordinal scores."""
    assert len(rater_a) == len(rater_b)
    n = len(rater_a)
    if n == 0:
        return 0.0
    categories = sorted(set(rater_a) | set(rater_b))
    idx = {c: i for i, c in enumerate(categories)}
    k = len(categories)
    confusion = [[0] * k for _ in range(k)]
    for a, b in zip(rater_a, rater_b, strict=True):
        confusion[idx[a]][idx[b]] += 1
    po = sum(confusion[i][i] for i in range(k)) / n
    row_marg = [sum(row) / n for row in confusion]
    col_marg = [sum(confusion[i][j] for i in range(k)) / n for j in range(k)]
    pe = sum(row_marg[i] * col_marg[i] for i in range(k))
    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


@dataclass
class IAAReport:
    n: int
    kappa_authenticity: float
    kappa_harm_severity: float
    min_kappa: float
    gate_threshold: float
    passed: bool


def inter_annotator_agreement(
    csv_path: str | Path, *, threshold: float = 0.70,
) -> IAAReport:
    """Cohen's kappa between the two human reviewers, computed from the raw
    wide annotation CSV (authenticity_score{1,2}, harm_severity_score{1,2}).

    Must read the ORIGINAL annotation CSV, not the long-format pipeline
    dataset: `csjail.convert_csv` averages the two reviewers' raw scores into
    a single `cs_authenticity` / `harm_severity` field before rows enter the
    pipeline, so the per-reviewer scores needed for kappa don't survive past
    that conversion.
    """
    import csv as _csv

    with Path(csv_path).open("r", encoding="utf-8", newline="") as f:
        rows = list(_csv.DictReader(f))
    auth1 = [int(r["authenticity_score1"]) for r in rows]
    auth2 = [int(r["authenticity_score2"]) for r in rows]
    sev1 = [int(r["harm_severity_score1"]) for r in rows]
    sev2 = [int(r["harm_severity_score2"]) for r in rows]
    k_auth = cohens_kappa(auth1, auth2)
    k_sev = cohens_kappa(sev1, sev2)
    min_k = min(k_auth, k_sev)
    return IAAReport(
        n=len(rows),
        kappa_authenticity=round(k_auth, 4),
        kappa_harm_severity=round(k_sev, 4),
        min_kappa=round(min_k, 4),
        gate_threshold=threshold,
        passed=min_k >= threshold,
    )


def dataset_stats(rows: list[Prompt]) -> dict:
    """Descriptive Table-1 stats over the (feature-attached) rows."""
    lengths = [len(r.prompt.split()) for r in rows]
    ratios = [r.urdu_word_ratio for r in rows if r.urdu_word_ratio is not None]
    cmis = [r.cmi for r in rows if r.cmi is not None]

    def _summ(xs: list[float]) -> dict:
        if not xs:
            return {"n": 0}
        a = np.array(xs, dtype=float)
        return {
            "n": int(a.size), "mean": round(float(a.mean()), 3),
            "min": round(float(a.min()), 3), "max": round(float(a.max()), 3),
        }

    return {
        "by_condition": dict(Counter(r.condition for r in rows)),
        "by_harm_category": dict(Counter(r.harm_category for r in rows)),
        "word_len": _summ([float(x) for x in lengths]),
        "urdu_word_ratio": _summ(ratios),
        "cmi": _summ(cmis),
    }
