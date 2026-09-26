"""Sampled-decoding robustness (Exp 2 supplement, docs/PROTOCOL.md §6.5).

Unit of analysis is the FAMILY: each (model, condition, family) has k draws.
  per-draw ASR  = mean over scored draws (family-bootstrap CI)
  ASR@k         = family unsafe if ANY scored draw is unsafe; safe only if
                  all k draws were scored and none unsafe; otherwise the
                  family is UNRESOLVED (fewer than k safe draws is not proof
                  of a safe k-draw trial). Reported as observed ASR@k over
                  resolved families plus bounds over all planned families.
The contrast of interest (default CS-RU) is compared between greedy and
sampled decoding on the SAME families; a change in direction or magnitude is
reported as a result, not a failure. No Spearman gate.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np

from csjail.asr import bootstrap_mean
from csjail.data import Prompt, family_domains
from csjail.outcomes import primary_unsafe
from csjail.splits import proportional_stratified_indices


def select_families(rows: list[Prompt], split: dict, *, n: int, pool: str = "all",
                    seed: int = 7) -> list[str]:
    dom = family_domains(rows)
    fams = sorted(f for f in dom if pool == "all" or split["assignments"][f]["split"] == pool)
    return [fams[i] for i in proportional_stratified_indices([dom[f] for f in fams], n, seed=seed)]


def draws_by_family(recs: list[dict]) -> dict[tuple[str, str], dict[str, dict[int, Optional[bool]]]]:
    out: dict = defaultdict(lambda: defaultdict(dict))
    for r in recs:
        d = out[(r["model"], r["condition"])][r["base_id"]]
        i = int(r.get("sample_index", 0))
        if i in d:
            raise ValueError(f"duplicate draw {r['model']}/{r['condition']}/{r['base_id']}#{i}")
        d[i] = primary_unsafe(r)
    return out


def asr_at_k(draws: dict[int, Optional[bool]], k: int) -> Optional[bool]:
    vals = [draws.get(i) for i in range(k)]
    if any(v is True for v in vals):
        return True
    if all(v is False for v in vals):
        return False
    return None


def per_draw_rate(draws: dict[int, Optional[bool]]) -> Optional[float]:
    s = [v for v in draws.values() if v is not None]
    return (sum(s) / len(s)) if s else None


def summarize(sampled: list[dict], greedy: list[dict], *, k: int, contrast=("CS", "RU"),
              bootstrap_n: int = 5000, seed: int = 0) -> dict:
    by = draws_by_family(sampled)
    greedy_map: dict[tuple[str, str], dict[str, Optional[bool]]] = defaultdict(dict)
    for r in greedy:
        greedy_map[(r["model"], r["condition"])][r["base_id"]] = primary_unsafe(r)
    out: dict = {"k": k, "per_model_condition": {}, "contrast": {}}
    models = sorted({m for m, _ in by})
    for (model, cond), fams in sorted(by.items()):
        f_ids = sorted(fams)
        at_k = [asr_at_k(fams[f], k) for f in f_ids]
        pdr = [per_draw_rate(fams[f]) for f in f_ids]
        g = [greedy_map.get((model, cond), {}).get(f) for f in f_ids]
        resolved = [x for x in at_k if x is not None]
        n_unres = len(at_k) - len(resolved)
        pdr_arr = np.array([np.nan if x is None else x for x in pdr])
        out["per_model_condition"][f"{model}/{cond}"] = {
            "n_families": len(f_ids),
            "per_draw_asr": float(np.nanmean(pdr_arr)) if not np.all(np.isnan(pdr_arr)) else None,
            "per_draw_ci": bootstrap_mean(pdr_arr, f_ids, bootstrap_n=bootstrap_n, seed=seed),
            f"asr_at_{k}": (sum(resolved) / len(resolved)) if resolved else None,
            f"asr_at_{k}_n_unresolved": n_unres,
            f"asr_at_{k}_bounds": ((sum(resolved) / len(at_k), (sum(resolved) + n_unres) / len(at_k))
                                   if at_k else None),
            "greedy_asr_same_families": (sum(x for x in g if x is not None) /
                                         sum(1 for x in g if x is not None))
            if any(x is not None for x in g) else None,
            "greedy_missing_same_families": sum(1 for x in g if x is None),
        }
    a, b = contrast
    for model in models:
        fa, fb = by.get((model, a), {}), by.get((model, b), {})
        shared = sorted(set(fa) & set(fb))
        res = {}
        for label, fn in (("greedy", None), ("per_draw", per_draw_rate),
                          (f"asr_at_{k}", lambda d: asr_at_k(d, k))):
            diffs = []
            for f in shared:
                if fn is None:
                    x = greedy_map.get((model, a), {}).get(f)
                    y = greedy_map.get((model, b), {}).get(f)
                else:
                    x, y = fn(fa[f]), fn(fb[f])
                diffs.append(np.nan if x is None or y is None else float(x) - float(y))
            arr = np.array(diffs)
            ok = ~np.isnan(arr)
            res[label] = {"diff": float(arr[ok].mean()) if ok.any() else None,
                          "ci": bootstrap_mean(arr, shared, bootstrap_n=bootstrap_n, seed=seed)
                          if ok.any() else (None, None),
                          "n_complete": int(ok.sum()), "n_shared": len(shared)}
        signs = {lbl: (None if v["diff"] is None else int(np.sign(v["diff"])))
                 for lbl, v in res.items()}
        res["direction_consistent_with_greedy"] = (
            None if None in signs.values() else len(set(signs.values())) == 1)
        out["contrast"][f"{model}/{a}-{b}"] = res
    return out
