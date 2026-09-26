"""Exp 6 — build DPO preference pairs with lineage and leakage protection.

A pair is {prompt, chosen, rejected} plus lineage:
  rejected  the TARGET model's own primary-unsafe response (full or materially
            harmful partial) from a greedy Exp 2 result on an eligible family
  chosen    a generated refusal that passed validate_chosen

Eligibility (enforced here AND again at the training boundary):
  - results must all come from the target model (mixed-model inputs rejected)
  - family in the split's train_pool; not sharing a group with eval/holdout
    families; not in an ablation-excluded domain
  - unified unsafe predicate (csjail.outcomes.primary_unsafe), never a raw
    numeric label

Selection is seeded and domain-aware (round-robin across domains over a
seeded per-domain shuffle), producing ONE ordering whose prefixes are the
nested N-curve subsets. `target_pairs` is a cap, never a quota: pairs are
never duplicated, padded, or borrowed from evaluation data.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Iterable, Optional

import numpy as np

from csjail.outcomes import primary_unsafe
from csjail.splits import assert_trainable, trainable_families
from csjail.utils.io import read_jsonl, write_jsonl

N_CURVE_DEFAULT = (50, 100, 200, "all")
MIN_CHOSEN_CHARS = 20


class PairBuildError(Exception):
    pass


def mine_rejected(results: list[dict], split: dict, *, model: str, condition: str,
                  exclude_domains: Iterable[str] = ()) -> tuple[list[dict], dict]:
    models = {r.get("model") for r in results}
    if models != {model}:
        raise PairBuildError(f"target-specific mining needs results from exactly {model!r}; "
                             f"got {sorted(map(str, models))} (narrow the input)")
    excl = set(exclude_domains)
    eligible = trainable_families(split)
    a = split["assignments"]
    nontrain_groups = {x["group_id"] for x in a.values() if x["split"] != "train_pool"}
    counts: Counter = Counter()
    out = []
    for r in results:
        if r.get("condition") != condition or r.get("arm", "A") != "A":
            continue
        counts["in_condition"] += 1
        f = r["base_id"]
        if int(r.get("sample_index", 0)) != 0 or int((r.get("sampling") or {}).get("n", 1)) != 1:
            counts["excluded_not_greedy_single"] += 1
            continue
        if f not in eligible or a[f]["group_id"] in nontrain_groups:
            counts["excluded_not_train_pool_or_group_relative"] += 1
            continue
        if r.get("domain_id") in excl:
            counts["excluded_domain"] += 1
            continue
        u = primary_unsafe(r)
        if u is None:
            counts["missing_outcome"] += 1
            continue
        if not u:
            counts["not_unsafe"] += 1
            continue
        if not (r.get("response") or "").strip():
            counts["empty_response"] += 1
            continue
        counts["mined"] += 1
        out.append({
            "base_id": f, "domain_id": r["domain_id"], "condition": condition,
            "group_id": a[f]["group_id"], "rejected": r["response"],
            "rejected_lineage": {k: r.get(k) for k in (
                "model", "arm", "gen_key", "response_sha256", "sample_index", "split_id",
                "dataset_version", "judge_fingerprint_id", "judge_compliance_degree", "_source")},
        })
    by_dom = Counter(m["domain_id"] for m in out)
    return out, {"counts": dict(counts), "mined_by_domain": dict(sorted(by_dom.items()))}


def validate_chosen(text: Optional[str], judgment) -> tuple[bool, str]:
    """A usable `chosen` is non-empty, judged (validly) NOT unsafe, and a
    genuine refusal/redirect (response_kind == refusal). Hedged/partial,
    gibberish, empty, irrelevant or merely 'safe_helpful' text is rejected."""
    if text is None or len(text.strip()) < MIN_CHOSEN_CHARS:
        return False, "empty_or_too_short"
    if judgment is None or not judgment.ok:
        return False, "judge_missing"
    if judgment.unsafe:
        return False, "unsafe"
    if judgment.response_kind != "refusal":
        return False, f"not_refusal:{judgment.response_kind}"
    return True, "ok"


def order_pairs(pairs: list[dict], *, seed: int) -> list[dict]:
    """Seeded, domain-aware ordering: shuffle within each domain (sorted
    input), then round-robin across sorted domains. Prefixes are nested."""
    by_dom: dict[str, list[dict]] = defaultdict(list)
    for p in sorted(pairs, key=lambda p: p["base_id"]):
        by_dom[p["domain_id"]].append(p)
    rng = np.random.default_rng(seed)
    queues = {d: [ps[i] for i in rng.permutation(len(ps))] for d, ps in sorted(by_dom.items())}
    out = []
    while any(queues.values()):
        for d in sorted(queues):
            if queues[d]:
                out.append(queues[d].pop(0))
    for i, p in enumerate(out):
        p["order_rank"] = i
    return out


def dedup(pairs: list[dict]) -> tuple[list[dict], int]:
    seen, out = set(), []
    for p in pairs:
        k = (p["prompt"], p["rejected"])
        if k in seen:
            continue
        seen.add(k)
        out.append(p)
    return out, len(pairs) - len(out)


def length_report(pairs: list[dict], tolerance: float) -> dict:
    if not pairs:
        return {"n": 0, "within_tolerance": None}
    ch = float(np.median([len(p["chosen"]) for p in pairs]))
    rj = float(np.median([len(p["rejected"]) for p in pairs]))
    ratio = ch / rj if rj else None
    return {"n": len(pairs), "median_chosen_chars": ch, "median_rejected_chars": rj,
            "ratio": ratio, "tolerance": tolerance,
            "within_tolerance": None if ratio is None else abs(ratio - 1) <= tolerance}


def supported_budgets(n_available: int, requested=N_CURVE_DEFAULT) -> dict:
    fixed = [b for b in requested if b != "all"]
    ok = [b for b in fixed if b <= n_available]
    return {"n_available": n_available, "runnable": ok + (["all"] if n_available else []),
            "not_supported": [b for b in fixed if b > n_available],
            "status": "EXPLORATORY_BELOW_50" if 0 < n_available < min(fixed) else
                      ("NO_PAIRS" if n_available == 0 else "OK")}


def matched_sets(cs_pairs: list[dict], en_pairs: list[dict], *, seed: int
                 ) -> tuple[list[dict], list[dict], dict]:
    """Matched B/C: restrict both to families with a valid pair in BOTH
    languages for this model, same count, same seeded family ordering."""
    cs_f = {p["base_id"]: p for p in cs_pairs}
    en_f = {p["base_id"]: p for p in en_pairs}
    shared = sorted(set(cs_f) & set(en_f))
    ordered = order_pairs([dict(cs_f[f]) for f in shared], seed=seed)
    fam_order = [p["base_id"] for p in ordered]
    en_ordered = []
    for i, f in enumerate(fam_order):
        e = dict(en_f[f]); e["order_rank"] = i
        en_ordered.append(e)
    rep = {"n_cs_valid": len(cs_f), "n_en_valid": len(en_f), "n_matched": len(shared),
           "selection_restriction": "families with valid model-derived pairs in both CS and EN",
           "matched_by_domain": dict(sorted(Counter(p["domain_id"] for p in ordered).items()))}
    return ordered, en_ordered, rep


def write_pairs(path: str, pairs: list[dict]) -> None:
    """Trainer input WITH lineage (the trainer reads prompt/chosen/rejected)."""
    write_jsonl(path, pairs)


def read_pairs(path: str) -> list[dict]:
    return read_jsonl(path)


def assert_pairs_trainable(pairs: list[dict], split: dict, *, model: Optional[str] = None,
                           exclude_domains: Iterable[str] = ()) -> None:
    """Training-boundary re-check of split/group/model/domain compatibility."""
    local = [p for p in pairs if p.get("source") != "external"]
    fams = [p["base_id"] for p in local if p.get("base_id")]
    if len(fams) != len(local):
        raise PairBuildError("benchmark-derived pairs without family lineage cannot be trained on")
    assert_trainable(split, fams)
    excl = set(exclude_domains)
    bad = [p["base_id"] for p in pairs if p.get("domain_id") in excl]
    if bad:
        raise PairBuildError(f"{len(bad)} pairs from an excluded domain, e.g. {bad[:3]}")
    if model is not None:
        wrong = [p["base_id"] for p in pairs
                 if (p.get("rejected_lineage") or {}).get("model") not in (None, model)
                 and p.get("source") != "external"]
        if wrong:
            raise PairBuildError(f"{len(wrong)} pairs mined from another model, e.g. {wrong[:3]}")


def pair_counts(pairs: list[dict]) -> dict:
    return {"n": len(pairs), "by_domain": dict(sorted(Counter(p.get("domain_id")
                                                              for p in pairs).items()))}


def load_external_english(path: str, *, limit: Optional[int] = None) -> list[dict]:
    """Off-the-shelf English preference set (practical baseline B_ext only;
    off-policy, different source -- never a substitute for matched B)."""
    out = []
    for r in read_jsonl(path):
        if all(r.get(k) for k in ("prompt", "chosen", "rejected")):
            out.append({"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"],
                        "source": "external", "base_id": None, "domain_id": None})
        if limit and len(out) >= limit:
            break
    return out


def asdict_safe(obj) -> dict:
    return asdict(obj)
